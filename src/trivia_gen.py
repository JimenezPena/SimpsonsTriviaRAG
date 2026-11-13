"""Simpsons Trivia Generator (RAG) — v2.

Modes:
  1) Trivia question: samples a random chunk from the vector DB and the LLM
     generates a question with 4 options and its answer (revealed after you answer).
  2) Free-form Q&A: you ask a question, the most similar chunks are retrieved
     and the LLM answers citing the episode and season.

CLI options (set at startup):
    python src/trivia_gen.py                    # normal mode
    python src/trivia_gen.py --verbose         # shows prompts and retrieved chunks
    python src/trivia_gen.py --verbose --topN=5
    python src/trivia_gen.py --topN=8          # retrieve top-8, send top-3 to the LLM

--topN widens RETRIEVAL for inspection purposes: the LLM still receives only
the TOP_K best chunks; --verbose shows the remaining topN - TOP_K chunks that
were retrieved but NOT sent.

Zero ingestion cost: the DB is already persisted; only 1 embedding per query.
"""
import os
import json
import time
import random
import sqlite3
import argparse
import urllib.request
import urllib.error
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

# ==================== CONFIGURATION ====================
# the DB created by ingest_batch.py; anchored to the project root so it does not
# depend on the working directory (gunicorn / Render start from anywhere)
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "batch_vector_db")
COLLECTION_NAME = "langchain"
LLM_MODEL = "gemini-3.5-flash-lite"
LLM_TIMEOUT = 25                 # seconds per LLM HTTP call (must stay well below gunicorn --timeout)
MAX_SEASON = 11
TOP_K = 3                        # chunks SENT to the LLM in free-form Q&A
# =======================================================

load_dotenv()
API_KEY = os.environ["GOOGLE_API_KEY"]


# ----------------------------------------------------------------------------
# Embeddings (same class as ingest.py: ideally a shared copy)
# ----------------------------------------------------------------------------
class BatchedGeminiEmbeddings(Embeddings):
    API_URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
               "{model}:batchEmbedContents")

    def __init__(self, model="gemini-embedding-001", batch_size=100):
        self.api_key = os.environ["GOOGLE_API_KEY"]
        self.model = model
        self.batch_size = batch_size

    def _batch_embed(self, texts, task_type):
        payload = {
            "requests": [
                {
                    "model": f"models/{self.model}",
                    "content": {"parts": [{"text": t}]},
                    "taskType": task_type,
                }
                for t in texts
            ]
        }
        url = self.API_URL.format(model=self.model) + f"?key={self.api_key}"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                embeddings = [e["values"] for e in data.get("embeddings", [])]
                if len(embeddings) != len(texts):
                    raise RuntimeError(f"Expected {len(texts)} embeddings, "
                                       f"got {len(embeddings)}")
                return embeddings
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 4:
                    time.sleep(min(60 * (attempt + 1), 90))
                    continue
                raise RuntimeError(f"HTTP {e.code}: "
                                   f"{e.read().decode('utf-8', 'ignore')}") from e
        raise RuntimeError("Retries exhausted")

    def embed_documents(self, texts):
        return self._batch_embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text):
        return self._batch_embed([text], "RETRIEVAL_QUERY")[0]


# ----------------------------------------------------------------------------
# LLM (direct REST, no SDK dependency)
# ----------------------------------------------------------------------------
def ask_llm(prompt, system=None):
    def build_payload(with_thinking=True):
        gc = {"temperature": 0.9, "maxOutputTokens": 2048}
        if with_thinking:
            # Gemini 3 family: thinkingLevel instead of thinkingBudget.
            # "minimal" = cheapest and fastest. Thinking tokens count inside
            # maxOutputTokens, so the budget is raised to 2048 + 2048.
            gc["thinkingConfig"] = {"thinkingLevel": "minimal"}
            gc["maxOutputTokens"] = 4096
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": gc,
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        return payload

    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{LLM_MODEL}:generateContent?key={API_KEY}")

    # First attempt with the thinking level; if the model rejects the setting
    # (HTTP 400), retry once without it (some models don't accept it)
    for with_thinking in (True, False):
        body = json.dumps(build_payload(with_thinking)).encode("utf-8")
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        try:
            with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "ignore")
            if e.code == 400 and with_thinking:
                print(f"⚠️  400 with thinkingConfig; retrying without it: {detail[:200]}")
                continue
            raise RuntimeError(f"LLM HTTP {e.code}: {detail}") from e
        except (urllib.error.URLError, OSError) as e:
            # timeouts, DNS or connection failures: surface as RuntimeError so
            # the web layer answers with JSON instead of hanging or crashing
            raise RuntimeError(f"LLM network error ({type(e).__name__}): {e}") from e
        except (KeyError, IndexError, ValueError) as e:
            raise RuntimeError(f"Unexpected LLM response: {e!r}") from e
    raise RuntimeError("LLM call failed after thinking fallback")


# ----------------------------------------------------------------------------
# Vector store
# ----------------------------------------------------------------------------
def load_vector_store():
    return Chroma(
        embedding_function=BatchedGeminiEmbeddings(),
        persist_directory=DB_PATH,
        collection_name=COLLECTION_NAME,
    )


def _ro_connect():
    """Read-only connection to Chroma's own SQLite file. Plain sqlite3 releases
    the GIL while querying and does not go through Chroma's Rust bindings."""
    path = os.path.join(DB_PATH, "chroma.sqlite3")
    if not os.path.exists(path):
        raise RuntimeError(f"Vector DB not found at {path}")
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)


_NOT_SYNOPSIS = ("NOT EXISTS (SELECT 1 FROM embedding_metadata k WHERE k.id = e.id "
                 "AND k.key = 'kind' AND k.string_value = 'synopsis')")


def db_count():
    """Number of SCRIPT chunks stored (plot summaries are not counted)."""
    con = _ro_connect()
    try:
        return con.execute(
            f"SELECT COUNT(*) FROM embeddings e WHERE {_NOT_SYNOPSIS}").fetchone()[0]
    finally:
        con.close()


def random_chunk(vector_store=None, season=None):
    """Samples a random chunk (optionally from a specific season).

    Reads the document and metadata straight from chroma.sqlite3 instead of
    calling collection.count()/get(): on Render those Rust-binding calls
    blocked the whole process. `vector_store` is kept only for compatibility.
    """
    con = _ro_connect()
    try:
        if season:
            row = con.execute(
                "SELECT e.id FROM embeddings e JOIN embedding_metadata m "
                f"ON m.id = e.id AND m.key = 'season' AND m.int_value = ? "
                f"WHERE {_NOT_SYNOPSIS} ORDER BY RANDOM() LIMIT 1",
                (int(season),)).fetchone()
        else:
            row = None
        if row is None:
            # no season requested, or none stored for it (the DB may hold fewer
            # seasons than MAX_SEASON): fall back to any chunk, as before
            row = con.execute(
                f"SELECT e.id FROM embeddings e WHERE {_NOT_SYNOPSIS} "
                "ORDER BY RANDOM() LIMIT 1").fetchone()
        if row is None:
            raise RuntimeError("The collection is empty.")
        doc, meta = None, {}
        for key, s, i, f, b in con.execute(
                "SELECT key, string_value, int_value, float_value, bool_value "
                "FROM embedding_metadata WHERE id = ?", (row[0],)):
            val = next((v for v in (s, i, f, b) if v is not None), None)
            if key == "chroma:document":
                doc = val
            else:
                meta[key] = val
        return doc, meta
    finally:
        con.close()

# ----------------------------------------------------------------------------
# Clip quiz mode: chunks with an aligned Instagram clip (quiz_pool.json)
# ----------------------------------------------------------------------------
QUIZ_POOL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "quiz_pool.json")
_quiz_pool = None


def load_quiz_pool():
    """shortcode -> {chunk_id, ig_url, shortcode, audio_lang, season, title,
    text_es, text_en, text_source}: one entry per clip. Loaded once, then
    cached."""
    global _quiz_pool
    if _quiz_pool is None:
        try:
            with open(QUIZ_POOL_PATH, encoding="utf-8") as f:
                _quiz_pool = json.load(f)
        except FileNotFoundError:
            raise RuntimeError(
                f"quiz_pool.json not found at {QUIZ_POOL_PATH}. "
                "Generate it with: "
                "PYTHONPATH=src python -m spanish_align.export_quiz_pool")
    return _quiz_pool


def random_quiz_chunk(lang=None):
    """Samples a random chunk WITH an aligned clip, from quiz_pool.json.
    No vector DB involved: plain random choice over the pool.
    lang ('es' | 'en'): only clips whose audio is in that language."""
    pool = load_quiz_pool()
    if not pool:
        raise RuntimeError("quiz_pool.json is empty")
    keys = [k for k, v in pool.items()
            if lang is None or v.get("audio_lang", "es") == lang]
    if not keys:
        raise LookupError(f"no clips with {lang} audio in quiz_pool.json")
    return pool[random.choice(keys)]
    

SYNOPSIS_KIND = "synopsis"     # metadata "kind" of the plot documents (add_synopses_to_db)


def clip_prompt(entry, lang=None):
    """Prompt of the clips-mode trivia: the fragment the clip says, plus its
    visual description when the pool entry has one."""
    text = entry["text_en"] if lang == "en" else entry["text_es"]
    prompt = (f"Fragmento de guion (temporada {entry['season']}, episodio "
              f"\"{entry['title']}\"):\n\n{text}\n\n")
    if entry.get("visual"):
        prompt += f"Descripción visual del clip (en inglés):\n{entry['visual']}\n\n"
    return prompt + "Genera la pregunta de trivia (castellano + inglés)."


def _where(season, *conds):
    conds = list(conds) + ([{"season": {"$eq": season}}] if season is not None else [])
    return None if not conds else conds[0] if len(conds) == 1 else {"$and": conds}


def retrieve(vector_store, question, season=None, k=TOP_K):
    """Free-form Q&A: retrieves the k SCRIPT chunks most similar to the question
    (the plot summaries stored in the same collection are excluded)."""
    # langchain-chroma uses `filter` in its public API (with Chroma's where
    # syntax); passing `where` causes weird errors depending on the version
    kwargs = {"k": k, "filter": _where(season, {"kind": {"$ne": SYNOPSIS_KIND}})}
    return vector_store.similarity_search_with_score(question, **kwargs)


def retrieve_synopsis(vector_store, question, season=None, k=1):
    """The k plot-summary paragraphs most similar to the question ([] if the
    DB has none, e.g. add_synopses_to_db has not been run)."""
    kwargs = {"k": k, "filter": _where(season, {"kind": {"$eq": SYNOPSIS_KIND}})}
    return vector_store.similarity_search_with_score(question, **kwargs)


def print_chunk(title, doc, score, full=True):
    """Pretty-prints one retrieved chunk with its distance score."""
    m = doc.metadata
    print(f"\n{title}")
    print(f"   📺 S{m['season']} \"{m['episode_title']}\" "
          f"(lines {m.get('line_start')}-{m.get('line_end')}, distance {score:.3f})")
    if full:
        print(f"   --- chunk text ---")
        for line in doc.page_content.splitlines():
            print(f"   {line}")
    else:
        snippet = doc.page_content.replace("\n", " ¶ ")
        print(f"   {snippet[:300]}{'...' if len(snippet) > 300 else ''}")


# ----------------------------------------------------------------------------
# Game modes
# ----------------------------------------------------------------------------
TRIVIA_SYSTEM = (
    "You are an expert on The Simpsons (classic seasons). You generate trivia "
    "questions from a script fragment you are given. The question must have an "
    "objective answer WITHIN the fragment: names, objects, places, what a "
    "character says, or what happens. Return EXACTLY this format:\n"
    "QUESTION: ...\nA) ...\nB) ...\nC) ...\nD) ...\n"
    "ANSWER: letter) text\nFUN FACT: one short, fun line related to the fragment."
)

TRIVIA_ES_SYSTEM = (
    "Eres un experto en Los Simpson (temporadas clásicas). A partir del "
    "fragmento de guion que se te da (en castellano o en inglés), generas una pregunta "
    "de trivia con respuesta objetiva DENTRO del fragmento: nombres, "
    "objetos, lugares, qué dice un personaje o qué ocurre. Puede venir también "
    "una DESCRIPCIÓN VISUAL del clip (en inglés), que cuenta lo que se ve: úsala "
    "para preguntar por gags visuales, acciones o lugares cuando eso dé una "
    "pregunta mejor que el diálogo, y para entender el contexto de lo que se dice. "
    "La descripción se ha generado automáticamente y puede equivocarse: pregunta "
    "solo por lo que sea claro en ella o esté confirmado por el diálogo, y no "
    "inventes nada que no aparezca ni en el fragmento ni en la descripción. "
    "Devuelve "
    "EXACTAMENTE este formato, primero en castellano y luego el mismo "
    "contenido en inglés (las opciones A-D deben ser las MISMAS y en el "
    "MISMO orden en ambos idiomas):\n"
    "PREGUNTA: ...\nA) ...\nB) ...\nC) ...\nD) ...\n"
    "RESPUESTA: letra) texto\nCURIOSIDAD: una línea breve y divertida.\n"
    "---\n"
    "QUESTION: ...\nA) ...\nB) ...\nC) ...\nD) ...\n"
    "ANSWER: letter) text\nFUN FACT: one short, fun line."
)
    
def play_trivia(vector_store, season=None, verbose=False):
    chunk, meta = random_chunk(vector_store, season)
    source = f"S{meta['season']} \"{meta['episode_title']}\""
    print(f"\n🎯 Question based on {source}\n")

    prompt = (f"Script fragment (season {meta['season']}, episode "
              f"\"{meta['episode_title']}\"):\n\n{chunk}\n\n"
              f"Generate the trivia question.")

    if verbose:
        print("=" * 70)
        print("🔍 VERBOSE — full prompt sent to the LLM")
        print(f"   system: {TRIVIA_SYSTEM}")
        print(f"   prompt:\n{prompt}")
        print("=" * 70)

    try:
        out = ask_llm(prompt, system=TRIVIA_SYSTEM)
    except RuntimeError as e:
        print(f"⚠️  LLM error: {e}")
        return

    # hide the answer line until the player answers
    hidden, answer_line = [], None
    for line in out.splitlines():
        if line.strip().upper().startswith("ANSWER:"):
            answer_line = line
            continue
        hidden.append(line)

    if answer_line is None:
        # truncated output or broken format: show what we got and warn
        print("\n".join(hidden))
        print("\n⚠️  The LLM did not generate the ANSWER line (truncated output?). "
              "Question discarded, try again.")
        return

    print("\n".join(hidden))

    user = input("\nYour answer (A/B/C/D): ").strip().upper()
    print(f"\n{answer_line}")
    correct = user == answer_line.split("ANSWER:")[1].strip()[:1].upper()
    print("🎉 Correct!" if correct else "😢 Wrong!")
    print(f"📺 Source: {source}")


ANSWER_SYSTEM = (
    "Answer questions about The Simpsons using ONLY the context provided: "
    "script fragments and, sometimes, a plot summary of an episode. Be brief "
    "and direct (max 4 sentences).\n"
    "If the context does not contain the answer, say so clearly instead of "
    "making it up. Finish by citing the episode the information comes from."
)


def ask_anything(vector_store, season=None, verbose=False, top_n=TOP_K):
    question = input("\n❓ Your question about The Simpsons: ").strip()
    if not question:
        return

    # retrieve the top_n best chunks, but only the TOP_K best go to the LLM
    results = retrieve(vector_store, question, season, k=max(top_n, TOP_K))
    if not results:
        print("I couldn't find anything for that question.")
        return

    sent, not_sent = results[:TOP_K], results[TOP_K:]

    if verbose:
        print("\n" + "=" * 70)
        print(f"🔍 VERBOSE — chunks sent to the LLM (top-{len(sent)})")
        for i, (doc, score) in enumerate(sent, 1):
            print_chunk(f"[{i}] SENT", doc, score, full=True)
        if not_sent:
            print(f"\n🔍 VERBOSE — retrieved but NOT sent (ranks {TOP_K + 1}-{len(results)})")
            for i, (doc, score) in enumerate(not_sent, TOP_K + 1):
                print_chunk(f"[{i}] NOT SENT", doc, score, full=True)
        print("=" * 70)

    context = "\n\n---\n\n".join(
        f"[Season {doc.metadata['season']}, \"{doc.metadata['episode_title']}\"]\n"
        f"{doc.page_content}"
        for doc, score in sent
    )
    prompt = f"Context:\n\n{context}\n\nQuestion: {question}"

    if verbose:
        print("🔍 VERBOSE — full prompt sent to the LLM:")
        print(f"   system: {ANSWER_SYSTEM}")
        print(prompt)
        print("=" * 70)

    try:
        out = ask_llm(prompt, system=ANSWER_SYSTEM)
    except RuntimeError as e:
        print(f"⚠️  LLM error: {e}")
        return
    print(f"\n💬 {out}\n")
    # source list (compact) always shown, verbose or not
    for i, (doc, score) in enumerate(results, 1):
        m = doc.metadata
        tag = "→ sent" if i <= TOP_K else "  not sent"
        print(f"   📺 [{i}] S{m['season']} \"{m['episode_title']}\" "
              f"(distance {score:.3f}) {tag}")


# ----------------------------------------------------------------------------
# Main menu
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Simpsons Trivia Generator (RAG)")
    parser.add_argument("--verbose", action="store_true",
                        help="show the prompts and the retrieved chunks")
    parser.add_argument("--topN", type=int, default=TOP_K,
                        help="retrieve the top-N chunks (only TOP_K are sent "
                             "to the LLM; the rest are shown with --verbose)")
    args = parser.parse_args()

    print("🍩 Simpsons Trivia Generator (RAG) — v2")
    print(f"   retrieval: top-{args.topN} | sent to LLM: top-{TOP_K} | "
          f"verbose: {'ON' if args.verbose else 'OFF'}")
    vector_store = load_vector_store()
    season = None

    while True:
        print("\n" + "=" * 50)
        print("1) 🎯 New trivia question")
        print("2) 💬 Free-form Q&A (RAG)")
        print(f"3) 🎬 Selected season: {season if season else 'all'}")
        print("4) 🚪 Exit")
        choice = input("Option: ").strip()

        try:
            if choice == "1":
                play_trivia(vector_store, season, verbose=args.verbose)
            elif choice == "2":
                ask_anything(vector_store, season,
                             verbose=args.verbose, top_n=args.topN)
            elif choice == "3":
                raw = input("Season (1-13, empty = all): ").strip()
                season = int(raw) if raw else None
            elif choice == "4":
                print("See you next time! 🍺")
                break
            else:
                print("Invalid option.")
        except Exception as e:
            import traceback
            print(f"⚠️  {e}")
            traceback.print_exc()  # exact origin of the failure, not just the message


if __name__ == "__main__":
    main()
