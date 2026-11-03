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
import argparse
import urllib.request
import urllib.error
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

# ==================== CONFIGURATION ====================
DB_PATH = "batch_vector_db"      # the DB created by ingest_batch.py
COLLECTION_NAME = "langchain"
LLM_MODEL = "gemini-3.5-flash-lite"
MAX_SEASON = 13
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
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "ignore")
            if e.code == 400 and with_thinking:
                print(f"⚠️  400 with thinkingConfig; retrying without it: {detail[:200]}")
                continue
            raise RuntimeError(f"LLM HTTP {e.code}: {detail}") from e
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


def random_chunk(vector_store, season=None):
    """Samples a random chunk (optionally from a specific season)."""
    where = {"season": {"$eq": season}} if season else None
    # sample simulation: Chroma has no "random" ordering;
    # we pick a random offset over the total of the collection
    col = vector_store._collection
    total = col.count()
    offset = random.randint(0, max(total - 1, 0))
    res = col.get(limit=1, offset=offset, where=where,
                  include=["documents", "metadatas"])
    docs, metas = res.get("documents", []), res.get("metadatas", [])
    if not docs:
        # retry without where, or bail out
        res = col.get(limit=1, offset=offset, include=["documents", "metadatas"])
        docs, metas = res.get("documents", []), res.get("metadatas", [])
        if not docs:
            raise RuntimeError("The collection is empty.")
    return docs[0], metas[0]


def retrieve(vector_store, question, season=None, k=TOP_K):
    """Free-form Q&A: retrieves the k chunks most similar to the question."""
    # langchain-chroma uses `filter` in its public API (with Chroma's where
    # syntax); passing `where` causes weird errors depending on the version
    kwargs = {"k": k}
    if season is not None:
        kwargs["filter"] = {"season": {"$eq": season}}
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
    "Answer questions about The Simpsons using ONLY the script context "
    "provided. Be brief and direct (max 4 sentences).\n"
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
