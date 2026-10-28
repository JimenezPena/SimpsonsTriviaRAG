"""Simpsons Trivia Generator (RAG) — draft v1.

Modos:
  1) Pregunta de trivia: muestrea un chunk aleatorio de la BD vectorial y el LLM
     genera una pregunta con 4 opciones y su respuesta (revelada tras responder).
  2) Consulta libre: haces una pregunta, se recuperan los chunks más similares y
     el LLM responde citando episodio y temporada.

Cero coste de ingesta: la BD ya está persistida; solo 1 embedding por consulta.
"""
import os
import json
import time
import random
import urllib.request
import urllib.error
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

# ==================== CONFIGURACIÓN ====================
DB_PATH = "vector_db"      # la BD creada por ingest_batch.py
COLLECTION_NAME = "langchain"
LLM_MODEL = "gemini-2.5-flash"
MAX_SEASON = 13
TOP_K = 3                        # chunks recuperados en consulta libre
# =======================================================

load_dotenv()
API_KEY = os.environ["GOOGLE_API_KEY"]


# ----------------------------------------------------------------------------
# Embeddings (misma clase que ingest_batch.py: copia compartida idealmente)
# ----------------------------------------------------------------------------
class BatchedGeminiEmbeddings(Embeddings):
    API_URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
               "{model}:batchEmbedContents")

    def __init__(self, model="gemini-embedding-2", batch_size=100):
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
                    raise RuntimeError(f"Esperaba {len(texts)} embeddings, "
                                       f"llegaron {len(embeddings)}")
                return embeddings
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 4:
                    time.sleep(min(60 * (attempt + 1), 90))
                    continue
                raise RuntimeError(f"HTTP {e.code}: "
                                   f"{e.read().decode('utf-8', 'ignore')}") from e
        raise RuntimeError("Reintentos agotados")

    def embed_documents(self, texts):
        return self._batch_embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text):
        return self._batch_embed([text], "RETRIEVAL_QUERY")[0]


# ----------------------------------------------------------------------------
# LLM (REST directo, sin dependencias del SDK)
# ----------------------------------------------------------------------------
def ask_llm(prompt, system=None):
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.9,
            "maxOutputTokens": 2048,
            # gemini-2.5-flash "piensa" antes de responder y sus tokens de
            # razonamiento se comen el presupuesto de salida: desactivado aquí
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{LLM_MODEL}:generateContent?key={API_KEY}")
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"LLM HTTP {e.code}: "
                           f"{e.read().decode('utf-8', 'ignore')}") from e


# ----------------------------------------------------------------------------
# Base vectorial
# ----------------------------------------------------------------------------
def load_vector_store():
    return Chroma(
        embedding_function=BatchedGeminiEmbeddings(),
        persist_directory=DB_PATH,
        collection_name=COLLECTION_NAME,
    )


def random_chunk(vector_store, season=None):
    """Muestrea un chunk aleatorio (opcionalmente de una temporada concreta)."""
    where = {"season": {"$eq": season}} if season else None
    # simulación de sample: pedir en orden "aleatorio" de Chroma no existe;
    # pedimos un offset aleatorio sobre el total de la colección
    col = vector_store._collection
    total = col.count()
    offset = random.randint(0, max(total - 1, 0))
    res = col.get(limit=1, offset=offset, where=where,
                  include=["documents", "metadatas"])
    docs, metas = res.get("documents", []), res.get("metadatas", [])
    if not docs:
        # reintentar sin where o avisar
        res = col.get(limit=1, offset=offset, include=["documents", "metadatas"])
        docs, metas = res.get("documents", []), res.get("metadatas", [])
        if not docs:
            raise RuntimeError("La colección está vacía.")
    return docs[0], metas[0]


def retrieve(vector_store, question, season=None):
    """Consulta libre: recupera los TOP_K chunks más parecidos a la pregunta."""
    # langchain-chroma usa `filter` en su API pública (con sintaxis where de
    # Chroma); pasar `where` causa errores raros según la versión
    kwargs = {"k": TOP_K}
    if season is not None:
        kwargs["filter"] = {"season": {"$eq": season}}
    return vector_store.similarity_search_with_score(question, **kwargs)


# ----------------------------------------------------------------------------
# Modos de juego
# ----------------------------------------------------------------------------
TRIVIA_SYSTEM = (
    "Eres un experto en Los Simpson (temporadas clásicas). Generas preguntas "
    "de trivia a partir de un fragmento de guion que se te da. La pregunta debe "
    "tener respuesta objetiva DENTRO del fragmento: nombres, objetos, lugares, "
    "qué dice un personaje o qué ocurre.\n"
    "IDIOMA: escribe TODO en español neutro. El guion está en inglés: traduce "
    "de forma natural e idiomática, nunca literal. Si citas una frase exacta, "
    "pon tu traducción y el original inglés entre paréntesis.\n"
    "Devuelve EXACTAMENTE este formato:\n"
    "PREGUNTA: ...\nA) ...\nB) ...\nC) ...\nD) ...\n"
    "RESPUESTA: letra) texto\nCURIOSIDAD: una línea breve y divertida "
    "relacionada con el fragmento."
)


def play_trivia(vector_store, season=None):
    chunk, meta = random_chunk(vector_store, season)
    source = f"T{meta['season']} «{meta['episode_title']}»"
    print(f"\n🎯 Pregunta basada en {source}\n")

    prompt = (f"Fragmento del guion (temporada {meta['season']}, episodio "
              f"«{meta['episode_title']}»):\n\n{chunk}\n\n"
              f"Genera la pregunta de trivia.")
    try:
        out = ask_llm(prompt, system=TRIVIA_SYSTEM)
    except RuntimeError as e:
        print(f"⚠️  Error del LLM: {e}")
        return

    # ocultar la línea de respuesta hasta que el jugador conteste
    hidden, answer_line = [], None
    for line in out.splitlines():
        if line.strip().upper().startswith("RESPUESTA:"):
            answer_line = line
            continue
        hidden.append(line)

    if answer_line is None:
        # salida truncada o formato roto: mostrar lo que haya y avisar
        print("\n".join(hidden))
        print("\n⚠️  El LLM no generó la línea RESPUESTA (¿salida cortada?). "
              "Pregunta descartada, prueba otra vez.")
        return

    print("\n".join(hidden))

    user = input("\nTu respuesta (A/B/C/D): ").strip().upper()
    print(f"\n{answer_line}")
    correct = user == answer_line.split("RESPUESTA:")[1].strip()[:1].upper()
    print("🎉 ¡Correcto!" if correct else "😢 ¡Fallaste!")
    print(f"📺 Fuente: {source}")


ANSWER_SYSTEM = (
    "Responde preguntas sobre Los Simpson usando SOLO el contexto de guion "
    "proporcionado. Sé breve y directo (máx. 4 frases).\n"
    "IDIOMA: responde SIEMPRE en español neutro. El guion está en inglés: "
    "traduce de forma natural e idiomática, nunca literal. Si citas una frase "
    "exacta, pon tu traducción y el original inglés entre paréntesis.\n"
    "Si el contexto no contiene la respuesta, dilo claramente en vez de "
    "inventar. Termina citando el episodio de donde sale la información."
)


def ask_anything(vector_store, season=None):
    question = input("\n❓ Tu pregunta sobre Los Simpson: ").strip()
    if not question:
        return
    results = retrieve(vector_store, question, season)
    if not results:
        print("No encontré nada para esa pregunta.")
        return

    context = "\n\n---\n\n".join(
        f"[Temporada {doc.metadata['season']}, «{doc.metadata['episode_title']}»]\n"
        f"{doc.page_content}"
        for doc, score in results
    )
    prompt = f"Contexto:\n\n{context}\n\nPregunta: {question}"
    try:
        out = ask_llm(prompt, system=ANSWER_SYSTEM)
    except RuntimeError as e:
        print(f"⚠️  Error del LLM: {e}")
        return
    print(f"\n💬 {out}\n")
    for doc, score in results:
        m = doc.metadata
        print(f"   📺 [{m['season']}] «{m['episode_title']}» (distancia {score:.3f})")


# ----------------------------------------------------------------------------
# Menú principal
# ----------------------------------------------------------------------------
def main():
    print("🍩 Simpsons Trivia Generator (RAG) — draft v1")
    vector_store = load_vector_store()
    season = None

    while True:
        print("\n" + "=" * 50)
        print("1) 🎯 Nueva pregunta de trivia")
        print("2) 💬 Consulta libre (RAG)")
        print(f"3) 🎬 Temporada elegida: {season if season else 'todas'}")
        print("4) 🚪 Salir")
        choice = input("Opción: ").strip()

        try:
            if choice == "1":
                play_trivia(vector_store, season)
            elif choice == "2":
                ask_anything(vector_store, season)
            elif choice == "3":
                raw = input("Temporada (1-13, vacío = todas): ").strip()
                season = int(raw) if raw else None
            elif choice == "4":
                print("¡Hasta la próxima! 🍺")
                break
            else:
                print("Opción no válida.")
        except Exception as e:
            import traceback
            print(f"⚠️  {e}")
            traceback.print_exc()  # origen exacto del fallo, no solo el mensaje


if __name__ == "__main__":
    main()
