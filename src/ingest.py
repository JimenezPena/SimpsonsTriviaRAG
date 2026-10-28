import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

# ==================== CONFIGURACIÓN ====================
DATA_LINES_PATH = "data/simpsons_script_lines.csv"
DATA_EPISODES_PATH = "data/simpsons_episodes.csv"
DB_PATH = "vector_db"
CHECKPOINT_PATH = os.path.join(DB_PATH, "ingested_episodes.json")

MAX_SEASON = 13          # Ignora cualquier línea de una temporada > 13
CHUNK_SIZE = 20          # Líneas de diálogo por chunk
CHUNK_OVERLAP = 5        # Solape entre chunks consecutivos (dentro de la escena)
SCENE_CONTEXT_LINES = 3  # Líneas finales de la escena anterior que se anteponen
                          # al primer chunk de cada escena nueva (continuidad de gags)
BATCH_SIZE = 100         # Textos por llamada a la API (límite de Google: 100)
SLEEP_BETWEEN_BATCHES = 20.0  # Ahora el lote es un episodio (~33 chunks ≈ 9k tokens):
                              # 3 episodios/min ≈ 27k TPM, dentro del límite de 30k.
                              # Ingesta completa: ~291 episodios ≈ 1h40m
# =======================================================

load_dotenv()


def load_data():
    """Carga el CSV de líneas y lo une con el de episodios para traer season y title."""
    print("📖 Leyendo los CSV...")
    lines = pd.read_csv(DATA_LINES_PATH, dtype={"speaking_line": str, "word_count": str})
    episodes = pd.read_csv(DATA_EPISODES_PATH)

    # speaking_line viene como string "true"/"false" en el CSV original
    lines = lines[lines["speaking_line"].astype(str).str.lower() == "true"]
    lines = lines.dropna(subset=["spoken_words"])

    # JOIN: episode_id (líneas) -> id (episodios). Trae season, title y number_in_series.
    df = lines.merge(
        episodes[["id", "season", "title", "number_in_series"]],
        how="inner",
        left_on="episode_id",
        right_on="id",
    )
    df = df.rename(columns={"id": "episode_row_id"})

    # Filtro de temporadas: descarta todo lo que supere MAX_SEASON
    df = df[df["season"] <= MAX_SEASON].copy()

    print(f"🧹 {len(df)} líneas de diálogo en temporadas 1-{MAX_SEASON}, "
          f"de {df['episode_id'].nunique()} episodios.")
    return df


def build_chunks(df):
    """Agrupa por episodio, corta cuando cambia la localización (cambio de escena)
    y crea chunks de ~CHUNK_SIZE líneas con solape de CHUNK_OVERLAP."""
    documents = []

    for episode_id, ep in df.groupby("episode_id", sort=True):
        ep = ep.sort_values("number")
        season = int(ep["season"].iloc[0])
        title = str(ep["title"].iloc[0])

        # 1) Segmentar en escenas: corte cuando cambia location_id
        scenes, current = [], []
        last_loc = None
        for _, row in ep.iterrows():
            loc = row["location_id"]
            if last_loc is not None and loc != last_loc and current:
                scenes.append(current)
                current = []
            current.append(row)
            last_loc = loc
        if current:
            scenes.append(current)

        # 2) Dentro de cada escena, trocear con solape; el primer chunk de cada
        #    escena lleva las últimas líneas de la escena anterior como contexto
        prev_tail = []  # últimas líneas de la escena anterior (continuidad de gags)
        for scene in scenes:
            scene = list(scene)
            step = CHUNK_SIZE - CHUNK_OVERLAP
            for start in range(0, len(scene), step):
                chunk = scene[start:start + CHUNK_SIZE]

                # Contexto previo solo en el primer chunk de la escena
                if start == 0 and prev_tail:
                    context_lines = "\n".join(
                        f"{r['raw_character_text'] or 'Alguien'}: {r['spoken_words']}"
                        for r in prev_tail
                    )
                    dialog = f"[contexto previo]\n{context_lines}\n---\n" + "\n".join(
                        f"{r['raw_character_text'] or 'Alguien'}: {r['spoken_words']}"
                        for r in chunk
                    )
                else:
                    dialog = "\n".join(
                        f"{r['raw_character_text'] or 'Alguien'}: {r['spoken_words']}"
                        for r in chunk
                    )

                # Contexto + chunk no debe superar el chunk normal + margen
                prev_tail = scene[-SCENE_CONTEXT_LINES:]
                locations = sorted({str(r["raw_location_text"]) for r in chunk})
                characters = sorted({str(r["raw_character_text"]) for r in chunk})

                page_content = (
                    f"[Los Simpson - Temporada {season}, Episodio: '{title}']\n"
                    f"Escena en {', '.join(locations)}\n\n{dialog}"
                )
                metadata = {
                    "season": season,
                    "episode_id": int(episode_id),
                    "episode_title": title,
                    "characters": characters,
                    "locations": locations,
                    "line_start": int(chunk[0]["number"]),
                    "line_end": int(chunk[-1]["number"]),
                }
                documents.append(Document(page_content=page_content, metadata=metadata))

    print(f"📄 {len(documents)} chunks generados.")
    return documents


def load_checkpoint():
    """Episodios completados en ejecuciones anteriores."""
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            return set(json.load(f))
    return set()


def save_checkpoint(done_episodes):
    tmp = CHECKPOINT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(sorted(done_episodes), f)
    os.replace(tmp, CHECKPOINT_PATH)  # escritura atómica


def chroma_episode_counts(vector_store):
    """Número de chunks por episode_id que ya viven en Chroma."""
    existing = vector_store.get(include=["metadatas"])
    counts = {}
    for m in existing.get("metadatas", []):
        eid = m.get("episode_id")
        if eid is not None:
            counts[eid] = counts.get(eid, 0) + 1
    return counts


def ingest(documents):
    """Vectoriza temporada a temporada, con checkpoint por episodio.

    - Un episodio en el checkpoint ya está completo: se omite.
    - Un episodio presente en Chroma pero NO en el checkpoint quedó a medias
      por un 429: se borra y se reinserta entero.
    - Cualquier error de cuota interrumpe limpiamente y guarda el progreso.
    """
    print("⚡ Generando embeddings y guardando en ChromaDB...")
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")  #alternativa: gemini-embedding-001
    vector_store = Chroma(embedding_function=embeddings, persist_directory=DB_PATH)

    done = load_checkpoint()
    if done:
        print(f"♻️  Checkpoint: {len(done)} episodios ya completados, se retoma desde ahí.")

    counts = chroma_episode_counts(vector_store)
    for eid in counts:
        if eid not in done:
            print(f"🧯 Episodio {eid} quedó a medias (429 previo): se reingesta entero.")
            vector_store.delete(where={"episode_id": eid})

    pending = [d for d in documents if d.metadata["episode_id"] not in done]
    print(f"📄 {len(pending)} chunks pendientes de {len(documents)} totales.")

    # Iterar temporada -> episodio: el checkpoint se marca EPISODIO a EPISODIO,
    # justo cuando el último lote de ese episodio se ha ingestado con éxito.
    seasons = sorted({d.metadata["season"] for d in pending})
    for season in seasons:
        season_pending = [d for d in pending if d.metadata["season"] == season]
        episodes = sorted({d.metadata["episode_id"] for d in season_pending})
        print(f"🎬 Temporada {season}: {len(episodes)} episodios, {len(season_pending)} chunks")

        for ep_n, eid in enumerate(episodes, 1):
            ep_docs = [d for d in season_pending if d.metadata["episode_id"] == eid]
            for i in range(0, len(ep_docs), BATCH_SIZE):
                batch = ep_docs[i:i + BATCH_SIZE]
                try:
                    vector_store.add_documents(batch)
                except Exception as e:
                    print(f"⚠️  Error en el episodio {eid} (temporada {season}): {e}")
                    print(f"💾 Progreso guardado ({len(done)} episodios). Relanza el "
                          f"script: retomará en el episodio {eid}, T{season}.")
                    save_checkpoint(done)
                    return
                time.sleep(SLEEP_BETWEEN_BATCHES)

            # Episodio completo: checkpoint inmediato
            done.add(eid)
            save_checkpoint(done)
            print(f"   ✅ T{season}x{ep_n}/{len(episodes)} — episodio {eid} "
                  f"«{ep_docs[0].metadata['episode_title']}» "
                  f"({len(done)} episodios, {len(ep_docs)} chunks)")

    print(f"🎉 Ingesta completa: {len(done)} episodios en ./{DB_PATH}!")


def main():
    print("🚀 Iniciando la ingesta de datos de Los Simpson...")

    for path in (DATA_LINES_PATH, DATA_EPISODES_PATH):
        if not os.path.exists(path):
            print(f"❌ Error: no se encuentra {path}. Descárgalo de Kaggle y colócalo ahí.")
            return

    df = load_data()
    documents = build_chunks(df)
    ingest(documents)


if __name__ == "__main__":
    main()
