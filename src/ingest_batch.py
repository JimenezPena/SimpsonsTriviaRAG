import os
import json
import time
import urllib.request
import urllib.error
import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

# ==================== CONFIGURATION ====================
DATA_LINES_PATH = "data/simpsons_script_lines.csv"
DATA_EPISODES_PATH = "data/simpsons_episodes.csv"
DB_PATH = "batch_vector_db"
CHECKPOINT_PATH = os.path.join(DB_PATH, "ingested_episodes.json")

MAX_SEASON = 13          # Discard any line from a season > 13
CHUNK_SIZE = 20          # Dialogue lines per chunk
CHUNK_OVERLAP = 5        # Overlap between consecutive chunks (within a scene)
SCENE_CONTEXT_LINES = 3  # Last lines of the previous scene prepended to the
                          # first chunk of each new scene (gag continuity)
BATCH_SIZE = 100         # Texts per batchEmbedContents call (Google limit: 100)
SLEEP_BETWEEN_BATCHES = 10.0  # 1 real request per episode (~9k tokens): the 30k
                              # TPM allows ~3 requests/min with ample headroom.
                              # Full ingestion: ~291 episodes ≈ 50 min
# =======================================================

load_dotenv()


class BatchedGeminiEmbeddings(Embeddings):
    """Embeddings with REAL batching via REST batchEmbedContents (no google-genai).

    LangChain/SDK embedders send one HTTP request per text
    (71 chunks = 71 RPM, saturating the free-tier quota). This embedder calls
    the batchEmbedContents endpoint directly: up to 100 texts per HTTP
    request, with backoff retries on 429.
    """

    API_URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
               "{model}:batchEmbedContents")

    def __init__(self, model="gemini-embedding-001", batch_size=100,
                 max_retries=5):
        self.api_key = os.environ["GOOGLE_API_KEY"]
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries

    def _batch_embed(self, texts, task_type):
        """One real HTTP request with up to self.batch_size texts."""
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

        for attempt in range(1, self.max_retries + 1):
            try:
                req = urllib.request.Request(
                    url, data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                embeddings = [e["values"] for e in data.get("embeddings", [])]
                if len(embeddings) != len(texts):
                    raise RuntimeError(
                        f"Expected {len(texts)} embeddings but got "
                        f"{len(embeddings)}. Response: {data}"
                    )
                return embeddings
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < self.max_retries:
                    wait = min(60 * attempt, 90)
                    print(f"      ⏳ 429 RPM/TPM, retry {attempt}/"
                          f"{self.max_retries} in {wait}s...")
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', 'ignore')}") from e
            except urllib.error.URLError as e:
                if attempt < self.max_retries:
                    time.sleep(10 * attempt)
                    continue
                raise RuntimeError(f"Network error: {e}") from e
        raise RuntimeError("Retries exhausted")

    def embed_documents(self, texts, task_type="RETRIEVAL_DOCUMENT"):
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            embeddings.extend(self._batch_embed(batch, task_type))
            if i + self.batch_size < len(texts):
                time.sleep(1)
        return embeddings

    def embed_query(self, text):
        # different task_type: document vs query is the correct setup for RAG
        return self._batch_embed([text], "RETRIEVAL_QUERY")[0]


def load_data():
    """Loads the script-lines CSV and joins it with the episodes CSV to bring in season and title."""
    print("📖 Reading the CSV files...")
    lines = pd.read_csv(DATA_LINES_PATH, dtype={"speaking_line": str, "word_count": str})
    episodes = pd.read_csv(DATA_EPISODES_PATH)

    # speaking_line comes as the string "true"/"false" in the original CSV
    lines = lines[lines["speaking_line"].astype(str).str.lower() == "true"]
    lines = lines.dropna(subset=["spoken_words"])

    # JOIN: episode_id (lines) -> id (episodes). Brings season, title and number_in_series.
    df = lines.merge(
        episodes[["id", "season", "title", "number_in_series"]],
        how="inner",
        left_on="episode_id",
        right_on="id",
    )
    df = df.rename(columns={"id": "episode_row_id"})

    # Season filter: discard anything above MAX_SEASON
    df = df[df["season"] <= MAX_SEASON].copy()

    print(f"🧹 {len(df)} dialogue lines in seasons 1-{MAX_SEASON}, "
          f"across {df['episode_id'].nunique()} episodes.")
    return df


def build_chunks(df):
    """Groups by episode, cuts on location changes (scene changes) and creates
    chunks of ~CHUNK_SIZE lines with an overlap of CHUNK_OVERLAP."""
    documents = []

    for episode_id, ep in df.groupby("episode_id", sort=True):
        ep = ep.sort_values("number")
        season = int(ep["season"].iloc[0])
        title = str(ep["title"].iloc[0])

        # 1) Segment into scenes: cut when location_id changes
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

        # 2) Within each scene, split with overlap; the first chunk of each
        #    scene carries the last lines of the previous scene as context
        prev_tail = []  # last lines of the previous scene (gag continuity)
        for scene in scenes:
            scene = list(scene)
            step = CHUNK_SIZE - CHUNK_OVERLAP
            for start in range(0, len(scene), step):
                chunk = scene[start:start + CHUNK_SIZE]

                # Previous context only in the first chunk of the scene
                if start == 0 and prev_tail:
                    context_lines = "\n".join(
                        f"{r['raw_character_text'] or 'Someone'}: {r['spoken_words']}"
                        for r in prev_tail
                    )
                    dialog = f"[previous context]\n{context_lines}\n---\n" + "\n".join(
                        f"{r['raw_character_text'] or 'Someone'}: {r['spoken_words']}"
                        for r in chunk
                    )
                else:
                    dialog = "\n".join(
                        f"{r['raw_character_text'] or 'Someone'}: {r['spoken_words']}"
                        for r in chunk
                    )

                # Context + chunk should not exceed the normal chunk + margin
                prev_tail = scene[-SCENE_CONTEXT_LINES:]
                locations = sorted({str(r["raw_location_text"]) for r in chunk})
                characters = sorted({str(r["raw_character_text"]) for r in chunk})

                page_content = (
                    f"[The Simpsons - Season {season}, Episode: '{title}']\n"
                    f"Scene at {', '.join(locations)}\n\n{dialog}"
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

    print(f"📄 {len(documents)} chunks generated.")
    return documents


def load_checkpoint():
    """Episodes completed in previous runs."""
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            return set(json.load(f))
    return set()


def save_checkpoint(done_episodes):
    tmp = CHECKPOINT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(sorted(done_episodes), f)
    os.replace(tmp, CHECKPOINT_PATH)  # atomic write


def chroma_episode_counts(vector_store):
    """Number of chunks per episode_id already living in Chroma."""
    existing = vector_store.get(include=["metadatas"])
    counts = {}
    for m in existing.get("metadatas", []):
        eid = m.get("episode_id")
        if eid is not None:
            counts[eid] = counts.get(eid, 0) + 1
    return counts


def ingest(documents):
    """Vectorizes season by season, with an episode-level checkpoint.

    - An episode in the checkpoint is already complete: it is skipped.
    - An episode present in Chroma but NOT in the checkpoint was left half-done
      by a 429: it is deleted and re-inserted whole.
    - Any quota error interrupts cleanly and saves progress.
    """
    print("⚡ Generating embeddings and saving to ChromaDB...")
    embeddings = BatchedGeminiEmbeddings(model="gemini-embedding-001")
    vector_store = Chroma(embedding_function=embeddings, persist_directory=DB_PATH)

    done = load_checkpoint()
    if done:
        print(f"♻️  Checkpoint: {len(done)} episodes already completed, resuming from there.")

    counts = chroma_episode_counts(vector_store)
    for eid in counts:
        if eid not in done:
            print(f"🧯 Episode {eid} was left half-done (previous 429): re-ingesting it whole.")
            vector_store.delete(where={"episode_id": eid})

    pending = [d for d in documents if d.metadata["episode_id"] not in done]
    print(f"📄 {len(pending)} chunks pending out of {len(documents)} total.")

    # Iterate season -> episode: the checkpoint is marked EPISODE by EPISODE,
    # right when the last batch of that episode has been ingested successfully.
    seasons = sorted({d.metadata["season"] for d in pending})
    for season in seasons:
        season_pending = [d for d in pending if d.metadata["season"] == season]
        episodes = sorted({d.metadata["episode_id"] for d in season_pending})
        print(f"🎬 Season {season}: {len(episodes)} episodes, {len(season_pending)} chunks")

        for ep_n, eid in enumerate(episodes, 1):
            ep_docs = [d for d in season_pending if d.metadata["episode_id"] == eid]

            # Full langchain bypass: Chroma.add_documents calls embed_documents
            # ONCE PER CHUNK in several versions, cancelling out the batching.
            # Here we compute the embeddings ourselves (1 real request per
            # episode) and insert them with upsert + deterministic IDs
            # (retrying a half-done episode never duplicates chunks).
            try:
                texts = [d.page_content for d in ep_docs]
                metas = [d.metadata for d in ep_docs]
                ids = [f"ep{eid}-c{i}" for i in range(len(ep_docs))]
                vectors = embeddings.embed_documents(texts)
                vector_store._collection.upsert(
                    ids=ids, embeddings=vectors,
                    documents=texts, metadatas=metas,
                )
            except Exception as e:
                print(f"⚠️  Error in episode {eid} (season {season}): {e}")
                print(f"💾 Progress saved ({len(done)} episodes). Relaunch the "
                      f"script: it will resume at episode {eid}, S{season}.")
                save_checkpoint(done)
                return
            time.sleep(SLEEP_BETWEEN_BATCHES)

            # Episode complete: immediate checkpoint
            done.add(eid)
            save_checkpoint(done)
            print(f"   ✅ S{season}x{ep_n}/{len(episodes)} — episode {eid} "
                  f"\"{ep_docs[0].metadata['episode_title']}\" "
                  f"({len(done)} episodes, {len(ep_docs)} chunks)")

    print(f"🎉 Ingestion complete: {len(done)} episodes in ./{DB_PATH}!")


def main():
    print("🚀 Starting The Simpsons data ingestion...")

    for path in (DATA_LINES_PATH, DATA_EPISODES_PATH):
        if not os.path.exists(path):
            print(f"❌ Error: cannot find the file at {path}. Download it from Kaggle and place it there.")
            return

    df = load_data()
    documents = build_chunks(df)
    ingest(documents)


if __name__ == "__main__":
    main()
