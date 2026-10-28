import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

# ==================== CONFIGURATION ====================
DATA_LINES_PATH = "data/simpsons_script_lines.csv"
DATA_EPISODES_PATH = "data/simpsons_episodes.csv"
DB_PATH = "vector_db"
CHECKPOINT_PATH = os.path.join(DB_PATH, "ingested_episodes.json")

MAX_SEASON = 13          # Discard any line from a season > 13
CHUNK_SIZE = 20          # Dialogue lines per chunk
CHUNK_OVERLAP = 5        # Overlap between consecutive chunks (within a scene)
SCENE_CONTEXT_LINES = 3  # Last lines of the previous scene prepended to the
                          # first chunk of each new scene (gag continuity)
BATCH_SIZE = 100         # Texts per API call (Google limit: 100)
SLEEP_BETWEEN_BATCHES = 20.0  # The batch is now one episode (~33 chunks ≈ 9k tokens):
                              # 3 episodes/min ≈ 27k TPM, within the 30k limit.
                              # Full ingestion: ~291 episodes ≈ 1h40m
# =======================================================

load_dotenv()


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
      by a 429: it is deleted and re-ingested whole.
    - Any quota error interrupts cleanly and saves progress.
    """
    print("⚡ Generating embeddings and saving to ChromaDB...")
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")  # alternative: gemini-embedding-001
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
            for i in range(0, len(ep_docs), BATCH_SIZE):
                batch = ep_docs[i:i + BATCH_SIZE]
                try:
                    vector_store.add_documents(batch)
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
                  f"«{ep_docs[0].metadata['episode_title']}» "
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
