"""OFFLINE TASK: export the English chunks from the existing ChromaDB.

Reads the chunks already indexed by ingest_batch.py (batch_vector_db/,
never re-derives them from the CSVs) and produces chunks_en.jsonl: one
JSON line per chunk, with its dialogue lines parsed out of page_content.
It also extracts the embeddings ALREADY STORED in Chroma (computed by
ingest_batch.py with taskType=RETRIEVAL_DOCUMENT) into
chunk_embeddings.json — no embedding is ever recalculated here.

    PYTHONPATH=src python -m spanish_align.export_from_db

Outputs:
    data/ingest/chunks_en.jsonl
    data/ingest/cache/chunk_embeddings.json
    {"chunk_id": "S05E10_L123-142", "episode_id": "S05E10",
     "season": 5, "episode": 10, "episode_title": "...",
     "kaggle_episode_id": 241, "line_start": 123, "line_end": 142,
     "lines": [{"speaker": "Homer", "text": "...", "context": false}]}
"""

import csv
import json

import chromadb

from .config import (
    DB_PATH, CHROMA_COLLECTION, EPISODES_CSV,
    CHUNKS_EN, CHUNK_EMBEDDINGS, ensure_dirs,
)


def get_collection():
    client = chromadb.PersistentClient(path=str(DB_PATH))
    if CHROMA_COLLECTION:
        return client.get_collection(CHROMA_COLLECTION)
    names = [c.name for c in client.list_collections()]
    if not names:
        raise SystemExit(f"No collections found in {DB_PATH}")
    if len(names) > 1:
        print(f"[WARN] multiple collections {names}; using '{names[0]}' "
              f"(set CHROMA_COLLECTION in config.py to override)")
    return client.get_collection(names[0])


def load_episode_codes() -> dict[int, dict]:
    """Kaggle episode id -> {season, episode, episode_id 'SxxExx'}.

    The CSV only carries number_in_series (global episode number), so the
    in-season episode number is derived by ranking episodes within each
    season by number_in_series (airing order)."""
    rows = []
    with open(EPISODES_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "id": int(row["id"]),
                "season": int(row["season"]),
                "number_in_series": float(row["number_in_series"]),
            })

    rows.sort(key=lambda r: r["number_in_series"])
    codes: dict[int, dict] = {}
    counter: dict[int, int] = {}
    for r in rows:
        counter[r["season"]] = counter.get(r["season"], 0) + 1
        ep = counter[r["season"]]
        codes[r["id"]] = {
            "season": r["season"],
            "episode": ep,
            "episode_id": f"S{r['season']:02d}E{ep:02d}",
        }
    return codes

def parse_page_content(content: str) -> list[dict]:
    """Split a chunk's page_content into dialogue lines.
    Lines carried as [previous context] are kept but flagged, so the
    alignment stage can skip lines that also belong to another chunk."""
    lines, in_context = [], False
    for ln in content.split("\n"):
        s = ln.strip()
        if not s or s.startswith("[The Simpsons") or s.startswith("Scene at"):
            continue
        if s == "[previous context]":
            in_context = True
            continue
        if s == "---":
            in_context = False
            continue
        speaker, _, text = s.partition(": ")
        lines.append({"speaker": speaker.strip() or "?",
                      "text": text.strip(),
                      "context": in_context})
    return lines


def main() -> None:
    ensure_dirs()
    codes = load_episode_codes()
    col = get_collection()

    # chromadb .get() paginates: loop until we have everything.
    # Embeddings come free from the DB (no API call, no recomputation).
    records, offset, batch = [], 0, 5000
    while True:
        page = col.get(limit=batch, offset=offset,
                       include=["documents", "metadatas", "embeddings"])
        if not page["ids"]:
            break
        records.extend(zip(page["ids"], page["documents"],
                           page["metadatas"], page["embeddings"]))
        if len(page["ids"]) < batch:
            break
        offset += batch

    print(f"Loaded {len(records)} chunks from {DB_PATH}")

    exported = 0
    embeddings: dict[str, list[float]] = {}
    with CHUNKS_EN.open("w", encoding="utf-8") as out:
        for _id, doc, meta, emb in records:
            kaggle_id = int(meta["episode_id"])
            code = codes.get(kaggle_id, {})
            lines = parse_page_content(doc)
            if not lines:
                continue
            rec = {
                "chunk_id": f"{code.get('episode_id', f'ep{kaggle_id}')}"
                            f"_L{meta['line_start']}-{meta['line_end']}",
                "episode_id": code.get("episode_id", f"ep{kaggle_id}"),
                "season": code.get("season", meta.get("season")),
                "episode": code.get("episode"),
                "episode_title": meta.get("episode_title", ""),
                "kaggle_episode_id": kaggle_id,
                "line_start": int(meta["line_start"]),
                "line_end": int(meta["line_end"]),
                "lines": lines,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            embeddings[rec["chunk_id"]] = emb.tolist()
            exported += 1

    CHUNK_EMBEDDINGS.parent.mkdir(parents=True, exist_ok=True)
    CHUNK_EMBEDDINGS.write_text(json.dumps(embeddings), encoding="utf-8")
    print(f"Done: {exported} chunks -> {CHUNKS_EN}")
    print(f"Embeddings extracted from DB -> {CHUNK_EMBEDDINGS} "
          f"(no API calls)")


if __name__ == "__main__":
    main()
