"""OFFLINE TASK: adds the Wikipedia plot paragraphs to the vector DB that the
web app uses (batch_vector_db/), so free-form questions can also draw on a
summary of what happens in each episode.

No API calls: the vectors come from embed_synopses (cache/synopsis_embeddings.json),
made with the same model and task type as the script chunks.

  * The paragraphs go in the SAME collection as the script chunks, marked
    with metadata kind="synopsis" (script chunks have no "kind"). The web app
    (trivia_gen) excludes them from the quiz and from the chunk retrieval and
    fetches the best paragraph separately.
  * Metadata: kind, season, episode_title, episode_code (SxxExx; not
    "episode_id", which the script chunks use with another type), part,
    n_parts, source, wiki_url.
  * Idempotent: it deletes the previous kind="synopsis" documents first, so
    it can be re-run after fetching more seasons. The script chunks are not
    touched.

    PYTHONPATH=src python -m spanish_align.add_synopses_to_db [--db batch_vector_db] [--dry-run]

Then pack the DB and replace the release asset (Render downloads it in build.sh):

    tar -czf db.tar.gz batch_vector_db
    gh release upload latest-db db.tar.gz --clobber --repo JimenezPena/SimpsonsTriviaRAG

(the plot text is from Wikipedia, CC BY-SA 4.0: wiki_url is stored with each document).
"""

import argparse
import json
from pathlib import Path

from .config import SYNOPSES
from .embed_synopses import load_docs, load_embeddings

COLLECTION = "langchain"
BATCH = 200


def build_records(docs: list[dict], vectors: dict, synopses: dict,
                  levels=("paragraph",)) -> tuple[list[dict], int]:
    """[{id, vector, document, metadata}] for the docs of those levels that
    have a vector, and how many were skipped for lacking one."""
    out, missing = [], 0
    for d in docs:
        if d["level"] not in levels:
            continue
        vec = vectors.get(d["doc_id"])
        if vec is None:
            missing += 1
            continue
        rec = synopses.get(d["episode_id"], {})
        out.append({
            "id": f"synopsis:{d['doc_id']}",
            "vector": vec,
            "document": d["text"],
            "metadata": {
                "kind": "synopsis", "season": int(d["season"]),
                "episode_title": d["title"], "episode_code": d["episode_id"],
                "part": int(d["part"]), "n_parts": int(d["n_parts"]),
                "source": "Wikipedia (CC BY-SA 4.0)",
                "wiki_url": rec.get("wiki_url") or "",
            },
        })
    return out, missing


def main() -> None:
    p = argparse.ArgumentParser(description="Add the episode plots to the vector DB")
    p.add_argument("--db", default="batch_vector_db")
    p.add_argument("--levels", default="paragraph",
                   help="comma list: paragraph, episode (default paragraph)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    docs, vectors = load_docs(), load_embeddings()
    synopses = json.loads(SYNOPSES.read_text(encoding="utf-8")) if SYNOPSES.exists() else {}
    records, missing = build_records(docs, vectors, synopses, tuple(args.levels.split(",")))
    print(f"{len(records)} plot documents ready"
          + (f" ({missing} skipped: no embedding, run embed_synopses)" if missing else ""))
    if not records:
        raise SystemExit("nothing to add")
    if args.dry_run:
        print("dry run: the DB was not touched")
        return

    import chromadb
    if not (Path(args.db) / "chroma.sqlite3").exists():
        raise SystemExit(f"{args.db}/chroma.sqlite3 not found")
    col = chromadb.PersistentClient(path=args.db).get_collection(COLLECTION)
    chunks = col.get(where={"kind": {"$ne": "synopsis"}}, limit=1, include=["embeddings"])
    if len(chunks["ids"]):
        dim = len(chunks["embeddings"][0])
        if dim != len(records[0]["vector"]):
            raise SystemExit(f"dimension mismatch: DB chunks have {dim}, plots "
                             f"{len(records[0]['vector'])}; nothing changed")
    before = col.count()
    col.delete(where={"kind": {"$eq": "synopsis"}})
    for i in range(0, len(records), BATCH):
        b = records[i:i + BATCH]
        col.add(ids=[r["id"] for r in b], embeddings=[r["vector"] for r in b],
                documents=[r["document"] for r in b], metadatas=[r["metadata"] for r in b])
    print(f"collection '{COLLECTION}': {before} -> {col.count()} documents "
          f"({len(records)} plot paragraphs, script chunks untouched)")


if __name__ == "__main__":
    main()
