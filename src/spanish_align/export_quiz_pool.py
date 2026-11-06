"""OFFLINE TASK: builds data/quiz_pool.json — the quiz mode's data source.

Joins the ingested clips (ig_links.jsonl: chunk_id -> ig_url) with the
literal Spanish translations (chunks_es.jsonl) and the originals
(chunks_en.jsonl). Every entry is a chunk that has an aligned clip, so
any question sampled from the pool has a video to show on reveal.

    PYTHONPATH=src python -m spanish_align.export_quiz_pool

Output: data/quiz_pool.json (small; commit it to the repo so Render
gets it for free — it only contains URLs and dialogue text).
"""

import json
from pathlib import Path

from .config import WORKDIR, CHUNKS_EN, CHUNKS_ES, ensure_dirs

IG_LINKS = WORKDIR / "ig_links.jsonl"
QUIZ_POOL = Path("data/quiz_pool.json")


def _load_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in
            path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    ensure_dirs()

    links = _load_jsonl(IG_LINKS)
    en = {c["chunk_id"]: c for c in _load_jsonl(CHUNKS_EN)}
    es = {c["chunk_id"]: c for c in _load_jsonl(CHUNKS_ES)}

    pool, skipped = {}, 0
    for link in links:
        cid = link["chunk_id"]
        c_en, c_es = en.get(cid), es.get(cid)
        if c_es is None or c_en is None:
            skipped += 1
            continue
        shortcode = link["ig_url"].rstrip("/").split("/")[-1]
        pool[cid] = {
            "chunk_id": cid,
            "ig_url": link["ig_url"],
            "shortcode": shortcode,
            "season": c_en["season"],
            "title": c_en["episode_title"],
            "text_es": "\n".join(f"{l['speaker']}: {l['text_es']}"
                                 for l in c_es["lines"] if not l.get("context")),
            "text_en": "\n".join(f"{l['speaker']}: {l['text']}"
                                 for l in c_en["lines"] if not l.get("context")),
        }

    QUIZ_POOL.write_text(json.dumps(pool, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    print(f"quiz_pool.json: {len(pool)} chunks with clip "
          f"| skipped (no translation yet): {skipped}")
    if skipped:
        print("Run translate_chunks for the missing episodes and re-export.")


if __name__ == "__main__":
    main()
