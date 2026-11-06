"""CLI orchestrator for the Instagram-clip Spanish-alignment pipeline.

Run from src/ (or with PYTHONPATH=src from the repo root):
    python -m spanish_align.cli ingest --url <IG-URL>   # single clip pipeline
    python -m spanish_align.cli ingest --urls clips.txt  # batch
    python -m spanish_align.cli review             # manual-review queue
    python -m spanish_align.cli cleanup            # delete local videos

Startup order (no embedding is ever recalculated; chunk embeddings are
read straight from the vector DB):
    1. python -m spanish_align.export_from_db     (offline, once)
    2. python -m spanish_align.translate_chunks   (offline, once; no embeddings)
    3. python -m spanish_align.cli ingest --urls clips.txt
"""

import argparse
import json
from pathlib import Path

from .config import (
    CHUNKS_DIR, REVIEW_QUEUE, IG_LINKS, CHUNKS_ES, ensure_dirs,
)
from .media import shortcode_from_url, delete_clip, media_stats
from .models import Chunk, parse_episode_id
from .chunk_match import identify_chunk_top_k, match_status
from .align import build_lines


# --------------------------------------------------------------------------
# Chunk-level Spanish data
# --------------------------------------------------------------------------

def load_chunk_es(chunk_id: str) -> dict | None:
    if not CHUNKS_ES.exists():
        return None
    for l in CHUNKS_ES.read_text(encoding="utf-8").splitlines():
        if l.strip() and json.loads(l)["chunk_id"] == chunk_id:
            return json.loads(l)
    return None


def load_literal_es(chunk_id: str) -> dict[int, str]:
    """Literal translations for the chunk: {line_index: text_es}."""
    translated = load_chunk_es(chunk_id)
    if not translated:
        return {}
    return {i: line["text_es"] for i, line in enumerate(translated["lines"])
            if "text_es" in line}


# --------------------------------------------------------------------------
# Single-clip ingestion
# --------------------------------------------------------------------------

def ingest(url: str, keep_video: bool = False, redo: bool = False) -> None:
    from .media import download_clip
    from .google_client import transcribe  # deferred: API key checked on use

    clip_id = shortcode_from_url(url)
    out = CHUNKS_DIR / f"{clip_id}.jsonl"
    if out.exists() and not redo:
        print(f"[SKIP] {clip_id}: already processed -> {out}")
        return
        
    video = download_clip(url)
    segments = transcribe(video)
    if not keep_video:
        delete_clip(url)  # the transcript stays cached as JSON

    # candidates, best first (a clip may span a chunk boundary)
    candidates = identify_chunk_top_k(segments, k=3)
    chunk, score = candidates[0]
    status = match_status(score)

    if status == "discard" or chunk is None:
        print(f"[SKIP] {clip_id}: no reliable match (score={score:.2f})")
        return

    chunk_record = Chunk(
        clip_id=clip_id, ig_url=url, ig_account=None,
        match_confidence=round(score, 3),
        match_method="chunk-embedding",
        episode_id=chunk["episode_id"],
        season=parse_episode_id(chunk["episode_id"])[0],
        episode=parse_episode_id(chunk["episode_id"])[1],
        episode_title=chunk.get("episode_title"),
    )
    # link registry: the DB chunk gets the ig_url (used by the main app)
    chunk_record.speakers = sorted({l["speaker"] for l in chunk["lines"]})

    if status == "accept":
        chunk_record.reviewed = True
        chunk_record.chunk_id = chunk["chunk_id"]
        chunk_record.runner_ups = [
            {"chunk_id": c["chunk_id"], "score": round(s, 3)}
            for c, s in candidates[1:] if c
        ]
        literal_es = load_literal_es(chunk["chunk_id"])
        chunk_record.lines = build_lines(segments, chunk["lines"], literal_es)
        register_ig_link(chunk["chunk_id"], url, score)
    else:
        queue_for_review(chunk["chunk_id"], url, score)

    out.write_text(chunk_record.to_jsonl() + "\n", encoding="utf-8")
    tag = "OK " if status == "accept" else "REV"
    print(f"[{tag}] {clip_id} -> {chunk['episode_id']} / {chunk['chunk_id']} "
          f"(conf={score:.2f})")


def register_ig_link(chunk_id: str, url: str, score: float) -> None:
    """chunk_id -> ig_url registry, consumed by the main RAG app to
    enrich the DB chunks with the clip link (and by quizzes/Q&A)."""
    rec = {
                "ig_url": url,
                "shortcode": clip_id,
                "chunk_id": chunk_id,        
                "episode_id": episode_id,      
                "score": round(conf, 3), 
                "status": "linked"
             }
    with IG_LINKS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def queue_for_review(chunk_id: str, url: str, score: float) -> None:
    rec = {
                "ig_url": url,
                "shortcode": clip_id,
                "chunk_id": chunk_id,       
                "episode_id": episode_id,        
                "score": round(conf, 3),
                "status": "pending"
            }
    with REVIEW_QUEUE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def cmd_ingest(args) -> None:
    urls = [args.url] if args.url else Path(args.urls).read_text().splitlines()
    for url in filter(None, (u.strip() for u in urls)):
        try:
            ingest(url, keep_video=args.keep_video, redo=args.redo)
        except Exception as e:  # a dead URL must not break the batch
            print(f"[ERR] {url}: {e}")


def cmd_review(_) -> None:
    if REVIEW_QUEUE.exists():
        print(REVIEW_QUEUE.read_text(encoding="utf-8"))

def review() -> None:
    """Interactive review of the clips pending in the review queue."""
    if not REVIEW_QUEUE.exists():
        print("No review queue file.")
        return
    items = [json.loads(l) for l in
             REVIEW_QUEUE.read_text(encoding="utf-8").splitlines() if l.strip()]
    pending = [i for i in items if i.get("status") == "pending"]
    print(f"Review queue: {len(pending)} pending / {len(items)} total\n")

    # chunk texts, to show the candidate next to the clip
    chunks = {}
    if CHUNKS_EN.exists():
        for l in CHUNKS_EN.read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            c = json.loads(l)
            chunks[c["chunk_id"]] = c

    reviewed = 0
    for item in pending:
        print("=" * 70)
        print(f"CLIP : {item['ig_url']}")
        print(f"MATCH: {item.get('episode_id')} / {item.get('chunk_id')} "
              f"(score {item.get('score', 0):.3f})")

        sc = item.get("shortcode")
        tpath = TRANSCRIPTS / f"{sc}.json"
        print("--- clip transcript (start) ---")
        if tpath.exists():
            t = json.loads(tpath.read_text(encoding="utf-8"))
            for line in t.get("lines", [])[:5]:
                print(f"  {line.get('speaker', '?')}: {line.get('text', '')}")
        else:
            print("  (no cached transcript)")

        c = chunks.get(item.get("chunk_id"))
        print("--- candidate chunk (first lines) ---")
        if c:
            for line in c.get("lines", [])[:5]:
                if not line.get("context"):
                    print(f"  {line.get('speaker', '?')}: {line.get('text', '')}")
        else:
            print("  (chunk not found in chunks_en.jsonl)")

        ans = input("[y] approve  [n] reject  [s] skip: ").strip().lower()
        if ans == "y":
            item["status"] = "approved"
            item["reviewed"] = True
            with IG_LINKS.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"chunk_id": item["chunk_id"],
                                    "ig_url": item["ig_url"],
                                    "score": item["score"]},
                                   ensure_ascii=False) + "\n")
            reviewed += 1
            print("  -> approved, added to ig_links.jsonl\n")
        elif ans == "n":
            item["status"] = "rejected"
            print("  -> rejected\n")
        else:
            print("  -> left pending\n")

    REVIEW_QUEUE.write_text(
        "\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n",
        encoding="utf-8")
    print(f"Reviewed: {reviewed} approved. "
          f"Queue saved ({sum(1 for i in items if i['status'] == 'pending')} pending).")
          
def cmd_cleanup(_) -> None:
    stats = media_stats()
    print(f"Video cache: {stats['count']} files, {stats['total_mb']} MB")
    if stats["count"]:
        for f in Path("data/ingest/media").glob("*.mp4"):
            f.unlink()
        print("Local videos deleted (transcripts kept)")


def main() -> None:
    p = argparse.ArgumentParser(description="Simpsons RAG — IG clip alignment")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("ingest", help="process clip(s)")
    s.add_argument("--url")
    s.add_argument("--urls", help="file with one URL per line")
    s.add_argument("--keep-video", action="store_true",
                   help="do not delete the video after transcription")
    s.add_argument("--redo", action="store_true",
                   help="reprocess clips that were already ingested")
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("review", help="show the review queue")
    s.set_defaults(func=cmd_review)

    s = sub.add_parser("cleanup", help="delete local videos")
    s.set_defaults(func=cmd_cleanup)

    ensure_dirs()
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
