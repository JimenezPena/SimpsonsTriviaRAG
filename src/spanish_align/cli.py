"""CLI orchestrator for the Instagram-clip Spanish-alignment pipeline.

Run from the project root with PYTHONPATH=src:
    PYTHONPATH=src python -m spanish_align.cli ingest --url <IG-URL>    # single clip
    PYTHONPATH=src python -m spanish_align.cli ingest --urls clips.txt  # batch
    PYTHONPATH=src python -m spanish_align.cli ingest --urls clips.txt --scope S05  # season hint
    #   (a line of the file may carry its own hint:  <url> S05E10)
    PYTHONPATH=src python -m spanish_align.cli review    # interactive review of doubtful clips
    PYTHONPATH=src python -m spanish_align.cli review --scope S05  # ([f] sets it per clip)
    PYTHONPATH=src python -m spanish_align.cli cleanup   # delete local videos

Startup order (no embedding is ever recalculated; chunk embeddings are
read straight from the vector DB):
    1. python -m spanish_align.export_from_db     (offline, once)
    2. python -m spanish_align.translate_chunks   (offline, once; no embeddings)
    3. python -m spanish_align.cli ingest --urls clips.txt
    4. python -m spanish_align.cli review         (clips with a doubtful match)
"""

import argparse
import json
from pathlib import Path

from .config import (
    CHUNKS_DIR, CHUNKS_ES, TRANSCRIPTS_DIR, MEDIA_DIR,
    REVIEW_QUEUE, IG_LINKS, ensure_dirs,
)
from .media import shortcode_from_url, delete_clip, media_stats
from .models import Chunk, parse_episode_id
from .chunk_match import (identify_chunk_top_k, match_status, parse_scope,
                          embed_clip)
from .align import build_lines


# --------------------------------------------------------------------------
# JSONL helpers
# --------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in
            path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _append_jsonl(path: Path, rec: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _write_jsonl(path: Path, items: list[dict]) -> None:
    """Rewrite the whole file atomically (a Ctrl-C mid-write cannot corrupt it)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items),
                   encoding="utf-8")
    tmp.replace(path)


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
# Registries: ig_links.jsonl (accepted) and review_queue.jsonl (doubtful)
# --------------------------------------------------------------------------

def register_ig_link(clip_id: str, url: str, chunk_id: str,
                     episode_id: str, score: float, clip_lang: str = "es") -> None:
    """chunk_id -> ig_url registry, consumed by export_quiz_pool and the
    main app. A (chunk, url) pair is registered only once."""
    if any(r.get("chunk_id") == chunk_id and r.get("ig_url") == url
           for r in _read_jsonl(IG_LINKS)):
        return
    _append_jsonl(IG_LINKS, {
        "ig_url": url,
        "shortcode": clip_id,
        "chunk_id": chunk_id,
        "episode_id": episode_id,
        "score": round(score, 3),
        "lang": clip_lang,
        "status": "linked",
    })


def queue_for_review(clip_id: str, url: str, chunk_id: str,
                     episode_id: str, score: float, clip_lang: str = "es",
                     scope: str | None = None) -> None:
    """Adds a doubtful clip to the review queue (once per clip)."""
    if any(r.get("shortcode") == clip_id or r.get("ig_url") == url
           for r in _read_jsonl(REVIEW_QUEUE)):
        return
    _append_jsonl(REVIEW_QUEUE, {
        "ig_url": url,
        "shortcode": clip_id,
        "chunk_id": chunk_id,
        "episode_id": episode_id,
        "score": round(score, 3),
        "lang": clip_lang,
        "status": "pending",
        **({"scope": scope} if scope else {}),
    })


# --------------------------------------------------------------------------
# Single-clip ingestion
# --------------------------------------------------------------------------

def _new_record(clip_id: str, url: str, chunk: dict, score: float) -> Chunk:
    season, episode = parse_episode_id(chunk["episode_id"])
    return Chunk(
        clip_id=clip_id, ig_url=url, ig_account=None,
        match_confidence=round(score, 3),
        match_method="chunk-embedding",
        episode_id=chunk["episode_id"],
        season=season, episode=episode,
        episode_title=chunk.get("episode_title"),
        speakers=sorted({l["speaker"] for l in chunk["lines"]}),
    )


def _align_and_link(record: Chunk, chunk: dict, segments: list[dict],
                    runner_ups: list[tuple[dict, float]] = (),
                    clip_lang: str = "es") -> None:
    """Confirms the clip<->chunk match: aligns the dub transcript with the
    chunk lines (dub text where it matches, literal translation elsewhere)
    and registers the clip in ig_links.jsonl."""
    record.reviewed = True
    record.chunk_id = chunk["chunk_id"]
    record.runner_ups = [{"chunk_id": c["chunk_id"], "score": round(s, 3)}
                         for c, s in runner_ups if c]
    record.audio_lang = clip_lang
    record.lines = build_lines(segments, chunk["lines"],
                               load_literal_es(chunk["chunk_id"]),
                               clip_lang=clip_lang)
    register_ig_link(record.clip_id, record.ig_url, chunk["chunk_id"],
                    chunk["episode_id"], record.match_confidence,
                    clip_lang=clip_lang)
                    

def ingest(url: str, keep_video: bool = False, redo: bool = False,
           scope: str | None = None) -> None:
    """scope: optional season/episode hint ('5', 'S05', 'S05E10'): only those
    chunks are considered when matching."""
    from .media import download_clip
    from .google_client import transcribe  # deferred: API key checked on use

    clip_id = shortcode_from_url(url)
    out = CHUNKS_DIR / f"{clip_id}.jsonl"
    if out.exists() and not redo:
        print(f"[SKIP] {clip_id}: already processed -> {out}")
        return

    cached = _read_transcript_cache(clip_id)
    if isinstance(cached, dict) and "segments" in cached:
        result = cached    # already transcribed: no download, no Gemini call
    else:                  # first time, or an old bare-list cache (no language)
        video = download_clip(url)
        result = transcribe(video)
        if not keep_video:
            delete_clip(url)   # the transcript stays cached as JSON
    segments, lang = result["segments"], result.get("lang", "es")

    # candidates, best first (a clip may span a chunk boundary)
    parse_scope(scope)   # fail early on a malformed hint
    candidates = identify_chunk_top_k(segments, k=3, scope=scope)
    if not candidates:
        print(f"[SKIP] {clip_id}: no chunks in scope '{scope}'")
        return
    chunk, score = candidates[0]
    status = match_status(score)

    if status == "discard" or chunk is None:
        print(f"[SKIP] {clip_id}: no reliable match (score={score:.2f})")
        return

    record = _new_record(clip_id, url, chunk, score)
    record.audio_lang = lang
    if status == "accept":
        _align_and_link(record, chunk, segments, candidates[1:], clip_lang=lang)
    else:
        queue_for_review(clip_id, url, chunk["chunk_id"],
                         chunk["episode_id"], score, clip_lang=lang,
                         scope=scope)

    out.write_text(record.to_jsonl() + "\n", encoding="utf-8")
    tag = "OK " if status == "accept" else "REV"
    print(f"[{tag}] {clip_id} -> {chunk['episode_id']} / {chunk['chunk_id']} "
          f"(conf={score:.2f})")


# --------------------------------------------------------------------------
# Interactive review of doubtful clips
# --------------------------------------------------------------------------

def _fmt_time(t) -> str:
    m, s = divmod(int(float(t)), 60)
    return f"{m}:{s:02d}"


def _read_transcript_cache(clip_id: str):
    """Raw cache content: a dict {"lang", "segments"} (current format), a bare
    list (old format, language unknown) or None when nothing is cached."""
    path = TRANSCRIPTS_DIR / f"{clip_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_transcript(clip_id: str) -> dict | None:
    """Cached transcript as {"lang": ..., "segments": [...]}.
    Legacy list caches are treated as Spanish."""
    data = _read_transcript_cache(clip_id)
    if data is None:
        return None
    if isinstance(data, dict):
        return data
    return {"lang": "es", "segments": data}



def _show_item(n: int, total: int, item: dict, segments: list[dict],
               candidates: list[tuple[dict, float]], lang: str = "es") -> None:
    print("=" * 70)
    print(f"[{n}/{total}] CLIP: {item['ig_url']}")
    print(f"--- clip transcript ({'original English audio' if lang == 'en' else 'Spanish dub'}) ---")
    for seg in segments[:6]:
        print(f"  [{_fmt_time(seg['start'])}] {seg['text']}")
    if len(segments) > 6:
        print(f"  ... (+{len(segments) - 6} more segments)")

    for k, (chunk, score) in enumerate(candidates, 1):
        print(f"--- candidate {k}: {chunk['episode_id']} "
              f"\"{chunk.get('episode_title', '')}\" / {chunk['chunk_id']} "
              f"(score {score:.3f}) ---")
        literal = load_literal_es(chunk["chunk_id"]) if k == 1 else {}
        shown = 0
        for i, line in enumerate(chunk["lines"]):
            if line.get("context"):
                continue
            if shown >= (6 if k == 1 else 2):
                break
            print(f"  {line.get('speaker', '?')}: {line['text']}")
            if i in literal:
                print(f"      ↳ {literal[i]}")
            shown += 1


def _approve(item: dict, segments: list[dict], chunk: dict, score: float,
             candidates: list[tuple[dict, float]], clip_lang: str = "es") -> None:
    """Confirms the chosen candidate: writes the aligned clip record and
    registers the link, exactly as an automatic accept would."""
    clip_id = item.get("shortcode") or shortcode_from_url(item["ig_url"])
    record = _new_record(clip_id, item["ig_url"], chunk, score)
    record.match_method = "manual-review"
    record.audio_lang = clip_lang
    others = [(c, s) for c, s in candidates
              if c["chunk_id"] != chunk["chunk_id"]]
    _align_and_link(record, chunk, segments, others, clip_lang=clip_lang)
    (CHUNKS_DIR / f"{clip_id}.jsonl").write_text(record.to_jsonl() + "\n",
                                                 encoding="utf-8")
    item.update(shortcode=clip_id, chunk_id=chunk["chunk_id"],
                episode_id=chunk["episode_id"], score=round(score, 3),
                status="approved")


def cmd_review(args) -> None:
    """Interactive review of the clips in review_queue.jsonl.

    The candidate chunks are recomputed from the cached transcript, so it
    also works with queue entries written by older versions (which only
    stored ig_url/score). Every decision is saved immediately: the review
    can be interrupted at any time and resumed later.

    Season/episode hint: --scope S05 (or 5, S05E10, 5,6 ...) restricts the
    candidates for the whole session; a clip queued by 'ingest --scope'
    keeps its own hint; and the [f] key sets/clears it for the current clip
    (e.g. you recognise it as season 5 and the right episode was not among
    the top 3). The hint is remembered with the clip."""
    items = _read_jsonl(REVIEW_QUEUE)
    pending = [i for i in items if i.get("status") == "pending"]
    print(f"Review queue: {len(pending)} pending / {len(items)} total")
    if not pending:
        return

    session_scope = getattr(args, "scope", None)
    parse_scope(session_scope)
    approved = rejected = 0
    try:
        for n, item in enumerate(pending, 1):
            clip_id = item.get("shortcode") or shortcode_from_url(item["ig_url"])
            transcript = _load_transcript(clip_id)
            if transcript is None:
                print(f"[{n}/{len(pending)}] {clip_id}: no cached transcript "
                      "(run 'ingest --redo' for it); left pending")
                continue
            segments = transcript["segments"]
            lang = transcript.get("lang", "es")
            scope = item.get("scope") or session_scope
            clip_emb = embed_clip(segments)     # once per clip, reused by [f]
            candidates = [(c, s) for c, s in identify_chunk_top_k(
                segments, k=3, scope=scope, clip_emb=clip_emb) if c]
            if not candidates:
                print(f"[{n}/{len(pending)}] {clip_id}: no candidate chunks"
                      f"{f' in scope {scope}' if scope else ''}; left pending")
                continue

            while True:
                _show_item(n, len(pending), item, segments, candidates, lang)
                if scope:
                    print(f"(search restricted to: {scope})")
                options = "/".join(str(k) for k in range(2, len(candidates) + 1))
                prompt = ("[y] approve #1  "
                          + (f"[{options}] approve that candidate  " if options else "")
                          + "[f] filter season/episode  [n] reject  [s] skip  [q] quit: ")
                valid = {"y", "n", "s", "q", "f"} | {str(k) for k in range(2, len(candidates) + 1)}
                ans = ""
                while ans not in valid:
                    ans = input(prompt).strip().lower()
                if ans != "f":
                    break
                text = input("  season/episode (e.g. 5, S05, S05E10, 5,6; "
                             "empty = no filter): ").strip()
                try:
                    new_scope = text or None
                    parse_scope(new_scope)
                except ValueError as e:
                    print(f"  {e}")
                    continue
                found = [(c, s) for c, s in identify_chunk_top_k(
                    segments, k=3, scope=new_scope, clip_emb=clip_emb) if c]
                if not found:
                    print("  no chunks match that filter; keeping the previous candidates")
                    continue
                scope, candidates = new_scope, found
                if scope:
                    item["scope"] = scope
                else:
                    item.pop("scope", None)
                    scope = session_scope
                    candidates = [(c, s) for c, s in identify_chunk_top_k(
                        segments, k=3, scope=scope, clip_emb=clip_emb) if c]

            if ans == "q":
                break
            if ans in ("y", *[str(k) for k in range(2, len(candidates) + 1)]):
                chunk, score = candidates[0 if ans == "y" else int(ans) - 1]
                try:
                    _approve(item, segments, chunk, score, candidates, clip_lang=lang)
                except Exception as e:  # e.g. API error: keep it pending
                    print(f"  -> [ERR] {e}; left pending\n")
                    continue
                approved += 1
                print(f"  -> approved: {chunk['chunk_id']}\n")
            elif ans == "n":
                item["status"] = "rejected"
                rejected += 1
                print("  -> rejected\n")
            else:
                print("  -> left pending\n")
            _write_jsonl(REVIEW_QUEUE, items)   # save after every decision
    except (KeyboardInterrupt, EOFError):
        print("\nInterrupted (decisions so far are saved).")

    left = sum(1 for i in items if i.get("status") == "pending")
    print(f"Approved: {approved} | rejected: {rejected} | still pending: {left}")


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def cmd_ingest(args) -> None:
    parse_scope(args.scope)
    urls = [args.url] if args.url else Path(args.urls).read_text().splitlines()
    for line in filter(None, (u.strip() for u in urls)):
        # a line may carry its own hint:  <url> S05  /  <url> S05E10
        url, _, hint = line.partition(" ")
        try:
            ingest(url, keep_video=args.keep_video, redo=args.redo,
                   scope=hint.strip() or args.scope)
        except Exception as e:  # a dead URL must not break the batch
            print(f"[ERR] {url}: {e}")


def cmd_cleanup(_args) -> None:
    stats = media_stats()
    print(f"Video cache: {stats['count']} files, {stats['total_mb']} MB")
    if stats["count"]:
        for f in MEDIA_DIR.glob("*.mp4"):
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
    s.add_argument("--scope", help="season/episode hint for all clips, e.g. "
                   "5, S05, S05E10, 5,6 (a line of --urls can override it: "
                   "'<url> S05E10')")
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("review", help="interactively review doubtful clip matches")
    s.add_argument("--scope", help="restrict candidates to a season/episode, "
                   "e.g. 5, S05, S05E10 ([f] changes it per clip)")
    s.set_defaults(func=cmd_review)

    s = sub.add_parser("cleanup", help="delete local videos")
    s.set_defaults(func=cmd_cleanup)

    ensure_dirs()
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
