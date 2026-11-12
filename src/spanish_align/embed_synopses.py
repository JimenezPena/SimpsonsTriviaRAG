"""OFFLINE TASK: embeds the Wikipedia plots (see fetch_synopses).

Each episode becomes several documents, all in the same set:

  * level "paragraph": one per plot paragraph (a paragraph is already a
    scene-sized unit, ~100 words). Paragraphs under MIN_WORDS are merged
    with a neighbour; ones over MAX_WORDS are split at sentence boundaries.
  * level "episode": the whole plot, one vector, for questions such as
    "what is this episode about?".

Continuity is NOT encoded with overlapping text but with
  * a header in the embedded text that carries the episode identity:
        The Simpsons, S03E22 "The Otto Show". Plot, part 2 of 4:
  * metadata that is stored but not embedded: episode_id, season, part,
    n_parts, level. They allow fetching the neighbouring paragraphs when
    answering and aggregating scores per episode when identifying a clip.

    PYTHONPATH=src python -m spanish_align.embed_synopses
    PYTHONPATH=src python -m spanish_align.embed_synopses --dry-run   # counts only, no API

Output: data/ingest/synopsis_docs.jsonl           (text + metadata)
        data/ingest/cache/synopsis_embeddings.json {doc_id: vector}
Embedding calls are cached by text, so a re-run only pays for what changed
(~1100 texts for seasons 1-10: about 11 batched requests).
"""

import argparse
import json
import math
import re

from .config import CACHE_DIR, SYNOPSES, WORKDIR, ensure_dirs
from .fetch_synopses import normalize_text

SYNOPSIS_DOCS = WORKDIR / "synopsis_docs.jsonl"
SYNOPSIS_EMBEDDINGS = CACHE_DIR / "synopsis_embeddings.json"
MIN_WORDS = 40
MAX_WORDS = 250


def _wc(text: str) -> int:
    return len(text.split())


def _split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(])", text) if s.strip()]


def _split_long(text: str) -> list[str]:
    """A paragraph over MAX_WORDS in near-equal parts, at sentence ends."""
    n = math.ceil(_wc(text) / MAX_WORDS)
    if n <= 1:
        return [text]
    target, parts, cur = _wc(text) / n, [], []
    for sent in _split_sentences(text):
        cur.append(sent)
        if sum(_wc(x) for x in cur) >= target and len(parts) < n - 1:
            parts.append(" ".join(cur)); cur = []
    if cur:
        parts.append(" ".join(cur))
    return parts


def split_plot(plot: str) -> list[str]:
    """Plot text -> paragraphs: merge the short ones, split the long ones."""
    paras = [p.strip() for p in plot.split("\n") if p.strip()]
    merged: list[str] = []
    carry = ""
    for p in paras:                      # a short paragraph joins the NEXT one
        p = (carry + " " + p).strip() if carry else p
        if _wc(p) < MIN_WORDS:
            carry = p
        else:
            merged.append(p); carry = ""
    if carry:                            # a short last one joins the previous
        if merged:
            merged[-1] += " " + carry
        else:
            merged.append(carry)
    out = []
    for p in merged:
        out += _split_long(p)
    return out


def _header(rec: dict, tail: str) -> str:
    return f'The Simpsons, {rec["episode_id"]} "{rec["title"]}". {tail}\n'


def build_docs(synopses: dict) -> list[dict]:
    docs = []
    for eid, rec in sorted(synopses.items()):
        plot = normalize_text(rec["plot"])
        base = {"episode_id": eid, "season": rec["season"],
                "episode": rec["episode"], "title": rec["title"]}
        parts = split_plot(plot)
        for i, text in enumerate(parts, 1):
            docs.append({**base, "doc_id": f"{eid}_P{i}", "level": "paragraph",
                         "part": i, "n_parts": len(parts), "text": text,
                         "embed_text": _header(rec, f"Plot, part {i} of {len(parts)}:") + text})
        whole = " ".join(parts)
        docs.append({**base, "doc_id": f"{eid}_EP", "level": "episode",
                     "part": 0, "n_parts": len(parts), "text": whole,
                     "embed_text": _header(rec, "Plot summary:") + whole})
    return docs


def load_docs() -> list[dict]:
    return [json.loads(l) for l in
            SYNOPSIS_DOCS.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_embeddings() -> dict[str, list[float]]:
    return json.loads(SYNOPSIS_EMBEDDINGS.read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser(description="Embed the episode plots")
    p.add_argument("--dry-run", action="store_true",
                   help="only build the documents and print the counts")
    args = p.parse_args()

    if not SYNOPSES.exists():
        raise SystemExit(f"{SYNOPSES} not found: run fetch_synopses first")
    ensure_dirs()
    synopses = json.loads(SYNOPSES.read_text(encoding="utf-8"))
    docs = build_docs(synopses)
    n_par = sum(d["level"] == "paragraph" for d in docs)
    words = [_wc(d["text"]) for d in docs if d["level"] == "paragraph"]
    print(f"{len(synopses)} episodes (seasons "
          f"{sorted({r['season'] for r in synopses.values()})}) -> {n_par} paragraph "
          f"docs + {len(docs) - n_par} episode docs; paragraph words: min "
          f"{min(words)}, max {max(words)}")
    SYNOPSIS_DOCS.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in docs),
                             encoding="utf-8")
    if args.dry_run:
        from .google_client import (EMBED_BATCH_LIMIT, _cache_key,
                                    _load_embedding_cache)
        cache = _load_embedding_cache(CACHE_DIR / "embeddings.json")
        todo = {d["embed_text"] for d in docs
                if _cache_key(d["embed_text"], "RETRIEVAL_DOCUMENT") not in cache}
        print(f"{len(docs) - len(todo)} of {len(docs)} texts already in the "
              f"embeddings cache; {len(todo)} would be sent to the API "
              f"(~{-(-len(todo) // EMBED_BATCH_LIMIT)} batched requests)")
        print(f"(dry run) documents written to {SYNOPSIS_DOCS}; nothing embedded")
        return

    from .google_client import QuotaExhausted, embed_texts   # late: --dry-run needs no API
    try:
        vectors = embed_texts([d["embed_text"] for d in docs],
                              task_type="RETRIEVAL_DOCUMENT")
    except (QuotaExhausted, KeyboardInterrupt, OSError) as e:
        # every finished batch is already in the embeddings cache: re-running
        # the same command only pays for what is left
        raise SystemExit(f"\nStopped before finishing: {e or type(e).__name__}\n"
                         "Progress is saved; run the same command again to resume.")
    SYNOPSIS_EMBEDDINGS.write_text(
        json.dumps({d["doc_id"]: v for d, v in zip(docs, vectors)}), encoding="utf-8")
    print(f"{len(vectors)} embeddings -> {SYNOPSIS_EMBEDDINGS}")


if __name__ == "__main__":
    main()
