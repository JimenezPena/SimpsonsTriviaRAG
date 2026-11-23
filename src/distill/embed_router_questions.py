"""Embed the router questions and attach their real top-3 retrieval distances.

    PYTHONPATH=src python -m distill.embed_router_questions                 # full run
    PYTHONPATH=src python -m distill.embed_router_questions --limit 10      # smoke test

Input : data/distill/router_questions.jsonl (gen_router_questions output).
Output: same file, updated IN PLACE: every record whose meta lacks
        "retrieved" runs the REAL retriever (trivia_gen.retrieve, script
        chunks only, same TOP_K as production) and stores meta.retrieved as
        gen_teacher does:

            [{"season", "episode_id", "episode_title", "line_start",
              "line_end", "distance"}, ...]

Resumable: records already carrying "retrieved" are skipped, so re-running
only processes the new questions (1 embedding per pending question, the same
retrieval path as gen_teacher and production). The vector DB must be
in place and GOOGLE_API_KEY set.
"""

import argparse
import json
import os
import time

from distill.gen_teacher import slim
from distill.retry import QuotaExhausted, call_with_retry

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_PATH = os.path.join(ROOT, "data", "distill", "router_questions.jsonl")


def annotate(row, hits):
    """hits: list of (Document, score) from trivia_gen.retrieve, best first."""
    row["meta"]["retrieved"] = [slim(d.metadata, s) for d, s in hits]
    return row


def run(args, tg, retry_sleep=time.sleep):
    def call(fn, *a, **k):
        return call_with_retry(fn, *a, max_retries=args.max_retries,
                               base=args.retry_base, cap=args.retry_cap,
                               sleep=retry_sleep, **k)

    if not os.path.exists(args.path):
        raise SystemExit(f"{args.path} not found; run gen_router_questions first")

    with open(args.path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    pending = [r for r in rows if "retrieved" not in (r.get("meta") or {})]
    if args.limit:
        pending = pending[:args.limit]
    print(f"{len(rows)} questions, {len(pending)} pending")
    if not pending:
        return

    store = call(tg.load_vector_store)         # needs the DB and GOOGLE_API_KEY

    n = 0
    for row in pending:
        # the exact same path as production / gen_teacher: 1 embedding per
        # query, script chunks only, TOP_K = 3
        try:
            hits = call(tg.retrieve, store, row["prompt"], None, k=tg.TOP_K)
        except QuotaExhausted:
            break
        if not hits:
            print(f"  no hits for {row['id']}, left pending")
            continue
        annotate(row, hits)
        n += 1

    with open(args.path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"annotated {n}/{len(pending)} questions -> {args.path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", default=DEFAULT_PATH)
    ap.add_argument("--limit", type=int, default=None, help="annotate at most N questions")
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--retry-base", type=float, default=5.0)
    ap.add_argument("--retry-cap", type=float, default=120.0)
    args = ap.parse_args()

    try:
        import trivia_gen as tg
    except QuotaExhausted:
        raise
    try:
        run(args, tg)
    except QuotaExhausted:
        print("quota exhausted; run again later, it resumes where it stopped")


if __name__ == "__main__":
    main()