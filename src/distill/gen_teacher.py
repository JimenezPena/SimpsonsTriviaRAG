"""Generate teacher data for distilling the free-form Q&A into a small model.

For every example (2 Gemini calls + 1 embedding call):
  1. sample a random script chunk (the "gold" chunk),
  2. the teacher writes a user-style question about it (or, with probability
     --unanswerable, one it does NOT answer: abstention examples),
  3. the REAL retriever runs on that question (so the context contains the
     retrieval misses production would have),
  4. the teacher answers with the production prompt (ANSWER_SYSTEM + context).

Output: JSONL, one record per line, resumable (re-running appends; a gold
chunk already present is never sampled again). Default location:
data/distill/teacher_qa.jsonl (git-ignored).

Run from the project root, with the vector DB in place and GOOGLE_API_KEY set:

    PYTHONPATH=src python -m distill.gen_teacher --n 10            # smoke test
    PYTHONPATH=src python -m distill.gen_teacher --n 200 --langs es,en

Every API call is retried with growing waits on 429 / 5xx / network errors
(distill/retry.py). A daily quota, or a 429 that outlasts all the retries,
stops the run cleanly: just run it again later and it resumes.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone

from distill.prompts import (LANG_NAMES, build_qa_prompt, build_question_prompt,
                             clean_question, format_context, QUESTION_SYSTEM)
from distill.retry import QuotaExhausted, call_with_retry

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUT = os.path.join(ROOT, "data", "distill", "teacher_qa.jsonl")


def chunk_key(meta):
    return (meta.get("episode_id"), meta.get("line_start"))


def overlaps(a, b):
    """Same episode and overlapping line ranges (chunks overlap by 5 lines)."""
    if a.get("episode_id") != b.get("episode_id"):
        return False
    try:
        return a["line_start"] <= b["line_end"] and b["line_start"] <= a["line_end"]
    except (KeyError, TypeError):
        return False


def slim(meta, distance=None):
    out = {k: meta.get(k) for k in
           ("season", "episode_id", "episode_title", "line_start", "line_end")}
    if distance is not None:
        out["distance"] = round(float(distance), 3)
    return out


def load_existing(path):
    ids, gold_keys = set(), set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue            # a half-written last line after a crash
                ids.add(rec["id"])
                gold_keys.add(chunk_key(rec["meta"]["gold"]))
    return ids, gold_keys


def _direct(fn, *a, **k):
    return fn(*a, **k)


def make_record(tg, store, text, gold_meta, lang, answerable, call=_direct,
                question_temp=0.9, answer_temp=0.2):
    """One teacher example. `call` wraps every network call (retries).
    Raises QuotaExhausted / RuntimeError / OSError when a call finally fails.

    Temperatures: the question writer stays warm (diverse inputs; noise is
    harmless there), the answer is cold (the student should imitate the
    teacher's most likely answer, not one random draw)."""
    q_prompt = build_question_prompt(text, gold_meta["season"],
                                     gold_meta["episode_title"], lang, answerable)
    question = clean_question(call(tg.ask_llm, q_prompt, system=QUESTION_SYSTEM,
                                   temperature=question_temp))
    if question is None:
        return None

    results = call(tg.retrieve, store, question, None, k=tg.TOP_K)
    if not results:
        return None
    sent = results[:tg.TOP_K]
    try:                                  # same optional plot lookup as the app
        plot = call(tg.plot_for_question, store, question, None, results[0][1])
    except QuotaExhausted:
        raise
    except Exception:
        plot = None

    context = format_context(
        [(d.metadata["season"], d.metadata["episode_title"], d.page_content)
         for d, _ in sent], plot)
    prompt = build_qa_prompt(context, question, lang)
    response = call(tg.ask_llm, prompt, system=tg.ANSWER_SYSTEM,
                    temperature=answer_temp)

    gold_in_context = any(overlaps(gold_meta, d.metadata) for d, _ in sent)
    gold_id = f"{gold_meta.get('episode_id')}:{gold_meta.get('line_start')}"
    return {
        "id": hashlib.sha1(f"{lang}|{question}|{gold_id}".encode()).hexdigest()[:12],
        "task": "qa",
        "lang": lang,
        "system": tg.ANSWER_SYSTEM,
        "prompt": prompt,
        "response": response,
        "meta": {
            "question": question,
            "kind": "answerable" if answerable else "unanswerable",
            "gold": slim(gold_meta),
            "gold_in_context": gold_in_context,
            "retrieved": [slim(d.metadata, s) for d, s in sent],
            "plot": ({"scope": plot.get("scope"), "season": plot.get("season"),
                      "title": plot.get("title")} if plot else None),
            "teacher": tg.LLM_MODEL,
            "temperature": {"question": question_temp, "answer": answer_temp},
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }


def run(args, tg, retry_sleep=time.sleep):
    """Main loop; `tg` is the trivia_gen module (or a stand-in in the tests)."""
    def call(fn, *a, **k):
        return call_with_retry(fn, *a, max_retries=args.max_retries,
                               base=args.retry_base, cap=args.retry_cap,
                               sleep=retry_sleep, **k)

    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    bad = [l for l in langs if l not in LANG_NAMES]
    if bad:
        raise SystemExit(f"unknown language(s) {bad}; known: {sorted(LANG_NAMES)}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ids, gold_keys = load_existing(args.out)
    n_before = len(ids)                  # languages keep cycling across resumed runs
    rng = random.Random(args.seed + n_before)
    store = tg.load_vector_store()
    stats = {"written": 0, "skipped": 0, "errors": 0, "quota_stop": False}

    with open(args.out, "a", encoding="utf-8") as out:
        while stats["written"] < args.n and stats["errors"] < args.max_errors:
            # fresh gold chunk (bounded retries: the pool may be nearly exhausted)
            for _ in range(25):
                text, meta = tg.random_chunk(None, args.season)
                if chunk_key(meta) not in gold_keys:
                    break
            else:
                print("could not find an unused chunk; stopping")
                break
            lang = langs[(n_before + stats["written"]) % len(langs)]
            answerable = rng.random() >= args.unanswerable
            try:
                rec = make_record(tg, store, text, meta, lang, answerable, call=call,
                                  question_temp=args.question_temp,
                                  answer_temp=args.answer_temp)
            except QuotaExhausted as e:
                print(f"quota exhausted ({str(e)[:120]}): run again later to resume")
                stats["quota_stop"] = True
                break
            except (RuntimeError, OSError) as e:
                stats["errors"] += 1
                print(f"gave up on one example ({stats['errors']}/{args.max_errors}): "
                      f"{str(e)[:200]}")
                continue
            if rec is None or rec["id"] in ids:
                stats["skipped"] += 1
                continue
            gold_keys.add(chunk_key(meta))
            ids.add(rec["id"])
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            stats["written"] += 1
            print(f"[{stats['written']}/{args.n}] {lang} "
                  f"{rec['meta']['kind'][:5]} gold_in_ctx={rec['meta']['gold_in_context']}"
                  f"  {rec['meta']['question'][:70]}")
            if args.sleep:
                time.sleep(args.sleep)
    return stats


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n", type=int, default=10, help="examples to add (default 10)")
    p.add_argument("--langs", default="es,en",
                   help="comma-separated, cycled one by one (default es,en)")
    p.add_argument("--unanswerable", type=float, default=0.15,
                   help="fraction of questions the gold chunk does NOT answer")
    p.add_argument("--season", type=int, default=None, help="restrict gold chunks")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--sleep", type=float, default=2.0, help="seconds between examples")
    p.add_argument("--question-temp", type=float, default=0.9,
                   help="temperature of the question writer (default 0.9)")
    p.add_argument("--answer-temp", type=float, default=0.2,
                   help="temperature of the teacher's answer (default 0.2)")
    p.add_argument("--max-retries", type=int, default=5,
                   help="retries per API call on 429/5xx/network errors (default 5)")
    p.add_argument("--retry-base", type=float, default=5.0,
                   help="first wait in seconds; doubles each retry (default 5)")
    p.add_argument("--retry-cap", type=float, default=120.0,
                   help="longest single wait in seconds (default 120)")
    p.add_argument("--max-errors", type=int, default=5,
                   help="examples abandoned after their retries before stopping")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    sys.path.insert(0, os.path.join(ROOT, "src"))
    import trivia_gen as tg          # needs GOOGLE_API_KEY and the vector DB
    stats = run(args, tg)
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
