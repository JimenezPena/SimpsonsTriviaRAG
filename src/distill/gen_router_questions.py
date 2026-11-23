"""Generate out-of-corpus questions for the router: general + nonrelated.

    PYTHONPATH=src python -m distill.gen_router_questions --calls 20
    PYTHONPATH=src distill/gen_router_questions --calls 20 --per-kind 5 --langs es,en

These are the NEGATIVE-ish classes the router needs to tell apart from Corpus:

  general     about The Simpsons, but the script DB cannot answer it (episode
              air years, "what colour are the Simpsons", merch, production,
              other media...). The retriever still returns plausible-looking
              chunks, so this class is what makes the distance threshold hard.
  nonrelated  nothing to do with the show at all (world cup, software, world
              facts...). Named NONRELATED (not "offtopic") because
              gen_teacher already stores a kind "offtopic" with a DIFFERENT
              meaning (in-corpus question next to unrelated context) and the
              router dataset must not collide with it.

API budget: one Gemini call produces --per-kind questions of EACH kind
(default 5 + 5), so ~calls * 1 request total. Questions arrive as strict
JSON; a failed parse or too-few questions returns None for that call and is
counted, not retried (just run again to top up).

Output: data/distill/router_questions.jsonl, JSONL, one record per question,
resumable (questions already present are never written twice; the model is
asked to avoid ultra-common questions to keep the yield up). Record shape
mirrors gen_teacher.py minus everything the router does not need:

    {"id", "task": "router", "lang", "prompt", "meta": {question, kind,
     generator, temperature, created}}

The next step (embed_router_questions) fills meta.retrieved with top-3
distances from the real DB.
"""

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone

from distill.retry import QuotaExhausted, call_with_retry

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUT = os.path.join(ROOT, "data", "distill", "router_questions.jsonl")

KINDS = ("general", "nonrelated")

BATCH_SYSTEM = (
    "You build a labelled test set for a router that classifies user questions "
    "about The Simpsons into: corpus / metadata / general / nonrelated.\n"
    "* general: about The Simpsons, but the answer is NOT in episode scripts "
    "(air dates, animation, voice actors, merchandising, production, the "
    "franchise in general).\n"
    "* nonrelated: nothing to do with The Simpsons or the episode content in "
    "any way (sports, technology, world facts, health, other shows...).\n"
    "Rules:\n"
    "- Write questions the way a real user types them: short, casual, no "
    "quotation marks around the show name.\n"
    "- Mix Spanish and English across the questions.\n"
    "- general questions must be self-contained and answerable with general "
    "knowledge of the show, NOT from dialogue lines.\n"
    "- Avoid the most obvious clichés; vary topics, characters and phrasing "
    "so questions do not repeat.\n"
    "Return ONLY JSON, exactly this shape, with exactly the requested numbers:\n"
    '{"general": [{"lang": "es|en", "question": "..."}], '
    '"nonrelated": [{"lang": "es|en", "question": "..."}]}'
)


def build_batch_prompt(n_per_kind, langs):
    return (f"Write {n_per_kind} 'general' questions and {n_per_kind} 'nonrelated' "
            f"questions, only in these languages: {', '.join(langs)}.")


def parse_batch(raw, n_per_kind):
    """-> (general, nonrelated) lists of {"lang", "question"} or None on failure.

    Tolerates ```json fences and requires the minimum expected count per
    kind, so a half-hearted answer is thrown away instead of polluting the
    set."""
    m = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return None
    out = {}
    for kind in KINDS:
        items = []
        for it in (data.get(kind) or [])[:n_per_kind]:
            lang = (it.get("lang") or "").strip().lower()
            q = (it.get("question") or "").strip()
            if lang in ("es", "en") and "?" in q and len(q) >= 8:
                items.append({"lang": lang, "question": q})
        if len(items) < max(1, n_per_kind - 1):     # tolerate one short
            return None
        out[kind] = items
    return out["general"], out["nonrelated"]


def load_existing(path):
    seen = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["meta"]["question"].strip().lower())
                except (ValueError, KeyError):
                    continue           # half-written last line after a crash
    return seen


def make_id(lang, question):
    return hashlib.sha1(f"{lang}|{question}".encode()).hexdigest()[:12]


def run(args, tg, retry_sleep=time.sleep):
    def call(fn, *a, **k):
        return call_with_retry(fn, *a, max_retries=args.max_retries,
                               base=args.retry_base, cap=args.retry_cap,
                               sleep=retry_sleep, **k)

    langs = [l.strip().lower() for l in args.langs.split(",") if l.strip()]
    bad = [l for l in langs if l not in ("es", "en")]
    if bad:
        raise SystemExit(f"unknown language(s) {bad}; use es and/or en")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    seen = load_existing(args.out)
    stats = {"calls": 0, "written": 0, "duplicates": 0, "failed_calls": 0}

    with open(args.out, "a", encoding="utf-8") as out:
        while stats["calls"] < args.calls and stats["failed_calls"] <= args.max_errors:
            prompt = build_batch_prompt(args.per_kind, langs)
            try:
                raw = call(tg.ask_llm, prompt, system=BATCH_SYSTEM,
                           temperature=1.0)
            except QuotaExhausted:
                print("daily quota exhausted; run again later to resume")
                break
            except (RuntimeError, OSError) as e:
                print(f"call failed for good: {str(e)[:120]}")
                stats["failed_calls"] += 1
                continue
            stats["calls"] += 1

            parsed = parse_batch(raw, args.per_kind)
            if parsed is None:
                print(f"call {stats['calls']}: unparseable answer, skipped")
                stats["failed_calls"] += 1
                continue
            for kind, items in zip(KINDS, parsed):
                for it in items:
                    question, lang = it["question"], it["lang"]
                    key = question.strip().lower()
                    if key in seen:
                        stats["duplicates"] += 1
                        continue
                    seen.add(key)
                    rec = {
                        "id": make_id(lang, question),
                        "task": "router",
                        "lang": lang,
                        "prompt": question,
                        "meta": {
                            "question": question,
                            "kind": kind,
                            "generator": tg.LLM_MODEL,
                            "temperature": 1.0,
                            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        },
                    }
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    stats["written"] += 1
            out.flush()
            print(f"call {stats['calls']}/{args.calls}: total written={stats['written']}")

    print(json.dumps(stats, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calls", type=int, default=20,
                    help="number of LLM calls (each yields per-kind general + per-kind nonrelated)")
    ap.add_argument("--per-kind", type=int, default=5)
    ap.add_argument("--langs", default="es,en", help="comma-separated: es,en")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--max-errors", type=int, default=5)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--retry-base", type=float, default=5.0)
    ap.add_argument("--retry-cap", type=float, default=120.0)
    args = ap.parse_args()

    import trivia_gen as tg          # needs DB? no: only ask_llm + LLM_MODEL
    run(args, tg)


if __name__ == "__main__":
    main()