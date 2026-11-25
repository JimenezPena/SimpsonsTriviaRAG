"""Evaluate the router on data/agentic/router_eval.jsonl (hand-written, labelled).

    PYTHONPATH=src python -m agentic.eval_router                 # offline: the gate only, 0 API calls
    PYTHONPATH=src python -m agentic.eval_router --llm --limit 5 # try it on 5 questions first
    PYTHONPATH=src python -m agentic.eval_router --llm           # what production does: gate + LLM
    PYTHONPATH=src python -m agentic.eval_router --llm --ungated # also LLM-route the gated-out ones
    PYTHONPATH=src python -m agentic.eval_router --zone data/distill/teacher_qa.jsonl \\
        data/distill/router_questions.jsonl --limit 30           # the questions above the
                                                                  # distance thresholds, 30 LLM calls
    ... --quiet                                                  # summary only

Valid answers are cached in data/agentic/router_cache.jsonl (git-ignored), keyed by
question and router prompt, so repeated runs cost nothing (unparsable answers
are never cached: a rerun retries them); a run stops after 3
consecutive API errors (quota) and resumes later.

It prints one block per question (ok / original tag / what the router decided
and what production would do), then the summary. `ok` is judged on what
production does: a catalog question must be answered with the right tool and
arguments, everything else must be left to the RAG.

Metrics (production behaviour = the router ACTS only when it says `metadata`
with a valid tool call):
  metadata solved   metadata questions answered with the right tool + arguments
  false actions     non-metadata questions the router wrongly answered from the catalog
  diverted          corpus questions the router sent to general / off-topic / catalog
                    (the harmful error: the RAG should have answered them)
  route accuracy    4-way route over the questions that reached the LLM
"""
import argparse
import collections
import hashlib
import json
import os
import sys
import time

from agentic import policy, router, tools
from agentic.catalog import ROOT, norm

DEFAULT_CASES = os.path.join(ROOT, "data", "agentic", "router_eval.jsonl")
DEFAULT_CACHE = os.path.join(ROOT, "data", "agentic", "router_cache.jsonl")


def load_cases(path=DEFAULT_CASES):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def args_match(expected, got):
    """Same arguments: None/missing are equal, titles compare normalized."""
    def clean(d):
        return {k: (norm(v) if isinstance(v, str) else v)
                for k, v in (d or {}).items() if v is not None}
    try:
        return clean(expected) == clean(got)
    except (TypeError, AttributeError):
        return False


def evaluate(cases, ask=None, use_gate=True, ungated=False, max_errors=3, sleep=0.0):
    """-> rows. Without `ask` only the gate is evaluated (no API)."""
    rows, errors = [], 0
    for c in cases:
        gate = router.looks_like_metadata(c["q"])
        row = {"case": c, "gate": gate, "decision": None}
        if ask is not None and (gate or ungated or not use_gate):
            d = router.decide(c["q"], ask, use_gate=False)
            row["decision"] = d
            errors = errors + 1 if d["route"] is None and d["raw"] is None else 0
            if errors >= max_errors:
                print(f"{max_errors} consecutive API errors: stopping (quota?); "
                      "run again later, cached answers are kept", file=sys.stderr)
                rows.append(row)
                break
            if sleep:
                time.sleep(sleep)
        rows.append(row)
    return rows


def acted(decision):
    """Would the router have answered itself? (valid metadata call)"""
    if not decision or decision["route"] != "metadata" or not decision["tool"]:
        return False
    _, err = tools.validate_call(decision["tool"], decision["args"])
    return err is None


def summarize(rows):
    meta = [r for r in rows if r["case"]["route"] == "metadata"]
    other = [r for r in rows if r["case"]["route"] != "metadata"]
    out = {
        "n": len(rows),
        "gate_recall": sum(r["gate"] for r in meta) / len(meta) if meta else None,
        "gate_pass_other": sum(r["gate"] for r in other) / len(other) if other else None,
    }
    judged = [r for r in rows if r["decision"] is not None]
    if judged:
        solved = sum(
            1 for r in meta if r["decision"] is not None and acted(r["decision"])
            and r["decision"]["tool"] == r["case"]["tool"]
            and args_match(r["case"]["args"], r["decision"]["args"]))
        out["metadata_solved"] = (solved, len(meta))
        out["missed_by_gate"] = sum(1 for r in meta if r["decision"] is None)
        out["false_actions"] = (
            sum(1 for r in other if acted(r["decision"])), len(other))
        corpus = [r for r in rows if r["case"]["route"] == "corpus" and r["decision"] is not None]
        out["diverted"] = (sum(1 for r in corpus if r["decision"]["route"] in
                               ("general", "offtopic") or acted(r["decision"])), len(corpus))
        usable = [r for r in judged if r["decision"]["route"] is not None]
        out["route_accuracy"] = (     # an unparsable answer counts as wrong
            sum(r["decision"]["route"] == r["case"]["route"] for r in usable), len(judged))
        out["unparsed"] = len(judged) - len(usable)
        conf = collections.Counter((r["case"]["route"], r["decision"]["route"]) for r in usable)
        out["confusion"] = {f"{a}->{b}": n for (a, b), n in sorted(conf.items())}
        out["calls"] = len(judged)
    return out


def _pct(x):
    return "n/a (no such questions in this sample)" if x is None else f"{x:.0%}"


def sample(cases, n):
    """First `n` cases taken round-robin across routes, so a small sample still
    contains every route (the file lists them grouped by route)."""
    if not n or n >= len(cases):
        return list(cases)
    queues = collections.OrderedDict()
    for c in cases:
        queues.setdefault(c["route"], []).append(c)
    out = []
    while len(out) < n:
        for q in queues.values():
            if q and len(out) < n:
                out.append(q.pop(0))
    return out


def outcome(row):
    """What production would do with this question and whether that is right.
    -> (ok, verdict text). Every route acts now: a catalog question needs the
    right tool and arguments, a general / off-topic one the matching route, and a
    corpus one must stay with the RAG (anything else is a diverted question)."""
    c, d = row["case"], row["decision"]
    if d is None:
        if row["gate"]:
            return None, "(gate passed: would call the router LLM; run with --llm)"
        if c["route"] == "corpus":
            return True, "-> RAG (gate: no LLM call)"
        return None, "(decided by the retrieval distance, not by the gate: use --zone)"
    if d["route"] is None:
        raw = " ".join((d["raw"] or "no answer from the API").split())
        return False, f"UNPARSABLE -> RAG | {raw[:200]}"
    if acted(d):
        call = f"{d['tool']}({', '.join(f'{k}={v!r}' for k, v in d['args'].items())})"
        good = c["route"] == "metadata" and d["tool"] == c["tool"] \
            and args_match(c["args"], d["args"])
        return good, f"{d['route']} -> answers itself: {call}"
    good = d["route"] == c["route"]
    action = {"corpus": "RAG", "general": "general-knowledge answer",
              "offtopic": "refusal (if distance allows)"}.get(d["route"], "RAG (invalid tool call)")
    return good, f"{d['route']} -> {action}"


def format_cases(rows):
    """One line per question: production outcome, original tag, router verdict."""
    lines = [f"{'#':>2}  ok  {'tag':<9} verdict", "-" * 100]
    for i, r in enumerate(rows, 1):
        ok, verdict = outcome(r)
        mark = "?" if ok is None else ("OK" if ok else "XX")
        lines.append(f"{i:>2}  {mark:<3} {r['case']['route']:<9} {verdict}")
        lines.append(f"      {r['case']['q']}")
    return "\n".join(lines)


def format_report(s):
    lines = [f"cases: {s['n']}",
             f"gate recall on metadata questions: {_pct(s['gate_recall'])}  "
             f"(questions the gate would let reach the router)",
             f"gate pass rate on the others: {_pct(s['gate_pass_other'])}  "
             f"(= extra LLM calls spent on non-metadata questions)"]
    if "calls" in s:
        lines += [f"LLM calls: {s['calls']} ({s['unparsed']} unparsable)",
                  "metadata solved (right tool + args): %d/%d" % s["metadata_solved"],
                  f"metadata missed by the gate: {s['missed_by_gate']}",
                  "false actions (wrongly answered from the catalog): %d/%d" % s["false_actions"],
                  "diverted (corpus questions sent to general / off-topic / catalog): %d/%d"
                  % s["diverted"],
                  "route accuracy: %d/%d" % s["route_accuracy"],
                  "confusion (expected->predicted): " + json.dumps(s["confusion"])]
    return "\n".join(lines)


def cached_ask(ask, path):
    cache = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if router.parse_route(r["raw"]) is not None:   # old garbage is retried
                        cache[r["key"]] = r["raw"]
                except (ValueError, KeyError):
                    continue

    def wrapped(prompt, system=None, temperature=0.0):
        key = hashlib.sha1(
            f"v{router.ROUTER_PROMPT_VERSION}|{hashlib.sha1((system or '').encode()).hexdigest()}|{prompt}"
            .encode()).hexdigest()
        if key in cache:
            return cache[key]
        raw = ask(prompt, system=system, temperature=temperature)
        if router.parse_route(raw) is not None:       # never cache garbage: the rerun retries it
            cache[key] = raw
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "raw": raw}, ensure_ascii=False) + "\n")
        return raw
    return wrapped


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--cases", default=DEFAULT_CASES)
    p.add_argument("--zone", nargs="+", metavar="FILE",
                   help="evaluate on the questions of these data files (gen_teacher / "
                        "gen_router_questions records, with distances) that are ABOVE the "
                        "distance thresholds, i.e. exactly what the router sees; implies --ungated")
    p.add_argument("--llm", action="store_true", help="call the router LLM (uses API quota)")
    p.add_argument("--ungated", action="store_true",
                   help="also route the questions the gate would skip")
    p.add_argument("--limit", type=int, default=0,
                   help="only N cases, taken round-robin across routes")
    p.add_argument("--sleep", type=float, default=2.0)
    p.add_argument("--cache", default=DEFAULT_CACHE)
    p.add_argument("--quiet", action="store_true", help="only the summary, not case by case")
    args = p.parse_args(argv)

    if args.zone:
        th = policy.thresholds(os.environ)
        cases, stats = policy.zone_cases(args.zone, th)
        print(policy.format_stats(stats, th), "\n")
        args.ungated = True
    else:
        cases = load_cases(args.cases)
    cases = sample(cases, args.limit)
    ask = None
    if args.llm:
        sys.path.insert(0, os.path.join(ROOT, "src"))
        import trivia_gen as tg            # needs GOOGLE_API_KEY
        os.makedirs(os.path.dirname(args.cache), exist_ok=True)
        ask = cached_ask(tg.ask_llm, args.cache)
    rows = evaluate(cases, ask, ungated=args.ungated, sleep=args.sleep if args.llm else 0)
    if not args.quiet:
        print(format_cases(rows), "\n")
    print(format_report(summarize(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
