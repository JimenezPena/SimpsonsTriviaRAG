"""Cheap, deterministic metrics to compare teacher / base student / LoRA student.

    PYTHONPATH=src python -m distill.metrics --data data/distill/test.jsonl \\
        --preds base=preds_base.jsonl lora=preds_lora.jsonl

The teacher row is built from the references in the data file itself. Columns:

  lang_ok        answer is in the requested language (stopword heuristic)
  <=6 sent       respects the "brief" instruction
  cites          non-abstaining answers that name an episode present in the
                 context (the teacher's prompt asks for a citation)
  abstain agree  student abstains exactly when the teacher abstains
  overconfident  among questions where the TEACHER abstains, how often the
                 student answers anyway (grounding failure; lower is better)
  chrF           similarity to the teacher's answer (needs `pip install sacrebleu`)
  p50 / p95 s    generation latency per answer (batch 1)
  tok/s          generated tokens per second

These are proxies, not quality. Read a sample of answers by hand and, when
the pipeline is validated, add an LLM judge; chrF in particular punishes
valid rephrasings.
"""
import argparse
import json
import re

from distill.build_dataset import MAX_SENTENCES, is_abstention, lang_ok, n_sentences
from distill.sft_data import read_jsonl

TITLE_RE = re.compile(r'\[(?:Season|Plot summary \(Wikipedia\), season) ?(\d+), "([^"]+)"\]')

try:                                  # optional dependency
    from sacrebleu.metrics import CHRF
    _CHRF = CHRF()
except Exception:                     # noqa: BLE001
    _CHRF = None


def context_titles(prompt):
    return {t.lower() for _, t in TITLE_RE.findall(prompt)}


def score_one(row, response):
    ref = row["messages"][-1]["content"]
    prompt = row["messages"][1]["content"]
    abst, ref_abst = is_abstention(response), is_abstention(ref)
    titles = context_titles(prompt)
    cites = None
    if not abst and titles:
        cites = any(t in response.lower() for t in titles)
    return {
        "lang_ok": lang_ok(response, row["meta"]["lang"]),
        "short": n_sentences(response) <= MAX_SENTENCES,
        "abstain_agree": abst == ref_abst,
        "ref_abstain": ref_abst,
        "answered": not abst,
        "cites": cites,
        "chrf": _CHRF.sentence_score(response, [ref]).score if _CHRF and response else None,
    }


def _mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def percentile(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(rows, preds):
    """rows: SFT rows; preds: id -> {"response", optional "latency", "n_tokens"}."""
    pairs = [(row, preds[row["id"]]) for row in rows if row["id"] in preds]
    scored = [score_one(r, p["response"]) for r, p in pairs]
    lat = [p["latency"] for _, p in pairs if p.get("latency")]
    toks = sum(p.get("n_tokens") or 0 for _, p in pairs if p.get("latency"))
    return {
        "n": len(pairs),
        "lang_ok": _mean(s["lang_ok"] for s in scored),
        "short": _mean(s["short"] for s in scored),
        "cites": _mean(s["cites"] for s in scored),
        "abstain_agree": _mean(s["abstain_agree"] for s in scored),
        "overconfident": _mean(s["answered"] for s in scored if s["ref_abstain"]),
        "chrf": _mean(s["chrf"] for s in scored),
        "p50": percentile(lat, 0.5), "p95": percentile(lat, 0.95),
        "tok_s": toks / sum(lat) if lat and toks else None,
    }


def by_group(rows, preds):
    """{'all': summary, 'es': summary, 'en': summary, ...}"""
    out = {"all": summarize(rows, preds)}
    for lang in sorted({r["meta"]["lang"] for r in rows}):
        out[lang] = summarize([r for r in rows if r["meta"]["lang"] == lang], preds)
    return out


def _fmt(x, pct=False, nd=2):
    if x is None:
        return "-"
    return f"{100 * x:.0f}%" if pct else f"{x:.{nd}f}"


def format_tables(results):
    """results: {system: by_group(...)} -> markdown, one table per group."""
    cols = [("n", "n", None), ("lang_ok", "lang ok", True), ("short", f"<={MAX_SENTENCES} sent", True),
            ("cites", "cites", True), ("abstain_agree", "abstain agree", True),
            ("overconfident", "overconfident", True), ("chrf", "chrF", False),
            ("p50", "p50 s", False), ("p95", "p95 s", False), ("tok_s", "tok/s", False)]
    groups = list(next(iter(results.values())))
    lines = []
    for g in groups:
        lines += [f"\n**{g}**\n", "| system | " + " | ".join(c[1] for c in cols) + " |",
                  "|---|" + "---|" * len(cols)]
        for system, res in results.items():
            s = res.get(g, {})
            cells = [str(s.get("n", 0)) if k == "n" else _fmt(s.get(k), pct)
                     for k, _, pct in cols]
            lines.append(f"| {system} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def load_preds(path):
    return {r["id"]: r for r in read_jsonl(path)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", required=True, help="test.jsonl or val.jsonl")
    p.add_argument("--preds", nargs="*", default=[], metavar="LABEL=FILE")
    p.add_argument("--no-teacher", action="store_true")
    p.add_argument("--json", help="also write the raw numbers here")
    args = p.parse_args(argv)

    rows = read_jsonl(args.data)
    all_preds = {}
    for item in args.preds:
        label, _, path = item.partition("=")
        all_preds[label] = load_preds(path)
    if all_preds:
        # every system is scored on the same questions (e.g. --limit 30 runs)
        common = set.intersection(*(set(p) for p in all_preds.values()))
        rows = [r for r in rows if r["id"] in common]
        print(f"{len(rows)} questions common to all systems")
    results = {}
    if not args.no_teacher:
        results["teacher"] = by_group(
            rows, {r["id"]: {"response": r["messages"][-1]["content"]} for r in rows})
    for label, preds in all_preds.items():
        results[label] = by_group(rows, preds)
    print(format_tables(results))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
