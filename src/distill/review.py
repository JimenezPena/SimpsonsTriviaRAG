"""Hand-review of the records that build_dataset flags or drops.

    PYTHONPATH=src python -m distill.review --flag abstained_with_gold --n 20
    PYTHONPATH=src python -m distill.review --flag wrong_lang --lang es --n 20
    PYTHONPATH=src python -m distill.review --flag abstained_with_gold --prompt-version 2

Writes a markdown file (default data/distill/review_<flag>.md) with one block
per case: the question, the teacher's answer and the evidence needed to judge
it (for abstained_with_gold, the retrieved chunk that contains the source of
the question; for wrong_lang, the stopword counts the heuristic used).
Fill in the `Verdict:` line of each block with the letter from the legend at
the top, then count them:

    PYTHONPATH=src python -m distill.review --tally data/distill/review_wrong_lang.md

Nothing is lost by the filters: build_dataset always re-reads
teacher_qa.jsonl, so if the review shows the heuristic is wrong, fixing it
recovers the dropped examples without new API calls.
"""
import argparse
import collections
import json
import os
import random
import re

from distill.build_dataset import flags_for, stopword_counts
from distill.gen_teacher import DEFAULT_OUT, ROOT, overlaps

LEGENDS = {
    "abstained_with_gold": [
        "A = the chunk does NOT answer the question (the question writer's fault)",
        "B = the chunk answers it; the teacher was too cautious",
        "C = ambiguous / only partly answered"],
    "wrong_lang": [
        "A = the answer really is in the wrong language",
        "B = right language; the heuristic is wrong (English quotes, names, short answer...)",
        "C = mixed languages"],
}
DEFAULT_LEGEND = ["A = the flag is right (the example should be dropped)",
                  "B = the flag is wrong (the example is fine)",
                  "C = unsure"]


def context_blocks(prompt):
    """Blocks of the context of a production prompt, split on the separator."""
    ctx = prompt.split("Context:\n\n", 1)[-1].rsplit("\n\nQuestion: ", 1)[0]
    return ctx.split("\n\n---\n\n")


def gold_chunks(rec):
    """Texts of the retrieved chunks that overlap the chunk the question was
    written from ([] if it was not retrieved or cannot be isolated)."""
    meta = rec["meta"]
    n = len(meta["retrieved"])
    blocks = context_blocks(rec["prompt"])
    if not n or len(blocks) < n:           # a chunk contained the separator itself
        return []
    return [b for b, r in zip(blocks[-n:], meta["retrieved"]) if overlaps(meta["gold"], r)]


def select(records, flag, lang=None, prompt_version=None):
    """-> [(record, hard_flags, soft_flags)] carrying `flag`."""
    out = []
    for rec in records:
        if lang and rec["lang"] != lang:
            continue
        if prompt_version is not None and rec["meta"].get("question_prompt", 1) != prompt_version:
            continue
        hard, soft = flags_for(rec)
        if flag in hard + soft:
            out.append((rec, hard, soft))
    return out


def _quote(text):
    return ["> " + line if line else ">" for line in (text or "(empty)").splitlines()]


def render(rec, flag, i):
    m = rec["meta"]
    g = m["gold"]
    lines = [f"## {i}. `{rec['id']}` · {rec['lang']} · {m['kind']} · "
             f"question prompt v{m.get('question_prompt', 1)}", "",
             f"**Question:** {m['question']}", "",
             "**Teacher answer:**", "", *_quote(rec["response"]), ""]
    if flag == "wrong_lang":
        counts = stopword_counts(rec["response"] or "")
        lines += ["Stopwords found: " + ", ".join(f"{l}={n}" for l, n in counts.items())
                  + f" (requested: **{rec['lang']}**)", ""]
    lines.append(f"Source of the question: S{g['season']} \"{g['episode_title']}\" "
                 f"lines {g['line_start']}-{g['line_end']}")
    lines.append("Retrieved: " + "; ".join(
        f"S{r['season']} \"{r['episode_title']}\"" + (f" d={r['distance']}" if "distance" in r else "")
        for r in m["retrieved"]))
    if flag != "wrong_lang":
        if m["gold_in_context"]:
            chunks = gold_chunks(rec)
            lines += ["", "**Retrieved chunk that contains the source of the question:**", ""]
            if chunks:
                lines += ["```", *chunks, "```"]
            else:
                lines.append("(could not be isolated from the prompt: read it in teacher_qa.jsonl)")
        else:
            lines += ["", "(the source chunk was NOT retrieved)"]
    lines += ["", "Verdict: ", "", "---", ""]
    return "\n".join(lines)


def build_report(rows, flag, total, shown_note=""):
    legend = LEGENDS.get(flag, DEFAULT_LEGEND)
    head = [f"# Review: `{flag}`", "", shown_note, "", "Fill in `Verdict:` with:", "",
            *[f"- {l}" for l in legend], "", "---", ""]
    body = [render(rec, flag, i) for i, (rec, _, _) in enumerate(rows, 1)]
    return "\n".join(head) + "\n".join(body)


def tally(text):
    """Counts the letters written after `Verdict:` in a reviewed report."""
    letters = re.findall(r"^Verdict:[ \t]*([A-Za-z])", text, flags=re.M)
    pending = len(re.findall(r"^Verdict:[ \t]*$", text, flags=re.M))
    return collections.Counter(l.upper() for l in letters), pending


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--inp", default=DEFAULT_OUT)
    p.add_argument("--flag", help="e.g. abstained_with_gold, wrong_lang, no_citation, "
                                  "answered_without_gold, unanswerable_but_answered")
    p.add_argument("--n", type=int, default=20, help="cases to export (0 = all)")
    p.add_argument("--lang", help="only this language")
    p.add_argument("--prompt-version", type=int,
                   help="only records from this question prompt (1 = no version stored)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out")
    p.add_argument("--tally", metavar="REVIEWED.md", help="count the verdicts in a reviewed file")
    args = p.parse_args(argv)

    if args.tally:
        with open(args.tally, encoding="utf-8") as f:
            counts, pending = tally(f.read())
        print(dict(sorted(counts.items())), f"| {pending} still empty")
        return 0
    if not args.flag:
        p.error("--flag is required (or use --tally)")

    with open(args.inp, encoding="utf-8") as f:
        records = []
        for line in f:
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
    pool = [r for r in records
            if (not args.lang or r["lang"] == args.lang)
            and (args.prompt_version is None
                 or r["meta"].get("question_prompt", 1) == args.prompt_version)]
    rows = select(pool, args.flag)
    sample = list(rows)
    random.Random(args.seed).shuffle(sample)
    if args.n:
        sample = sample[:args.n]
    out = args.out or os.path.join(ROOT, "data", "distill", f"review_{args.flag}.md")
    note = (f"{len(rows)} of {len(pool)} records are flagged; "
            f"showing {len(sample)} (random sample, seed {args.seed}).")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(build_report(sample, args.flag, len(pool), note))
    print(note, "->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
