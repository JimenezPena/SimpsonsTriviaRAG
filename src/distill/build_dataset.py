"""Teacher JSONL -> filtered, episode-split SFT files (train / val / test).

    PYTHONPATH=src python -m distill.build_dataset            # keep soft-flagged
    PYTHONPATH=src python -m distill.build_dataset --strict   # drop them too

Filtering (all deterministic, no API calls):
  hard flags -> always dropped: empty, too_long, wrong_lang
  soft flags -> kept unless --strict; they mark answers the context may not
                support (the dangerous ones to distill):
      answered_without_gold   the gold chunk was NOT retrieved but the teacher
                              answered instead of saying it does not know
      abstained_with_gold     the gold chunk WAS retrieved and the teacher
                              still said it does not know
      unanswerable_but_answered   near-miss question the chunk does not answer,
                              yet the teacher answered
      offtopic_but_answered   context was random chunks from other episodes,
                              yet the teacher answered (from its own knowledge)
      no_citation             a real answer that cites none of the episodes
                              in its context

Split: by EPISODE (hash of episode_id), never by example. Chunks overlap by
5 lines and one episode shares jokes and names across chunks, so a random
split leaks test content into training.
"""
import argparse
import collections
import hashlib
import json
import os
import re

from distill.gen_teacher import DEFAULT_OUT, ROOT

MAX_SENTENCES = 6      # ANSWER_SYSTEM asks for max 4 + the citation

_STOP = {
    "es": set("el la los las de que en y un una es se por con para del no lo al como más pero su".split()),
    "en": set("the of and to in is that it for with was as his he on are not by this at from".split()),
    "ca": set("el la els les de que en i un una és es per amb per al com més però seu".split()),
    "de": set("der die das und ist nicht ein eine mit von zu den dem auf für sich es im".split()),
}
_ABSTAIN = re.compile(
    r"no (se )?(contiene|menciona|indica|dice|hay|aparece|puedo|encuentro)|"
    r"(el|este) contexto no|no (consta|figura)|"
    r"(does not|doesn't|do not|don't) (contain|mention|say|specify|provide|include|know)|"
    r"not (mentioned|specified|provided|stated|found)|no information|"
    r"(enth[äa]lt|erw[äa]hnt) (keine|nicht)|no (hi ha|conté|s'esmenta)", re.I)


def words(text):
    return re.findall(r"[^\W\d_]+", text.lower())


def lang_ok(text, lang):
    """Crude check: more stopwords of `lang` than of any other language."""
    toks = words(text)
    own = sum(t in _STOP.get(lang, ()) for t in toks)
    other = max((sum(t in s for t in toks) for l, s in _STOP.items() if l != lang),
                default=0)
    return own >= 2 and own >= other


def stopword_counts(text):
    """{lang: stopwords of that language found in text}: the evidence behind lang_ok."""
    toks = words(text)
    return {l: sum(t in s for t in toks) for l, s in sorted(_STOP.items())}


def n_sentences(text):
    return len([s for s in re.split(r"[.!?]+(?:\s|$)", text) if s.strip()])


def is_abstention(text):
    return bool(_ABSTAIN.search(text))


def flags_for(rec):
    """-> (hard, soft) lists of flag names."""
    resp = (rec.get("response") or "").strip()
    meta = rec["meta"]
    hard, soft = [], []
    if not resp:
        return ["empty"], soft
    if n_sentences(resp) > MAX_SENTENCES:
        hard.append("too_long")
    if not lang_ok(resp, rec["lang"]):
        hard.append("wrong_lang")

    abst = is_abstention(resp)
    answerable = meta["kind"] == "answerable"
    if answerable and not meta["gold_in_context"] and not abst:
        soft.append("answered_without_gold")
    if answerable and meta["gold_in_context"] and abst:
        soft.append("abstained_with_gold")
    if not answerable and not abst:       # unanswerable_ / offtopic_but_answered
        soft.append(f"{meta['kind']}_but_answered")
    if not abst:
        titles = {r["episode_title"].lower() for r in meta["retrieved"]
                  if r.get("episode_title")}
        if titles and not any(t in resp.lower() for t in titles):
            soft.append("no_citation")
    return hard, soft


def split_of(episode_id, seed=0, val=0.1, test=0.1):
    """Stable assignment by hash: adding data later never moves an episode."""
    h = int(hashlib.sha1(f"{seed}:{episode_id}".encode()).hexdigest(), 16) % 10_000
    h /= 10_000
    return "test" if h < test else "val" if h < test + val else "train"


def to_sft(rec, flags):
    return {
        "id": rec["id"],
        "messages": [{"role": "system", "content": rec["system"]},
                     {"role": "user", "content": rec["prompt"]},
                     {"role": "assistant", "content": rec["response"].strip()}],
        "meta": {"lang": rec["lang"], "kind": rec["meta"]["kind"],
                 "gold": rec["meta"]["gold"],
                 "gold_in_context": rec["meta"]["gold_in_context"],
                 "question": rec["meta"]["question"], "flags": flags},
    }


def build(records, seed=0, strict=False):
    """-> (dict split -> [sft records], Counter of drop reasons)."""
    splits = {"train": [], "val": [], "test": []}
    dropped = collections.Counter()
    for rec in records:
        hard, soft = flags_for(rec)
        if hard:
            dropped.update(hard)
            continue
        if strict and soft:
            dropped.update(soft)
            continue
        split = split_of(rec["meta"]["gold"]["episode_id"], seed)
        splits[split].append(to_sft(rec, soft))
    return splits, dropped


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--inp", default=DEFAULT_OUT)
    p.add_argument("--outdir", default=os.path.join(ROOT, "data", "distill"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--strict", action="store_true")
    args = p.parse_args(argv)

    with open(args.inp, encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]
    splits, dropped = build(records, args.seed, args.strict)

    os.makedirs(args.outdir, exist_ok=True)
    for name, rows in splits.items():
        with open(os.path.join(args.outdir, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    kept = sum(map(len, splits.values()))
    print(f"read {len(records)} | kept {kept} | "
          + " ".join(f"{k}={len(v)}" for k, v in splits.items()))
    print("dropped:", dict(dropped) or "-")
    langs = collections.Counter(r["meta"]["lang"] for v in splits.values() for r in v)
    kinds = collections.Counter(r["meta"]["kind"] for v in splits.values() for r in v)
    hit = collections.defaultdict(lambda: [0, 0])      # answerable questions only
    for v in splits.values():
        for r in v:
            if r["meta"]["kind"] == "answerable":
                h = hit[r["meta"]["lang"]]
                h[0] += bool(r["meta"]["gold_in_context"])
                h[1] += 1
    soft = collections.Counter(f for v in splits.values() for r in v for f in r["meta"]["flags"])
    print("by language:", dict(langs))
    print("by kind:", dict(kinds))
    print("answerable, gold chunk retrieved:",
          {l: f"{a}/{b} ({100 * a / b:.0f}%)" for l, (a, b) in sorted(hit.items())} or "-")
    print("soft flags kept:", dict(soft) or "-")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
