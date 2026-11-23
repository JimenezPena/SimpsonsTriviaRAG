"""Teacher JSONL -> filtered, episode-split SFT files (train / val / test).

    PYTHONPATH=src python -m distill.build_dataset            # keep soft-flagged
    PYTHONPATH=src python -m distill.build_dataset --strict   # drop them too

Filtering (all deterministic, no API calls):
  hard flags -> always dropped: empty, too_long, wrong_lang (only with positive
                evidence: the answer's own prose, without quotes or citation,
                is mostly in another language)
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
      foreign_quote           a quoted passage (>= 5 words) in another language
                              than the answer, typically English script lines
                              inside a Spanish answer

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

# Stopwords per supported language (es / en only for now; to add a language,
# add it here and in prompts.LANG_NAMES). Words that are common in both
# languages ("a", "no", "me", "son") are left out on purpose.
_STOP = {
    "es": set("""el la los las de del que en y un una unos unas es se por con para al lo
                 como más pero su sus sin sobre entre cuando también ya le les ser fue
                 está están son ha han hay muy si ni ese esa esto esta estos quien qué
                 cómo dónde cuál""".split()),
    "en": set("""the of and to in is that it for with was as his he on are not by this at
                 from who whom what which does did has have had her she they their them
                 will would be been were but or an you your""".split()),
}
# Abstention = the answer says the CONTEXT does not contain the answer. The
# negation must share a sentence with a reference to the context/text, so a
# real answer like "Homer says he doesn't know why they left" is not one.
_CTX_REF = re.compile(
    r"\b(?:context|contexto|excerpt|extracto|passage|pasaje|fragment|fragmento|provided|"
    r"proporcionad[oa]s?|given|text|texto|script|gui[oó]n)\b", re.I)
_NO_CONTENT = re.compile(
    r"\b(?:does not|doesn't|do not|don't|did not|cannot|can't|is not|isn't|are not|aren't)\s+"
    r"(?:\w+\s+){0,2}?(?:contain|mention|say|specify|provide|include|state|indicate|reveal|"
    r"give|name|identify|show|have|offer|explain|answer|determine|find)"
    r"|\b(?:not|no)\s+(?:been\s+)?(?:mentioned|specified|provided|stated|found|given|included|indicated)\b"
    r"|\bno (?:\w+\s+)?(?:mention|information|indication|reference)\b"
    r"|\bno\s+(?:se\s+)?(?:contiene|menciona|indica|dice|aparece|proporciona|incluye|especifica|"
    r"figura|consta|revela|nombra|identifica|explica|ofrece|aclara|encuentra|detalla|hay)\b"
    r"|\bno es posible\b|\bno se puede\b", re.I)
# Phrasings that are abstentions even without naming the context.
_ABSTAIN_ALWAYS = re.compile(
    r"\bno (?:consta|figura)\b|\bno (?:hay|tengo|dispongo de|existe) (?:ninguna )?informaci[oó]n\b|"
    r"\bthere is no (?:information|mention|indication)\b|"
    r"(?:^|[.!?]\s+)no se (?:menciona|indica|especifica|dice|sabe|conoce|aclara)\b|"
    r"(?:^|[.!?]\s+)(?:it is|it's) not (?:mentioned|specified|stated|known)\b|"
    r"\b(?:i|we) (?:do not|don't|cannot|can't) (?:know|answer|find|determine)\b|\bno lo s[eé]\b",
    re.I)

# Text that does not carry the answer's language: quoted script lines / titles
# and the episode citation line ("Source: ...", "Episode: ...", "Fuente: ...").
_QUOTED = re.compile(r'"[^"]*"|“[^”]*”|«[^»]*»|‘[^’]*’|(?<!\w)\'[^\'\n]{2,}\'(?!\w)')
_CITATION = re.compile(r"(?im)(?:^|\()\s*(?:source|fuente|episode|episodio)s?\s*:.*$")


def words(text):
    return re.findall(r"[^\W\d_]+", text.lower())


def stopword_counts(text):
    """{lang: stopwords of that language found in text}: the evidence behind lang_ok."""
    toks = words(text)
    return {l: sum(t in s for t in toks) for l, s in sorted(_STOP.items())}


def prose(text):
    """The answer without quoted script lines / titles and the citation line:
    only what the teacher wrote itself carries the answer's language."""
    return _CITATION.sub("", _QUOTED.sub(" ", text))


def lang_ok(text, lang):
    """False only with POSITIVE evidence of another language: its stopwords
    (counted on prose(text)) number at least 2 and exceed those of `lang`.
    Short answers with little evidence pass, so they are not dropped for
    having a single stopword; an empty answer never passes."""
    if not (text or "").strip():
        return False
    if lang not in _STOP:
        return True                      # unsupported language: cannot judge
    counts = stopword_counts(prose(text))
    other = max((n for l, n in counts.items() if l != lang), default=0)
    return not (other >= 2 and other > counts[lang])


def foreign_quotes(text, lang, ignore_titles=()):
    """Quoted segments (>= 5 words) clearly written in another language than
    the requested one, e.g. English script lines inside a Spanish answer.
    Episode titles (`ignore_titles`) and the citation line are not counted."""
    ignore = {t.lower() for t in ignore_titles}
    out = []
    for m in _QUOTED.finditer(_CITATION.sub("", text)):
        seg = m.group(0)[1:-1]
        if seg.strip(" .,;:!?").lower() in ignore or len(words(seg)) < 5:
            continue
        c = stopword_counts(seg)
        other = max((n for l, n in c.items() if l != lang), default=0)
        if other >= 2 and other > c.get(lang, 0):
            out.append(seg)
    return out


def n_sentences(text):
    return len([s for s in re.split(r"[.!?]+(?:\s|$)", text) if s.strip()])


def is_abstention(text):
    """True when the answer says the context does not contain the answer
    (also when it then adds what the context does say). Quoted script lines
    are ignored."""
    t = _QUOTED.sub(" ", text or "")
    if _ABSTAIN_ALWAYS.search(t):
        return True
    return any(_CTX_REF.search(s) and _NO_CONTENT.search(s)
               for s in re.split(r"(?<=[.!?;:])\s+|\n+", t))


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
    titles = {r["episode_title"].lower() for r in meta["retrieved"] if r.get("episode_title")}
    if not abst:
        if titles and not any(t in resp.lower() for t in titles):
            soft.append("no_citation")
    gold_title = (meta.get("gold") or {}).get("episode_title")
    if foreign_quotes(resp, rec["lang"], titles | ({gold_title} if gold_title else set())):
        soft.append("foreign_quote")        # e.g. English script lines in a Spanish answer
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


def build(records, seed=0, strict=False, val=0.1, test=0.1):
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
        split = split_of(rec["meta"]["gold"]["episode_id"], seed, val, test)
        splits[split].append(to_sft(rec, soft))
    return splits, dropped


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--inp", default=DEFAULT_OUT)
    p.add_argument("--outdir", default=os.path.join(ROOT, "data", "distill"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--strict", action="store_true")
    p.add_argument("--val-frac", type=float, default=0.1,
                   help="share of EPISODES (not examples) sent to val (default 0.1)")
    p.add_argument("--test-frac", type=float, default=0.1,
                   help="share of EPISODES sent to test (default 0.1)")
    args = p.parse_args(argv)
    if args.val_frac < 0 or args.test_frac < 0 or args.val_frac + args.test_frac >= 1:
        p.error("--val-frac and --test-frac must be >= 0 and add up to less than 1")

    with open(args.inp, encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]
    splits, dropped = build(records, args.seed, args.strict, args.val_frac, args.test_frac)

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
