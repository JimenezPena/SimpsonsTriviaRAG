"""Does the synopsis help to identify a clip's EPISODE?

Ground truth: the clips already confirmed in ig_links.jsonl (their
transcripts are in the cache). For each one the clip transcript is used as a
query and the episodes are ranked by:

  baseline        best chunk of the episode (the current method)
  syn_episode     similarity to the whole-plot vector
  syn_para_max    best plot paragraph
  syn_para_top2   mean of the two best paragraphs
  fusion          baseline + syn_para_max (reciprocal-rank fusion)
  fusion3         baseline + syn_para_max + syn_episode

Reported: how often the true episode is 1st / in the top 3 / top 5, and its
mean rank, for all the clips and split by the confidence with which each clip
was linked:

  auto    linked automatically (match score >= MATCH_AUTO_ACCEPT): the
          baseline was confident, and the link was not checked by a human
  review  the match was doubtful (score below the threshold, or approved
          through the review queue): the baseline was weak here and a human
          decided, so this is where a synopsis could add something

Needs: export_from_db, embed_synopses and the clip transcripts.

    PYTHONPATH=src python -m spanish_align.eval_synopses
    PYTHONPATH=src python -m spanish_align.eval_synopses --verbose   # per clip

Caveat: the confirmed links were found with the baseline method (plus your
review), so the baseline starts with an advantage; a synopsis variant that
merely ties it is already useful, because it can fix what the baseline got
wrong or left in the review queue.
"""

import argparse
import json
from collections import defaultdict

import numpy as np

from .config import (CHUNK_EMBEDDINGS, CHUNKS_EN, IG_LINKS, MATCH_AUTO_ACCEPT,
                     REVIEW_QUEUE, TRANSCRIPTS_DIR)
from .embed_synopses import load_docs, load_embeddings

RRF_K = 10


def _unit(vectors) -> np.ndarray:
    m = np.asarray(vectors, dtype=float)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def episode_scores(query: np.ndarray, doc_vecs: np.ndarray, doc_eps: list[str],
                   agg: str) -> dict[str, float]:
    """Cosine of one (unit) query to every doc, aggregated per episode:
    'max' or 'top2' (mean of the two best)."""
    sims = doc_vecs @ query
    per_ep: dict[str, list[float]] = defaultdict(list)
    for s, e in zip(sims, doc_eps):
        per_ep[e].append(float(s))
    if agg == "max":
        return {e: max(v) for e, v in per_ep.items()}
    return {e: float(np.mean(sorted(v, reverse=True)[:2])) for e, v in per_ep.items()}


def ranking(scores: dict[str, float]) -> list[str]:
    return sorted(scores, key=scores.get, reverse=True)


def rrf(rankings: list[list[str]], k: int = RRF_K) -> list[str]:
    fused: dict[str, float] = defaultdict(float)
    for r in rankings:
        for pos, e in enumerate(r, 1):
            fused[e] += 1.0 / (k + pos)
    return sorted(fused, key=fused.get, reverse=True)


def _segments(path) -> list[dict] | None:
    data = json.loads(path.read_text(encoding="utf-8"))
    segs = data["segments"] if isinstance(data, dict) else data
    return segs or None


def _group(score, shortcode: str, approved: set[str]) -> str:
    """'review' | 'auto' | 'unknown' from the link's match score and the review queue."""
    if shortcode in approved:
        return "review"
    if score is None:
        return "unknown"
    return "auto" if score >= MATCH_AUTO_ACCEPT else "review"


def load_truth() -> list[tuple[str, str, list[dict], float | None, str]]:
    """[(shortcode, true episode_id, segments, match score, group)] for linked
    clips with a transcript; group is 'auto', 'review' or 'unknown'."""
    approved = set()
    if REVIEW_QUEUE.exists():
        for line in REVIEW_QUEUE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                if item.get("status") == "approved":
                    approved.add(item.get("shortcode"))
    out, seen = [], set()
    for line in IG_LINKS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        link = json.loads(line)
        sc = link.get("shortcode")
        path = TRANSCRIPTS_DIR / f"{sc}.json"
        if link.get("status") != "linked" or sc in seen or not path.exists():
            continue
        segs = _segments(path)
        if segs:
            seen.add(sc)
            score = link.get("score")
            out.append((sc, link["episode_id"], segs, score, _group(score, sc, approved)))
    return out


def evaluate(truth, queries: np.ndarray, chunk_vecs: np.ndarray, chunk_eps: list[str],
             docs: list[dict], doc_vecs: np.ndarray) -> dict:
    """Returns {variant: [rank of the true episode for each clip]} (1 = best)
    and the per-clip rankings, over the episodes that have both chunks and a
    synopsis."""
    universe = set(chunk_eps) & {d["episode_id"] for d in docs}
    par = [i for i, d in enumerate(docs) if d["level"] == "paragraph" and d["episode_id"] in universe]
    epi = [i for i, d in enumerate(docs) if d["level"] == "episode" and d["episode_id"] in universe]
    keep = [i for i, e in enumerate(chunk_eps) if e in universe]
    par_v, par_e = doc_vecs[par], [docs[i]["episode_id"] for i in par]
    epi_v, epi_e = doc_vecs[epi], [docs[i]["episode_id"] for i in epi]
    ch_v, ch_e = chunk_vecs[keep], [chunk_eps[i] for i in keep]

    ranks: dict[str, list] = defaultdict(list)
    per_clip, skipped = [], []
    for (sc, true, _, score, group), q in zip(truth, queries):
        if true not in universe:
            skipped.append((sc, true))
            continue
        r = {
            "baseline": ranking(episode_scores(q, ch_v, ch_e, "max")),
            "syn_episode": ranking(episode_scores(q, epi_v, epi_e, "max")),
            "syn_para_max": ranking(episode_scores(q, par_v, par_e, "max")),
            "syn_para_top2": ranking(episode_scores(q, par_v, par_e, "top2")),
        }
        r["fusion"] = rrf([r["baseline"], r["syn_para_max"]])
        r["fusion3"] = rrf([r["baseline"], r["syn_para_max"], r["syn_episode"]])
        for name, order in r.items():
            ranks[name].append(order.index(true) + 1)
        per_clip.append((sc, true, {n: o.index(true) + 1 for n, o in r.items()}, score, group))
    return {"ranks": dict(ranks), "per_clip": per_clip, "skipped": skipped,
            "universe": len(universe)}


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate synopsis-based episode identification")
    p.add_argument("--verbose", action="store_true", help="rank of each variant, per clip")
    args = p.parse_args()

    for path, how in ((IG_LINKS, "cli ingest"), (CHUNK_EMBEDDINGS, "export_from_db"),
                      (CHUNKS_EN, "export_from_db")):
        if not path.exists():
            raise SystemExit(f"{path} not found (run {how})")
    truth = load_truth()
    if not truth:
        raise SystemExit("no linked clip with a cached transcript to evaluate")

    from .google_client import embed_texts      # late: the maths above needs no API
    queries = _unit(embed_texts([" ".join(s["text"] for s in segs) for _, _, segs, _, _ in truth],
                                task_type="RETRIEVAL_QUERY"))
    ep_of = {}
    for line in CHUNKS_EN.read_text(encoding="utf-8").splitlines():
        if line.strip():
            c = json.loads(line)
            ep_of[c["chunk_id"]] = c["episode_id"]
    emb = json.loads(CHUNK_EMBEDDINGS.read_text(encoding="utf-8"))
    cids = [c for c in emb if c in ep_of]
    chunk_vecs, chunk_eps = _unit([emb[c] for c in cids]), [ep_of[c] for c in cids]
    docs, demb = load_docs(), load_embeddings()
    all_docs = len(docs)
    docs = [d for d in docs if d["doc_id"] in demb]
    if len(docs) < all_docs:
        print(f"WARNING: {all_docs - len(docs)} of {all_docs} synopsis docs have no "
              "embedding (embed_synopses did not finish, or was run before "
              "fetch_synopses was refreshed): run embed_synopses again.")
    if not docs:
        raise SystemExit("no synopsis embeddings: run fetch_synopses and embed_synopses")
    doc_vecs = _unit([demb[d["doc_id"]] for d in docs])

    season_of = lambda e: int(e[1:3])
    seasons = lambda eps: sorted({season_of(e) for e in eps})
    print(f"synopses cover seasons {seasons(d['episode_id'] for d in docs)} | chunks "
          f"cover {seasons(chunk_eps)} | clips are from {seasons(t[1] for t in truth)}")

    res = evaluate(truth, queries, chunk_vecs, chunk_eps, docs, doc_vecs)
    n = len(res["per_clip"])
    if n == 0:
        miss = sorted({e for _, e in res["skipped"]})
        raise SystemExit(
            f"\n0 clips evaluated: none of the {len(truth)} clips belongs to an episode "
            "that has both chunks and a synopsis embedded.\n"
            f"Episodes of the clips: {', '.join(miss[:12])}{' ...' if len(miss) > 12 else ''}\n"
            "Check the seasons above: fetch_synopses/embed_synopses must cover the "
            "clips' seasons.")
    print(f"{n} clips evaluated (of {len(truth)} with transcript) over "
          f"{res['universe']} episodes\n")
    names = list(res["ranks"])
    _table("ALL CLIPS", [rk for _, _, rk, _, _ in res["per_clip"]], names)
    labels = {"auto": f"AUTO-ACCEPTED (score >= {MATCH_AUTO_ACCEPT})",
              "review": f"TO REVIEW / REVIEWED (score < {MATCH_AUTO_ACCEPT} or approved in review)",
              "unknown": "NO SCORE RECORDED"}
    for group, label in labels.items():
        sub = [rk for _, _, rk, _, g in res["per_clip"] if g == group]
        if sub or group != "unknown":
            _table(label, sub, names)
    if args.verbose:
        print("\nrank of the true episode per clip:")
        print(f"{'clip':<14} {'episode':<8} {'score':>6} {'group':<7} "
              + " ".join(f"{x[:12]:>12}" for x in names))
        for sc, true, rk, score, group in sorted(
                res["per_clip"],        # doubtful ones first, lowest score first
                key=lambda t: ({"review": 0, "auto": 1}.get(t[4], 2),
                               t[3] if t[3] is not None else 0)):
            sc_txt = f"{score:.2f}" if score is not None else "  -"
            print(f"{sc:<14} {true:<8} {sc_txt:>6} {group:<7} "
                  + " ".join(f"{rk[x]:>12}" for x in names))


def _table(title: str, per_clip_ranks: list[dict], names: list[str]) -> None:
    print(f"{title}: {len(per_clip_ranks)} clips")
    if not per_clip_ranks:
        print("  (none)\n")
        return
    print(f"{'variant':<15} {'top1':>6} {'top3':>6} {'top5':>6} {'mean rank':>10}")
    for name in names:
        a = np.asarray([rk[name] for rk in per_clip_ranks])
        print(f"{name:<15} {np.mean(a == 1):>6.0%} {np.mean(a <= 3):>6.0%} "
              f"{np.mean(a <= 5):>6.0%} {a.mean():>10.1f}")
    print()


if __name__ == "__main__":
    main()
