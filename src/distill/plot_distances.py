"""Distances of the top-k retrieved chunks per router class, overlaid.

    PYTHONPATH=src python -m distill.plot_distances \
        --data data/distill/teacher_qa.jsonl data/distill/router_questions.jsonl \
        --out src/distill/images/distances.png

Accepts SEVERAL data files (mix of gen_teacher and gen_router_questions
records; anything with meta.retrieved works) and plots every class found in
meta.kind in the SAME figures (answerable / unanswerable / general /
nonrelated). The legacy kind "offtopic" stored in teacher_qa carries NO
distances in meta.retrieved, so those rows are skipped automatically.

  * rank-1 histogram, one step-fill per class   -> where the distributions
    cross is the natural threshold
  * rank-1 ECDF per class (zoomable read of the
    same overlap; with --threshold, marks the chosen cut and prints the
    resulting per-class keep/corpus rates)
  * box plot of rank 1-3 per class              -> how fast distance grows
    with the rank

Also prints a percentile table for ranks 1-3, one block per class, so the
threshold can be read straight off the percentiles (e.g. the p95 of the
answerable rank-1, checked against the general CDF).

Matplotlib is an optional dependency (`pip install matplotlib`).
"""

import argparse
import json
import os

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # noqa: BLE001
    matplotlib = plt = None

TOP_K = 3
COLORS = {"answerable": "#2ca02c", "unanswerable": "#ff7f0e",
          "general": "#d62728", "nonrelated": "#7f7f7f",
          "offtopic": "#9467bd"}  # legacy kind in teacher_qa: no distances


def read_rows(paths):
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield path, json.loads(line)


def collect_distances(rows, kinds, min_rank=TOP_K):
  """kind -> {id: list of distances}.

    Skips chunks without a "distance" key (teacher rows of the old offtopic
    kind store retrieved metadata WITHOUT distances) and drops any question
    with fewer than min_rank measured distances, so a class is never skewed
    by rows that came through a different path."""
  dists = {}
  for _, row in rows:
    meta = row.get("meta") or {}
    kind = meta.get("kind")
    if not kind or (kinds and kind not in kinds):
      continue
    measured = [
        float(r["distance"])
        for r in (meta.get("retrieved") or [])
        if isinstance(r, dict) and r.get("distance") is not None
    ]
    if len(measured) >= min_rank:
      dists.setdefault(kind, {})[row["id"]] = measured[:min_rank]
  return dists

def percentile(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def percentile_table(dists):
    for kind in sorted(dists):
        rows = dists[kind]
        print(f"\n== {kind} (n={len(rows)}) ==")
        print(f"{'pct':>6} " + " ".join(f"{'rank ' + str(i):>9}" for i in range(1, TOP_K + 1)))
        for q in (0.05, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99):
            cells = []
            for i in range(TOP_K):
                xs = [d[i] for d in rows.values() if len(d) > i]
                cells.append(f"{percentile(xs, q):9.3f}" if xs else " " * 9)
            print(f"{q:>6.2f} " + " ".join(cells))


def keep_rates(dists, threshold):
    print(f"\n== rank-1 <= {threshold:.3f} -> keep in Corpus ==")
    for kind in sorted(dists):
        xs = [d[0] for d in dists[kind].values()]
        n = len(xs)
        pct = 100.0 * sum(x <= threshold for x in xs) / n if n else 0.0
        print(f"  {kind:<13} {pct:5.1f}%  (n={n})")


def _color(kind):
    return COLORS.get(kind, None) or plt.cm.tab10(hash(kind) % 10)


def plot(dists, out, threshold=None):
    by_rank = {k: [[d[i] for d in v.values() if len(d) > i] for i in range(TOP_K)]
               for k, v in dists.items()}

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))

    # 1) rank-1 histogram, step + light fill so overlaps stay readable
    ax = axes[0]
    for kind in sorted(by_rank):
        xs = by_rank[kind][0]
        if xs:
            ax.hist(xs, bins=40, histtype="stepfilled", alpha=0.35,
                    color=_color(kind), label=f"{kind} (n={len(xs)})")
            ax.hist(xs, bins=40, histtype="step", color=_color(kind))
    if threshold is not None:
        ax.axvline(threshold, color="black", ls="--", lw=1)
    ax.set_title("Rank-1 distance by class")
    ax.set_xlabel("distance to question")
    ax.set_ylabel("questions")
    ax.legend(fontsize=8)

    # 2) rank-1 ECDF: exact read of the overlap between classes
    ax = axes[1]
    for kind in sorted(by_rank):
        xs = sorted(by_rank[kind][0])
        if not xs:
            continue
        ys = [(i + 1) / len(xs) for i in range(len(xs))]
        ax.step(xs, ys, where="post", color=_color(kind), label=kind)
    if threshold is not None:
        ax.axvline(threshold, color="black", ls="--", lw=1)
    ax.set_title("Rank-1 ECDF (crossing = threshold zone)")
    ax.set_xlabel("rank-1 distance")
    ax.set_ylabel("fraction of questions")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # 3) box plot per class and rank
    ax = axes[2]
    labels, data, colors = [], [], []
    for kind in sorted(by_rank):
        for i, xs in enumerate(by_rank[kind]):
            if xs:
                labels.append(f"{kind}\nr{i + 1}")
                data.append(xs)
                colors.append(_color(kind))
    boxes = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    for patch, c in zip(boxes["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)
    ax.set_title("Distance by chunk rank and class")
    ax.tick_params(axis="x", labelsize=7)

    fig.suptitle("Router threshold study: top-3 chunk distances vs question, per class")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", nargs="+",
                    default=["data/distill/teacher_qa.jsonl",
                             "data/distill/router_questions.jsonl"],
                    help="one or more data files (meta.retrieved + meta.kind)")
    ap.add_argument("--kinds", default=None,
                    help="comma-separated subset of meta.kind to plot, e.g. answerable,general")
    ap.add_argument("--threshold", type=float, default=None,
                    help="draw this rank-1 threshold and print per-class keep rates")
    ap.add_argument("--out", default="src/distill/images/distances.png")
    args = ap.parse_args()

    if plt is None:
        raise SystemExit("matplotlib is required: pip install matplotlib")

    kinds = {k.strip() for k in args.kinds.split(",")} if args.kinds else None
    rows = list(read_rows(args.data))
    dists = collect_distances(rows, kinds)
    if not dists:
        found = sorted({str((row.get("meta") or {}).get("kind")) for _, row in rows})
        raise SystemExit(f"no rows matched; kinds found across the files: {found}")

    percentile_table(dists)
    if args.threshold is not None:
        keep_rates(dists, args.threshold)
    plot(dists, os.path.abspath(args.out), threshold=args.threshold)


if __name__ == "__main__":
    main()
