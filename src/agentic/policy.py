"""When does a question go to the router? Retrieval-distance policy.

The RAG always retrieves first. If the best chunk is close, the question is
answered by the RAG as before (no extra LLM call). If every chunk is far, the
router decides between corpus / general / offtopic (and metadata, if the keyword
gate missed it). Thresholds come from the per-language calibration of the
distance distributions (answerable vs general vs unrelated questions):

    es  0.59   p95 of answerable, p05 of general
    en  0.58   p90 of answerable, p25 of general

They are a first estimate: override them without touching code with the
environment variables ROUTER_DIST_ES / ROUTER_DIST_EN.

A second, higher threshold guards the only harmful action, refusing a question
as off-topic: the router's `offtopic` verdict is honoured only when the best
distance is above ROUTER_REFUSE_DIST (default 0.70: unrelated questions start
around 0.75 while answerable ones almost never exceed 0.65). Below it, an
`offtopic` verdict falls back to the RAG. 0 disables the guard.

Distances are Chroma distances (lower = closer), as stored in meta.retrieved.
"""
import collections
import json

DEFAULT_DIST = {"es": 0.59, "en": 0.58}
DEFAULT_REFUSE_DIST = 0.70
KIND_ROUTE = {"answerable": "corpus", "unanswerable": "corpus",
              "general": "general", "nonrelated": "offtopic"}


def _env_float(environ, key, default):
    try:
        return float(environ[key])
    except (KeyError, TypeError, ValueError):
        return default


def thresholds(environ=None):
    """{"es": float, "en": float}, defaults overridden by ROUTER_DIST_ES / ROUTER_DIST_EN."""
    env = environ if environ is not None else {}
    return {l: _env_float(env, f"ROUTER_DIST_{l.upper()}", d) for l, d in DEFAULT_DIST.items()}


def refuse_distance(environ=None):
    env = environ if environ is not None else {}
    return max(0.0, _env_float(env, "ROUTER_REFUSE_DIST", DEFAULT_REFUSE_DIST))


def best_distance(results):
    """Smallest distance of retrieve()'s [(doc, distance), ...] (None if empty)."""
    ds = [s for _, s in results or [] if s is not None]
    return min(ds) if ds else None


def threshold_for(lang, th):
    """Threshold of `lang`; an unknown language uses the larger one (fewer router calls)."""
    return th.get(lang, max(th.values()))


def reaches_router(best, lang, th):
    return best is not None and best > threshold_for(lang, th)


# --- evaluation set from stored distances --------------------------------------
def _rows(paths):
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except ValueError:
                        continue


def row_distance(row):
    ds = [r["distance"] for r in (row.get("meta") or {}).get("retrieved") or []
          if isinstance(r, dict) and r.get("distance") is not None]
    return min(ds) if ds else None


def zone_cases(paths, th):
    """Questions the router would actually see: those of the data files (gen_teacher
    and gen_router_questions records) whose best distance is above the threshold of
    their language. -> (cases, stats); cases look like router_eval.jsonl rows plus
    `distance` and `kind`; stats = {(lang, kind): [above, total]}."""
    cases, stats, seen = [], collections.OrderedDict(), set()
    for row in _rows(paths):
        meta = row.get("meta") or {}
        kind, lang, d = meta.get("kind"), row.get("lang"), row_distance(row)
        q = meta.get("question") or row.get("prompt")
        if kind not in KIND_ROUTE or d is None or not q or lang not in th:
            continue
        key = (lang, kind)
        stats.setdefault(key, [0, 0])
        stats[key][1] += 1
        if d > th[lang] and q not in seen:
            seen.add(q)
            stats[key][0] += 1
            cases.append({"q": q, "lang": lang, "route": KIND_ROUTE[kind], "tool": None,
                          "args": {}, "distance": round(d, 3), "kind": kind})
    return cases, stats


def format_stats(stats, th):
    lines = [f"questions above the threshold (es {th['es']}, en {th['en']}) = reach the router:"]
    for (lang, kind), (above, total) in sorted(stats.items()):
        lines.append(f"  {lang} {kind:<13} {above:>4}/{total:<4} ({above / total:.0%})")
    return "\n".join(lines)
