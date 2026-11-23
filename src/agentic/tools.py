"""Closed set of metadata tools. The router only CHOOSES one and its arguments
(as JSON); arguments are validated here and the result is rendered from a
template, so the model never writes the number or title that ends up in the
answer. Every tool returns {"tool", "ok", "data" | "error", ...}."""
import re

MAX_LIST = 30

# name -> {"desc": ..., "args": {arg: "int" | "int?" | "str"}}   ("?" = optional)
SPECS = {
    "count_seasons": {"desc": "how many seasons the catalog covers", "args": {}},
    "count_episodes": {"desc": "how many episodes a season has; without `season`, in total",
                       "args": {"season": "int?"}},
    "list_episodes": {"desc": "titles of the episodes of a season", "args": {"season": "int"}},
    "episode_title": {"desc": "title of episode number `episode` of a season",
                      "args": {"season": "int", "episode": "int"}},
    "find_episode": {"desc": "season and number of an episode given its (English) title",
                     "args": {"title": "str"}},
}


def _as_int(v):
    if isinstance(v, bool):
        raise ValueError
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str) and re.fullmatch(r"\s*\d{1,4}\s*", v):
        return int(v)
    raise ValueError


def validate_call(name, args):
    """-> (clean_args, None) or (None, reason). Unknown tools / arguments, wrong
    types and missing required arguments are rejected (the router falls back
    to plain RAG); whether a season EXISTS is the tool's business."""
    spec = SPECS.get(name)
    if spec is None:
        return None, f"unknown tool {name!r}"
    if not isinstance(args, dict):
        return None, "args must be an object"
    extra = set(args) - set(spec["args"])
    if extra:
        return None, f"unexpected arguments {sorted(extra)}"
    clean = {}
    for arg, kind in spec["args"].items():
        optional = kind.endswith("?")
        v = args.get(arg)
        if v is None:
            if optional:
                continue
            return None, f"missing argument {arg!r}"
        try:
            clean[arg] = _as_int(v) if kind.startswith("int") else _text(v)
        except ValueError:
            return None, f"bad value for {arg!r}: {v!r}"
    return clean, None


def _text(v):
    if not isinstance(v, str) or not v.strip() or len(v) > 200:
        raise ValueError
    return v.strip()


def _season_missing(catalog, season, tool):
    return {"tool": tool, "ok": False, "error": "season_not_in_catalog", "season": season,
            "seasons": catalog.seasons}


def run(name, args, catalog):
    clean, err = validate_call(name, args)
    if err:
        raise ValueError(err)
    return globals()["_" + name](catalog, **clean)


def _count_seasons(catalog):
    s = catalog.seasons
    return {"tool": "count_seasons", "ok": True,
            "data": {"seasons": len(s), "first": s[0], "last": s[-1]}}


def _count_episodes(catalog, season=None):
    if season is None:
        return {"tool": "count_episodes", "ok": True,
                "data": {"season": None, "episodes": len(catalog.episodes),
                         "seasons": len(catalog.seasons)}}
    eps = catalog.in_season(season)
    if not eps:
        return _season_missing(catalog, season, "count_episodes")
    return {"tool": "count_episodes", "ok": True,
            "data": {"season": season, "episodes": len(eps),
                     "indexed": catalog.indexed_count(season)}}


def _list_episodes(catalog, season):
    eps = catalog.in_season(season)
    if not eps:
        return _season_missing(catalog, season, "list_episodes")
    return {"tool": "list_episodes", "ok": True,
            "data": {"season": season, "episodes": len(eps), "truncated": len(eps) > MAX_LIST,
                     "items": [(e["episode"], e["title"]) for e in eps[:MAX_LIST]]}}


def _episode_title(catalog, season, episode):
    if not catalog.in_season(season):
        return _season_missing(catalog, season, "episode_title")
    e = catalog.by_number(season, episode)
    if e is None:
        return {"tool": "episode_title", "ok": False, "error": "episode_not_in_season",
                "season": season, "episode": episode,
                "episodes": len(catalog.in_season(season))}
    return {"tool": "episode_title", "ok": True,
            "data": {"season": season, "episode": episode, "title": e["title"]}}


def _find_episode(catalog, title):
    hits = catalog.find_title(title)
    if not hits:
        return {"tool": "find_episode", "ok": False, "error": "title_not_found"}
    if len(hits) > 1:
        return {"tool": "find_episode", "ok": False, "error": "ambiguous_title",
                "candidates": [(e["season"], e["episode"], e["title"]) for e in hits[:5]],
                "n_candidates": len(hits)}
    e = hits[0]
    return {"tool": "find_episode", "ok": True,
            "data": {"season": e["season"], "episode": e["episode"], "title": e["title"]}}


# --- answers from templates (no LLM) ------------------------------------------
def _pl(n, one, many):
    return one if n == 1 else many


def render(result, lang):
    """Plain-text answer in `lang` ('es' | 'en') for a tool result."""
    es = lang == "es"
    t, ok = result["tool"], result["ok"]
    if not ok:
        return _render_error(result, es)
    d = result["data"]
    if t == "count_seasons":
        a = (f"El catálogo cubre {d['seasons']} temporadas (de la {d['first']} a la {d['last']})."
             if es else f"The catalog covers {d['seasons']} seasons ({d['first']} to {d['last']}).")
    elif t == "count_episodes" and d["season"] is None:
        a = (f"El catálogo tiene {d['episodes']} episodios en {d['seasons']} temporadas."
             if es else f"The catalog has {d['episodes']} episodes across {d['seasons']} seasons.")
    elif t == "count_episodes":
        n = d["episodes"]
        a = (f"La temporada {d['season']} tiene {n} {_pl(n, 'episodio', 'episodios')} en el catálogo."
             if es else f"Season {d['season']} has {n} {_pl(n, 'episode', 'episodes')} in the catalog.")
        if d.get("indexed") is not None and d["indexed"] != n:
            a += (f" De ellos, {d['indexed']} tienen el guion indexado."
                  if es else f" Of those, {d['indexed']} have their script indexed.")
    elif t == "list_episodes":
        lines = "\n".join(f"{i}. {title}" for i, title in d["items"])
        head = (f"Episodios de la temporada {d['season']} ({d['episodes']}):"
                if es else f"Season {d['season']} episodes ({d['episodes']}):")
        more = (f"\n(+{d['episodes'] - len(d['items'])} más)" if es
                else f"\n(+{d['episodes'] - len(d['items'])} more)") if d["truncated"] else ""
        a = f"{head}\n{lines}{more}"
    elif t == "episode_title":
        a = (f"El episodio {d['episode']} de la temporada {d['season']} es \"{d['title']}\"."
             if es else f"Episode {d['episode']} of season {d['season']} is \"{d['title']}\".")
    else:  # find_episode
        a = (f"\"{d['title']}\" es el episodio {d['episode']} de la temporada {d['season']}."
             if es else f"\"{d['title']}\" is episode {d['episode']} of season {d['season']}.")
    src = ("Fuente: catálogo de episodios del proyecto." if es
           else "Source: the project's episode catalog.")
    return f"{a}\n{src}"


def _render_error(r, es):
    e = r["error"]
    if e == "season_not_in_catalog":
        s = r["seasons"]
        return (f"La temporada {r['season']} no está en el catálogo, que cubre de la {s[0]} a la {s[-1]}."
                if es else f"Season {r['season']} is not in the catalog, which covers {s[0]} to {s[-1]}.")
    if e == "episode_not_in_season":
        n = r["episodes"]
        return (f"La temporada {r['season']} tiene {n} episodios en el catálogo; no existe el número {r['episode']}."
                if es else f"Season {r['season']} has {n} episodes in the catalog; there is no number {r['episode']}.")
    if e == "ambiguous_title":
        opts = "; ".join(f"\"{t}\" (T{s}, ep. {n})" if es else f"\"{t}\" (S{s}, ep. {n})"
                         for s, n, t in r["candidates"])
        return (f"Varios episodios coinciden: {opts}. ¿Cuál quieres decir?" if es
                else f"Several episodes match: {opts}. Which one do you mean?")
    return ("No encuentro ese título en el catálogo (los títulos están en inglés)." if es
            else "I can't find that title in the catalog.")
