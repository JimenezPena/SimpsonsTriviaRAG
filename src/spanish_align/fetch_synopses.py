"""OFFLINE TASK: downloads the plot of every episode from Wikipedia.

For each season, lists the articles of "Category:The Simpsons season N
episodes" through the MediaWiki API, and for each article takes
  * season / episode-in-season from its infobox,
  * the "Plot" section (falls back to Synopsis/Premise/Summary, then to the
    article's lead; the source is recorded in plot_source).
Only the Plot is kept (not the rest of the article). Each episode's infobox
number is then contrasted with chunks_en.jsonl (the vector DB export): if the
DB has a different title at that SxxExx a WARNING is printed, and the episode
is stored under the DB's number when the title identifies it unambiguously;
what cannot be matched is reported, not guessed.

    PYTHONPATH=src python -m spanish_align.fetch_synopses              # seasons 1-10
    PYTHONPATH=src python -m spanish_align.fetch_synopses --seasons 5-6
    PYTHONPATH=src python -m spanish_align.fetch_synopses --refresh    # ignore the cache

Output: data/ingest/episode_synopses.json  {episode_id: {...}}
A small digest per article (Plot + infobox numbers) is cached in
data/ingest/cache/wiki/ (re-runs are free and resumable). Text © Wikipedia contributors, CC BY-SA 4.0: keep the wiki_url
if you ever show it.
"""

import argparse
import difflib
import hashlib
import http.client
import json
import re
import time
import urllib.parse
import urllib.request

from .config import CHUNKS_EN, SYNOPSES, WIKI_CACHE, ensure_dirs

API = "https://en.wikipedia.org/w/api.php"
# Wikipedia asks API clients to identify themselves
USER_AGENT = ("SimpsonsTriviaRAG/1.0 "
              "(https://github.com/JimenezPena/SimpsonsTriviaRAG; personal research project)")
PAUSE = 0.3                      # seconds between requests
PLOT_HEADINGS = ("plot", "synopsis", "premise", "summary", "plot summary")


# --------------------------------------------------------------------------
# API access
# --------------------------------------------------------------------------

def _api(params: dict, retries: int = 4) -> dict:
    url = API + "?" + urllib.parse.urlencode(
        {**params, "format": "json", "formatversion": 2})
    for attempt in range(1, retries + 1):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read())
            time.sleep(PAUSE)
            return data
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                wait = 10 * attempt
                print(f"    [warn] HTTP {e.code}: retry {attempt}/{retries} in {wait}s")
                time.sleep(wait)
                continue
            raise
        except (urllib.error.URLError, http.client.HTTPException,
                ConnectionError, TimeoutError) as e:
            if attempt < retries:
                wait = 10 * attempt
                print(f"    [warn] {type(e).__name__}: retry {attempt}/{retries} in {wait}s")
                time.sleep(wait)
                continue
            raise


def category_members(season: int) -> list[str]:
    """Article titles in 'Category:The Simpsons season N episodes'."""
    titles, cont = [], {}
    while True:
        data = _api({"action": "query", "list": "categorymembers",
                     "cmtitle": f"Category:The Simpsons season {season} episodes",
                     "cmnamespace": 0, "cmlimit": 500, **cont})
        titles += [m["title"] for m in data.get("query", {}).get("categorymembers", [])]
        cont = data.get("continue", {})
        if not cont:
            return titles


def fetch_article(title: str, refresh: bool = False) -> dict | None:
    """Digest of one article (disk-cached): {'title', 'infobox', 'plot',
    'plot_source', 'wikitext_len', 'templates'}. Only the Plot and a few
    infobox numbers are kept, not the article."""
    WIKI_CACHE.mkdir(parents=True, exist_ok=True)
    cache = WIKI_CACHE / (hashlib.sha1(title.encode("utf-8")).hexdigest() + ".json")
    if cache.exists() and not refresh:
        cached = json.loads(cache.read_text(encoding="utf-8"))
        if "wikitext" in cached:         # old cache format: the whole article
            if cached["wikitext"]:
                art = digest(cached["title"], cached.get("extract", ""), cached["wikitext"])
                cache.write_text(json.dumps(art, ensure_ascii=False), encoding="utf-8")
                return art
        elif cached.get("wikitext_len"):
            return cached
    data = _api({"action": "query", "titles": title, "redirects": 1,
                 "prop": "extracts|revisions", "explaintext": 1,
                 "exsectionformat": "wiki", "rvprop": "content",
                 "rvslots": "main"})
    pages = data.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        return None
    page = pages[0]
    revs = page.get("revisions") or [{}]
    wikitext = revs[0].get("slots", {}).get("main", {}).get("content", "")
    art = digest(page["title"], page.get("extract", ""), wikitext)
    if wikitext:                         # an empty wikitext is a glitch: do not cache it
        cache.write_text(json.dumps(art, ensure_ascii=False), encoding="utf-8")
    return art


def digest(title: str, extract: str, wikitext: str) -> dict:
    plot, source = extract_plot(extract)
    return {"title": title, "infobox": parse_infobox(wikitext), "plot": plot,
            "plot_source": source, "wikitext_len": len(wikitext),
            "templates": _template_names(wikitext)}


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def _template_names(wikitext: str) -> list[str]:
    return re.findall(r"\{\{\s*([^|{}\n]{1,60})", wikitext[:6000])[:8]


def _infobox_params(wikitext: str) -> dict[str, str] | None:
    """Top-level |name = value parameters of the first '{{Infobox ... episode'
    template. Nested templates ({{cite book |title=...}} inside a ref) and
    [[wikilinks|with pipes]] are skipped, so their parameters are never
    mistaken for the infobox's own."""
    m = re.search(r"\{\{\s*Infobox[^|{}\n]*episode", wikitext, re.I)
    if not m:
        return None
    parts, cur, depth, link, i = [], [], 1, 0, m.end()
    while i < len(wikitext) and depth > 0:
        two = wikitext[i:i + 2]
        if two == "{{":
            depth += 1; cur.append(two); i += 2; continue
        if two == "}}":
            depth -= 1; i += 2
            if depth == 0:
                break
            cur.append(two); continue
        if two == "[[":
            link += 1; cur.append(two); i += 2; continue
        if two == "]]":
            link = max(0, link - 1); cur.append(two); i += 2; continue
        ch = wikitext[i]
        if ch == "|" and depth == 1 and link == 0:
            parts.append("".join(cur)); cur = []
        else:
            cur.append(ch)
        i += 1
    parts.append("".join(cur))
    params = {}
    for part in parts[1:]:
        if "=" in part:
            k, v = part.split("=", 1)
            params.setdefault(k.strip().lower(), v.strip())
    return params


def parse_infobox(wikitext: str) -> dict | None:
    """{'season': int, 'episode': int, 'title': str|None, 'production': str|None}
    from the episode infobox; None if it is missing or has no numbers."""
    params = _infobox_params(wikitext)
    if not params:
        return None

    def num(field):
        f = re.search(r"\d+", params.get(field, ""))
        return int(f.group(0)) if f else None

    season, episode = num("season"), num("episode")
    if season is None or episode is None:
        return None
    title = re.sub(r"[\[\]'\"]|<[^>]+>", "", params.get("title", "")).strip() or None
    return {"season": season, "episode": episode, "title": title,
            "production": params.get("production") or None}


def extract_plot(extract: str) -> tuple[str, str]:
    """(plot text, source). Level-2 sections only; subsections stay inside."""
    parts = re.split(r"^==\s*([^=\n].*?)\s*==\s*$", extract, flags=re.M)
    lead, sections = parts[0], list(zip(parts[1::2], parts[2::2]))

    def clean(text):
        text = re.sub(r"^=+.*?=+\s*$", "", text, flags=re.M)     # sub-headings
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    for name in PLOT_HEADINGS:
        for head, body in sections:
            if head.strip().lower() == name and clean(body):
                return clean(body), name
    return clean(lead), "lead"


def _norm(title: str) -> str:
    t = re.sub(r"\((?:the simpsons|simpsons)[^)]*\)", "", title.lower())
    t = re.sub(r"\((?:part|episode)[^)]*\)", "", t)
    return re.sub(r"[^a-z0-9]+", "", t.replace("&", "and"))


def _same_title(a: str, b: str) -> bool:
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return False
    return na == nb or difflib.SequenceMatcher(None, na, nb).ratio() >= 0.85


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def _clean_title(wiki_title: str) -> str:
    """'Itchy & Scratchy Land (The Simpsons episode)' -> 'Itchy & Scratchy Land'."""
    return re.sub(r"\s*\((?:The )?Simpsons[^)]*\)", "", wiki_title).strip()


def _record(eid, season, episode, art) -> dict:
    return {
        "episode_id": eid, "season": season, "episode": episode,
        "title": _clean_title(art["title"]),
        "wiki_title": art["title"],
        "wiki_url": "https://en.wikipedia.org/wiki/"
                    + urllib.parse.quote(art["title"].replace(" ", "_"), safe="_(),!'"),
        "plot_source": art["plot_source"], "words": len(art["plot"].split()),
        "plot": art["plot"],
    }


def build(seasons: list[int], refresh: bool = False) -> tuple[dict, list[dict], list[str]]:
    """Returns ({episode_id: record}, unresolved, problems). The number comes
    from the article's infobox. 'unresolved' are articles whose infobox gave
    no season/episode, or whose number another article already holds: they
    carry the season of the category they were listed in and get their
    number from the title match against the DB (check_against_chunks)."""
    records, unresolved, problems = {}, [], []
    for season in seasons:
        members = category_members(season)
        print(f"Season {season}: {len(members)} articles in the category")
        if not members:
            problems.append(f"season {season}: empty category")
        for title in members:
            art = fetch_article(title, refresh)
            if art is None:
                problems.append(f"{title}: article not found")
                continue
            if not art["plot"]:
                problems.append(f"{title}: no plot found (wikitext {art['wikitext_len']} "
                                f"chars, templates: {art['templates']})")
                continue
            info = art["infobox"]
            if info is None:
                unresolved.append(_record(None, season, None, art))
                continue
            if info["season"] != season and info["season"] not in seasons:
                continue          # listed in this category but belongs to a season not requested
            eid = f"S{info['season']:02d}E{info['episode']:02d}"
            rec = _record(eid, info["season"], info["episode"], art)
            if eid in records and records[eid]["wiki_title"] != art["title"]:
                unresolved.append({**rec, "episode_id": None, "episode": None,
                                   "season": season})
                continue
            records[eid] = rec
    return records, unresolved, problems


def check_against_chunks(records: dict, unresolved: list[dict] = ()) -> list[str]:
    """Contrasts the infobox numbers with the DB (chunks_en.jsonl) and warns:
      * the title at that SxxExx in the DB differs from the article's: if the
        same title sits at ONE other number of that season, the record is
        stored under the DB's number (the one the chunks use) and the
        Wikipedia one is kept in 'wiki_episode_id'; otherwise it stays as is,
        flagged 'unverified';
      * articles without usable infobox numbers are placed by title.
    Everything that does not match is listed; nothing is guessed."""
    if not CHUNKS_EN.exists():
        return ["chunks_en.jsonl not found: numbers not checked"] + [
            f"'{r['title']}' (season {r['season']}): episode number unknown, "
            "needs chunks_en.jsonl to place it" for r in unresolved]
    db = {}
    for line in CHUNKS_EN.read_text(encoding="utf-8").splitlines():
        if line.strip():
            c = json.loads(line)
            db[c["episode_id"]] = (c["season"], c["episode_title"])
    notes = []

    # 1) infobox number vs DB title
    moves = {}
    for eid, rec in records.items():
        if eid in db and _same_title(db[eid][1], rec["title"]):
            continue
        cands = [e for e, (s, t) in db.items()
                 if s == rec["season"] and _same_title(t, rec["title"])]
        here = db.get(eid, (None, None))[1]
        if len(cands) == 1:
            moves[eid] = cands[0]
            notes.append(f"WARNING {eid} '{rec['title']}': the DB has "
                         f"{here!r} at {eid} but this title is {cands[0]} -> stored as {cands[0]}")
        else:
            rec["unverified"] = True
            notes.append(f"WARNING {eid} '{rec['title']}': the DB has {here!r} at {eid} "
                         f"and no unique match by title; left as is, check by hand")
    targets = list(moves.values())
    for old, new in moves.items():
        if targets.count(new) > 1:
            records[old]["unverified"] = True
            notes.append(f"WARNING {old} '{records[old]['title']}': several articles map to "
                         f"{new}; left as is, check by hand")
    movable = {o: n for o, n in moves.items() if targets.count(n) == 1}
    moved = {n: {**records.pop(o), "episode_id": n, "episode": int(n.split("E")[1]),
                 "wiki_episode_id": o} for o, n in movable.items()}
    for n, rec in moved.items():
        if n in records:      # an untouched record holds it: do not overwrite
            rec["episode_id"], rec["episode"] = rec["wiki_episode_id"], int(rec["wiki_episode_id"].split("E")[1])
            rec["unverified"] = True
            records[rec["episode_id"]] = rec
            notes.append(f"WARNING {n}: already held by another article; "
                         f"'{rec['title']}' kept at {rec['episode_id']}")
        else:
            records[n] = rec

    # 2) articles without usable numbers: by title
    for rec in unresolved:
        cands = [e for e, (s, t) in db.items()
                 if s == rec["season"] and _same_title(t, rec["title"])
                 and e not in records]
        if len(cands) == 1:
            eid = cands[0]
            rec.update(episode_id=eid, episode=int(eid.split("E")[1]),
                       resolved_by="title")
            records[eid] = rec
            notes.append(f"'{rec['title']}' -> {eid} (no usable infobox numbers: placed by title)")
        else:
            notes.append(f"WARNING '{rec['title']}' (season {rec['season']}): no usable infobox "
                         f"numbers and {len(cands)} title matches in the DB; skipped")

    missing = sorted(e for e in db if e not in records)
    if missing:
        notes.append(f"{len(missing)} episodes of the DB without synopsis: "
                     + ", ".join(f"{e} '{db[e][1]}'" for e in missing[:15])
                     + (" ..." if len(missing) > 15 else ""))
    return notes


def _parse_seasons(text: str) -> list[int]:
    if "-" in text:
        a, b = text.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in text.split(",")]


def main() -> None:
    p = argparse.ArgumentParser(description="Episode plots from Wikipedia")
    p.add_argument("--seasons", default="1-10", help="e.g. 1-10, 5, 5,6 (default 1-10)")
    p.add_argument("--refresh", action="store_true", help="ignore the local cache")
    args = p.parse_args()

    ensure_dirs()
    records, unresolved, problems = build(_parse_seasons(args.seasons), args.refresh)
    notes = check_against_chunks(records, unresolved)

    SYNOPSES.write_text(json.dumps(dict(sorted(records.items())),
                                   ensure_ascii=False, indent=1), encoding="utf-8")
    words = [r["words"] for r in records.values()]
    srcs = {}
    for r in records.values():
        srcs[r["plot_source"]] = srcs.get(r["plot_source"], 0) + 1
    print(f"\n{len(records)} synopses -> {SYNOPSES}")
    if words:
        print(f"words per plot: min {min(words)}, median {sorted(words)[len(words)//2]}, max {max(words)}"
              f" | sections used: {srcs}")
    for label, items in (("Problems", problems), ("Against the DB", notes)):
        if items:
            print(f"\n{label} ({len(items)}):")
            for it in items:
                print(f"  - {it}")


if __name__ == "__main__":
    main()
