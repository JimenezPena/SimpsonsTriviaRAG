"""Episode catalog: the structured data behind the metadata tools.

Source of truth: data/ingest/episode_synopses.json (committed, one entry per
episode with season, number in season and title). It is a CATALOG of what the
project knows, not the real-world episode list, so answers say "in the
catalog". Optionally the catalog also knows which episodes have script chunks
in the vector DB (`indexed`), read straight from chroma.sqlite3.
"""
import json
import os
import re
import sqlite3
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_PATH = os.path.join(ROOT, "data", "ingest", "episode_synopses.json")


def norm(text):
    """Lowercase, no accents, no apostrophes, punctuation -> single spaces."""
    t = unicodedata.normalize("NFKD", text or "")
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    t = re.sub(r"['’‘`]", "", t)
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


class Catalog:
    def __init__(self, episodes, indexed=None):
        """episodes: [{"season", "episode", "title", ...}];
        indexed: {season: set(normalized titles with script chunks)} or None."""
        self.episodes = sorted(
            ({"season": int(e["season"]), "episode": int(e["episode"]),
              "title": e["title"], "episode_id": e.get("episode_id")} for e in episodes),
            key=lambda e: (e["season"], e["episode"]))
        self.indexed = indexed
        self._by_season = {}
        for e in self.episodes:
            self._by_season.setdefault(e["season"], []).append(e)

    @classmethod
    def load(cls, path=DEFAULT_PATH, indexed=None):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(list(data.values()) if isinstance(data, dict) else data, indexed)

    @property
    def seasons(self):
        return sorted(self._by_season)

    def in_season(self, season):
        return list(self._by_season.get(season, []))

    def by_number(self, season, number):
        return next((e for e in self._by_season.get(season, []) if e["episode"] == number), None)

    def find_title(self, text):
        """Episodes whose title matches `text`: the exact (normalized) title if
        there is one, otherwise every title that contains it (>= 4 characters).
        Never fuzzy: a wrong guess is worse than 'not found'."""
        q = norm(text)
        if not q:
            return []
        exact = [e for e in self.episodes if norm(e["title"]) == q]
        if exact:
            return exact
        if len(q) < 4:
            return []
        return [e for e in self.episodes if q in norm(e["title"])]

    def indexed_count(self, season):
        """Episodes of `season` with script chunks in the DB (None if unknown)."""
        if self.indexed is None:
            return None
        have = self.indexed.get(season, set())
        return sum(norm(e["title"]) in have for e in self._by_season.get(season, []))


def load_indexed_from_chroma(db_path):
    """{season: {normalized episode titles with script chunks}} from Chroma's own
    SQLite file (read-only; plot summaries, kind='synopsis', are not scripts)."""
    path = os.path.join(db_path, "chroma.sqlite3")
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)
    try:
        rows = con.execute(
            "SELECT DISTINCT s.int_value, t.string_value FROM embeddings e "
            "JOIN embedding_metadata s ON s.id = e.id AND s.key = 'season' "
            "JOIN embedding_metadata t ON t.id = e.id AND t.key = 'episode_title' "
            "WHERE NOT EXISTS (SELECT 1 FROM embedding_metadata k WHERE k.id = e.id "
            "AND k.key = 'kind' AND k.string_value = 'synopsis')").fetchall()
    finally:
        con.close()
    out = {}
    for season, title in rows:
        if season is not None and title:
            out.setdefault(int(season), set()).add(norm(title))
    return out
