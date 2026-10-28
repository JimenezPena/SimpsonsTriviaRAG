"""Chunk inspector for the SimpsonsTriviaRAG vector database.

Reads directly from ChromaDB (no embedding API calls: consumes no quota).

Usage:
    python src/crosscheck_chunks.py                          # global summary + per-episode table
    python src/crosscheck_chunks.py --season 3               # season 3 only
    python src/crosscheck_chunks.py --episode 43             # only the episode with episode_id=43
    python src/crosscheck_chunks.py --show --episode 43      # prints the TEXT of its chunks
    python src/crosscheck_chunks.py --show --season 2        # prints the chunks of a whole season
    python src/crosscheck_chunks.py --compare                 # episodes that SHOULD be present (per the
                                                      # CSV and MAX_SEASON) but are missing from the DB
"""
import os
import argparse
import pandas as pd
import chromadb

DB_PATH = "vector_db"
COLLECTION_NAME = "langchain"   # default collection name used by langchain-chroma
EPISODES_CSV = "data/simpsons_episodes.csv"
MAX_SEASON = 13                 # must match ingest.py


def load_chunks():
    client = chromadb.PersistentClient(path=DB_PATH)
    col = client.get_collection(COLLECTION_NAME)
    data = col.get(include=["metadatas", "documents"])
    chunks = []
    for doc, meta in zip(data["documents"], data["metadatas"]):
        chunks.append({"text": doc, **meta})
    return chunks


def group_by_episode(chunks):
    episodes = {}
    for c in chunks:
        eid = c.get("episode_id")
        if eid is None:
            continue
        episodes.setdefault(eid, []).append(c)
    return episodes


def print_stats(chunks, season=None, episode=None):
    episodes = group_by_episode(chunks)
    rows = []
    for eid, echunks in episodes.items():
        e = echunks[0]
        if season is not None and e.get("season") != season:
            continue
        if episode is not None and eid != episode:
            continue
        rows.append({
            "episode_id": eid,
            "season": e.get("season"),
            "title": e.get("episode_title", "?"),
            "chunks": len(echunks),
            "lines": max((c.get("line_end", 0) for c in echunks), default=0),
        })

    if not rows:
        print("No chunks match the filter.")
        return

    rows.sort(key=lambda r: (r["season"], r["episode_id"]))
    counts = [r["chunks"] for r in rows]
    print(f"{'S':>2} {'episode_id':>10}  {'chunks':>6}  title")
    print("-" * 70)
    for r in rows:
        flag = " ⚠️" if r["chunks"] < 10 else ""
        print(f"{r['season']:>2} {r['episode_id']:>10}  {r['chunks']:>6}  "
              f"{r['title']}{flag}")

    print("-" * 70)
    print(f"Episodes listed: {len(rows)}")
    print(f"Chunks: total={sum(counts)}, average={sum(counts)/len(counts):.1f}, "
          f"min={min(counts)}, max={max(counts)}")
    low = [r for r in rows if r["chunks"] < 10]
    if low:
        print(f"\n⚠️  Episodes with fewer than 10 chunks (⚠️ above):")
        for r in low:
            print(f"   S{r['season']} — {r['episode_id']} \"{r['title']}\" "
                  f"({r['chunks']} chunks, last line {r['lines']})")


def print_chunks(chunks, season=None, episode=None, limit_lines=None):
    episodes = group_by_episode(chunks)
    shown = 0
    for eid in sorted(episodes):
        echunks = episodes[eid]
        e = echunks[0]
        if season is not None and e.get("season") != season:
            continue
        if episode is not None and eid != episode:
            continue
        print(f"\n{'=' * 70}")
        print(f"S{e.get('season')} — episode {eid} \"{e.get('episode_title', '?')}\" "
              f"({len(echunks)} chunks)")
        print(f"{'=' * 70}")
        echunks.sort(key=lambda c: c.get("line_start", 0))
        for i, c in enumerate(echunks, 1):
            print(f"\n--- chunk {i}/{len(echunks)} "
                  f"[lines {c.get('line_start')}–{c.get('line_end')}] ---")
            print(c["text"])
            shown += 1
    print(f"\n({shown} chunks shown)")


def print_missing(chunks):
    """Compares the expected episodes (CSV + MAX_SEASON) with those present in the DB."""
    if not os.path.exists(EPISODES_CSV):
        print(f"Cannot find {EPISODES_CSV}; comparison impossible.")
        return
    eps = pd.read_csv(EPISODES_CSV)
    eps = eps[eps["season"] <= MAX_SEASON]
    expected = set(eps["id"])
    present = set(group_by_episode(chunks).keys())
    missing = sorted(expected - present)
    extra = sorted(present - expected)

    print(f"Expected (S1-{MAX_SEASON}): {len(expected)} episodes")
    print(f"Present in the DB: {len(present)} episodes")
    if missing:
        print(f"\n❌ Not ingested yet ({len(missing)}):")
        titles = eps.set_index("id")["title"]
        for eid in missing:
            season = eps.loc[eps["id"] == eid, "season"].iloc[0]
            print(f"   S{season} — {eid} \"{titles[eid]}\"")
    else:
        print("\n✅ All expected episodes are in the DB.")
    if extra:
        print(f"\n⚠️ Episodes in the DB outside the range: {extra}")


def main():
    parser = argparse.ArgumentParser(description="Chunk inspector (local ChromaDB)")
    parser.add_argument("--season", type=int, help="Filter by season (e.g. 3)")
    parser.add_argument("--episode", type=int,
                        help="Filter by episode_id (the internal id, e.g. 43)")
    parser.add_argument("--show", action="store_true",
                        help="Print the TEXT of the chunks instead of just counting")
    parser.add_argument("--compare", action="store_true",
                        help="List expected episodes missing from the DB")
    args = parser.parse_args()

    if not os.path.exists(DB_PATH):
        print(f"❌ No database found at ./{DB_PATH}")
        return

    chunks = load_chunks()
    if not chunks:
        print("The collection exists but it is empty.")
        return

    if args.compare:
        print_missing(chunks)
    elif args.show:
        print_chunks(chunks, season=args.season, episode=args.episode)
    else:
        print_stats(chunks, season=args.season, episode=args.episode)


if __name__ == "__main__":
    main()
