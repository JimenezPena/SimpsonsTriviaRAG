"""Inspector de chunks de la base vectorial de SimpsonsTriviaRAG.

Lee directamente de ChromaDB (sin llamadas a la API de embeddings: no consume cuota).

Uso:
    python src/crosscheck_chunks.py                          # resumen global + tabla por episodio
    python src/crosscheck_chunks.py --season 3               # solo la temporada 3
    python src/crosscheck_chunks.py --episode 43             # solo el episodio con episode_id=43
    python src/crosscheck_chunks.py --show --episode 43      # imprime el TEXTO de sus chunks
    python src/crosscheck_chunks.py --show --season 2        # imprime los chunks de toda la temporada
    python src/crosscheck_chunks.py --compare                 # episodios que DEBERÍAN estar (según el CSV
                                                      # y MAX_SEASON) y no están en la BD
"""
import os
import argparse
import pandas as pd
import chromadb

DB_PATH = "vector_db"
COLLECTION_NAME = "langchain"   # nombre por defecto que usa langchain-chroma
EPISODES_CSV = "data/simpsons_episodes.csv"
MAX_SEASON = 13                 # debe coincidir con ingest.py


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
        print("No hay chunks que coincidan con el filtro.")
        return

    rows.sort(key=lambda r: (r["season"], r["episode_id"]))
    counts = [r["chunks"] for r in rows]
    print(f"{'T':>2} {'episode_id':>10}  {'chunks':>6}  título")
    print("-" * 70)
    for r in rows:
        flag = " ⚠️" if r["chunks"] < 10 else ""
        print(f"{r['season']:>2} {r['episode_id']:>10}  {r['chunks']:>6}  "
              f"{r['title']}{flag}")

    print("-" * 70)
    print(f"Episodios listados: {len(rows)}")
    print(f"Chunks: total={sum(counts)}, media={sum(counts)/len(counts):.1f}, "
          f"mín={min(counts)}, máx={max(counts)}")
    low = [r for r in rows if r["chunks"] < 10]
    if low:
        print(f"\n⚠️  Episodios con menos de 10 chunks (⚠️ arriba):")
        for r in low:
            print(f"   T{r['season']} — {r['episode_id']} «{r['title']}» "
                  f"({r['chunks']} chunks, última línea {r['lines']})")


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
        print(f"T{e.get('season')} — episodio {eid} «{e.get('episode_title', '?')}» "
              f"({len(echunks)} chunks)")
        print(f"{'=' * 70}")
        echunks.sort(key=lambda c: c.get("line_start", 0))
        for i, c in enumerate(echunks, 1):
            print(f"\n--- chunk {i}/{len(echunks)} "
                  f"[líneas {c.get('line_start')}–{c.get('line_end')}] ---")
            print(c["text"])
            shown += 1
    print(f"\n({shown} chunks mostrados)")


def print_missing(chunks):
    """Compara los episodios esperados (CSV + MAX_SEASON) con los presentes en la BD."""
    if not os.path.exists(EPISODES_CSV):
        print(f"No se encuentra {EPISODES_CSV}; imposible comparar.")
        return
    eps = pd.read_csv(EPISODES_CSV)
    eps = eps[eps["season"] <= MAX_SEASON]
    expected = set(eps["id"])
    present = set(group_by_episode(chunks).keys())
    missing = sorted(expected - present)
    extra = sorted(present - expected)

    print(f"Esperados (T1-{MAX_SEASON}): {len(expected)} episodios")
    print(f"Presentes en la BD: {len(present)} episodios")
    if missing:
        print(f"\n❌ Sin ingerir ({len(missing)}):")
        titles = eps.set_index("id")["title"]
        for eid in missing:
            season = eps.loc[eps["id"] == eid, "season"].iloc[0]
            print(f"   T{season} — {eid} «{titles[eid]}»")
    else:
        print("\n✅ Todos los episodios esperados están en la BD.")
    if extra:
        print(f"\n⚠️ Episodios en la BD fuera de rango: {extra}")


def main():
    parser = argparse.ArgumentParser(description="Inspector de chunks (ChromaDB local)")
    parser.add_argument("--season", type=int, help="Filtrar por temporada (p. ej. 3)")
    parser.add_argument("--episode", type=int,
                        help="Filtrar por episode_id (el id interno, p. ej. 43)")
    parser.add_argument("--show", action="store_true",
                        help="Imprimir el TEXTO de los chunks en vez de solo contar")
    parser.add_argument("--compare", action="store_true",
                        help="Listar episodios esperados que faltan en la BD")
    args = parser.parse_args()

    if not os.path.exists(DB_PATH):
        print(f"❌ No existe la base de datos en ./{DB_PATH}")
        return

    chunks = load_chunks()
    if not chunks:
        print("La colección existe pero está vacía.")
        return

    if args.compare:
        print_missing(chunks)
    elif args.show:
        print_chunks(chunks, season=args.season, episode=args.episode)
    else:
        print_stats(chunks, season=args.season, episode=args.episode)


if __name__ == "__main__":
    main()
