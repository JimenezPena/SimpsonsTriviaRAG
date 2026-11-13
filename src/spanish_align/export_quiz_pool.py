"""OFFLINE TASK: builds data/quiz_pool.json — the quiz mode's data source.

One entry per confirmed clip (ig_links.jsonl: chunk_id -> ig_url), so any
question sampled from the pool has a video to show on reveal. The text is
what the CLIP says, not the whole chunk:

  * Spanish dub clip   -> text_es = the dub lines that aligned with the
                          script (translation_source="dub"); segments that
                          did not align are left out, because their literal
                          translation would belong to a different line.
  * English-audio clip -> text_en = the audio transcript; text_es = literal
                          translation of the lines it aligned with.
  * Each entry also carries "visual" (what is SEEN in the clip, from
    `cli describe`) when the clip has one.
  * No usable aligned lines (e.g. a clip record from an old version) -> the
    whole chunk's literal translation, counted in the summary.

Entries are keyed by clip (shortcode): two videos of the same chunk are
two entries.

    PYTHONPATH=src python -m spanish_align.export_quiz_pool

Output: data/quiz_pool.json (small; commit it to the repo so Render
gets it for free — it only contains URLs and dialogue text).
"""

import json
from collections import Counter
from pathlib import Path

from .config import (CHUNKS_DIR, CHUNKS_EN, CHUNKS_ES, IG_LINKS, TRANSCRIPTS_DIR,
                     ensure_dirs)

QUIZ_POOL = Path("data/quiz_pool.json")


def _load_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in
            path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _clip_record(shortcode):
    p = CHUNKS_DIR / f"{shortcode}.jsonl"
    if not p.exists():
        return None
    first = p.read_text(encoding="utf-8").splitlines()[:1]
    return json.loads(first[0]) if first else None


def _visual(shortcode):
    """The clip's visual description (cli describe), or None."""
    p = TRANSCRIPTS_DIR / f"{shortcode}.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    return (data.get("visual") or None) if isinstance(data, dict) else None


def _join(rows):
    """'Speaker: text' per row; consecutive duplicates collapse (two dub
    segments aligned to the same script line)."""
    out = []
    for speaker, text in rows:
        row = f"{speaker}: {text}"
        if not out or out[-1] != row:
            out.append(row)
    return "\n".join(out)


def _chunk_es(chunk_es):
    return _join((l["speaker"], l["text_es"]) for l in chunk_es["lines"]
                 if not l.get("context"))


def _chunk_en(chunk_en):
    return _join((l["speaker"], l["text"]) for l in chunk_en["lines"]
                 if not l.get("context"))


def _clip_text(rec, lang):
    """(text_es, text_en, source) taken from the clip's own aligned lines;
    None for the parts the clip cannot provide."""
    lines = (rec or {}).get("lines") or []
    if lang == "en":
        audio = [l for l in lines if l.get("text_en")]
        text_en = _join((l["speaker"], l["text_en"]) for l in audio) or None
        text_es = _join((l["speaker"], l["text_es"]) for l in audio
                        if l.get("text_es")) or None
        return text_es, text_en, "audio"
    dub = [l for l in lines
           if l.get("translation_source") == "dub" and l.get("text_es")]
    text_es = _join((l["speaker"], l["text_es"]) for l in dub) or None
    return text_es, None, "dub"


def main() -> None:
    ensure_dirs()
    links = _load_jsonl(IG_LINKS)
    en = {c["chunk_id"]: c for c in _load_jsonl(CHUNKS_EN)}
    es = {c["chunk_id"]: c for c in _load_jsonl(CHUNKS_ES)}

    pool, no_translation, stale = {}, 0, 0
    sources = Counter()
    for link in links:
        cid = link["chunk_id"]
        c_en, c_es = en.get(cid), es.get(cid)
        if c_es is None or c_en is None:
            no_translation += 1
            continue
        shortcode = link.get("shortcode") or link["ig_url"].rstrip("/").split("/")[-1]
        rec = _clip_record(shortcode)
        if rec and rec.get("chunk_id") and rec["chunk_id"] != cid:
            stale += 1          # the clip was re-matched to another chunk
            continue
        lang = ((rec or {}).get("audio_lang") or link.get("lang")
                or link.get("audio_lang") or "es")

        clip_es, clip_en, source = _clip_text(rec, lang)
        if clip_es is None and clip_en is None:
            source = "chunk-literal"          # fallback: the whole chunk
        pool[shortcode] = {
            "chunk_id": cid, "ig_url": link["ig_url"], "shortcode": shortcode,
            "audio_lang": lang,
            "season": c_en["season"], "title": c_en["episode_title"],
            "text_es": clip_es or _chunk_es(c_es),
            "text_en": clip_en or _chunk_en(c_en),
            "text_source": source,
        }
        visual = _visual(shortcode)
        if visual:
            pool[shortcode]["visual"] = visual
        sources[source] += 1

    QUIZ_POOL.write_text(json.dumps(pool, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    print(f"quiz_pool.json: {len(pool)} clips "
          f"({', '.join(f'{n} {s}' for s, n in sorted(sources.items())) or 'none'})")
    if sources["chunk-literal"]:
        print(f"  {sources['chunk-literal']} entries use the whole chunk's literal "
              "translation (no aligned lines): reprocess those clips with "
              "'cli ingest --redo'.")
    n_vis = sum(1 for e in pool.values() if e.get("visual"))
    print(f"  {n_vis} of {len(pool)} clips have a visual description"
          + ("" if n_vis == len(pool) else " (cli describe adds the rest)"))
    if stale:
        print(f"  skipped {stale} stale ig_links entries (clip re-matched to another chunk).")
    if no_translation:
        print(f"  skipped {no_translation} (no translation yet): "
              "run translate_chunks for the missing episodes and re-export.")


if __name__ == "__main__":
    main()
