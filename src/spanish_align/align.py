"""Line-by-line alignment: dub transcript vs. English chunk lines.

For each transcribed segment, finds the semantically closest line in the
matched chunk. Above the threshold, the dub transcript "wins"
(translation_source="dub"). Below it, the precomputed literal
translation from chunks_es.jsonl is used (translation_source="literal").
"""

import numpy as np

from .models import Line
from .config import ALIGN_THRESHOLD
from .google_client import embed_texts
from .chunk_match import cosine


def align_line(text_en: str, text_es_dub: str) -> float:
    """Semantic similarity between a script line and a dub line."""
    a, b = embed_texts([text_en, text_es_dub])
    return cosine(a, b)


def build_lines(
    segments: list[dict],
    chunk_lines: list[dict],
    literal_es: dict[int, str],
    clip_lang: str = "es",
) -> list[Line]:
    """clip_lang="es": dub transcript -> text_es (source "dub").
    clip_lang="en": the transcript IS the original -> text_en
    (source "audio"); the literal translation covers text_es."""
    dubs = [s["text"] for s in segments]
    ens = [l["text"] for l in chunk_lines if not l.get("context")]
    embs = embed_texts(dubs + ens, task_type="SEMANTIC_SIMILARITY")
    dub_embs, en_embs = embs[:len(dubs)], embs[len(dubs):]
    en_arr = np.asarray(en_embs)
    en_idx = [i for i, l in enumerate(chunk_lines) if not l.get("context")]

    lines: list[Line] = []
    for seg, dub_emb in zip(segments, dub_embs):
        denom = np.linalg.norm(en_arr, axis=1) * np.linalg.norm(dub_emb)
        scores = (en_arr @ dub_emb) / np.where(denom == 0, 1e-9, denom)
        best_j = int(np.argmax(scores))
        best_i = en_idx[best_j]

        if scores[best_j] >= ALIGN_THRESHOLD:
            if clip_lang == "en":
                lines.append(Line(
                    speaker=chunk_lines[best_i].get("speaker", "?"),
                    text_es=literal_es.get(best_i) or "",
                    start=seg["start"], end=seg["end"],
                    text_en=seg["text"],
                    translation_source="audio",
                ))
            else:
                lines.append(Line(
                    speaker=chunk_lines[best_i].get("speaker", "?"),
                    text_es=seg["text"],
                    start=seg["start"], end=seg["end"],
                    text_en=chunk_lines[best_i]["text"],
                    translation_source="dub",
                ))
        else:
            lit = literal_es.get(best_i)
            lines.append(Line(
                speaker=chunk_lines[best_i].get("speaker", "?"),
                text_es=lit if lit else "",
                start=seg["start"], end=seg["end"],
                text_en=chunk_lines[best_i]["text"],
                translation_source="literal" if lit else "missing",
            ))
    return lines
