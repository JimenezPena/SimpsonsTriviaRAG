"""Chunk identification for a clip.

Matches the clip's dub transcript directly against the chunks exported from the vector DB (gemini-embedding-001 is
multilingual, so the Spanish transcript can be compared with the English
chunks). Chunk embeddings are extracted ONCE from the vector DB (where
ingest_batch.py already computed them) — no embedding is ever
recalculated; identifying a clip only requires 1 new embedding (the
clip transcript itself).
"""

import json
import unicodedata

import numpy as np

from .config import (
    CHUNKS_EN, CHUNK_EMBEDDINGS,
    MATCH_AUTO_ACCEPT, MATCH_REVIEW,
)
from .google_client import embed_texts

CHARACTER_ALIASES = {
    "homero": "Homer", "homer": "Homer", "marge": "Marge",
    "bart": "Bart", "lisa": "Lisa", "maggie": "Maggie",
    "milhouse": "Milhouse", "burns": "Mr. Burns",
    "senor burns": "Mr. Burns", "señor burns": "Mr. Burns",
    "ned flanders": "Ned Flanders", "flanders": "Ned Flanders",
    "apu": "Apu", "moe": "Moe", "skinner": "Skinner", "krusty": "Krusty",
    "abuelo": "Grampa", "grampa": "Grampa",
}


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKD", text).lower()
    return "".join(c for c in t if not unicodedata.combining(c))


def detect_speakers(text: str) -> set[str]:
    """Cheap heuristic: character mentions in the text."""
    t = normalize(text)
    return {alias for key, alias in CHARACTER_ALIASES.items() if key in t}


def cosine(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def load_chunks_en() -> list[dict]:
    return [json.loads(l) for l in
            CHUNKS_EN.read_text(encoding="utf-8").splitlines() if l.strip()]


# --------------------------------------------------------------------------
# Offline precomputation of chunk embeddings
# --------------------------------------------------------------------------

def _load_chunk_embeddings() -> dict[str, list[float]]:
    """Embeddings extracted from the vector DB by export_from_db.py —
    never recalculated: they were computed by ingest_batch.py."""
    if not CHUNK_EMBEDDINGS.exists():
        raise SystemExit(
            "No chunk embeddings found. Run first:\n"
            "  python -m spanish_align.export_from_db"
        )
    return json.loads(CHUNK_EMBEDDINGS.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Clip matching
# --------------------------------------------------------------------------

def identify_chunk(segments: list[dict]) -> tuple[dict | None, float]:
    """Returns (chunk, score). Cosine + bonus for shared characters.
    A clip can span two chunks; the caller may also want the runner-up."""
    full_text = " ".join(s["text"] for s in segments)
    clip_emb = embed_texts([full_text], task_type="RETRIEVAL_QUERY")[0]
    clip_chars = detect_speakers(full_text)

    embs = _load_chunk_embeddings()
    chunks = {c["chunk_id"]: c for c in load_chunks_en()}

    best_id, best_score = None, 0.0
    for cid, emb in embs.items():
        score = cosine(clip_emb, emb)
        chunk = chunks.get(cid, {})
        chunk_chars = {normalize(s) for s in
                       (l["speaker"] for l in chunk.get("lines", []))}
        if clip_chars and clip_chars & chunk_chars:
            score = min(1.0, score + 0.05 * len(clip_chars & chunk_chars))
        if score > best_score:
            best_id, best_score = cid, score
    return chunks.get(best_id), best_score


def identify_chunk_top_k(segments: list[dict], k: int = 3) -> list[tuple[dict, float]]:
    """Top-k candidate chunks (a clip may span a chunk boundary)."""
    full_text = " ".join(s["text"] for s in segments)
    clip_emb = embed_texts([full_text], task_type="RETRIEVAL_QUERY")[0]

    embs = _load_chunk_embeddings()
    chunks = {c["chunk_id"]: c for c in load_chunks_en()}
    scored = sorted(
        ((chunks.get(cid), cosine(clip_emb, emb)) for cid, emb in embs.items()),
        key=lambda t: t[1], reverse=True,
    )
    return scored[:k]


def match_status(score: float) -> str:
    """Classifies the score: 'accept' | 'review' | 'discard'."""
    if score >= MATCH_AUTO_ACCEPT:
        return "accept"
    if score >= MATCH_REVIEW:
        return "review"
    return "discard"