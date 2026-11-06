"""Central configuration: paths, thresholds and Google models.

Aligned with the main SimpsonsTriviaRAG repo:
  - the English chunks live in the existing ChromaDB (vector_db/), built
    by ingest.py; this pipeline never re-derives them from the CSVs.
  - embeddings: gemini-embedding-001 (batchEmbedContents, up to 100 texts)
  - LLM: gemini-3.5-flash-lite
"""

import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise SystemExit("Set the GOOGLE_API_KEY environment variable")

# Google models (Gemini API) — keep in sync with the main project
EMBED_MODEL = "gemini-embedding-001"
GEMINI_MODEL = "gemini-3.1-flash-lite"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models"
EMBED_BATCH_LIMIT = 100          # max texts per batchEmbedContents request

# Existing vector DB (same path as ingest_batch.py)
DB_PATH = "batch_vector_db"
CHROMA_COLLECTION = None         # None = use the first collection found

# Episodes CSV: only to map Kaggle episode_id -> season/number (SxxExx code)
EPISODES_CSV = Path("data/simpsons_episodes.csv")

# Working paths (outputs live under data/ingest/)
WORKDIR = Path("data/ingest")
MEDIA_DIR = WORKDIR / "media"            # video cache (deletable)
TRANSCRIPTS_DIR = WORKDIR / "transcripts"
CHUNKS_DIR = WORKDIR / "chunks"
CACHE_DIR = WORKDIR / "cache"

CHUNKS_EN = WORKDIR / "chunks_en.jsonl"          # export of existing DB chunks
CHUNKS_ES = WORKDIR / "chunks_es.jsonl"          # translations (by chunk_id)
REVIEW_QUEUE = WORKDIR / "review_queue.jsonl"
IG_LINKS = WORKDIR / "ig_links.jsonl"            # chunk_id -> ig_url registry
CHUNK_EMBEDDINGS = CACHE_DIR / "chunk_embeddings.json"

# Thresholds (tune after validating phase F1 with ~20 clips)
MATCH_AUTO_ACCEPT = 0.75    # above: automatic match
MATCH_REVIEW = 0.55          # between REVIEW and AUTO_ACCEPT: manual review queue
ALIGN_THRESHOLD = 0.60       # per-line literal-vs-dub similarity


def ensure_dirs() -> None:
    for d in (MEDIA_DIR, TRANSCRIPTS_DIR, CHUNKS_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
