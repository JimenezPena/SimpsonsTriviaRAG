"""Gemini API client: embeddings, audio transcription and translation.

All the heavy lifting goes through here, with disk caches to avoid
repeated calls (embeddings) and re-transcriptions (transcripts).
"""

import base64
import hashlib
import http.client
import json
import time
import urllib.request

from .config import (
    GOOGLE_API_KEY, GEMINI_MODEL, GEMINI_URL, EMBED_MODEL, EMBED_BATCH_LIMIT,
    CACHE_DIR, TRANSCRIPTS_DIR, ensure_dirs,
)


# --------------------------------------------------------------------------
# Low-level helpers
# --------------------------------------------------------------------------

def _post(model: str, method: str, payload: dict,
          timeout: int = 120, retries: int = 4) -> dict:
    """POST to the Gemini API with retries: backoff on 429 rate limits
    and reconnection on transient network errors (RemoteDisconnected,
    URLError, timeouts) — same pattern as ingest_batch.py."""
    url = f"{GEMINI_URL}/{model}:{method}?key={GOOGLE_API_KEY}"
    body = json.dumps(payload).encode("utf-8")
    for attempt in range(1, retries + 1):
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries:
                time.sleep(min(60 * attempt, 90))
                continue
            if e.code in (500, 502, 503, 504) and attempt < retries:
                wait = 15 * attempt
                print(f"    [warn] HTTP {e.code}: "
                      f"retry {attempt}/{retries} in {wait}s")
                time.sleep(wait)
                continue
            raise
        except (urllib.error.URLError, http.client.HTTPException,
                ConnectionError, TimeoutError) as e:
            if attempt < retries:
                wait = 10 * attempt
                print(f"    [warn] {type(e).__name__}: "
                      f"retry {attempt}/{retries} in {wait}s")
                time.sleep(wait)
                continue
            raise


def _strip_fences(raw: str) -> str:
    return (raw.strip()
            .removeprefix("```json").removeprefix("```")
            .removesuffix("```").strip())


# --------------------------------------------------------------------------
# Embeddings (with disk cache, batched within the API limit)
# --------------------------------------------------------------------------

def embed_texts(texts: list[str],
                task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """gemini-embedding-001, batched in chunks of EMBED_BATCH_LIMIT.

    task_type follows the same convention as ingest_batch.py:
      RETRIEVAL_DOCUMENT  — chunks/scripts being indexed or compared against
      RETRIEVAL_QUERY     — the clip transcript used as a search query
      SEMANTIC_SIMILARITY — symmetric line-by-line comparison (alignment)
    Cached on disk by (text, task_type) hash: nothing is embedded twice.
    """
    ensure_dirs()
    cache_file = CACHE_DIR / "embeddings.json"
    cache: dict[str, list[float]] = {}
    if cache_file.exists():
        cache = json.loads(cache_file.read_text(encoding="utf-8"))

    keys = [_cache_key(t, task_type) for t in texts]
    missing_idx = [i for i, k in enumerate(keys) if k not in cache]
    for m in range(0, len(missing_idx), EMBED_BATCH_LIMIT):
        batch_idx = missing_idx[m:m + EMBED_BATCH_LIMIT]
        payload = {
            "requests": [
                {"model": f"models/{EMBED_MODEL}",
                 "content": {"parts": [{"text": texts[i][:8000]}]},
                 "taskType": task_type}
                for i in batch_idx
            ]
        }
        url = f"{GEMINI_URL}/{EMBED_MODEL}:batchEmbedContents?key={GOOGLE_API_KEY}"
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            resp = json.loads(r.read())
        for i, emb in zip(batch_idx, resp["embeddings"]):
            cache[keys[i]] = emb["values"]
    if missing_idx:
        cache_file.write_text(json.dumps(cache), encoding="utf-8")
    return [cache[k] for k in keys]


def _cache_key(text: str, task_type: str) -> str:
    return hashlib.sha256(f"{task_type}\x00{text}".encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Audio transcription (video -> segments with timestamps)
# --------------------------------------------------------------------------

def transcribe(video) -> dict:
    """Returns {"lang": "es"|"en", "segments": [{start, end, text}]}.
    Cached; old list-format caches are ignored and re-transcribed."""
    ensure_dirs()
    cache = TRANSCRIPTS_DIR / f"{video.stem}.json"
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "lang" in data:
            return data
        # old format (bare list): language unknown -> re-transcribe below

    with open(video, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    prompt = (
        "Transcribe ALL the dialogue in this clip from The Simpsons. First "
        "detect the spoken language: 'es' for the Spanish dub, 'en' for the "
        "original English audio. Reply ONLY with a JSON object, no markdown:\n"
        '{"lang": "<es|en>", "segments": [{"start": <seconds float>, '
        '"end": <seconds float>, "text": "<line>"}]}\n'
        "One element per dialogue line. Ignore music, sound effects and "
        "canned laughter. Use the audio timeline for start/end."
    )
    payload = {
        "contents": [{"parts": [
            {"text": prompt},
            {"inline_data": {"mimeType": "video/mp4", "data": b64}},
        ]}],
        "generationConfig": {"temperature": 0.0},
    }
    resp = _post(GEMINI_MODEL, "generateContent", payload)
    raw = resp["candidates"][0]["content"]["parts"][0]["text"]
    data = json.loads(_strip_fences(raw))
    cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data

# --------------------------------------------------------------------------
# Literal EN -> ES translation
# --------------------------------------------------------------------------

def translate_literal(lines_en: list[str]) -> list[str]:
    """Literal translation (not the official dub), line by line."""
    prompt = (
        "Translate each line of this The Simpsons script from English to "
        "Spanish. Use LITERAL translation (not the official dub): keep the "
        "original meaning as close as possible. Reply ONLY with a JSON array "
        "of strings, one translation per input line, same order. No markdown.\n\n"
        + json.dumps(lines_en, ensure_ascii=False)
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0},
    }
    resp = _post(GEMINI_MODEL, "generateContent", payload)
    raw = resp["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(_strip_fences(raw))
