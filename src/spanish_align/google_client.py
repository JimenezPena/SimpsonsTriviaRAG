"""Gemini API client: embeddings, audio transcription and translation.

All the heavy lifting goes through here, with disk caches to avoid
repeated calls (embeddings) and re-transcriptions (transcripts).
"""

import base64
import hashlib
import http.client
import json
import os
import re
import time
import urllib.error
import urllib.request

from .config import (
    GOOGLE_API_KEY, GEMINI_MODEL, GEMINI_URL, EMBED_MODEL, EMBED_BATCH_LIMIT,
    CACHE_DIR, TRANSCRIPTS_DIR, ensure_dirs,
)


# --------------------------------------------------------------------------
# Low-level helpers
# --------------------------------------------------------------------------

class QuotaExhausted(RuntimeError):
    """A DAILY quota is used up: waiting a few minutes will not help."""


_RATE_LIMIT_HITS = 0        # 429s that were waited out (the callers pace themselves with it)


def _rate_limit_info(e: urllib.error.HTTPError) -> tuple[float | None, bool]:
    """(seconds the API asks us to wait, is it a per-day quota?) from a 429."""
    try:
        body = e.read().decode("utf-8", "replace")
    except Exception:
        body = ""
    delay = None
    m = re.search(r'"retryDelay"\s*:\s*"([\d.]+)s"', body)
    if m:
        delay = float(m.group(1))
    elif e.headers is not None and e.headers.get("Retry-After", "").replace(".", "").isdigit():
        delay = float(e.headers["Retry-After"])
    daily = bool(re.search(r"PerDay|per day|daily", body, re.I))
    return delay, daily


def _post(model: str, method: str, payload: dict,
          timeout: int = 120, retries: int = 6) -> dict:
    """POST to the Gemini API, surviving the usual failures:
      * 429 rate limit: waits what the API asks for (retryDelay /
        Retry-After), else a growing backoff; a PER-DAY quota raises
        QuotaExhausted at once instead of waiting;
      * 500/502/503/504 and network errors (RemoteDisconnected, URLError,
        timeouts, resets): retried with a growing pause;
      * anything else (400, 403...) is raised immediately.
    The last error is re-raised once the retries run out."""
    global _RATE_LIMIT_HITS
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
            if e.code == 429:
                delay, daily = _rate_limit_info(e)
                if daily:
                    raise QuotaExhausted(
                        f"Gemini daily quota exhausted ({model}): try again "
                        "tomorrow or use another API key/plan") from e
                if attempt >= retries:
                    raise
                wait = min((delay + 2) if delay else 20 * attempt, 120)
                _RATE_LIMIT_HITS += 1
                print(f"    [warn] HTTP 429 (rate limit): retry "
                      f"{attempt}/{retries} in {wait:.0f}s")
                time.sleep(wait)
                continue
            if e.code in (500, 502, 503, 504) and attempt < retries:
                wait = 15 * attempt
                print(f"    [warn] HTTP {e.code}: "
                      f"retry {attempt}/{retries} in {wait}s")
                time.sleep(wait)
                continue
            raise
        except (urllib.error.URLError, http.client.HTTPException,
                ConnectionError, TimeoutError, json.JSONDecodeError) as e:
            if attempt < retries:
                wait = min(10 * attempt, 60)
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

EMBED_PAUSE = 1.0           # seconds between batches (longer after a 429)


def _load_embedding_cache(cache_file) -> dict[str, list[float]]:
    if not cache_file.exists():
        return {}
    try:
        return json.loads(cache_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        bad = cache_file.with_suffix(".corrupt.json")
        os.replace(cache_file, bad)
        print(f"    [warn] {cache_file.name} was unreadable (interrupted write?); "
              f"moved to {bad.name}, starting a new cache")
        return {}


def _save_embedding_cache(cache_file, cache) -> None:
    tmp = cache_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache), encoding="utf-8")
    os.replace(tmp, cache_file)      # atomic: an interruption never leaves half a file


def _embed_batch(texts: list[str], task_type: str) -> list[list[float]]:
    """One batchEmbedContents request; if it keeps failing, the batch is
    split in two (a too-large or too-slow request often goes through in
    halves). A single text that fails is raised."""
    payload = {
        "requests": [
            {"model": f"models/{EMBED_MODEL}",
             "content": {"parts": [{"text": t[:8000]}]},
             "taskType": task_type}
            for t in texts
        ]
    }
    try:
        resp = _post(EMBED_MODEL, "batchEmbedContents", payload, timeout=90)
        vecs = [e["values"] for e in resp["embeddings"]]
        if len(vecs) != len(texts):
            raise ValueError(f"asked for {len(texts)} embeddings, got {len(vecs)}")
        return vecs
    except QuotaExhausted:
        raise
    except (urllib.error.URLError, http.client.HTTPException, ConnectionError,
            TimeoutError, ValueError, KeyError) as e:
        if len(texts) == 1:
            raise
        print(f"    [warn] batch of {len(texts)} failed ({type(e).__name__}: {e}); "
              "splitting it in two")
        mid = len(texts) // 2
        return _embed_batch(texts[:mid], task_type) + _embed_batch(texts[mid:], task_type)


def embed_texts(texts: list[str],
                task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """gemini-embedding-001, batched in chunks of EMBED_BATCH_LIMIT.

    task_type follows the same convention as ingest_batch.py:
      RETRIEVAL_DOCUMENT  — chunks/scripts being indexed or compared against
      RETRIEVAL_QUERY     — the clip transcript used as a search query
      SEMANTIC_SIMILARITY — symmetric line-by-line comparison (alignment)
    Cached on disk by (text, task_type) hash: nothing is embedded twice.

    Fault tolerant: each request retries on rate limits, 5xx and network
    errors (see _post); the cache is saved after EVERY batch, so a failure
    half-way (or Ctrl-C) loses nothing and the same command resumes where it
    stopped; a per-day quota raises QuotaExhausted with the progress kept.
    """
    ensure_dirs()
    cache_file = CACHE_DIR / "embeddings.json"
    cache = _load_embedding_cache(cache_file)

    keys = [_cache_key(t, task_type) for t in texts]
    missing_idx = [i for i, k in enumerate(keys) if k not in cache]
    # identical texts in the same call are embedded once
    todo, first = [], {}
    for i in missing_idx:
        if keys[i] not in first:
            first[keys[i]] = i
            todo.append(i)
    pause = EMBED_PAUSE
    for n, m in enumerate(range(0, len(todo), EMBED_BATCH_LIMIT)):
        batch_idx = todo[m:m + EMBED_BATCH_LIMIT]
        hits_before = _RATE_LIMIT_HITS
        try:
            vecs = _embed_batch([texts[i] for i in batch_idx], task_type)
        except BaseException:
            raise             # earlier batches are already on disk
        for i, v in zip(batch_idx, vecs):
            cache[keys[i]] = v
        _save_embedding_cache(cache_file, cache)
        done = min(m + EMBED_BATCH_LIMIT, len(todo))
        if len(todo) > EMBED_BATCH_LIMIT:
            print(f"    embedded {done}/{len(todo)}")
        if _RATE_LIMIT_HITS > hits_before:       # the API pushed back: go slower
            pause = min(pause * 2 + 5, 60)
        if done < len(todo):
            time.sleep(pause)
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

def _translate_once(lines_en: list[str]) -> list[str]:
    """One API call. Raises ValueError if the reply is not a JSON array of
    exactly len(lines_en) strings (truncated output, wrong count...)."""
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
    try:
        cand = resp["candidates"][0]
        raw = cand["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        # blocked / empty reply: no text at all
        why = (resp.get("promptFeedback", {}).get("blockReason")
               or (resp.get("candidates") or [{}])[0].get("finishReason"))
        raise ValueError(f"no text in the reply (reason: {why})")
    try:
        out = json.loads(_strip_fences(raw))
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON (finishReason={cand.get('finishReason')}): {e}")
    if (not isinstance(out, list) or len(out) != len(lines_en)
            or not all(isinstance(x, str) for x in out)):
        n = len(out) if isinstance(out, list) else type(out).__name__
        raise ValueError(f"expected {len(lines_en)} strings, got {n}")
    return out


def translate_literal(lines_en: list[str]) -> list[str]:
    """Literal translation (not the official dub), line by line.

    The reply is validated (valid JSON, same number of lines: a count
    mismatch would silently shift every translation). On failure the call is
    retried once, then the batch is split in two and each half translated
    separately; a single line that still fails (e.g. blocked by the API's
    content filters) is kept in English and reported, so one bad line cannot
    abort a long run."""
    if not lines_en:
        return []
    last = None
    for _ in range(2):
        try:
            return _translate_once(lines_en)
        except ValueError as e:
            last = e
    if len(lines_en) == 1:
        print(f"    [warn] could not translate {lines_en[0]!r}: {last}; "
              "kept in English")
        return list(lines_en)
    print(f"    [warn] batch of {len(lines_en)} lines failed ({last}); "
          "splitting it in two")
    mid = len(lines_en) // 2
    return translate_literal(lines_en[:mid]) + translate_literal(lines_en[mid:])
