"""Retries with incremental backoff for flaky Gemini calls.

    call_with_retry(tg.ask_llm, prompt, system=..., temperature=0.2)

What gets retried (errors surface from trivia_gen as RuntimeError, or as a
raw OSError/URLError from the embedding client):

  * HTTP 429 per-minute rate limit   -> wait and retry; if it still fails after
                                        every retry, QuotaExhausted
  * HTTP 429 DAILY quota ("PerDay")  -> no point waiting: QuotaExhausted now
  * HTTP 408 / 5xx, network errors, timeouts, empty/odd responses
                                     -> wait and retry; re-raised at the end
  * anything else (400, 401, 403, 404...) -> raised immediately

The wait grows as base * factor**attempt (5, 10, 20, 40, 80 s by default),
capped, with a little jitter, and never shorter than the "retryDelay" hint
Gemini puts in 429 bodies.
"""
import random
import re
import time


class QuotaExhausted(RuntimeError):
    """The quota is gone (daily) or 429 outlasted every retry: stop the run
    cleanly and resume later."""


_HTTP = re.compile(r"HTTP (\d{3})")
_HINT = re.compile(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"')
MAX_HINT = 300.0       # never trust a server hint beyond 5 minutes


def classify(err):
    """-> 'daily_quota' | 'rate_limit' | 'transient' | 'fatal'."""
    msg = str(err)
    m = _HTTP.search(msg)
    code = int(m.group(1)) if m else None
    if code == 429:
        return "daily_quota" if "perday" in msg.lower() else "rate_limit"
    if code is not None:
        return "transient" if code == 408 or code >= 500 else "fatal"
    if (isinstance(err, OSError) or "network error" in msg
            or "Unexpected LLM response" in msg):
        return "transient"
    return "fatal"


def backoff_delay(attempt, base, factor, cap, hint=None):
    delay = min(cap, base * factor ** attempt)
    return max(delay, min(hint, MAX_HINT)) if hint else delay


def call_with_retry(fn, *args, max_retries=5, base=5.0, factor=2.0, cap=120.0,
                    jitter=0.25, sleep=time.sleep, rand=random.random,
                    log=print, **kwargs):
    """Calls fn(*args, **kwargs), retrying as described in the module doc.
    `sleep`, `rand` and `log` are injectable so tests run instantly."""
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except (RuntimeError, OSError) as e:
            kind = classify(e)
            if kind == "daily_quota":
                raise QuotaExhausted(str(e)) from e
            if kind == "fatal":
                raise
            if attempt == max_retries:
                if kind == "rate_limit":
                    raise QuotaExhausted(str(e)) from e
                raise
            hint = _HINT.search(str(e))
            delay = backoff_delay(attempt, base, factor, cap,
                                  float(hint.group(1)) if hint else None)
            delay *= 1 + jitter * rand()
            log(f"   {kind}: {str(e)[:90]!r} -> retry {attempt + 1}/{max_retries} "
                f"in {delay:.0f}s")
            sleep(delay)
