"""Text of a Gemini generateContent response.

Joins EVERY text part of the first candidate (the answer can come split in
several parts; reading only parts[0] silently truncates it) and skips thought
parts. Warns when the model did not finish normally (finishReason other than
STOP, e.g. MAX_TOKENS or SAFETY) so a cut-off answer leaves a trace in the log.
Raises KeyError / IndexError / ValueError on an unexpected shape, as before.
"""


def candidate_text(data, log=print):
    cand = data["candidates"][0]
    parts = cand["content"]["parts"]
    text = "".join(p["text"] for p in parts
                   if isinstance(p, dict) and "text" in p and not p.get("thought"))
    if not text.strip():
        raise ValueError("response without text parts")
    reason = cand.get("finishReason")
    if reason not in (None, "STOP"):
        log(f"[llm] finishReason={reason}, {len(parts)} part(s), {len(text)} chars", flush=True)
    return text.strip()
