"""Baseline (no-RAG) question answering for SimpsonsTriviaRAG.

Answers questions with the SAME LLM and the SAME generation parameters as the
free-form Q&A mode in trivia_gen.py, but WITHOUT any retrieved context — the
system prompt is only: "You are an expert in the Simpsons TV series."

Use it to measure how much the RAG pipeline actually adds: run the same
question through both scripts and compare answers side by side.
"""
import os
import json
import urllib.request
import urllib.error
from dotenv import load_dotenv

# ==================== CONFIGURATION ====================
LLM_MODEL = "gemini-3.5-flash-lite"   # must match trivia_gen.py
# =======================================================

load_dotenv()
API_KEY = os.environ["GOOGLE_API_KEY"]

# Baseline system prompt: identical for every question, no retrieved context
BASELINE_SYSTEM = "You are an expert in the Simpsons TV series. Answer the following question."


def ask_llm(prompt, system=None):
    """Same LLM call as trivia_gen.py: same temperature, thinking and maxOutputTokens."""
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.9,
            "maxOutputTokens": 4096,
            # Gemini 3 family: thinkingLevel instead of thinkingBudget.
            # "minimal" = cheapest and fastest. Thinking tokens count inside
            # maxOutputTokens, so the budget is raised to 2048 + 2048.
            "thinkingConfig": {"thinkingLevel": "minimal"},
        },
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{LLM_MODEL}:generateContent?key={API_KEY}")
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "ignore")
        if e.code == 400 and "thinkingConfig" in payload["generationConfig"]:
            # Some models reject the thinking setting: retry once without it
            print(f"⚠️  400 with thinkingConfig; retrying without it: {detail[:200]}")
            del payload["generationConfig"]["thinkingConfig"]
            payload["generationConfig"]["maxOutputTokens"] = 2048
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=body,
                                         headers={"Content-Type": "application/json"},
                                         method="POST")
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            except urllib.error.HTTPError as e2:
                raise RuntimeError(f"LLM HTTP {e2.code}: "
                                   f"{e2.read().decode('utf-8', 'ignore')}") from e2
        raise RuntimeError(f"LLM HTTP {e.code}: {detail}") from e


def answer_question(question):
    """Asks the LLM with no retrieval: pure parametric knowledge."""
    return ask_llm(question, system=BASELINE_SYSTEM)


def main():
    print("🍩 Simpsons Baseline Generator (no RAG) — for RAG comparison")
    print("   Same LLM and parameters as trivia_gen.py, but WITHOUT retrieved context.\n")

    while True:
        question = input("❓ Your question about The Simpsons (empty to exit): ").strip()
        if not question:
            print("See you next time! 🍺")
            break
        try:
            out = answer_question(question)
        except RuntimeError as e:
            print(f"⚠️  LLM error: {e}")
            continue
        print(f"\n💬 {out}\n")


if __name__ == "__main__":
    main()
