"""Web backend for SimpsonsTriviaRAG: a thin Flask wrapper around the logic
in trivia_gen.py (retrieval + LLM), exposing JSON endpoints for the frontend.

Run from the project root:
    pip install flask
    python src/app.py
    # open http://localhost:5000
"""
import os
import re
import uuid
import sys

sys.path.insert(0, os.path.dirname(__file__))   # allow `import trivia_gen`

from flask import Flask, request, jsonify, send_from_directory

import trivia_gen as tg   # reuses: load_vector_store, random_chunk, retrieve,
                          # ask_llm, TRIVIA_SYSTEM, ANSWER_SYSTEM, TOP_K, MAX_SEASON

app = Flask(__name__, static_folder=None)

# the vector store is loaded ONCE at startup, not per request
VECTOR_STORE = tg.load_vector_store()
if VECTOR_STORE._collection.count() == 0:
    # Chroma silently creates an empty DB when the folder is missing; fail
    # loudly at boot instead of returning confusing errors on every request
    raise RuntimeError(f"Vector DB is empty or missing at {tg.DB_PATH}. "
                       "On Render it is downloaded from the GitHub release by build.sh.")

# trivia questions in flight: the answer stays on the server until reveal
QUESTIONS = {}


# ----------------------------------------------------------------------------
# Parsing helpers
# ----------------------------------------------------------------------------
def parse_trivia(out):
    """Splits the LLM output into question, options, answer and fun fact."""
    question, options, answer, fun_fact = None, [], None, None
    for line in out.splitlines():
        s = line.strip()
        if not s:
            continue
        upper = s.upper()
        if upper.startswith("QUESTION:"):
            question = s.split(":", 1)[1].strip()
        elif re.match(r"^[A-D]\)", s):
            options.append(s)
        elif upper.startswith("ANSWER:"):
            answer = s.split(":", 1)[1].strip()
        elif upper.startswith("FUN FACT:"):
            fun_fact = s.split(":", 1)[1].strip()
    return question, options, answer, fun_fact


def chunk_info(doc, score):
    m = doc.metadata
    return {
        "season": m.get("season"),
        "title": m.get("episode_title"),
        "line_start": m.get("line_start"),
        "line_end": m.get("line_end"),
        "distance": round(float(score), 3),
        "text": doc.page_content,
    }


# ----------------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------------
@app.get("/")
def index():
    # serves index.html living next to this file
    return send_from_directory(os.path.dirname(__file__), "index.html")


@app.get("/images/<path:filename>")
def images(filename):
    # static images (wallpaper, logo, Homer thinking gifs) next to this file
    return send_from_directory(os.path.join(os.path.dirname(__file__), "images"),
                               filename)


@app.get("/api/config")
def config():
    return jsonify({"max_season": tg.MAX_SEASON, "top_k": tg.TOP_K})


@app.post("/api/trivia")
def api_trivia():
    """Generates a new trivia question. The answer is NOT sent to the client."""
    season = request.json.get("season") if request.json else None
    chunk, meta = tg.random_chunk(VECTOR_STORE, season)
    source = {"season": meta["season"], "title": meta["episode_title"]}

    prompt = (f"Script fragment (season {meta['season']}, episode "
              f"\"{meta['episode_title']}\"):\n\n{chunk}\n\n"
              f"Generate the trivia question.")
    try:
        out = tg.ask_llm(prompt, system=tg.TRIVIA_SYSTEM)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 502

    question, options, answer, fun_fact = parse_trivia(out)
    if not question or len(options) < 2 or not answer:
        # malformed/truncated output: the frontend can just ask again
        return jsonify({"error": "malformed LLM output, try again"}), 502

    qid = uuid.uuid4().hex
    QUESTIONS[qid] = {"answer": answer, "fun_fact": fun_fact, "source": source}
    return jsonify({"qid": qid, "question": question, "options": options,
                    "source": source})


@app.post("/api/trivia/reveal")
def api_trivia_reveal():
    """Reveals the answer of a question and checks the player's guess."""
    body = request.json or {}
    qid, guess = body.get("qid"), (body.get("guess") or "").strip().upper()
    q = QUESTIONS.pop(qid, None)
    if q is None:
        return jsonify({"error": "unknown or already answered question"}), 404
    correct = bool(guess) and guess == q["answer"].strip()[:1].upper()
    return jsonify({"correct": correct, "answer": q["answer"],
                    "fun_fact": q["fun_fact"], "source": q["source"]})


@app.post("/api/ask")
def api_ask():
    """Free-form Q&A: retrieves the top chunks and answers with the LLM."""
    body = request.json or {}
    question = (body.get("question") or "").strip()
    season = body.get("season")
    top_n = int(body.get("top_n") or tg.TOP_K)
    if not question:
        return jsonify({"error": "empty question"}), 400

    results = tg.retrieve(VECTOR_STORE, question, season,
                          k=max(top_n, tg.TOP_K))
    if not results:
        return jsonify({"answer": "Nothing found for that question.",
                        "sources": []})

    sent = results[:tg.TOP_K]
    context = "\n\n---\n\n".join(
        f"[Season {doc.metadata['season']}, \"{doc.metadata['episode_title']}\"]\n"
        f"{doc.page_content}"
        for doc, score in sent
    )
    prompt = f"Context:\n\n{context}\n\nQuestion: {question}"
    try:
        answer = tg.ask_llm(prompt, system=tg.ANSWER_SYSTEM)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 502

    sources = []
    for i, (doc, score) in enumerate(results, 1):
        info = chunk_info(doc, score)
        info["rank"] = i
        info["sent_to_llm"] = i <= tg.TOP_K
        sources.append(info)
    return jsonify({"answer": answer, "sources": sources})


if __name__ == "__main__":
    # LOCAL development server (python src/app.py).
    # PRODUCTION (Render): gunicorn src.app:app (see Render start command);
    # PORT is injected by Render, so the local default stays at 5000.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1" if not os.environ.get("PORT") else "0.0.0.0",
            port=port, threaded=True)