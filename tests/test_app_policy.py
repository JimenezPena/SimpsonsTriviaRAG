"""End-to-end /api/ask with the distance policy: Flask test client, trivia_gen
stubbed (no Chroma, no API, no cost)."""
import importlib
import json
import sys
import types

import pytest


class Doc:
    def __init__(self, text="chunk"):
        self.page_content = text
        self.metadata = {"season": 3, "episode_title": "Ep", "line_start": 1, "line_end": 5}


def make_app(monkeypatch, distance, llm_replies, router="on", extra_env=None):
    """Imports a fresh src/app.py. Returns (app, calls) where calls lists the
    (system-tag, prompt) of every LLM call; llm_replies pops one per call."""
    calls = []
    replies = list(llm_replies)

    def ask_llm(prompt, system=None, temperature=None):
        calls.append((system, prompt))
        if not replies:
            raise RuntimeError("no more replies")
        r = replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    tg = types.ModuleType("trivia_gen")
    tg.db_count = lambda: 1
    tg.DB_PATH = "/nonexistent"
    tg.TOP_K = 3
    tg.MAX_SEASON = 11
    tg.ANSWER_SYSTEM = "ANSWER_SYSTEM"
    tg.load_vector_store = lambda: object()
    tg.retrieve = lambda store, q, season, k=3: [(Doc(), distance), (Doc(), distance + 0.05)]
    tg.plot_for_question = lambda *a, **k: None
    tg.ask_llm = ask_llm
    monkeypatch.setitem(sys.modules, "trivia_gen", tg)
    monkeypatch.setenv("ROUTER", router)
    for k in ("ROUTER_DIST_ES", "ROUTER_DIST_EN", "ROUTER_REFUSE_DIST"):
        monkeypatch.delenv(k, raising=False)
    for k, v in (extra_env or {}).items():
        monkeypatch.setenv(k, v)
    sys.modules.pop("app", None)
    app = importlib.import_module("app")
    app.app.config["TESTING"] = True
    return app, calls


def ask(app, question, lang="es", season=None):
    body = {"question": question, "lang": lang}
    if season is not None:
        body["season"] = season
    r = app.app.test_client().post("/api/ask", json=body)
    return r.status_code, r.get_json()


def route_json(route, **kw):
    return json.dumps({"route": route, **kw})


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    sys.modules.pop("app", None)


def test_close_question_never_calls_router(monkeypatch):
    app, calls = make_app(monkeypatch, 0.45, ["RAG answer"])
    code, out = ask(app, "¿Qué dijo Homer en la taberna?")
    assert code == 200 and out["answer"] == "RAG answer"
    assert len(calls) == 1 and calls[0][0] == "ANSWER_SYSTEM"


def test_far_question_offtopic_refused(monkeypatch):
    app, calls = make_app(monkeypatch, 0.85, [route_json("offtopic")])
    code, out = ask(app, "¿Capital de Francia?")
    assert out["route"]["route"] == "offtopic" and out["route"]["trigger"] == "distance"
    assert out["sources"] == [] and len(calls) == 1


def test_zone_offtopic_below_refuse_guard_goes_to_rag(monkeypatch):
    app, calls = make_app(monkeypatch, 0.62, [route_json("offtopic"), "RAG answer"])
    code, out = ask(app, "pregunta ambigua")
    assert out["answer"] == "RAG answer" and len(calls) == 2


def test_guard_can_be_disabled(monkeypatch):
    app, calls = make_app(monkeypatch, 0.62, [route_json("offtopic")],
                          extra_env={"ROUTER_REFUSE_DIST": "0"})
    code, out = ask(app, "pregunta ambigua")
    assert out["route"]["route"] == "offtopic"


def test_general_answer_is_labelled(monkeypatch):
    app, calls = make_app(monkeypatch, 0.66, [route_json("general"), "Son amarillos."])
    code, out = ask(app, "¿De qué color son los Simpson?")
    assert out["route"]["route"] == "general" and "Son amarillos." in out["answer"]
    assert out["answer"] != "Son amarillos."          # the label is added by code
    assert len(calls) == 2


def test_corpus_verdict_goes_to_rag(monkeypatch):
    app, calls = make_app(monkeypatch, 0.62, [route_json("corpus"), "RAG answer"])
    code, out = ask(app, "algo del guion")
    assert out["answer"] == "RAG answer"


def test_season_filter_skips_policy(monkeypatch):
    app, calls = make_app(monkeypatch, 0.9, ["RAG answer"])
    code, out = ask(app, "¿Capital de Francia?", season=3)
    assert out["answer"] == "RAG answer" and len(calls) == 1


def test_router_off_skips_policy(monkeypatch):
    app, calls = make_app(monkeypatch, 0.9, ["RAG answer"], router="off")
    code, out = ask(app, "¿Capital de Francia?")
    assert out["answer"] == "RAG answer" and len(calls) == 1


def test_router_error_falls_back_to_rag(monkeypatch):
    app, calls = make_app(monkeypatch, 0.9, [RuntimeError("503"), "RAG answer"])
    code, out = ask(app, "¿Capital de Francia?")
    assert out["answer"] == "RAG answer"


def test_language_thresholds(monkeypatch):
    # 0.585: above the English cut-off (0.58), below the Spanish one (0.59)
    app, calls = make_app(monkeypatch, 0.585, ["RAG answer"])
    assert ask(app, "q", lang="es")[1]["answer"] == "RAG answer" and len(calls) == 1
    app, calls = make_app(monkeypatch, 0.585, [route_json("general"), "General."])
    assert ask(app, "q", lang="en")[1]["route"]["route"] == "general"


def test_gated_metadata_question_reuses_decision(monkeypatch):
    # keyword-gated metadata question: one router call even if distance is high
    app, calls = make_app(monkeypatch, 0.9,
                          [route_json("metadata", tool="count_episodes", args={"season": 10})])
    code, out = ask(app, "¿cuántos episodios tiene la temporada 10?")
    assert out["route"]["route"] == "metadata" and len(calls) == 1
