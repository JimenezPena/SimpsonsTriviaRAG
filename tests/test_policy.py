"""Offline tests: retrieval-distance policy, general / off-topic routes, zone evaluation."""
import json
import os

import pytest

from agentic import answers, eval_router as ev, policy, router
from agentic.catalog import Catalog

CAT = Catalog.load()
TH = policy.thresholds({})


def fake_ask(*replies):
    """Returns the replies in order (an Exception instance is raised)."""
    calls, queue = [], list(replies)

    def ask(prompt, system=None, temperature=None):
        calls.append({"prompt": prompt, "system": system, "temperature": temperature})
        r = queue.pop(0)
        if isinstance(r, Exception):
            raise r
        return r
    ask.calls = calls
    return ask


def verdict(route, tool=None, args=None):
    return json.dumps({"route": route, "tool": tool, "args": args or {}})


# --- thresholds ------------------------------------------------------------------
def test_default_thresholds_and_env_overrides():
    assert policy.thresholds({}) == {"es": 0.59, "en": 0.58}
    assert policy.thresholds({"ROUTER_DIST_ES": "0.61", "ROUTER_DIST_EN": "0.55"}) == {"es": 0.61, "en": 0.55}
    assert policy.thresholds({"ROUTER_DIST_ES": "abc", "ROUTER_DIST_EN": ""}) == {"es": 0.59, "en": 0.58}
    assert policy.refuse_distance({}) == 0.70
    assert policy.refuse_distance({"ROUTER_REFUSE_DIST": "0.8"}) == 0.8
    assert policy.refuse_distance({"ROUTER_REFUSE_DIST": "-1"}) == 0.0     # 0 disables the guard


def test_best_distance_and_who_reaches_the_router():
    assert policy.best_distance([("d1", 0.7), ("d2", 0.5), ("d3", 0.6)]) == 0.5
    assert policy.best_distance([]) is None and policy.best_distance(None) is None
    assert policy.reaches_router(0.60, "es", TH) and not policy.reaches_router(0.59, "es", TH)   # strictly above
    assert policy.reaches_router(0.59, "en", TH) and not policy.reaches_router(0.58, "en", TH)
    assert not policy.reaches_router(None, "es", TH)
    assert policy.threshold_for("ca", TH) == 0.59                                                # unknown: the larger one


# --- answers ---------------------------------------------------------------------
def test_label_is_added_by_code_and_prompt_asks_for_the_language():
    assert answers.label_general(" Matt Groening. ", "es") == (
        "Respuesta de conocimiento general (no procede de los guiones):\nMatt Groening.")
    assert answers.label_general("x", "en").startswith("General-knowledge answer")
    assert "Spanish" in answers.build_general_prompt("¿Quién creó la serie?", "es")
    assert "English" in answers.build_general_prompt("Who created it?", "en")
    assert "Never invent" in answers.GENERAL_SYSTEM and "not sure" in answers.GENERAL_SYSTEM
    assert answers.offtopic_reply("es") != answers.offtopic_reply("en")


# --- answer_in_zone ----------------------------------------------------------------
def zone(ask, best=0.80, catalog=CAT, decision=None, lang="es", refuse=0.70):
    return router.answer_in_zone("pregunta", lang, ask, catalog, best, refuse, decision=decision)


def test_offtopic_is_refused_without_a_second_call_only_far_enough():
    ask = fake_ask(verdict("offtopic"))
    out = zone(ask, best=0.85)
    assert out["answer"] == answers.offtopic_reply("es") and out["sources"] == []
    assert out["route"] == {"route": "offtopic", "trigger": "distance", "distance": 0.85}
    assert len(ask.calls) == 1                                          # the router call is the only one
    # the guard: an offtopic verdict at a distance where answerable questions live -> the RAG
    assert zone(fake_ask(verdict("offtopic")), best=0.65) is None
    assert zone(fake_ask(verdict("offtopic")), best=0.70) is None       # <= guard
    assert zone(fake_ask(verdict("offtopic")), best=0.65, refuse=0.0) is not None   # guard disabled


def test_general_answers_from_the_model_with_the_label():
    ask = fake_ask(verdict("general"), "Matt Groening la creó.")
    out = zone(ask, best=0.66)
    assert out["answer"].startswith("Respuesta de conocimiento general") and out["answer"].endswith("Matt Groening la creó.")
    assert out["route"]["route"] == "general" and out["route"]["distance"] == 0.66
    router_call, answer_call = ask.calls
    assert router_call["system"] == router.ROUTER_SYSTEM and router_call["temperature"] == 0.0
    assert answer_call["system"] == answers.GENERAL_SYSTEM and answer_call["temperature"] == 0.2
    assert "Question: pregunta" in answer_call["prompt"]


@pytest.mark.parametrize("replies", [
    (verdict("corpus"),),                                   # a retrieval miss: the RAG answers
    ("not json",), ("",), (RuntimeError("HTTP 503"),),      # router failures
    (verdict("general"), RuntimeError("HTTP 503")),         # the general answer fails
    (verdict("general"), "   "),                            # empty general answer
    (verdict("metadata", "drop_tables", {}),),              # invented tool
])
def test_every_failure_falls_back_to_the_rag(replies):
    assert zone(fake_ask(*replies)) is None


def test_metadata_in_the_zone_uses_the_catalog_and_needs_one():
    ask = fake_ask(verdict("metadata", "count_episodes", {"season": 10}))
    out = zone(ask, best=0.62)
    assert "23 episodios" in out["answer"]
    assert out["route"]["trigger"] == "distance" and out["route"]["distance"] == 0.62
    assert zone(fake_ask(verdict("metadata", "count_episodes", {"season": 10})), catalog=None) is None


def test_a_decision_already_taken_is_reused_not_asked_again():
    ask = fake_ask("must not be used")
    d = {"route": "offtopic", "tool": None, "args": {}, "raw": verdict("offtopic"), "gated_out": False}
    assert zone(ask, best=0.9, decision=d)["route"]["route"] == "offtopic" and not ask.calls
    failed = {"route": None, "tool": None, "args": {}, "raw": None, "gated_out": False}   # the API failed before
    assert zone(ask, decision=failed) is None and not ask.calls
    gated = {"route": "corpus", "tool": None, "args": {}, "raw": None, "gated_out": True}  # never consulted
    ask2 = fake_ask(verdict("offtopic"))
    assert zone(ask2, best=0.9, decision=gated) is not None and len(ask2.calls) == 1


# --- zone evaluation set ----------------------------------------------------------
def row(kind, lang, q, dists, key="question"):
    meta = {"kind": kind, "retrieved": [{"distance": d} for d in dists]}
    meta[key] = q
    return {"id": q, "lang": lang, "prompt": q, "meta": meta}


def test_zone_cases_pick_what_the_router_sees(tmp_path):
    a, b = tmp_path / "teacher.jsonl", tmp_path / "router.jsonl"
    a.write_text("\n".join(json.dumps(r) for r in [
        row("answerable", "es", "ans-near", [0.5, 0.6]),          # best 0.5 <= 0.59: not seen
        row("answerable", "es", "ans-far", [0.62, 0.7]),           # best 0.62 > 0.59: seen, labelled corpus
        row("unanswerable", "en", "near-miss", [0.60, 0.7]),       # 0.60 > 0.58
        row("offtopic", "es", "legacy", []),                       # no distances: skipped
    ]) + "\n{broken", encoding="utf-8")
    b.write_text("\n".join(json.dumps(r) for r in [
        row("general", "es", "gen-in", [0.58]),                    # <= 0.59: not seen
        row("general", "en", "gen-out", [0.59]),                   # > 0.58
        row("nonrelated", "es", "far", [0.9, 0.95]),
        row("nonrelated", "es", "far", [0.9, 0.95]),               # duplicate question
        row("general", "ca", "other-language", [0.9]),             # language without threshold
    ]), encoding="utf-8")
    cases, stats = policy.zone_cases([str(a), str(b)], TH)
    got = {c["q"]: (c["route"], c["lang"], c["distance"]) for c in cases}
    assert got == {"ans-far": ("corpus", "es", 0.62), "near-miss": ("corpus", "en", 0.6),
                   "gen-out": ("general", "en", 0.59), "far": ("offtopic", "es", 0.9)}
    assert stats[("es", "answerable")] == [1, 2] and stats[("es", "general")] == [0, 1]
    assert "es answerable" in policy.format_stats(stats, TH) and "50%" in policy.format_stats(stats, TH)


def test_outcome_judges_every_route_and_counts_diverted_corpus_questions():
    cases = [{"q": f"q{i}", "lang": "es", "route": r, "tool": None, "args": {}}
             for i, r in enumerate(["corpus", "corpus", "general", "offtopic"])]
    answers_ = {"q0": verdict("corpus"), "q1": verdict("offtopic"),      # the harmful one
                "q2": verdict("general"), "q3": verdict("general")}      # a refusal case answered, not refused

    def ask(prompt, system=None, temperature=None):
        return answers_[prompt[3:]]
    rows = ev.evaluate(cases, ask, ungated=True)
    text = ev.format_cases(rows)
    assert text.count("OK ") == 2 and text.count("XX ") == 2
    s = ev.summarize(rows)
    assert s["diverted"] == (1, 2) and s["route_accuracy"] == (2, 4)
    assert "diverted" in ev.format_report(s)


def test_gated_out_questions_without_distance_are_not_judged_offline():
    rows = ev.evaluate([{"q": "¿Qué dice Homer?", "lang": "es", "route": "general", "tool": None, "args": {}},
                        {"q": "¿Qué dice Homer?", "lang": "es", "route": "corpus", "tool": None, "args": {}}])
    text = ev.format_cases(rows)
    assert "use --zone" in text and "-> RAG (gate: no LLM call)" in text


def test_cli_zone_mode_needs_no_api_offline(tmp_path, capsys):
    f = tmp_path / "r.jsonl"
    f.write_text("\n".join(json.dumps(row(k, "en", f"{k}-{i}", [0.9])) for k in ("general", "nonrelated")
                           for i in range(3)), encoding="utf-8")
    ev.main(["--zone", str(f), "--limit", "4"])
    out = capsys.readouterr().out
    assert "reach the router" in out and "cases: 4" in out
