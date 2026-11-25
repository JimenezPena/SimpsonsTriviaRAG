"""Offline tests for the metadata router (no API, no vector DB)."""
import collections
import json
import os
import sqlite3

import pytest

from agentic import eval_router as ev
from agentic import router, tools
from agentic.catalog import DEFAULT_PATH, Catalog, load_indexed_from_chroma, norm

CAT = Catalog.load()
RAW = json.load(open(DEFAULT_PATH, encoding="utf-8"))


def fake_ask(reply):
    calls = []

    def ask(prompt, system=None, temperature=None):
        calls.append((prompt, system, temperature))
        if isinstance(reply, Exception):
            raise reply
        return reply
    ask.calls = calls
    return ask


def route_json(route="metadata", tool=None, args=None):
    return json.dumps({"route": route, "tool": tool, "args": args or {}})


# --- catalog and tools ---------------------------------------------------------
def test_counts_match_the_raw_file_independently():
    per_season = collections.Counter(v["season"] for v in RAW.values())
    for season, n in per_season.items():
        assert tools.run("count_episodes", {"season": season}, CAT)["data"]["episodes"] == n
    assert tools.run("count_episodes", {}, CAT)["data"]["episodes"] == len(RAW)
    seasons = tools.run("count_seasons", {}, CAT)["data"]
    assert seasons["seasons"] == len(per_season) and seasons["last"] == max(per_season)


def test_episode_lookups_round_trip():
    for v in list(RAW.values())[::17]:
        t = tools.run("episode_title", {"season": v["season"], "episode": v["episode"]}, CAT)
        assert t["data"]["title"] == v["title"]
        f = tools.run("find_episode", {"title": v["title"]}, CAT)
        assert f["ok"] and (f["data"]["season"], f["data"]["episode"]) == (v["season"], v["episode"])


def test_find_title_is_exact_first_and_never_fuzzy():
    assert norm("Homer's Odyssey") == norm("homers odyssey") == "homers odyssey"
    assert tools.run("find_episode", {"title": "homers odyssey"}, CAT)["data"]["episode"] == 3
    # an exact title wins even though other titles contain it
    r = tools.run("find_episode", {"title": "Who Shot Mr. Burns?"}, CAT)
    assert r["ok"] and r["data"]["title"] == "Who Shot Mr. Burns?"
    amb = tools.run("find_episode", {"title": "burns"}, CAT)
    assert not amb["ok"] and amb["error"] == "ambiguous_title" and amb["n_candidates"] > 1
    assert tools.run("find_episode", {"title": "Hmmm Odysey"}, CAT)["error"] == "title_not_found"
    assert tools.run("find_episode", {"title": "of"}, CAT)["error"] == "title_not_found"


def test_missing_season_or_episode_is_reported_not_invented():
    assert tools.run("count_episodes", {"season": 30}, CAT)["error"] == "season_not_in_catalog"
    assert tools.run("list_episodes", {"season": 0}, CAT)["error"] == "season_not_in_catalog"
    r = tools.run("episode_title", {"season": 1, "episode": 99}, CAT)
    assert r["error"] == "episode_not_in_season" and r["episodes"] == 13


@pytest.mark.parametrize("name,args,ok", [
    ("count_episodes", {"season": "10"}, True),
    ("count_episodes", {"season": 10.0}, True),
    ("count_episodes", {"season": None}, True),
    ("count_episodes", {"season": "ten"}, False),
    ("count_episodes", {"season": True}, False),
    ("count_episodes", {"seasn": 1}, False),
    ("episode_title", {"season": 1}, False),
    ("find_episode", {"title": ""}, False),
    ("find_episode", {"title": "x" * 300}, False),
    ("drop_tables", {}, False),
    ("count_seasons", [], False),
])
def test_validate_call(name, args, ok):
    clean, err = tools.validate_call(name, args)
    assert (err is None) == ok


def test_render_es_en_and_plurals():
    one = Catalog([{"season": 1, "episode": 1, "title": "Only One"}])
    assert "1 episodio en" in tools.render(tools.run("count_episodes", {"season": 1}, one), "es")
    assert "1 episode in" in tools.render(tools.run("count_episodes", {"season": 1}, one), "en")
    es = tools.render(tools.run("count_episodes", {"season": 10}, CAT), "es")
    assert "temporada 10 tiene 23 episodios" in es and "Fuente" in es
    en = tools.render(tools.run("count_episodes", {"season": 10}, CAT), "en")
    assert "Season 10 has 23 episodes" in en and "Source" in en
    assert "no está en el catálogo" in tools.render(
        tools.run("count_episodes", {"season": 30}, CAT), "es")
    long = tools.render(tools.run("list_episodes", {"season": 3}, CAT), "en")
    assert long.count("\n") >= 24 and "(+" not in long          # 24 episodes < MAX_LIST


def test_list_is_truncated_when_long(monkeypatch):
    monkeypatch.setattr(tools, "MAX_LIST", 5)
    text = tools.render(tools.run("list_episodes", {"season": 3}, CAT), "en")
    assert "(+19 more)" in text and text.count("\n") == 7


def _fake_chroma(tmp_path):
    con = sqlite3.connect(tmp_path / "chroma.sqlite3")
    con.execute("CREATE TABLE embeddings (id INTEGER PRIMARY KEY)")
    con.execute("CREATE TABLE embedding_metadata (id INT, key TEXT, string_value TEXT, int_value INT)")
    rows = [(1, 1, "Bart the Genius"), (2, 1, "Bart the Genius"), (3, 1, "Homer's Odyssey"),
            (4, 2, "Bart Gets an \"F\"")]
    for i, season, title in rows:
        con.execute("INSERT INTO embeddings VALUES (?)", (i,))
        con.execute("INSERT INTO embedding_metadata VALUES (?, 'season', NULL, ?)", (i, season))
        con.execute("INSERT INTO embedding_metadata VALUES (?, 'episode_title', ?, NULL)", (i, title))
    con.execute("INSERT INTO embeddings VALUES (9)")                # a plot summary: not a script
    con.execute("INSERT INTO embedding_metadata VALUES (9, 'season', NULL, 1)")
    con.execute("INSERT INTO embedding_metadata VALUES (9, 'episode_title', 'Moaning Lisa', NULL)")
    con.execute("INSERT INTO embedding_metadata VALUES (9, 'kind', 'synopsis', NULL)")
    con.commit()
    con.close()
    return str(tmp_path)


def test_indexed_coverage_comes_from_script_chunks_only(tmp_path):
    indexed = load_indexed_from_chroma(_fake_chroma(tmp_path))
    assert indexed == {1: {"bart the genius", "homers odyssey"}, 2: {"bart gets an f"}}
    cat = Catalog.load(indexed=indexed)
    assert cat.indexed_count(1) == 2 and cat.indexed_count(2) == 1 and cat.indexed_count(3) == 0
    text = tools.render(tools.run("count_episodes", {"season": 1}, cat), "es")
    assert "tienen el guion indexado" in text and "De ellos, 2" in text
    assert CAT.indexed_count(1) is None                              # unknown: nothing is claimed
    assert "indexado" not in tools.render(tools.run("count_episodes", {"season": 1}, CAT), "es")


# --- router ---------------------------------------------------------------------
def test_parse_route_is_tolerant_and_strict():
    p = router.parse_route('```json\n{"route": "metadata", "tool": "count_seasons", "args": {}}\n```')
    assert p == {"route": "metadata", "tool": "count_seasons", "args": {}}
    assert router.parse_route('Sure! {"route":"corpus","tool":null,"args":null} done')["args"] == {}
    for bad in ["", None, "no json", '{"route": "metadata"', '{"route": "magic"}', "[1, 2]",
                '{"tool": "count_seasons"}']:
        assert router.parse_route(bad) is None


def test_gate_lets_catalog_questions_through_and_skips_plain_ones():
    for q in ["¿Cuántos episodios tiene la temporada 10?", "How many seasons?",
              "¿Qué número de capítulo es?", "Which season is X in?"]:
        assert router.looks_like_metadata(q), q
    for q in ["¿Qué le dice Marge a Homer?", "Why does Bart sell his soul?", ""]:
        assert not router.looks_like_metadata(q), q


def test_gated_out_questions_cost_no_llm_call():
    ask = fake_ask(route_json())
    d = router.decide("¿Qué le dice Marge a Homer?", ask)
    assert d["route"] == "corpus" and d["gated_out"] and not ask.calls
    assert router.answer_if_metadata("Why does Bart sell his soul?", "en", ask, CAT) is None
    assert not ask.calls


def test_router_prompt_lists_every_tool_and_route():
    for name in tools.SPECS:
        assert name in router.ROUTER_SYSTEM
    for route in router.ROUTES:
        assert f'"{route}"' in router.ROUTER_SYSTEM


def test_answer_if_metadata_happy_path_uses_temperature_zero():
    ask = fake_ask(route_json(tool="count_episodes", args={"season": 10}))
    out = router.answer_if_metadata("¿Cuántos episodios tiene la temporada 10?", "es", ask, CAT)
    assert "23 episodios" in out["answer"] and out["sources"] == []
    assert out["route"] == {"route": "metadata", "tool": "count_episodes",
                            "args": {"season": 10}, "ok": True}
    assert ask.calls[0][2] == 0.0 and ask.calls[0][1] == router.ROUTER_SYSTEM


def test_answer_if_metadata_falls_back_to_the_rag():
    q = "How many episodes in season 2?"
    for reply in [route_json("corpus"), route_json("general"), route_json("offtopic"),
                  route_json(tool=None),                                      # metadata, no tool
                  route_json(tool="delete_everything", args={}),              # invented tool
                  route_json(tool="count_episodes", args={"season": "two"}),  # bad argument
                  route_json(tool="episode_title", args={"season": 1}),       # missing argument
                  "I think it is about the catalog", "", RuntimeError("HTTP 429")]:
        assert router.answer_if_metadata(q, "en", fake_ask(reply), CAT) is None, reply


def test_unknown_season_is_answered_honestly_not_by_the_rag():
    ask = fake_ask(route_json(tool="count_episodes", args={"season": 30}))
    out = router.answer_if_metadata("¿Cuántos episodios tiene la temporada 30?", "es", ask, CAT)
    assert "no está en el catálogo" in out["answer"] and out["route"]["ok"] is False


# --- evaluation set and app wiring ----------------------------------------------
CASES = ev.load_cases()


def test_eval_labels_are_valid_and_balanced():
    assert len(CASES) >= 40
    assert {c["route"] for c in CASES} == set(router.ROUTES)
    assert {c["lang"] for c in CASES} == {"es", "en"}
    assert len({c["q"] for c in CASES}) == len(CASES)
    for c in CASES:
        if c["route"] == "metadata":
            clean, err = tools.validate_call(c["tool"], c["args"])
            assert err is None, c
            if c["tool"] == "find_episode":                   # the label must be findable
                assert CAT.find_title(c["args"]["title"]), c
        else:
            assert c["tool"] is None and c["args"] == {}


def test_gate_passes_every_labelled_metadata_question():
    s = ev.summarize(ev.evaluate(CASES))
    assert s["gate_recall"] == 1.0 and 0 < s["gate_pass_other"] < 1


def test_eval_with_a_perfect_and_a_bad_router():
    by_q = {c["q"]: c for c in CASES}

    def oracle(prompt, system=None, temperature=None):
        c = by_q[prompt[3:]]
        return route_json(c["route"], c["tool"], c["args"])
    s = ev.summarize(ev.evaluate(CASES, oracle, ungated=True))
    assert s["metadata_solved"][0] == s["metadata_solved"][1] and s["false_actions"][0] == 0
    assert s["route_accuracy"][0] == s["route_accuracy"][1] == len(CASES)

    def always_catalog(prompt, system=None, temperature=None):
        return route_json(tool="count_seasons")
    s = ev.summarize(ev.evaluate(CASES, always_catalog, ungated=True))
    assert s["false_actions"][0] == s["false_actions"][1] > 0       # the harmful error is counted
    assert "false actions" in ev.format_report(s)


def test_eval_stops_after_consecutive_api_errors():
    ask = fake_ask(RuntimeError("HTTP 429"))
    rows = ev.evaluate(CASES, ask, ungated=True, max_errors=3)
    assert len(ask.calls) == 3 and len(rows) == 3


def test_cached_ask_does_not_repeat_calls(tmp_path):
    ask = fake_ask(route_json("corpus"))
    cache = str(tmp_path / "c.jsonl")
    a = ev.cached_ask(ask, cache)
    assert a("Q: x", system="S") == a("Q: x", system="S")
    b = ev.cached_ask(fake_ask("other"), cache)                      # a new process: reads the file
    assert b("Q: x", system="S") == route_json("corpus") and len(ask.calls) == 1


def test_app_py_wiring_is_optional_and_falls_back():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(here, "src", "app.py"), encoding="utf-8").read()
    assert 'os.environ.get("ROUTER"' in src and "ROUTER_ON and get_catalog() is not None" in src
    gate, retrieve = src.index("agentic_router.decide("), src.index("tg.retrieve(get_store()")
    zone = src.index("agentic_router.answer_in_zone(")
    assert gate < retrieve < zone                      # keyword gate, then retrieval, then the distance policy
    assert "except Exception" in src[gate - 200:retrieve] and "except Exception" in src[zone - 200:zone + 400]
    assert "season is None and agentic_policy.reaches_router" in src   # season-filtered searches skip it
    assert "decision=decision" in src                  # one router call per question at most


def test_report_survives_a_sample_without_non_metadata_questions():
    only_meta = [c for c in CASES if c["route"] == "metadata"][:3]
    text = ev.format_report(ev.summarize(ev.evaluate(only_meta)))     # used to raise TypeError
    assert "n/a" in text and "gate recall" in text
    only_other = [c for c in CASES if c["route"] == "corpus"][:3]
    assert "n/a" in ev.format_report(ev.summarize(ev.evaluate(only_other)))


def test_limit_samples_every_route():
    s = ev.sample(CASES, 10)
    assert len(s) == 10 and {c["route"] for c in s} == set(router.ROUTES)
    assert ev.sample(CASES, 0) == CASES and ev.sample(CASES, 999) == CASES
    assert len({c["q"] for c in s}) == 10


def test_case_by_case_output_shows_tag_verdict_and_raw_garbage():
    by_q = {c["q"]: c for c in CASES}
    pick = [by_q["¿Cuántos episodios tiene la temporada 10?"],            # metadata, solved
            by_q["How many episodes does Breaking Bad season 2 have?"],   # gated, offtopic
            by_q["¿Qué le dice Marge a Homer sobre las bolas de bolos?"], # gated out
            by_q["¿Cuántas temporadas hay?"]]                             # metadata, garbage

    def ask(prompt, system=None, temperature=None):
        q = prompt[3:]
        if "Breaking" in q:
            return route_json("metadata", "count_episodes", {"season": 2})   # the harmful error
        if "temporadas" in q:
            return "Lo siento, no puedo ayudar con eso " + "x" * 200
        return route_json("metadata", "count_episodes", {"season": 10})
    rows = ev.evaluate(pick, ask)
    text = ev.format_cases(rows)
    assert "OK  metadata  metadata -> answers itself: count_episodes(season=10)" in text
    assert "XX  offtopic  metadata -> answers itself: count_episodes(season=2)" in text
    assert "OK  corpus    -> RAG (gate: no LLM call)" in text
    assert "XX  metadata  UNPARSABLE -> RAG | Lo siento" in text
    assert "¿Cuántas temporadas hay?" in text and len(text.splitlines()) == 2 + 2 * 4
    s = ev.summarize(rows)
    assert s["route_accuracy"] == (1, 3) and s["false_actions"] == (1, 2)   # garbage counts as wrong


def test_wrong_arguments_are_not_ok_and_invalid_calls_go_to_the_rag():
    c = [x for x in CASES if x["tool"] == "count_episodes" and x["args"] == {"season": 3}][0]
    bad = ev.evaluate([c], lambda p, system=None, temperature=None:
                      route_json("metadata", "count_episodes", {"season": 4}))
    assert "XX  metadata" in ev.format_cases(bad)
    inv = ev.evaluate([c], lambda p, system=None, temperature=None:
                      route_json("metadata", "count_episodes", {"season": "tres"}))
    assert "(invalid tool call)" in ev.format_cases(inv)


def test_unparsable_answers_are_never_cached(tmp_path):
    cache = str(tmp_path / "c.jsonl")
    garbage = ev.cached_ask(fake_ask('{"route": "metadata", "tool": "count_episodes", "args": {"season'), cache)
    garbage("Q: x", system="S")
    assert not os.path.exists(cache) or not open(cache).read()
    # an old garbage line already in the file is ignored too, so it is retried
    with open(cache, "w") as f:
        f.write(json.dumps({"key": "k", "raw": "garbage"}) + "\n")
    good = fake_ask(route_json("corpus"))
    assert ev.cached_ask(good, cache)("Q: x", system="S") == route_json("corpus") and good.calls


def test_unparsable_output_is_shown_in_full_up_to_200_chars():
    raw = '{"route": "metadata", "tool": "count_episodes", "args": {"season ' + "y" * 150
    rows = ev.evaluate([CASES[0]], fake_ask(raw))
    assert "y" * 130 in ev.format_cases(rows)
