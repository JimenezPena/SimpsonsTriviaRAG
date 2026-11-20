"""Offline tests for the distillation data pipeline (no API, no vector DB).

    python -m pytest tests -q
"""
import json
import os
import random
import types

import pytest

from distill import build_dataset as bd
from distill import gen_teacher as gt
from distill import prompts
from distill import retry as rt

NO_SLEEP = lambda s: None          # noqa: E731  (retries run instantly in tests)

APP_PY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "src", "app.py")


# ----------------------------------------------------------------------------
# A stand-in for trivia_gen: 4 episodes x 3 overlapping chunks, no network
# ----------------------------------------------------------------------------
class Doc:
    def __init__(self, text, meta):
        self.page_content, self.metadata = text, meta


class FakeTG:
    LLM_MODEL = "fake-teacher"
    TOP_K = 3
    ANSWER_SYSTEM = "Answer questions about The Simpsons using ONLY the context."

    def __init__(self, quota_after=None, errors=None):
        self.rng = random.Random(0)
        self.quota_after, self.calls = quota_after, 0
        self.errors = list(errors or [])      # raised, in order, by the first calls
        self.temps = []                       # temperature received by each call
        self.docs = []
        for ep in range(1, 5):
            for i, start in enumerate((1, 16, 31)):
                meta = {"season": ep, "episode_id": ep, "episode_title": f"Episode {ep}",
                        "line_start": start, "line_end": start + 19}
                self.docs.append(Doc(f"[ep {ep}] Homer: line block {i}", meta))

    def load_vector_store(self):
        return object()

    def random_chunk(self, store, season=None):
        d = self.rng.choice(self.docs)
        return d.page_content, d.metadata

    retrieve_calls = 0

    def retrieve(self, store, question, season=None, k=3):
        self.retrieve_calls += 1
        hit = [d for d in self.docs if d.metadata["episode_title"] in question]
        rest = [d for d in self.docs if d not in hit]
        return [(d, 0.1 * i) for i, d in enumerate((hit + rest)[:k])]

    def plot_for_question(self, *a, **k):
        return None

    def ask_llm(self, prompt, system=None, temperature=None):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        self.temps.append(temperature)
        if self.quota_after is not None and self.calls > self.quota_after:
            raise RuntimeError("LLM HTTP 429: quota")
        if system == prompts.QUESTION_SYSTEM:
            title = prompt.split('episode "')[1].split('"')[0]
            return f"\"¿Qué hace Homer en {title}?\""
        title = prompt.split('"')[1]
        if "Write the answer in English" in prompt:
            return f'Homer does something in the scene of the show. Source: "{title}".'
        return f'Homer hace algo en la escena que es de el programa. Fuente: "{title}".'


def args_for(tmp_path, **over):
    base = dict(n=6, langs="es,en", unanswerable=0.0, offtopic=0.0, season=None,
                out=str(tmp_path / "teacher.jsonl"), sleep=0, max_errors=5, seed=0,
                question_temp=0.9, answer_temp=0.2,
                max_retries=2, retry_base=1.0, retry_cap=10.0)
    base.update(over)
    return types.SimpleNamespace(**base)


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


# ----------------------------------------------------------------------------
# gen_teacher
# ----------------------------------------------------------------------------
def test_generates_resumes_and_never_repeats_gold(tmp_path):
    a = args_for(tmp_path)
    stats = gt.run(a, FakeTG(), retry_sleep=NO_SLEEP)
    recs = read(a.out)
    assert stats["written"] == 6 and len(recs) == 6
    assert [r["lang"] for r in recs] == ["es", "en"] * 3            # cycled
    r = recs[0]
    assert {"id", "task", "lang", "system", "prompt", "response", "meta"} <= set(r)
    assert r["prompt"].startswith("Context:\n\n[Season ")
    assert r["prompt"].endswith("Write the answer in Spanish (castellano).")
    assert len(r["meta"]["retrieved"]) == 3 and r["meta"]["teacher"] == "fake-teacher"

    a2 = args_for(tmp_path, n=3)                                    # resume
    gt.run(a2, FakeTG(), retry_sleep=NO_SLEEP)
    recs = read(a.out)
    keys = [gt.chunk_key(r["meta"]["gold"]) for r in recs]
    assert len(recs) == 9 and len(set(keys)) == 9                    # no gold twice
    assert len({r["id"] for r in recs}) == 9


def test_quota_stop_keeps_what_was_written(tmp_path):
    a = args_for(tmp_path, n=10)
    slept = []
    stats = gt.run(a, FakeTG(quota_after=5), retry_sleep=slept.append)  # 2 calls/example
    assert stats["quota_stop"] and stats["written"] == 2
    assert len(read(a.out)) == 2
    assert len(slept) == a.max_retries            # it did wait before giving up


def test_temperatures_are_passed_and_recorded(tmp_path):
    a = args_for(tmp_path, n=2, question_temp=0.8, answer_temp=0.1)
    tg = FakeTG()
    gt.run(a, tg, retry_sleep=NO_SLEEP)
    assert tg.temps == [0.8, 0.1, 0.8, 0.1]       # question, answer, question, answer
    assert read(a.out)[0]["meta"]["temperature"] == {"question": 0.8, "answer": 0.1}


def test_transient_errors_are_retried_with_growing_waits(tmp_path):
    a = args_for(tmp_path, n=1)
    tg = FakeTG(errors=[RuntimeError("LLM HTTP 503: unavailable"),
                        RuntimeError("LLM network error (URLError): timed out")])
    slept = []
    stats = gt.run(a, tg, retry_sleep=slept.append)
    assert stats["written"] == 1 and stats["errors"] == 0
    assert len(slept) == 2 and slept[1] > slept[0]


def test_fatal_error_is_not_retried(tmp_path):
    a = args_for(tmp_path, n=1, max_errors=1)
    slept = []
    stats = gt.run(a, FakeTG(errors=[RuntimeError("LLM HTTP 400: bad request")]),
                   retry_sleep=slept.append)
    assert stats["errors"] == 1 and stats["written"] == 0 and slept == []


# ----------------------------------------------------------------------------
# retry
# ----------------------------------------------------------------------------
def seq(*items):
    """A function that raises/returns the given items, one per call."""
    items = list(items)

    def fn():
        x = items.pop(0)
        if isinstance(x, BaseException):
            raise x
        return x
    return fn


QUIET = dict(log=lambda *_: None, jitter=0)


def test_backoff_is_incremental_and_capped():
    slept = []
    fn = seq(*[RuntimeError("LLM HTTP 429: rate")] * 4, "ok")
    out = rt.call_with_retry(fn, max_retries=5, base=5, factor=2, cap=15,
                             sleep=slept.append, **QUIET)
    assert out == "ok" and slept == [5, 10, 15, 15]


def test_server_retry_delay_hint_is_a_floor():
    slept = []
    body = 'LLM HTTP 429: {"details": [{"retryDelay": "34s"}]}'
    rt.call_with_retry(seq(RuntimeError(body), "ok"), base=1, sleep=slept.append, **QUIET)
    assert slept == [34.0]


def test_daily_quota_is_not_retried():
    msg = 'LLM HTTP 429: {"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}'
    with pytest.raises(rt.QuotaExhausted):
        rt.call_with_retry(seq(RuntimeError(msg), "ok"),
                           sleep=lambda s: pytest.fail("must not wait"), **QUIET)


def test_persistent_429_ends_as_quota_exhausted():
    with pytest.raises(rt.QuotaExhausted):
        rt.call_with_retry(seq(*[RuntimeError("LLM HTTP 429: x")] * 3),
                           max_retries=2, sleep=NO_SLEEP, **QUIET)


def test_persistent_5xx_is_reraised_not_quota():
    with pytest.raises(RuntimeError) as ei:
        rt.call_with_retry(seq(*[RuntimeError("LLM HTTP 503: x")] * 3),
                           max_retries=2, sleep=NO_SLEEP, **QUIET)
    assert not isinstance(ei.value, rt.QuotaExhausted)


def test_raw_os_errors_are_retried_and_4xx_are_fatal():
    assert rt.call_with_retry(seq(TimeoutError("timed out"), "ok"),
                              sleep=NO_SLEEP, **QUIET) == "ok"
    with pytest.raises(RuntimeError):
        rt.call_with_retry(seq(RuntimeError("LLM HTTP 400: bad"), "ok"),
                           sleep=lambda s: pytest.fail("must not wait"), **QUIET)


def test_offtopic_context_comes_from_other_episodes(tmp_path):
    a = args_for(tmp_path, n=8, offtopic=1.0)
    tg = FakeTG()
    stats = gt.run(a, tg, retry_sleep=NO_SLEEP)
    recs = read(a.out)
    assert stats["written"] == 8 and tg.retrieve_calls == 0      # no retrieval at all
    for r in recs:
        m = r["meta"]
        assert m["kind"] == "offtopic" and m["gold_in_context"] is False
        assert m["plot"] is None and "Plot summary" not in r["prompt"]
        assert len(m["retrieved"]) == tg.TOP_K
        assert all(c["episode_id"] != m["gold"]["episode_id"] for c in m["retrieved"])
        assert m["question_prompt"] == prompts.QUESTION_PROMPT_VERSION
        # the context really is what the student sees
        for c in m["retrieved"]:
            assert f'[Season {c["season"]}, "{c["episode_title"]}"]' in r["prompt"]


def test_offtopic_gives_up_when_no_other_episode_exists():
    class OneEpisode(FakeTG):
        def __init__(self):
            super().__init__()
            self.docs = [d for d in self.docs if d.metadata["episode_id"] == 1]

    tg = OneEpisode()
    text, meta = tg.random_chunk(None, None)
    assert gt.random_context(tg, meta, 3) is None
    assert gt.make_record(tg, None, text, meta, "es", "offtopic", call=gt._direct) is None


def test_kind_mix_follows_the_fractions(tmp_path):
    a = args_for(tmp_path, n=4, offtopic=0.5, unanswerable=0.5)   # never "answerable"
    gt.run(a, FakeTG(), retry_sleep=NO_SLEEP)
    assert {r["meta"]["kind"] for r in read(a.out)} <= {"offtopic", "unanswerable"}


def test_question_prompt_asks_for_self_contained_questions():
    s = prompts.QUESTION_SYSTEM
    assert "SELF-CONTAINED" in s and "distinctive" in s and "the hotel" in s


def test_gold_in_context_uses_line_overlap():
    a = {"episode_id": 1, "line_start": 16, "line_end": 35}
    assert gt.overlaps(a, {"episode_id": 1, "line_start": 31, "line_end": 50})
    assert not gt.overlaps(a, {"episode_id": 1, "line_start": 36, "line_end": 55})
    assert not gt.overlaps(a, {"episode_id": 2, "line_start": 16, "line_end": 35})


def test_clean_question():
    assert prompts.clean_question('"¿Quién es Moe?"\nextra') == "¿Quién es Moe?"
    assert prompts.clean_question("Homer eats a donut") is None
    assert prompts.clean_question("") is None


# ----------------------------------------------------------------------------
# build_dataset
# ----------------------------------------------------------------------------
def make_rec(i, lang="es", kind="answerable", gold_in=True, resp=None, ep=1):
    title = f"Episode {ep}"
    if resp is None:
        resp = (f'Homer hace algo en la escena que es de el programa. Fuente: "{title}".'
                if lang == "es" else
                f'Homer does something in the scene of the show. Source: "{title}".')
    return {"id": f"id{i}", "task": "qa", "lang": lang, "system": "S", "prompt": "P",
            "response": resp,
            "meta": {"question": "q?", "kind": kind, "gold_in_context": gold_in,
                     "gold": {"episode_id": ep, "line_start": 1, "line_end": 20},
                     "retrieved": [{"episode_title": title}]}}


def test_hard_flags_are_dropped():
    recs = [make_rec(1), make_rec(2, resp=""),
            make_rec(3, resp="The quick brown fox is in the house and the dog is on the mat."),
            make_rec(4, resp="Uno. Dos. Tres. Cuatro. Cinco. Seis. Siete en el que es."),
            make_rec(5, lang="en")]
    splits, dropped = bd.build(recs)
    assert sum(map(len, splits.values())) == 2
    assert dropped["empty"] == 1 and dropped["wrong_lang"] == 1 and dropped["too_long"] == 1


def test_soft_flags_and_strict():
    es_abst = "El contexto no contiene la respuesta a eso en el episodio."
    recs = [
        make_rec(1, gold_in=False),                                  # answered_without_gold
        make_rec(2, gold_in=False, resp=es_abst),                    # good abstention
        make_rec(3, gold_in=True, resp=es_abst),                     # abstained_with_gold
        make_rec(4, kind="unanswerable"),                            # unanswerable_but_answered
        make_rec(5, resp="Homer hace algo en la escena que es de el programa."),  # no_citation
        make_rec(6),                                                 # clean
    ]
    recs.append(make_rec(7, kind="offtopic", gold_in=False))                 # answered anyway
    recs.append(make_rec(8, kind="offtopic", gold_in=False, resp=es_abst))   # correct abstention
    flags = {r["id"]: bd.flags_for(r)[1] for r in recs}
    assert flags["id7"] == ["offtopic_but_answered"] and flags["id8"] == []
    assert flags["id1"] == ["answered_without_gold"]
    assert flags["id2"] == []
    assert flags["id3"] == ["abstained_with_gold"]
    assert flags["id4"] == ["unanswerable_but_answered"]
    assert flags["id5"] == ["no_citation"]
    assert flags["id6"] == []
    loose, _ = bd.build(recs)
    strict, dropped = bd.build(recs, strict=True)
    assert sum(map(len, loose.values())) == 8
    assert sum(map(len, strict.values())) == 3 and sum(dropped.values()) == 5  # kept: 2, 6, 8


def test_split_is_by_episode_and_stable():
    recs = [make_rec(i, ep=ep) for i, ep in enumerate(list(range(1, 201)) * 2)]
    splits, _ = bd.build(recs)
    eps = {k: {r["meta"]["gold"]["episode_id"] for r in v} for k, v in splits.items()}
    assert not (eps["train"] & eps["val"]) and not (eps["train"] & eps["test"]) \
        and not (eps["val"] & eps["test"])                           # no leakage
    n = sum(map(len, splits.values()))
    assert 0.7 < len(splits["train"]) / n < 0.9                      # ~80/10/10
    assert len(splits["val"]) > 0 and len(splits["test"]) > 0
    # stable: the same episode lands in the same split whatever else is present
    again, _ = bd.build(recs[:50])
    for k, v in again.items():
        assert {r["meta"]["gold"]["episode_id"] for r in v} <= eps[k]


def test_sft_format():
    splits, _ = bd.build([make_rec(1)])
    row = next(r for v in splits.values() for r in v)
    assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant"]
    assert row["meta"]["lang"] == "es" and row["meta"]["flags"] == []


# ----------------------------------------------------------------------------
# train/serve drift guard
# ----------------------------------------------------------------------------
def test_prompt_layout_matches_app_py():
    """The student must be trained on what production sends. If app.py's
    prompt strings change, this fails: update distill/prompts.py and regenerate."""
    src = open(APP_PY, encoding="utf-8").read()
    for fragment in (
        r'''f"[Season {doc.metadata['season']}, \"{doc.metadata['episode_title']}\"]\n"''',
        r'''"\n\n---\n\n".join(''',
        r'''f"[Plot summary (Wikipedia), season {plot['season']}, "''',
        r'''f"Context:\n\n{context}\n\nQuestion: {question}\n\n"''',
        r'''f"Write the answer in {reply_in}."''',
        '"Spanish (castellano)" if lang == "es" else "English"',
    ):
        assert fragment in src, f"app.py no longer contains: {fragment}"


def test_prompt_layout_literal():
    ctx = prompts.format_context(
        [(3, "A", "texto A"), (4, "B", "texto B")],
        {"season": 3, "title": "A", "paragraphs": ["p1", "p2"]})
    assert ctx == ('[Plot summary (Wikipedia), season 3, "A"]\np1\n\np2\n\n---\n\n'
                   '[Season 3, "A"]\ntexto A\n\n---\n\n[Season 4, "B"]\ntexto B')
    assert prompts.build_qa_prompt("C", "Q?", "en") == \
        "Context:\n\nC\n\nQuestion: Q?\n\nWrite the answer in English."
