"""Offline tests for the hand-review exporter."""
import json

from distill import prompts
from distill import review as rv

ABST = "El contexto no contiene la respuesta a eso en el episodio."
EN_ANSWER = "The quick brown fox is in the house and the dog is on the mat."


def rec(i, lang="es", kind="answerable", gold_in=True, resp=ABST, version=2):
    chunks = [(7, "Lisa the Vegetarian", "Lisa: I'm a vegetarian\nHomer: roast pig!"),
              (3, "Other", "Bart: Ay caramba")]
    ctx = prompts.format_context(chunks, {"season": 7, "title": "Lisa the Vegetarian",
                                          "paragraphs": ["PLOT TEXT"]})
    retrieved = [
        {"season": 7, "episode_id": 1, "episode_title": "Lisa the Vegetarian",
         "line_start": 1, "line_end": 20, "distance": 0.1},
        {"season": 3, "episode_id": 2, "episode_title": "Other",
         "line_start": 5, "line_end": 24, "distance": 0.4}]
    gold = {"season": 7, "episode_id": 1, "episode_title": "Lisa the Vegetarian",
            "line_start": 10, "line_end": 29}
    meta = {"question": f"pregunta {i}?", "kind": kind, "gold": gold,
            "gold_in_context": gold_in, "retrieved": retrieved}
    if version:
        meta["question_prompt"] = version
    return {"id": f"id{i}", "lang": lang, "system": "S",
            "prompt": prompts.build_qa_prompt(ctx, f"pregunta {i}?", lang),
            "response": resp, "meta": meta}


def test_gold_chunk_is_isolated_from_plot_and_other_chunks():
    r = rec(1)
    assert rv.gold_chunks(r) == ["[Season 7, \"Lisa the Vegetarian\"]\n"
                                 "Lisa: I'm a vegetarian\nHomer: roast pig!"]


def test_gold_chunk_empty_when_the_separator_breaks_the_split():
    r = rec(1)
    r["prompt"] = r["prompt"].replace("Homer: roast pig!", "a\n\n---\n\nb")  # 4 blocks, 2 retrieved
    assert len(rv.context_blocks(r["prompt"])) >= 2
    r["meta"]["retrieved"] = r["meta"]["retrieved"] * 3                      # more than blocks
    assert rv.gold_chunks(r) == []


def test_select_by_flag_language_and_prompt_version():
    recs = [rec(1), rec(2, gold_in=False), rec(3, lang="en", resp=ABST), rec(4, version=None),
            rec(5, resp="Homer hace algo en la escena que es de el programa. Fuente: \"Other\".")]
    flagged = rv.select(recs, "abstained_with_gold")
    assert [r["id"] for r, _, _ in flagged] == ["id1", "id3", "id4"]
    assert [r["id"] for r, _, _ in rv.select(recs, "abstained_with_gold", lang="en")] == ["id3"]
    assert [r["id"] for r, _, _ in rv.select(recs, "abstained_with_gold", prompt_version=1)] == ["id4"]
    assert [r["id"] for r, _, _ in rv.select(recs, "abstained_with_gold", prompt_version=2)] == ["id1", "id3"]


def test_wrong_lang_is_a_hard_flag_and_shows_the_counts():
    r = rec(1, resp=EN_ANSWER)
    assert [x[0]["id"] for x in rv.select([r], "wrong_lang")] == ["id1"]
    text = rv.render(r, "wrong_lang", 1)
    assert "requested: **es**" in text and "en=" in text and "es=" in text
    assert "```" not in text                      # no chunk needed for language problems
    assert "Verdict: " in text


def test_render_abstained_shows_the_source_chunk():
    text = rv.render(rec(1), "abstained_with_gold", 1)
    assert "Lisa: I'm a vegetarian" in text and "Ay caramba" not in text and "PLOT TEXT" not in text
    assert "pregunta 1?" in text and "> El contexto no contiene" in text
    assert "question prompt v2" in text
    assert "NOT retrieved" in rv.render(rec(2, gold_in=False), "answered_without_gold", 1)


def test_tally():
    report = "Verdict: A\nVerdict: b\nVerdict:   \nVerdict: A - note\nVerdict:\n"
    counts, pending = rv.tally(report)
    assert counts == {"A": 2, "B": 1} and pending == 2


def test_cli_writes_a_reproducible_sample(tmp_path, capsys):
    inp = tmp_path / "teacher.jsonl"
    inp.write_text("\n".join(json.dumps(rec(i)) for i in range(1, 11)) + "\n{half-written", encoding="utf-8")
    out1, out2 = tmp_path / "a.md", tmp_path / "b.md"
    for out in (out1, out2):
        rv.main(["--inp", str(inp), "--flag", "abstained_with_gold", "--n", "4", "--out", str(out)])
    assert out1.read_text() == out2.read_text()
    text = out1.read_text()
    assert text.count("Verdict: \n") == 4 and "10 of 10 records are flagged; showing 4" in text
    assert "A = the chunk does NOT answer" in text
    rv.main(["--tally", str(out1)])
    assert "still empty" in capsys.readouterr().out.splitlines()[-1]
