"""Offline tests for SFT tokenisation and student metrics (no GPU, no models)."""
from distill import metrics
from distill import sft_data as sd

MSGS = [{"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
        {"role": "assistant", "content": "ans"}]


class FakeTok:
    """ChatML-like template; one token per character."""
    eos_token = "<|im_end|>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kw):
        assert tokenize is False and kw.get("enable_thinking") is False
        out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return out + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


class RewritingTok(FakeTok):
    """A template that rewrites the conversation when the last turn is the answer."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kw):
        out = super().apply_chat_template(messages, tokenize, add_generation_prompt, **kw)
        return "REWRITTEN" + out if messages[-1]["role"] == "assistant" else out


# ----------------------------------------------------------------------------
def test_split_uses_the_template_when_consistent():
    prompt, completion, ok = sd.split_prompt_completion(FakeTok(), MSGS)
    assert prompt.endswith("<|im_start|>assistant\n") and "ans" not in prompt
    assert completion == "ans<|im_end|>\n" and ok


def test_split_falls_back_when_template_rewrites():
    prompt, completion, ok = sd.split_prompt_completion(RewritingTok(), MSGS)
    assert completion == "ans<|im_end|>" and not ok
    assert prompt == sd.inference_prompt(RewritingTok(), MSGS)


def test_inference_prompt_ignores_a_trailing_answer():
    assert sd.inference_prompt(FakeTok(), MSGS) == sd.inference_prompt(FakeTok(), MSGS[:2])


def test_labels_mask_only_the_prompt():
    ex, ok = sd.build_example(FakeTok(), MSGS, max_len=1000)
    n_prompt = len(sd.inference_prompt(FakeTok(), MSGS))
    assert ex["labels"][:n_prompt] == [-100] * n_prompt
    assert ex["labels"][n_prompt:] == ex["input_ids"][n_prompt:]
    assert "".join(map(chr, ex["input_ids"][n_prompt:])) == "ans<|im_end|>\n"
    assert len(ex["attention_mask"]) == len(ex["input_ids"])


def test_too_long_is_skipped_not_truncated():
    assert sd.build_example(FakeTok(), MSGS, max_len=10) is None
    rows = [{"messages": MSGS}] * 3
    ex, skipped, inconsistent, lengths = sd.build_dataset(FakeTok(), rows, max_len=10)
    assert ex == [] and skipped == 3
    ex, skipped, inconsistent, lengths = sd.build_dataset(RewritingTok(), rows, max_len=1000)
    assert len(ex) == 3 and inconsistent == 3 and len(lengths) == 3


def test_pad_batch():
    a = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3]}
    b = {"input_ids": [4], "attention_mask": [1], "labels": [4]}
    out = sd.pad_batch([a, b], pad_id=0)
    assert out["input_ids"] == [[1, 2, 3], [4, 0, 0]]
    assert out["attention_mask"] == [[1, 1, 1], [1, 0, 0]]
    assert out["labels"] == [[-100, 2, 3], [4, -100, -100]]


# ----------------------------------------------------------------------------
ES_OK = 'Homer hace algo en la escena que es de el programa. Fuente: "Lisa the Vegetarian".'
ES_ABST = "El contexto no contiene la respuesta a eso en el episodio."


def row(i, ref, lang="es", kind="answerable"):
    user = ('Context:\n\n[Season 7, "Lisa the Vegetarian"]\ntexto\n\n---\n\n'
            '[Plot summary (Wikipedia), season 3, "Other Episode"]\nresumen\n\n'
            "Question: q?\n\nWrite the answer in Spanish (castellano).")
    return {"id": f"r{i}", "meta": {"lang": lang, "kind": kind},
            "messages": [{"role": "system", "content": "S"},
                         {"role": "user", "content": user},
                         {"role": "assistant", "content": ref}]}


def test_context_titles_include_plot_headers():
    assert metrics.context_titles(row(0, "x")["messages"][1]["content"]) == \
        {"lisa the vegetarian", "other episode"}


def test_score_one_flags():
    r = row(0, ES_OK)
    good = metrics.score_one(r, ES_OK)
    assert good["lang_ok"] and good["short"] and good["cites"] is True
    assert good["abstain_agree"] and not good["ref_abstain"]
    nocite = metrics.score_one(r, "Homer hace algo en la escena que es de el programa.")
    assert nocite["cites"] is False
    abst = metrics.score_one(r, ES_ABST)
    assert abst["cites"] is None and not abst["abstain_agree"]
    assert not metrics.score_one(r, "")["lang_ok"]


def test_overconfident_counts_only_where_the_teacher_abstains():
    rows = [row(1, ES_ABST), row(2, ES_ABST), row(3, ES_OK)]
    preds = {"r1": {"response": ES_OK},        # teacher abstains, student answers
             "r2": {"response": ES_ABST},      # both abstain
             "r3": {"response": ES_OK}}
    s = metrics.summarize(rows, preds)
    assert s["n"] == 3 and s["overconfident"] == 0.5
    assert abs(s["abstain_agree"] - 2 / 3) < 1e-9


def test_latency_and_throughput():
    rows = [row(1, ES_OK), row(2, ES_OK)]
    preds = {"r1": {"response": ES_OK, "latency": 1.0, "n_tokens": 20},
             "r2": {"response": ES_OK, "latency": 3.0, "n_tokens": 20}}
    s = metrics.summarize(rows, preds)
    assert s["p50"] == 2.0 and s["tok_s"] == 10.0
    assert metrics.summarize(rows, {"r1": {"response": ES_OK}})["tok_s"] is None


def test_groups_and_table():
    rows = [row(1, ES_OK, "es"), row(2, "Homer does something in the scene of the show.", "en")]
    refs = {r["id"]: {"response": r["messages"][-1]["content"]} for r in rows}
    res = {"teacher": metrics.by_group(rows, refs)}
    assert set(res["teacher"]) == {"all", "en", "es"}
    table = metrics.format_tables(res)
    assert "| teacher |" in table and "**es**" in table and "lang ok" in table


def test_cli_scores_every_system_on_the_common_questions(tmp_path, capsys):
    import json
    rows = [row(i, ES_OK) for i in range(1, 5)]
    data = tmp_path / "val.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    a = tmp_path / "a.jsonl"      # 2 questions
    b = tmp_path / "b.jsonl"      # 3 questions, 2 in common with a
    a.write_text("\n".join(json.dumps({"id": f"r{i}", "response": ES_OK}) for i in (1, 2)))
    b.write_text("\n".join(json.dumps({"id": f"r{i}", "response": ES_ABST}) for i in (2, 3, 4)))
    metrics.main(["--data", str(data), "--preds", f"a={a}", f"b={b}"])
    out = capsys.readouterr().out
    assert "1 questions common to all systems" in out       # only r2
    assert "| teacher | 1 |" in out and "| a | 1 |" in out and "| b | 1 |" in out


def test_percentile():
    assert metrics.percentile([1, 2, 3, 4], 0.5) == 2.5
    assert metrics.percentile([], 0.5) is None
