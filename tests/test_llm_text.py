import pytest

from llm_text import candidate_text


def resp(parts, reason="STOP"):
    return {"candidates": [{"content": {"parts": parts}, "finishReason": reason}]}


def test_joins_every_text_part_and_skips_thoughts():
    data = resp([{"text": "thinking...", "thought": True}, {"text": '{"route": "a", '},
                 {"text": '"args": {}}'}])
    assert candidate_text(data) == '{"route": "a", "args": {}}'


def test_warns_when_the_model_did_not_finish(capsys):
    logs = []
    candidate_text(resp([{"text": "cut"}], "MAX_TOKENS"), log=lambda *a, **k: logs.append(a[0]))
    assert "finishReason=MAX_TOKENS" in logs[0]
    candidate_text(resp([{"text": "ok"}]), log=lambda *a, **k: logs.append("no"))
    assert len(logs) == 1


@pytest.mark.parametrize("data", [{}, {"candidates": []}, {"candidates": [{"content": {}}]},
                                  resp([]), resp([{"text": "  "}])])
def test_bad_shapes_raise_what_ask_llm_already_catches(data):
    with pytest.raises((KeyError, IndexError, ValueError)):
        candidate_text(data)
