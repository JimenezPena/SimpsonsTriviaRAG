"""Tokenisation for SFT, independent of any training library (testable offline).

The model is trained on exactly what it will see at inference:

    <chat template of system + user + generation prompt>   <- masked (label -100)
    <assistant answer + end-of-turn token>                  <- the only loss

`enable_thinking=False` is passed to the chat template: Qwen3/3.5 read it (no
reasoning block), other templates ignore it.
"""
import json


def read_jsonl(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def _render(tok, messages, **kw):
    return tok.apply_chat_template(messages, tokenize=False,
                                   enable_thinking=False, **kw)


def inference_prompt(tok, messages):
    """Text to feed the model at inference: the conversation WITHOUT the
    assistant answer (if present) plus the generation prompt."""
    if messages and messages[-1]["role"] == "assistant":
        messages = messages[:-1]
    return _render(tok, messages, add_generation_prompt=True)


def split_prompt_completion(tok, messages):
    """-> (prompt, completion, template_consistent).

    completion = what the template appends after the generation prompt for the
    assistant turn (answer + end-of-turn token). If the template does not
    render the full conversation as `prompt + something` (it happens with
    templates that rewrite the last turn), fall back to answer + eos_token and
    report template_consistent=False so the caller can warn."""
    prompt = inference_prompt(tok, messages)
    full = _render(tok, messages)
    if full.startswith(prompt) and len(full) > len(prompt):
        return prompt, full[len(prompt):], True
    return (prompt, messages[-1]["content"] + (getattr(tok, "eos_token", None) or ""),
            False)


def build_example(tok, messages, max_len):
    """-> (example, template_consistent) or None when longer than max_len
    (examples are skipped, never truncated: a cut prompt teaches nothing and a
    cut answer teaches to stop early)."""
    prompt, completion, ok = split_prompt_completion(tok, messages)
    t = getattr(tok, "tokenizer", tok)       # multimodal processors wrap one
    p = t(prompt, add_special_tokens=False)["input_ids"]
    c = t(completion, add_special_tokens=False)["input_ids"]
    if len(p) + len(c) > max_len:
        return None
    return ({"input_ids": p + c, "labels": [-100] * len(p) + c,
             "attention_mask": [1] * (len(p) + len(c))}, ok)


def pad_batch(examples, pad_id):
    """Right-pads a list of examples to the longest one (plain lists)."""
    n = max(len(e["input_ids"]) for e in examples)
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for e in examples:
        pad = n - len(e["input_ids"])
        out["input_ids"].append(e["input_ids"] + [pad_id] * pad)
        out["attention_mask"].append(e["attention_mask"] + [0] * pad)
        out["labels"].append(e["labels"] + [-100] * pad)
    return out


def build_dataset(tok, rows, max_len):
    """-> (examples, n_skipped_too_long, n_template_inconsistent, lengths)."""
    examples, skipped, inconsistent, lengths = [], 0, 0, []
    for row in rows:
        built = build_example(tok, row["messages"], max_len)
        if built is None:
            skipped += 1
            continue
        ex, ok = built
        inconsistent += not ok
        lengths.append(len(ex["input_ids"]))
        examples.append(ex)
    return examples, skipped, inconsistent, lengths
