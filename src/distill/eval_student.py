"""Generate answers with a student model (base or base + LoRA adapter) and
score them against the teacher.

    # base model, no training: the "base student" row of the table
    PYTHONPATH=src python -m distill.eval_student --model unsloth/Qwen3-1.7B \\
        --data data/distill/val.jsonl --limit 30 --out preds_base.jsonl

    # with the trained adapter
    PYTHONPATH=src python -m distill.eval_student --model unsloth/Qwen3-1.7B \\
        --adapter /content/drive/MyDrive/simpsons_distill/qwen3-1.7b/adapter \\
        --data data/distill/test.jsonl --out preds_lora.jsonl

Greedy decoding, batch size 1 (so latency is per answer). Resumable: ids
already in --out are skipped. Compare several systems with
`python -m distill.metrics`. Latency on a Colab T4 is only meaningful to
compare models with each other, not as a production figure.

Not run in the sandbox it was written in (no GPU): see train_lora.py.
"""
import argparse
import json
import os
import time

from distill import metrics
from distill.sft_data import inference_prompt, read_jsonl


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", default="unsloth/Qwen3-1.7B", help="base model")
    p.add_argument("--adapter", help="LoRA adapter dir (from train_lora); omit for the base model")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--no-4bit", action="store_true")
    args = p.parse_args(argv)

    from unsloth import FastLanguageModel          # before transformers
    import torch

    use_4bit = not args.no_4bit and "qwen3.5" not in args.model.lower()
    load_kw = {"load_in_4bit": True} if use_4bit else {"load_in_4bit": False,
                                                       "load_in_16bit": True}
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter or args.model,      # an adapter dir loads its base too
        max_seq_length=args.max_seq_len, dtype=None, **load_kw)
    FastLanguageModel.for_inference(model)
    tok = getattr(tokenizer, "tokenizer", tokenizer)

    rows = read_jsonl(args.data, args.limit)
    done = set()
    if os.path.exists(args.out):
        done = {r["id"] for r in read_jsonl(args.out)}
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(todo)} to generate ({len(done)} already in {args.out})")

    with open(args.out, "a", encoding="utf-8") as out:
        for i, row in enumerate(todo, 1):
            text = inference_prompt(tokenizer, row["messages"])
            inputs = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
            if inputs["input_ids"].shape[1] > args.max_seq_len:
                print(f"skip {row['id']}: prompt longer than --max-seq-len")
                continue
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, use_cache=True)
            torch.cuda.synchronize()
            latency = time.time() - t0
            new = gen[0][inputs["input_ids"].shape[1]:]
            response = tok.decode(new, skip_special_tokens=True).strip()
            out.write(json.dumps({"id": row["id"], "response": response,
                                  "latency": round(latency, 3),
                                  "n_tokens": int(new.shape[0])}, ensure_ascii=False) + "\n")
            out.flush()
            if i % 10 == 0 or i == len(todo):
                print(f"[{i}/{len(todo)}] last {latency:.1f}s, {int(new.shape[0])} tokens")

    preds = metrics.load_preds(args.out)
    res = metrics.by_group([r for r in rows if r["id"] in preds], preds)
    print(metrics.format_tables({os.path.basename(args.out): res}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
