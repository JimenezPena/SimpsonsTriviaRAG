"""LoRA fine-tuning of a small student model on the teacher data (Unsloth).

Meant for a free Colab T4 (see notebooks/distill_colab.ipynb):

    PYTHONPATH=src python -m distill.train_lora \\
        --model unsloth/Qwen3-1.7B \\
        --train data/distill/train.jsonl --val data/distill/val.jsonl \\
        --out /content/drive/MyDrive/simpsons_distill/qwen3-1.7b

* Loss only on the assistant answer (prompt tokens are masked).
* Checkpoints every --save-steps to --out, and the script RESUMES from the
  latest one if you run it again (Colab sessions get cut).
* The final adapter goes to <out>/adapter.
* QLoRA (4-bit) by default. Unsloth advises against 4-bit for Qwen3.5, so for
  names containing "qwen3.5" the model is loaded in 16-bit instead (and needs
  transformers v5; see the notebook). Force with --no-4bit.

NOTE: written against Unsloth's documented API but not run in the sandbox it
was written in (no GPU there): the first run on Colab is the real test. Use
--max-steps 5 for a smoke test.
"""
import argparse
import glob
import json
import os
import time

from distill.sft_data import build_dataset, pad_batch, read_jsonl


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", default="unsloth/Qwen3-1.7B")
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-seq-len", type=int, default=4096,
                   help="examples longer than this are skipped, not truncated")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--save-steps", type=int, default=25)
    p.add_argument("--max-steps", type=int, default=-1, help="smoke test: stop early")
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--seed", type=int, default=3407)
    args = p.parse_args(argv)

    # Unsloth must be imported before transformers / peft
    from unsloth import FastLanguageModel
    import torch
    from transformers import Trainer, TrainingArguments

    use_4bit = not args.no_4bit and "qwen3.5" not in args.model.lower()
    if not use_4bit:
        print("loading in 16-bit (no QLoRA)")
    load_kw = {"load_in_4bit": True} if use_4bit else {"load_in_4bit": False,
                                                       "load_in_16bit": True}
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=args.max_seq_len, dtype=None, **load_kw)
    model = FastLanguageModel.get_peft_model(
        model, r=args.rank, lora_alpha=args.alpha, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=args.seed)

    data = {}
    for name, path in (("train", args.train), ("val", args.val)):
        ex, skipped, inconsistent, lengths = build_dataset(
            tokenizer, read_jsonl(path), args.max_seq_len)
        data[name] = ex
        lengths = sorted(lengths)
        print(f"{name}: {len(ex)} examples, {skipped} skipped (> {args.max_seq_len} tokens), "
              f"tokens p50={lengths[len(lengths) // 2] if lengths else 0} "
              f"max={lengths[-1] if lengths else 0}")
        if inconsistent:
            print(f"  WARNING: {inconsistent} examples use the fallback template split; "
                  "check that the chat template appends the end-of-turn token")
    if not data["train"]:
        raise SystemExit("no training examples")

    class ListDataset(torch.utils.data.Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            return self.items[i]

    tok = getattr(tokenizer, "tokenizer", tokenizer)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def collate(batch):
        return {k: torch.tensor(v) for k, v in pad_batch(batch, pad_id).items()}

    # `warmup_ratio` was removed in recent transformers: compute the steps
    # ourselves (an integer works in every version)
    steps_per_epoch = max(1, -(-len(data["train"]) // (args.batch * args.grad_accum)))
    total_steps = args.max_steps if args.max_steps > 0 else int(steps_per_epoch * args.epochs)
    warmup_steps = max(1, int(0.05 * total_steps))
    print(f"{steps_per_epoch} steps/epoch, {total_steps} steps total, {warmup_steps} warmup")

    native_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    os.makedirs(args.out, exist_ok=True)
    ckpt_dir = os.path.join(args.out, "checkpoints")
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=ckpt_dir,
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=args.batch,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs, max_steps=args.max_steps,
            learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=warmup_steps,
            weight_decay=0.01, optim="adamw_8bit",
            fp16=not native_bf16, bf16=native_bf16,
            logging_steps=5, eval_strategy="epoch",
            save_strategy="steps", save_steps=args.save_steps, save_total_limit=2,
            seed=args.seed, report_to="none", remove_unused_columns=False),
        train_dataset=ListDataset(data["train"]),
        eval_dataset=ListDataset(data["val"]) if data["val"] else None,
        data_collator=collate)

    resume = bool(glob.glob(os.path.join(ckpt_dir, "checkpoint-*")))
    if resume:
        print("resuming from the latest checkpoint")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=True if resume else None)
    minutes = (time.time() - t0) / 60

    adapter_dir = os.path.join(args.out, "adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    info = {"model": args.model, "adapter": adapter_dir, "args": vars(args),
            "n_train": len(data["train"]), "n_val": len(data["val"]),
            "minutes_this_run": round(minutes, 1),
            "last_log": trainer.state.log_history[-3:]}
    with open(os.path.join(args.out, "train_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1, default=str)
    print(f"adapter saved to {adapter_dir} ({minutes:.1f} min this run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
