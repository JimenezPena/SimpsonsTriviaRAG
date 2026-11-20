# Distilling the free-form Q&A into a small local model

Goal: replace the Gemini call in `/api/ask` with a small open model (1-3B)
fine-tuned with LoRA, using the current system as the **teacher**. The student
gets the same input as production (system prompt + retrieved context +
question) and learns to write the teacher's answer. Retrieval stays as it is.

Pipeline: `gen_teacher` → `build_dataset` → `train_lora` → `eval_student` / `metrics`.
Steps 1-2 run locally; 3-4 run on a free Colab T4 through
`notebooks/distill_colab.ipynb`.

## 1. Generate teacher data

```bash
PYTHONPATH=src python -m distill.gen_teacher --n 10      # smoke test first
PYTHONPATH=src python -m distill.gen_teacher --n 200 --langs es,en
```

Each example costs 2 Gemini calls (write the question, answer it) + 1
embedding call.

* **Temperatures:** `--question-temp 0.9` (diverse inputs) and `--answer-temp 0.2`
  (low-noise targets). They are stored in each record's `meta.temperature`.
  If the model misbehaves at low temperature (repetition, truncation), raise
  `--answer-temp` and compare on a 10-example run first.
* **Retries:** every API call is retried on 429 / 5xx / network errors with
  growing waits (`--retry-base 5` doubling up to `--retry-cap 120`, `--max-retries 5`),
  honouring the `retryDelay` hint Gemini sends with 429s. 400/401/403/404 are
  not retried. A *daily* quota (or a 429 that survives every retry) stops the
  run cleanly; it is resumable, so it can be spread over several days of
  free-tier quota. An example that fails after its retries is skipped; the run
  aborts after `--max-errors` of them.

Output `data/distill/teacher_qa.jsonl` (git-ignored on purpose: check
Gemini's terms on training models with its outputs before publishing the data
or the weights). One record per line:

| field | content |
|---|---|
| `id`, `task`, `lang` | `qa`; `es` / `en` (also `ca` / `de` are accepted) |
| `system`, `prompt`, `response` | exactly what production sends / what the teacher answered |
| `meta.question`, `meta.kind` | the synthetic question; `answerable` or `unanswerable` |
| `meta.gold`, `meta.gold_in_context` | the chunk the question was written from, and whether the retriever returned it (line-range overlap) |
| `meta.retrieved` | the chunks in the prompt, with distances |

Questions are written in the target language over the (English) script chunks,
so the Spanish examples exercise cross-lingual retrieval too: `gold_in_context`
by language is already a first number for the multilingual table.

## 2. Filter and split

```bash
PYTHONPATH=src python -m distill.build_dataset            # keeps soft-flagged
PYTHONPATH=src python -m distill.build_dataset --strict   # recommended for training
```

Writes `train.jsonl`, `val.jsonl`, `test.jsonl` (`messages` format for TRL /
`apply_chat_template`, plus `meta`). The split is **by episode**, never by
example: chunks overlap by 5 lines, so a random split would leak. Flags are
explained at the top of `build_dataset.py`.

## 3. Train the student (Colab)

Open `notebooks/distill_colab.ipynb` on a T4 runtime. Put `train/val/test.jsonl`
in Drive (`MyDrive/simpsons_distill/data/`: git-ignored, not in the repo). The
notebook evaluates the **base** model zero-shot, smoke-tests training for 5
steps, trains with LoRA (checkpoints to Drive every 25 steps, resumes if the
session is cut) and evaluates the adapter.

Choosing the student: run the base-model evaluation for each candidate on the
same 30 validation questions first (a few minutes each), then train the two
best. Default `unsloth/Qwen3-1.7B` (QLoRA fits a T4 comfortably). Candidates:
`unsloth/Gemma-3-4b-it` (quality ceiling), `unsloth/Llama-3.2-1B-Instruct`
(size floor; no Catalan). Qwen3.5 is possible but needs transformers v5 and
16-bit LoRA (Unsloth advises against 4-bit for it), and Colab ships neither.

Training loss is only on the answer tokens; examples longer than
`--max-seq-len` (4096) are skipped, never truncated; the script prints token
lengths so you can tune it.

## 4. Compare systems

`python -m distill.metrics --data val.jsonl --preds base=... lora=...` prints
one table per language with the teacher as reference row, scoring every system
on the same questions. Metrics are listed in `metrics.py`; they are proxies, so
read some answers by hand.

`train_lora.py` and `eval_student.py` were written against Unsloth's documented
API but have not been run (no GPU where they were written): the first Colab
run is their real test, hence the 5-step smoke test.

## Drift guard

The prompt layout lives in `distill/prompts.py` and mirrors `api_ask` in
`app.py`; `tests/test_distill.py::test_prompt_layout_matches_app_py` fails if
`app.py` changes. Run the tests with `python -m pytest tests -q` (offline).

## Known limitations

* Language and abstention checks are heuristics; an LLM judge for grounding
  is the natural next filter once the pipeline is validated.
