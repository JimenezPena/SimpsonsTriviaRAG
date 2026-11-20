# Distilling the free-form Q&A into a small local model

Goal: replace the Gemini call in `/api/ask` with a small open model (1-3B)
fine-tuned with LoRA, using the current system as the **teacher**. The student
gets the same input as production (system prompt + retrieved context +
question) and learns to write the teacher's answer. Retrieval stays as it is.

Pipeline: `gen_teacher` → `build_dataset` → *(train LoRA)* → *(evaluate)*.
Only the first two are implemented so far.

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

## Drift guard

The prompt layout lives in `distill/prompts.py` and mirrors `api_ask` in
`app.py`; `tests/test_distill.py::test_prompt_layout_matches_app_py` fails if
`app.py` changes. Run the tests with `python -m pytest tests -q` (offline).

## Known limitations

* Language and abstention checks are heuristics; an LLM judge for grounding
  is the natural next filter once the pipeline is validated.
