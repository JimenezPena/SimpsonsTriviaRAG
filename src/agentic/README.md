# Router for the free-form Q&A

Some questions cannot be answered by retrieval ("How many episodes does season
10 have?": no script chunk says it). The router sends each question to the
mechanism that can answer it:

| route | example | handled by |
|---|---|---|
| `corpus` | "What does Homer say about the bowling balls?" | the existing RAG |
| `metadata` | "How many episodes does season 10 have?" | **closed tools over the episode catalog** (this patch) |
| `general` | "What colour are the Simpsons?" | not acted on yet: goes to the RAG |
| `offtopic` | "Who won the world cup?" | not acted on yet: goes to the RAG |

The router is **off by default**; `ROUTER=on` enables it in `app.py`. Only a
valid `metadata` decision changes anything; every other route, and every
failure (API error, garbage output, unknown tool, bad arguments), falls back to
the RAG, so it can only help.

## How a metadata question is answered

1. A keyword gate (`temporada|season|episodio|cuántos|how many|...`) skips the
   LLM for questions that cannot be about the catalog: no extra call for most
   questions.
2. One LLM call (temperature 0) returns JSON: `{"route", "tool", "args"}`.
3. `tools.validate_call` checks the tool name and argument types. The model only
   *chooses*; it never writes the number or the title.
4. The tool runs on the catalog and the answer is rendered from a **template** in
   the user's language. No second LLM call, so no room for hallucination.

Tools (`tools.SPECS`): `count_seasons`, `count_episodes(season?)`,
`list_episodes(season)`, `episode_title(season, episode)`,
`find_episode(title)` (exact title, English; ambiguous titles return the
candidates; never fuzzy).

## What "the catalog" means

Data: `data/ingest/episode_synopses.json` (committed). Answers say "in the
catalog" on purpose: it is what the project knows, not the real-world list. When
the vector DB is available, the catalog also knows how many of those episodes
have **script chunks** and says so when they differ ("23 in the catalog; 20
have their script indexed"). Seasons outside the catalog get an explicit "not
in the catalog" answer instead of a guess.

## Evaluating it

`data/agentic/router_eval.jsonl` has 42 hand-written labelled questions (es/en,
4 routes, including hard negatives such as "How many episodes does Breaking Bad
season 2 have?").

```bash
PYTHONPATH=src python -m agentic.eval_router                  # offline: the gate only, 0 calls
PYTHONPATH=src python -m agentic.eval_router --llm --limit 5  # try 5 first
PYTHONPATH=src python -m agentic.eval_router --llm --show-errors
```

Key numbers: *metadata solved* (right tool and arguments), *false actions*
(non-catalog questions wrongly answered from the catalog: the harmful error)
and the gate's recall. LLM outputs are cached in `data/agentic/router_cache.jsonl`
(git-ignored): reruns are free and an API-quota stop resumes later.

The gate was tuned on questions written by the same person who wrote the gate:
its 100% recall on the evaluation set is optimistic. Add real user questions to
the set as they appear.

## General and off-topic questions: the retrieval-distance policy

`/api/ask` retrieves first. If the best chunk is **farther** than a per-language
threshold, the question is probably not about the scripts and the router
(one LLM call) decides: `general` (answered from the model's knowledge, labelled
as such by the code, not by the model), `offtopic` (polite refusal),
`metadata` (catalog tool) or `corpus` (back to the RAG). Questions closer than
the threshold never pay an extra call.

| | es | en |
|---|---|---|
| threshold (Chroma distance) | 0.59 (p95 answerable / p05 general) | 0.58 (p90 answerable / p25 general) |

Details worth knowing:

- Override with `ROUTER_DIST_ES`, `ROUTER_DIST_EN`. Unknown language: the larger one.
- **Refuse guard** (`ROUTER_REFUSE_DIST`, default 0.70, `0` disables): an
  `offtopic` verdict is honoured only if the best distance is above it; in the
  0.58-0.70 band the RAG answers (its abstention handles it). Chosen from the
  pooled percentiles (general p95 = 0.738, nonrelated p5 = 0.754).
- Searches restricted to a season are not subject to the policy (their
  distances have another distribution).
- The keyword gate still runs before retrieval; its decision is reused, so a
  question never costs two router calls.
- The request's `lang` is taken as the question's language.
- Distance does not separate *unanswerable* from *answerable* (that is the
  abstention job of the RAG); `general` and `answerable` overlap in 0.55-0.65,
  which is exactly where the LLM decides.

Evaluate it (offline stats are free; the LLM part is cached and limited):

```bash
PYTHONPATH=src python -m agentic.eval_router --zone data/distill/teacher_qa.jsonl \
    data/distill/router_questions.jsonl                 # offline: how many cases reach the router
PYTHONPATH=src python -m agentic.eval_router --zone data/distill/teacher_qa.jsonl \
    data/distill/router_questions.jsonl --llm --limit 30
```

The report adds *diverted* (corpus questions wrongly sent to general/offtopic/catalog:
the harmful error of this policy).

## Next

Grounding `general` in an external source, re-calibrating the thresholds on
more questions, and the router as an extra task for the LoRA student.
