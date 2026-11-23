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

## Next

`general` (answer from model knowledge, labelled as such, or grounded in an
external source) and `offtopic` (polite refusal). The route labels are already
produced and evaluated; the router is also a natural extra task for the LoRA
student.
