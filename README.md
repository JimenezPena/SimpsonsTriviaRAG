# 🍩 Simpsons RAG Trivia Generator

A lightweight **RAG (Retrieval-Augmented Generation)** system built on the historical *The Simpsons* scripts. It generates trivia questions and answers free-form questions through embeddings and LLMs, and it can be played from the terminal or from a small **Flask web app** (deployable on Render).

On top of the script-based RAG, an optional pipeline (`src/spanish_align`) takes **Instagram clips of the Spanish dub**, transcribes them, finds the script chunk each clip belongs to and aligns it line by line. The web quiz uses those chunks to ask questions **in Spanish and English** and shows the original clip once you answer.

---

## 🛠️ Architecture & Technologies

* **Orchestration:** Python + LangChain (Chroma integration only)
* **Embeddings:** Google Gemini API (`gemini-embedding-001`) via `batchEmbedContents` REST calls — up to 100 texts per request, so a full ingestion stays well within the free-tier quotas
* **Vector Database:** ChromaDB (local persistence in `batch_vector_db/`)
* **LLM:** Google Gemini API (`gemini-3.5-flash-lite`) via REST. The clip pipeline has its own model setting in `src/spanish_align/config.py` (`GEMINI_MODEL`)
* **Web:** Flask (`src/app.py`) + a single-page frontend (`src/index.html`), served with gunicorn on Render
* **Clips:** `yt-dlp` to download Instagram reels, Gemini to transcribe the audio
* **Dataset:** [The Simpsons Dataset (Kaggle)](https://www.kaggle.com/datasets/prashant111/the-simpsons-dataset/)

---

## ✨ Key features

### Script RAG
* **Scene-aware chunking** — script lines are grouped by episode and split on scene changes (`location_id`), in blocks of ~20 dialogue lines with a 5-line overlap. The first chunk of each scene includes the last lines of the previous one as context, so gags that span a scene cut keep their continuity.
* **Season filtering** — only classic episodes are indexed (`MAX_SEASON = 10` by default; configurable).
* **Resumable ingestion** — episode-level checkpoint (`ingested_episodes.json`). If the API quota interrupts the run, relaunching the script resumes exactly where it stopped. Partially ingested episodes are detected and rebuilt automatically, and deterministic chunk IDs (`upsert`) prevent duplicates.
* **Quota-friendly** — one real API request per episode (~30 chunks), with retries and backoff on `429 RESOURCE_EXHAUSTED`.

### Game modes
* **Trivia** — a random chunk is sampled from the database and the LLM generates a multiple-choice question about it (answer revealed after you guess).
* **Clip quiz (web)** — the question is generated from what an Instagram clip actually says (the Spanish dub text where it was aligned with the script), in Spanish and English (the language can be switched in the UI). After you answer, the clip is embedded below the verdict.
* **Free-form Q&A** — ask anything about the show; the top-3 most similar chunks are retrieved and the LLM answers using only that context, citing the source episode.

### Spanish clip pipeline (`src/spanish_align`)
* **Clip collection** — `yt-dlp` lists the reels of an Instagram account, deduplicated against what is already known.
* **Transcription** — each video is sent to Gemini, which detects whether it is the **Spanish dub or the original English audio** (`audio_lang`) and returns timestamped segments. Transcripts are cached; the video is deleted right after (no videos are kept or committed).
* **Chunk identification** — the clip transcript is embedded once and compared (cosine similarity) against the chunk embeddings **already stored in the vector DB**, so no chunk is ever re-embedded. The top-3 candidates are kept because a clip can span a chunk boundary.
* **Confidence tiers** — score ≥ 0.75 is accepted automatically, 0.55–0.75 goes to a manual review queue, below 0.55 is discarded (thresholds in `config.py`).
* **Line alignment** — every transcript segment is matched to its closest English line of the chunk. Above a similarity of 0.60, a Spanish clip's line is marked `translation_source="dub"` (the official Spanish text) and an English clip's line `"audio"` (the original line, with the literal translation as `text_es`). Below it, the line falls back to the precomputed `"literal"` translation (or `"missing"` if there is none).
* **Literal translations** — all chunks can be translated EN→ES once, offline, in large batches (resumable).

---

## 📂 Repository Structure

```text
SimpsonsTriviaRAG/
│
├── data/
│   ├── *.csv                      # Kaggle dataset (git-ignored)
│   ├── quiz_pool.json             # One entry per clip: URL + what the clip says (committed, used by the web quiz)
│   └── ingest/                    # Clip pipeline working folder
│       ├── ig_links.jsonl         # chunk_id -> Instagram URL registry (committed)
│       ├── review_queue.jsonl     # Clips with a doubtful match (committed)
│       ├── chunks_en.jsonl        # English chunks exported from the DB (git-ignored)
│       ├── chunks_es.jsonl        # Literal ES translations (git-ignored)
│       ├── chunks/                # One aligned record per clip (git-ignored)
│       ├── transcripts/           # Cached transcripts (git-ignored)
│       ├── cache/                 # Embedding caches (git-ignored)
│       └── media/                 # Temporary video downloads (git-ignored)
├── batch_vector_db/               # Local vector database (git-ignored)
├── src/
│   ├── ingest_batch.py            # Chunking, vectorization and ingestion into ChromaDB
│   ├── trivia_gen.py              # Core logic + terminal game (trivia and free-form Q&A)
│   ├── app.py                     # Flask web server (JSON API + static frontend)
│   ├── index.html                 # Web frontend
│   ├── crosscheck_chunks.py       # Inspect and print chunks per season/episode
│   ├── baseline_Q_and_A.py        # Same Q&A without retrieval, to measure what the RAG adds
│   └── spanish_align/             # Instagram clips -> aligned Spanish chunks
│       ├── config.py              # Paths, thresholds and Google models
│       ├── collect_urls.py        # List the reels of an Instagram account
│       ├── export_from_db.py      # Export chunks + stored embeddings from ChromaDB
│       ├── translate_chunks.py    # Literal EN->ES translation of all chunks (offline)
│       ├── media.py               # yt-dlp download and video cache
│       ├── google_client.py       # Gemini: embeddings, transcription, translation
│       ├── chunk_match.py         # Clip -> chunk identification
│       ├── align.py               # Line-by-line dub/literal alignment
│       ├── models.py              # Line and Chunk records
│       ├── cli.py                 # Orchestrator: ingest / review / cleanup
│       └── export_quiz_pool.py    # Builds data/quiz_pool.json for the web quiz
│
├── .gitignore                     # Files excluded from version control
├── requirements.txt               # Full dependencies (ingestion + clip pipeline + web)
├── requirements-web.txt           # Runtime-only dependencies for Render
├── build.sh                       # Render build step (installs and downloads the vector DB)
├── render.yaml                    # Render Blueprint
└── README.md                      # Main documentation
```

---

## 🚀 Getting Started

All commands are run **from the project root**.

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   The clip pipeline also needs [`yt-dlp`](https://github.com/yt-dlp/yt-dlp) available on your `PATH`.

2. **Configure your API key**

   Create a `.env` file in the project root:

   ```text
   GOOGLE_API_KEY=your_gemini_api_key
   ```

3. **Download the dataset**

   Get the CSVs from [The Simpsons Dataset (Kaggle)](https://www.kaggle.com/datasets/prashant111/the-simpsons-dataset/) and place them under `data/` (`simpsons_script_lines.csv` and `simpsons_episodes.csv`).

4. **Build the vector database**

   ```bash
   python src/ingest_batch.py
   ```

   Ingestion can be interrupted and relaunched at any time; progress is saved per episode.

5. **Play in the terminal**

   ```bash
   python src/trivia_gen.py
   ```

   Two modes are available: trivia, and free-form Q&A (see *Game modes* above).

6. **Or run the web app**

   ```bash
   python src/app.py
   # open http://127.0.0.1:5001
   ```

   The web app needs `batch_vector_db/` (trivia and Q&A) and `data/quiz_pool.json` (clip quiz).

---

## 🎬 Spanish clip pipeline

Run the commands from the project root with `PYTHONPATH=src`. The steps are meant to be run in this order; steps 1 and 2 are done once.

```bash
# 1. Export the English chunks and their stored embeddings from the vector DB (no API calls)
PYTHONPATH=src python -m spanish_align.export_from_db

# 2. Literal EN->ES translation of every chunk (resumable; --only S01 or --only S01E10 to limit it)
PYTHONPATH=src python -m spanish_align.translate_chunks

# 3. (optional) Collect reel URLs from an account -> data/ingest/clips_collected.txt
PYTHONPATH=src python -m spanish_align.collect_urls simpsonscast --limit 30

# 4. Download, transcribe, match and align the clips
PYTHONPATH=src python -m spanish_align.cli ingest --urls data/ingest/clips_collected.txt
PYTHONPATH=src python -m spanish_align.cli ingest --url https://www.instagram.com/<account>/reel/<id>/

# 5. Build the quiz data source used by the web app (commit the result)
PYTHONPATH=src python -m spanish_align.export_quiz_pool
```

Other commands:

```bash
PYTHONPATH=src python -m spanish_align.cli review    # interactively approve/reject the doubtful matches
PYTHONPATH=src python -m spanish_align.cli cleanup   # delete locally cached videos (transcripts are kept)
```

`ingest` skips clips that were already processed (use `--redo` to reprocess them) and `--keep-video` keeps the downloaded video instead of deleting it. A clip whose transcript is already cached is not downloaded or transcribed again, so reprocessing only repeats the matching and alignment; to re-transcribe a clip, delete its file in `data/ingest/transcripts/`. `review` shows each doubtful clip with up to 3 candidate chunks: `y` approves the first, `2`/`3` approve another candidate, `n` rejects, `s` skips, `q` quits; every decision is saved immediately.

**Reprocessing the review queue** (e.g. after adding seasons to the vector DB): refresh the chunk data first, then re-match the queued clips. Transcripts are cached, so nothing is downloaded or transcribed again.

```bash
PYTHONPATH=src python -m spanish_align.export_from_db      # new chunks + embeddings
PYTHONPATH=src python -m spanish_align.translate_chunks    # literal ES for the new chunks (resumable)
PYTHONPATH=src python -m spanish_align.cli reprocess-review
PYTHONPATH=src python -m spanish_align.cli review          # whatever is still doubtful
```

Clips that are now accepted leave the queue and are linked; the rest are re-queued with their new best candidate (the old queue is saved as `review_queue.jsonl.bak`). The command aborts if the embeddings or translations are visibly incomplete (`--force` overrides it).

**Season/episode hint.** If you know (or suspect) where a clip comes from, restrict the search so only those chunks compete: `ingest --urls clips.txt --scope S05` (also `5`, `S05E10`, `5x10`, or a list such as `5,6`); a line of the URL file can carry its own hint (`https://www.instagram.com/reel/XXXX/ S05E10`) and overrides `--scope`. In `review`, `--scope S05` restricts the whole session and the `f` key sets or clears the filter for the current clip (the hint is remembered in the queue). The hint narrows the candidates; it does not lower the thresholds, so a clip that really is from another episode still scores low.

**Reprocessing everything from scratch:** move `data/ingest/ig_links.jsonl`, `data/ingest/review_queue.jsonl` and `data/ingest/chunks/` somewhere else (a backup folder) and run `ingest --urls <your list>` again, then `review` and `export_quiz_pool`. Starting from empty registries avoids stale links to chunks a clip no longer matches. Instagram rate-limits anonymous listing; if `collect_urls` returns errors or nothing, add `--cookies-from-browser safari` (or `chrome`, `firefox`).

**Where the results go**

| File | Content |
|------|---------|
| `data/ingest/chunks/<shortcode>.jsonl` | One record per clip: matched episode/`chunk_id`, `audio_lang`, confidence, runner-up candidates, and its aligned lines (`text_es`, `text_en`, `start`, `end`, `translation_source`) |
| `data/ingest/ig_links.jsonl` | `chunk_id -> ig_url` registry for the accepted clips |
| `data/ingest/review_queue.jsonl` | Clips whose match needs a human look |
| `data/quiz_pool.json` | One entry per clip, keyed by shortcode, with `text_es`/`text_en` taken from the clip's own aligned lines (`text_source` says how: `dub`, `audio`, or `chunk-literal` when the clip has no aligned lines and the whole chunk's literal translation is used) |

---

## 🌐 Web app

`src/app.py` is a thin Flask wrapper around `trivia_gen.py`:

| Endpoint | Purpose |
|----------|---------|
| `GET /` | Frontend (`index.html`) |
| `GET /api/config` | Max season and number of chunks used for Q&A |
| `POST /api/trivia` | New question. `{"mode": "clips"}` uses a chunk with a clip (bilingual); without `mode` it samples any chunk |
| `POST /api/trivia/reveal` | Checks the guess and returns the answer, the fun fact and, in clip mode, the Instagram clip |
| `POST /api/ask` | Free-form Q&A with retrieval |

The correct answer stays on the server until it is revealed. The clip quiz does not use the vector DB at all (it samples `quiz_pool.json`), the classic trivia reads chunks straight from `chroma.sqlite3`, and ChromaDB itself is only loaded on the first free-form question.

---

## ☁️ Deploying on Render

The repository includes a Render Blueprint (`render.yaml`):

1. Push the repository to GitHub and, in Render, choose **New → Blueprint** pointing at it.
2. Set the `GOOGLE_API_KEY` environment variable in the Render dashboard.
3. `build.sh` installs `requirements-web.txt` and downloads the vector database from the GitHub release `latest-db` (`db.tar.gz`, too heavy to live in the repo) into `batch_vector_db/`.
4. The service starts with `gunicorn --workers 1 --threads 4 --timeout 120 src.app:app`. A **single worker** is required because the pending quiz answers are kept in memory.

To publish a new vector DB, upload a new `db.tar.gz` to the `latest-db` release (it must extract to a `batch_vector_db/` folder) and redeploy. `data/quiz_pool.json` is committed, so new clips reach the site by committing the regenerated file.

---

## 🚧 Status & known limitations

* **Dub segments that do not align are left out of the quiz text.** For a Spanish clip only the lines that aligned with the script (`dub`) go into `text_es`; the others are dropped because their literal translation would belong to a different line. A clip with no aligned lines at all falls back to the whole chunk's literal translation (`text_source = "chunk-literal"`), which may cover more than the video shows.
* `data/ingest/chunks/` is git-ignored, so `export_quiz_pool` has to be run on the machine that ran the pipeline; only the resulting `data/quiz_pool.json` is committed.
* The clip on the quiz is shown with Instagram's own embed player, so it needs the reel to be publicly available.

---

## 📄 License

No-License
