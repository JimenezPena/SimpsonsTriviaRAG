# 🍩 Simpsons RAG Trivia Generator

A lightweight, **local and free RAG (Retrieval-Augmented Generation)** system that uses the historical *The Simpsons* scripts to generate trivia questions and contextual curiosities through embeddings and LLMs.

---

## 🛠️ Architecture & Technologies

* **Orchestration:** Python + LangChain (Chroma integration only)
* **Embeddings:** Google Gemini API (`gemini-embedding-2`) via `batchEmbedContents` REST calls — up to 100 texts per request, so a full ingestion stays well within the free-tier quotas
* **Vector Database:** ChromaDB (local persistence)
* **LLM:** Google Gemini API (`gemini-3.5-flash-lite`) via REST
* **Dataset:** [The Simpsons Dataset (Kaggle)](https://www.kaggle.com/datasets/prashant111/the-simpsons-dataset/)

---

## ✨ Key features

* **Scene-aware chunking** — script lines are grouped by episode and split on scene changes (`location_id`), in blocks of ~20 dialogue lines with a 5-line overlap. The first chunk of each scene includes the last lines of the previous one as context, so gags that span a scene cut keep their continuity.
* **Season filtering** — only classic episodes are indexed (`MAX_SEASON = 13` by default; configurable).
* **Resumable ingestion** — episode-level checkpoint (`ingested_episodes.json`). If the API quota interrupts the run, relaunching the script resumes exactly where it stopped. Partially ingested episodes are detected and rebuilt automatically, and deterministic chunk IDs (`upsert`) prevent duplicates.
* **Quota-friendly** — one real API request per episode (~30 chunks), with retries and backoff on `429 RESOURCE_EXHAUSTED`.

---

## 📂 Repository Structure

```text
simpsons-rag-trivia/
│
├── data/                  # .csv files from the original dataset (git-ignored)
├── vector_db/             # Locally persisted vector database (git-ignored)
├── src/                   # Source code
│   ├── __init__.py
│   ├── ingest.py          # Chunking, vectorization and data ingestion script
│   ├── trivia_gen.py      # Main script: trivia generation and quiz game
│   └── crosscheck_chunks  # Inspect and print chunks per season/episode
│
├── .gitignore             # Files excluded from version control
├── requirements.txt       # Project dependencies
└── README.md              # Main documentation
```

---

## 🚀 Getting Started

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure your API key**

   Create a `.env` file in the project root:

   ```text
   GOOGLE_API_KEY=your_gemini_api_key
   ```

3. **Download the dataset**

   Get the CSVs from [The Simpsons Dataset (Kaggle)](https://www.kaggle.com/datasets/prashant111/the-simpsons-dataset/) and place them under `data/`.

4. **Build the vector database**

   ```bash
   python src/ingest.py
   ```

   Ingestion can be interrupted and relaunched at any time; progress is saved per episode.

5. **Play!**

   ```bash
   python src/trivia_gen.py
   ```

   Two modes are available:
   * **Trivia** — a random chunk is sampled from the database and the LLM generates a multiple-choice question about it (answer revealed after you guess).
   * **Free-form Q&A** — ask anything about the show; the top-3 most similar chunks are retrieved and the LLM answers using only that context, citing the source episode.

---

## 📄 License

MIT
