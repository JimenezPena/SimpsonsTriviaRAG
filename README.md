# 🍩 Simpsons RAG Trivia Generator

Un sistema **RAG (Retrieval-Augmented Generation)** ligero, local y gratuito que utiliza los guiones históricos de *Los Simpson* para generar preguntas de trivia y curiosidades contextuales mediante embeddings y LLMs.

---

## 🛠️ Arquitectura y Tecnologías

* **Orquestador:** LangChain
* **Embeddings:** Google Generative AI (`text-embedding-004`)
* **Base de Datos Vectorial:** ChromaDB (Persistencia local)
* **LLM:** Google Gemini API (`gemini-2.5-flash`)
* **Dataset:** [The Simpsons Dataset (Kaggle)](https://www.kaggle.com/datasets/prashant111/the-simpsons-dataset/)

---

## 📂 Estructura del Repositorio

```text
simpsons-rag-trivia/
│
├── data/                  # Contiene los archivos .csv del dataset original (ignorado en git)
├── vector_db/             # Base de datos vectorial persistida localmente (ignorada en git)
├── src/                   # Código fuente del proyecto
│   ├── __init__.py
│   ├── ingest.py          		# Script de chunking, vectorización e ingesta de datos
│   └── trivia_gen.py      		# Script principal para generación y juego de trivias
│   └── crosscheck_chunks      # Script para comprobar y mostrar chunks por episodio/temporada
│
├── .gitignore             # Archivos excluidos del control de versiones
├── requirements.txt       # Dependencias del proyecto
└── README.md              # Documentación principal
