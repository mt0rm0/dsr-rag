# Ask Your Documents!
### A Hands-On RAG Workshop

Build a complete Retrieval Augmented Generation system from scratch — from raw documents to a working local chatbot — using open-source tools and a locally running LLM. No API keys required.

![RAG demo](images/ragshow.gif)

---

## What you'll build

By the end of this workshop you'll have a fully working local RAG system: load your own documents, embed them into a vector database, and query them through a chat interface powered by a local LLM. More importantly, you'll understand every step of the pipeline well enough to tune and debug it yourself.

---

## Schedule

| Time | Module | Duration |
|------|--------|----------|
| 9:30 | **Module 0** — Setup & orientation | 30 min |
| 10:00 | **Module 1** — RAG concepts | 60 min |
| 11:00 | *Break* | 15 min |
| 11:15 | **Module 2** — Embeddings & vector search | 60 min |
| 12:15 | **Module 3** — Chunking & ingestion | 45 min |
| 13:00 | *Lunch* | 60 min |
| 14:00 | **Module 4** — Full RAG pipeline | 60 min |
| 15:00 | *Break* | 15 min |
| 15:15 | **Module 5** — Improving RAG | 60 min |
| 16:15 | **Module 6** — Evaluation | 45 min |
| 17:00 | **Module 7** — Capstone | 30 min |
| 17:30 | *End* | |

If a group moves faster than expected, bonus modules are available — see [Bonus modules](#bonus-modules) below.

---

## Module overview

**Module 0 — Setup & orientation** (`notebooks/00_setup.ipynb`)
A single notebook that verifies your environment is ready: Python version, Ollama connectivity, model availability, and ChromaDB. Run this before the workshop starts. Also includes a quick demo of the finished app so you know where you're headed.

**Module 1 — RAG concepts** (`notebooks/01_rag_concepts.ipynb`)
Why does RAG exist and what problem does it solve? We walk through the architecture, then build a minimal RAG system from scratch using only NumPy — no frameworks — so every step is visible before we abstract it away.

**Module 2 — Embeddings & vector search** (`notebooks/02_embeddings_vector_search.ipynb`)
How sentence embeddings work, which similarity metrics to use and why, and how vector databases index and retrieve at scale. Three hands-on exercises: compare two embedding models, filter by metadata, and visualise embedding clusters with UMAP.

**Module 3 — Chunking & ingestion** (`notebooks/03_chunking_ingestion.ipynb`)
How you split documents has a bigger impact on retrieval quality than almost any other decision. We explore fixed-size, overlap, and context-aware chunking strategies, inspect the custom chunker built into this repo, and ingest a set of documents into ChromaDB.

**Module 4 — Full RAG pipeline** (`notebooks/04_rag_pipeline.ipynb`)
Wire everything together: document ingestion → embedding → retrieval → prompt construction → LLM response. Run the Gradio GUI and explore how temperature, top-k, and prompt phrasing affect the output.

**Module 5 — Improving RAG** (`notebooks/05_improving_rag.ipynb`)
Four techniques that meaningfully improve retrieval quality, each with a runnable before/after comparison: re-ranking with a CrossEncoder, HyDE (Hypothetical Document Embeddings), Multi-Query retrieval, and RAG-Fusion.

**Module 6 — Evaluation** (`notebooks/06_evaluation.ipynb`)
How do you know if your RAG system is actually good? Build a small gold-standard eval set, implement Recall@k, and run RAGAS on a sample. Leave with a reproducible score for your own system.

**Module 7 — Capstone** (`notebooks/07_capstone.ipynb`)
A guided mini-project: bring your own documents, ingest them, apply one improvement technique from Module 5, and measure the before/after delta with the eval tools from Module 6. Pairs share a one-minute finding at the end.

---

## Bonus modules

These are self-contained extensions for groups that move through the core material faster than expected. None of them are required — if you don't cover them in the workshop, they work just as well for self-study afterwards.

**Bonus A — RAG with Frameworks** (`notebooks/bonus_a_frameworks.ipynb`)
*Best inserted after Module 4.*
You've built the pipeline by hand — now see how LangChain and LlamaIndex implement the same thing in a fraction of the code. Requires bonus dependencies: `uv sync --extra bonus`.

**Bonus B — Hybrid Search** (`notebooks/bonus_b_hybrid_search.ipynb`)
*Best inserted after Module 5.*
Pure vector search has a known weakness: exact matches on keywords, product codes, names, and dates often score poorly on cosine similarity. Hybrid search combines vector search with BM25 and merges the ranked lists using Reciprocal Rank Fusion. Requires bonus dependencies: `uv sync --extra bonus`.

**Bonus C — Agentic RAG** (`notebooks/bonus_c_agentic_rag.ipynb`)
*Best inserted after Module 6.*
Standard RAG does one fixed retrieve-then-generate pass. Agentic RAG gives the LLM a retrieval tool it can call, inspect, and call again if the first result isn't good enough. Uses only core dependencies.

---

## Prerequisites

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — Python package manager
- Basic Python familiarity (loops, functions, imports)
- No prior NLP or ML experience required

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/your-org/dsr-rag.git
cd dsr-rag
```

**2. Install uv** (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**3. Create the environment and install dependencies**

```bash
uv sync
```

This creates a `.venv` virtualenv, installs all dependencies from `pyproject.toml`, and installs the `ragsst` package in editable mode — all in one step.

**4. Install Ollama and pull a model**

Follow the instructions at [ollama.com/download](https://ollama.com/download) for your operating system, then pull the default model:

```bash
ollama pull llama3.2
```

**5. Verify your setup**

```bash
uv run jupyter notebook notebooks/00_setup.ipynb
```

Run the single check cell. It will print ✓ or ✗ for each dependency. Fix any issues before the workshop starts.

> **Auto-install:** Alternatively, run `bash bin/install.sh` to do all of the above in one go.

### Installing bonus module dependencies

The bonus notebooks require additional packages. Install them with:

```bash
uv sync --extra bonus
```

---

## Running the app

After completing Module 4 you can launch the full GUI at any time:

```bash
uv run python local-rag-gui.py
```

Or the command-line version:

```bash
uv run python local-rag-cli.py
```

If the LLM server is not running, start it first in a separate terminal:

```bash
ollama serve
```

---

## Repository structure

```
dsr-rag/
│
├── notebooks/                          # Workshop notebooks (students work here)
│   ├── 00_setup.ipynb
│   ├── 01_rag_concepts.ipynb
│   ├── 02_embeddings_vector_search.ipynb
│   ├── 03_chunking_ingestion.ipynb
│   ├── 04_rag_pipeline.ipynb
│   ├── 05_improving_rag.ipynb
│   ├── 06_evaluation.ipynb
│   ├── 07_capstone.ipynb
│   ├── bonus_a_frameworks.ipynb
│   ├── bonus_b_hybrid_search.ipynb
│   └── bonus_c_agentic_rag.ipynb
│
├── src/
│   └── ragsst/                         # Core library used across notebooks
│       ├── __init__.py
│       ├── ragtool.py                  # Main RAGTool class
│       ├── utils.py                    # Chunking and file I/O
│       ├── parameters.py               # Centralised configuration
│       └── interface.py                # Gradio interface
│
├── tests/
│   ├── __init__.py
│   ├── test_utils.py                   # Tests for chunking and file I/O
│   └── test_ragtool.py                 # Tests for RAGTool core methods
│
├── data/                               # Sample documents for exercises
├── local-rag-gui.py                    # Launch the standalone Gradio app
├── local-rag-cli.py                    # Command-line version
├── pyproject.toml                      # Project metadata and dependencies
└── bin/
    └── install.sh                      # Auto-installer
```

---

## Running the tests

```bash
uv run pytest
```

With coverage:

```bash
uv run pytest --cov=src/ragsst --cov-report=term-missing
```

---

## Configuration

All tuneable parameters live in `src/ragsst/parameters.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EMBEDDING_MODELS` | `["multi-qa-mpnet-base-cos-v1", ...]` | Available sentence transformer models |
| `LLM_CHOICES` | `["llama3.2", ...]` | Available Ollama models |
| `COLLECTION_NAME` | `my_docs` | Default ChromaDB collection name |
| `DATA_PATH` | `data` | Path to your documents |
| `KEYWORD_SEARCH` | `True` | Fall back to keyword search when semantic search returns nothing |
| `FILTER_BY_KEYWORD` | `True` | Refine semantic results with keyword filtering |

---

## Development

Format and lint with ruff:

```bash
uv run ruff format .
uv run ruff check .
```

---

## Resources

- [Sentence Transformers](https://www.sbert.net/) — embedding models used in this workshop
- [ChromaDB docs](https://docs.trychroma.com/) — vector database
- [Ollama](https://ollama.com/) — local LLM runner
- [RAGAS](https://docs.ragas.io/) — RAG evaluation framework
- [uv docs](https://docs.astral.sh/uv/) — Python package manager used in this project
- [LangChain docs](https://python.langchain.com/) — framework covered in Bonus A
- [LlamaIndex docs](https://docs.llamaindex.ai/) — framework covered in Bonus A
- [RAG resources](https://github.com/mrdbourke/rag-resources) — curated reading list

---

## License

[GPLv3](./LICENSE)
