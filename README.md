# Ask Your Documents!
### A Hands-On RAG Workshop

Build a complete Retrieval Augmented Generation system from scratch, from raw documents to a working local chatbot, using open-source tools and a locally running LLM. No API keys required.

![RAG demo](images/ragshow.gif)

---

## What you'll build

By the end of this workshop you'll have a fully working local RAG system: load your own documents, embed them into a vector database, and query them through a chat interface powered by a local LLM. More importantly, you'll understand every step of the pipeline well enough to tune and debug it yourself.

---

## Schedule

| Time | Module | Duration |
|------|--------|----------|
| 9:30 | **Module 0**: Welcome, expectations, start the installation | 20 min |
| 9:50 | **Module 1**: RAG concepts | 50 min |
| 10:40 | *Break* | 15 min |
| 10:55 | **Module 2**: Embeddings & vector search | 55 min |
| 11:50 | **Module 3**: Chunking & ingestion | 55 min |
| 12:45 | *Lunch* | 60 min |
| 13:45 | **Module 4**: Full RAG pipeline, failure modes, prompt injection | 60 min |
| 14:45 | **Module 5**: Evaluation (baseline) | 30 min |
| 15:15 | *Break* | 15 min |
| 15:30 | **Module 6**: Improving RAG | 50 min |
| 16:20 | *Break* | 10 min |
| 16:30 | **Module 7**: Capstone | 40 min |
| 17:10 | Closing round | 20 min |
| 17:30 | *End* | |

**Setup runs during the workshop.** We start the [installation](#installation) together in
Module 0, and the downloads (several GB: PyTorch, the LLM, the embedding models) run in the
background while Module 1 covers the concepts. If you can, run it before the workshop day:
it saves bandwidth on the venue wifi and you can start Module 1 hands-on right away.

If a group moves faster than expected, bonus modules are available, see [Bonus modules](#bonus-modules) below.

---

## Module overview

**Module 0: Welcome & setup** (`notebooks/00_setup.ipynb`)
Welcome and expectations, then we start the installation. The notebook checks that your
environment is ready (Python, packages, Ollama, the LLM) and downloads all embedding and
re-ranking models. Downloads keep running while Module 1 starts with the concepts.

**Module 1: RAG concepts** (`notebooks/01_rag_concepts.ipynb`)
Starts by asking a bare LLM a question it cannot know the answer to. Then: why RAG exists, the
architecture, and a minimal RAG system built from scratch with NumPy, so every step is visible
before we abstract it away.

**Module 2: Embeddings & vector search** (`notebooks/02_embeddings_vector_search.ipynb`)
How sentence embeddings work, which similarity metric to use and why, and how ChromaDB stores,
filters and retrieves vectors. Compare two embedding models and visualise embedding clusters with UMAP.

**Module 3: Chunking & ingestion** (`notebooks/03_chunking_ingestion.ipynb`)
Fixed-size, sliding-window and structure-aware chunking, and a side-by-side test of how strategy
and chunk size change what gets retrieved. Ends with ingesting the sample documents into ChromaDB.

**Module 4: Full RAG pipeline** (`notebooks/04_rag_pipeline.ipynb`)
Retrieval, prompt construction and generation, inspected step by step. How top-k, threshold,
temperature and prompt phrasing affect the answer, the Gradio GUI, typical failure modes, and why
your documents are an attack surface (indirect prompt injection).

**Module 5: Evaluation** (`notebooks/05_evaluation.ipynb`)
Measure before you improve: a gold dataset, Recall@k, and a diagnosis of the questions the
baseline gets wrong. Optional: answer accuracy and LLM-as-judge. Outlook on RAGAS.

**Module 6: Improving RAG** (`notebooks/06_improving_rag.ipynb`)
Re-ranking with a cross-encoder, RAG-Fusion (multi-query + Reciprocal Rank Fusion) and, if time
allows, HyDE. Every technique is scored against the Module 5 baseline, including its latency.

**Module 7: Capstone** (`notebooks/07_capstone.ipynb`)
A short guided project on your own documents (or the samples): mini gold set, baseline, one
technique, measured. Findings are shared in a gallery walk, followed by a closing round.

---

## Bonus modules

These are self-contained extensions for groups that move through the core material faster than expected. None of them are required: if you don't cover them in the workshop, they work just as well for self-study afterwards.

**Bonus A: RAG with Frameworks** (`notebooks/bonus_a_frameworks.ipynb`)
*Best inserted after Module 4.*
You've built the pipeline by hand; now see how LangChain and LlamaIndex implement the same thing in a fraction of the code. Requires bonus dependencies: `uv sync --extra bonus`.

**Bonus B: Hybrid Search** (`notebooks/bonus_b_hybrid_search.ipynb`)
*Best inserted after Module 6.*
Pure vector search has a known weakness: exact matches on keywords, product codes, names, and dates often score poorly on cosine similarity. Hybrid search combines vector search with BM25 and merges the ranked lists using Reciprocal Rank Fusion. Requires bonus dependencies: `uv sync --extra bonus`.

**Bonus C: Agentic RAG** (`notebooks/bonus_c_agentic_rag.ipynb`)
*Best inserted after Module 6 or 7.*
Standard RAG does one fixed retrieve-then-generate pass. Agentic RAG gives the LLM a retrieval tool it can call, inspect, and call again if the first result isn't good enough. Uses only core dependencies.

---

## Prerequisites

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/getting-started/installation/): Python package manager
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

This creates a `.venv` virtualenv, installs all dependencies from `pyproject.toml`, and installs the `ragsst` package in editable mode, all in one step.

**4. Install Ollama and pull the model**

Follow the instructions at [ollama.com/download](https://ollama.com/download) for your operating system, then pull the workshop model:

```bash
ollama pull llama3.2:3b
```

**5. Verify your setup**

```bash
uv run jupyter notebook notebooks/00_setup.ipynb
```

Run the two check cells. The first prints ✓ or ✗ for each dependency; the second downloads the
embedding and re-ranking models (about 700 MB). If something is marked ✗, ask a trainer.

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
│   ├── 05_evaluation.ipynb
│   ├── 06_improving_rag.ipynb
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
│       ├── evaluation.py               # Recall@k helpers (Modules 5 to 7)
│       ├── parameters.py               # Centralised configuration
│       └── interface.py                # Gradio interface
│
├── tests/
│   ├── __init__.py
│   ├── test_evaluation.py              # Tests for the evaluation helpers
│   ├── test_utils.py                   # Tests for chunking and file I/O
│   └── test_ragtool.py                 # Tests for RAGTool core methods
│
├── data/
│   ├── sample_docs/                    # Sample documents for exercises
│   └── gold_dataset.json               # Gold questions for Modules 5 and 6
├── local-rag-gui.py                    # Launch the standalone Gradio app
├── local-rag-cli.py                    # Command-line version
├── pyproject.toml                      # Project metadata and dependencies
└── bin/
    └── install.sh                      # Auto-installer
```

---

## Running the tests

```bash
uv run --extra dev pytest
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
| `LLM_CHOICES` | `["llama3.2:3b", ...]` | Available Ollama models |
| `COLLECTION_NAME` | `my_docs` | Default ChromaDB collection name |
| `DATA_PATH` | `data` | Path to your documents |
| `KEYWORD_SEARCH` | `True` | Fall back to keyword search when semantic search returns nothing |
| `FILTER_BY_KEYWORD` | `True` | Refine semantic results with keyword filtering |

---

## Notes for trainers

- **Timing assumes a CPU-only laptop.** One RAG call with `llama3.2:3b` takes about 1 s on a
  laptop GPU and 10 to 30 s on a typical laptop CPU. Modules 4 to 6 contain 10 to 15 LLM calls
  each; the schedule includes that waiting time, exercises with many calls do not.
- **Rhythm.** Every module keeps the input short and alternates with hands-on or pair work
  (warm-up in Module 1, "explain it to your neighbour", "spot the failure", think-pair-share on
  the misses, the "decide and explain" recommendation). Cut theory before you cut these.
- **Expected numbers.** On the shipped gold set (12 questions) we measured Recall@1 / Recall@3 of
  0.67 / 0.92 for the baseline and 0.83 / 1.00 with re-ranking (about 10 ms extra per query on a
  GPU). RAG-Fusion varies between runs (0.67 to 0.83 Recall@1) because the rephrasings come from
  the LLM; that variation is a teaching point, not a bug. HyDE scored *below* the baseline
  (0.42 to 0.50), which shows that techniques can hurt. Both LLM-based techniques add 0.7 to 2.5 s
  per query on a GPU and 10 s or more on a CPU.
- **Setup during Module 1.** The first 20 to 25 minutes of Module 1 are theory, so downloads can
  finish in the background. Run the warm-up (1.0) on the projector, so people whose model is still
  downloading can follow along. Many laptops pulling several GB at once can overload venue wifi:
  check the bandwidth beforehand, ask participants to install at home if possible, and keep
  the models on a USB stick as a fallback (`~/.ollama/models` and `~/.cache/huggingface`).
- **If you are short on time**, skip HyDE in Module 6 and section 5.6 in Module 5. Do not skip
  Module 5: Module 6 depends on its baseline.
- **Level.** The day goes from basic concepts to evaluation and re-ranking. Participants without
  any Python experience will struggle from Module 4 onwards; pair them with someone who has it.

---

## Development

Format and lint with ruff:

```bash
uv run ruff format .
uv run ruff check .
```

---

## Resources

- [Sentence Transformers](https://www.sbert.net/): embedding models used in this workshop
- [ChromaDB docs](https://docs.trychroma.com/): vector database
- [Ollama](https://ollama.com/): local LLM runner
- [RAGAS](https://docs.ragas.io/): RAG evaluation framework
- [uv docs](https://docs.astral.sh/uv/): Python package manager used in this project
- [LangChain docs](https://python.langchain.com/): framework covered in Bonus A
- [LlamaIndex docs](https://docs.llamaindex.ai/): framework covered in Bonus A
- [RAG resources](https://github.com/mrdbourke/rag-resources): curated reading list

---

## License

[GPLv3](./LICENSE)
