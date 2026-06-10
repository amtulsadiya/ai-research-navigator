# AI Research Navigator

A citation-grounded RAG system for AI/ML learners. Ask questions about 50 curated AI/ML papers, courses, and blogs — get answers where every claim traces back to a real source.

## What it does

- Answers questions about AI/ML research using 50 curated documents
- Every factual claim carries an inline citation `[1][2]` linking to the source chunk
- Refuses gracefully when the question is out of scope or confidence is low
- Routes queries intelligently through 6 specialised agent nodes (LangGraph)
- Supports concept explanations, paper deep-dives, comparisons, recent developments, and reading lists

## Stack

| Component | Technology |
|---|---|
| Vector store | Qdrant (local Docker) |
| Embeddings | BAAI/bge-m3 (local, no API) |
| LLM | Google Gemini 2.0 Flash |
| Agent orchestration | LangGraph |
| Config | pydantic-settings |
| Logging | structlog |
| Dependency management | uv |

## Prerequisites

- Python 3.11
- Docker Desktop running
- Google AI Studio API key ([get one free](https://aistudio.google.com/apikey))
- uv installed (`pip install uv`)

## Quick start

```bash
# 1. Clone and enter
git clone <repo-url>
cd ai-research-navigator

# 2. Install dependencies
uv sync --all-extras

# 3. Configure environment
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY

# 4. Start Qdrant
docker compose up -d

# 5. Install pre-commit hooks
uv run pre-commit install

# 6. Fetch corpus documents
cd corpus
pip install requests trafilatura html2text
python complete_corpus.py
cd ..

# 7. Ingest corpus (takes ~60 min on CPU — bge-m3 embedding)
uv run python -m research_navigator.ingest ingest

# 8. Verify ingestion
uv run python -m research_navigator.ingest stats

# 9. Ask a question
uv run python -m research_navigator.agents query "What is the attention mechanism?"
```

## Usage

### Ask a question
```bash
uv run python -m research_navigator.agents query "How does RLHF work?"
uv run python -m research_navigator.agents query "Compare GPT and BERT"
uv run python -m research_navigator.agents query "Recommend papers on transformers"
```

### Test all 6 agent routes
```bash
uv run python -m research_navigator.agents test
```

### Visualise the agent graph
```bash
uv run python -m research_navigator.agents visualize
```

### Ingestion commands
```bash
uv run python -m research_navigator.ingest ingest       # full corpus
uv run python -m research_navigator.ingest validate     # check all files exist
uv run python -m research_navigator.ingest stats        # collection stats
uv run python -m research_navigator.ingest reindex --doc-ids arxiv-1706.03762  # single doc
```

### Run evaluation harness
```bash
make eval
# or
uv run python -m research_navigator.eval
```

### Run tests
```bash
make test
# or
uv run pytest tests/ -v
```

### Lint and type check
```bash
make lint
```

## Project structure

```
ai-research-navigator/
├── src/research_navigator/
│   ├── config.py              # pydantic-settings config
│   ├── logger.py              # structlog setup
│   ├── ingest/                # M1: parsing, chunking, embedding, Qdrant upsert
│   │   ├── models.py          # Pydantic data models
│   │   ├── parser.py          # PDF + Markdown parsers
│   │   ├── chunker.py         # Per-type chunking strategies
│   │   ├── embedder.py        # bge-m3 local embeddings
│   │   ├── qdrant_store.py    # Collection schema + upserts
│   │   └── pipeline.py        # Ingestion orchestrator
│   ├── retrieve/              # M2: query understanding + hybrid retrieval
│   │   ├── query_understanding.py  # Filter extraction via Gemini
│   │   ├── retriever.py       # Dense + sparse hybrid search
│   │   └── pipeline.py        # Query orchestrator
│   ├── generate/              # M2: citation building + generation
│   │   ├── citation_builder.py     # Deduplication + formatting
│   │   └── generator.py       # Gemini generation + refusal
│   ├── agents/                # M3: LangGraph state machine
│   │   ├── state.py           # AgentState TypedDict
│   │   ├── tools.py           # corpus_metadata_lookup, get_current_year
│   │   ├── nodes.py           # 7 agent nodes
│   │   └── graph.py           # LangGraph graph builder
│   └── eval/                  # M4: evaluation harness
│       └── harness.py         # Metrics + report generator
├── corpus/
│   ├── manifest.json          # Document metadata
│   ├── complete_corpus.py     # Fetch arXiv PDFs + blog posts
│   └── documents/             # 50 documents (gitignored)
├── eval/
│   ├── golden_set.json        # 40 evaluation questions
│   └── report.md              # Generated evaluation report
├── tests/
│   ├── unit/                  # Unit tests (no external services)
│   └── acceptance/            # Acceptance tests (need API)
├── docs/adr/                  # Architecture Decision Records
├── docker-compose.yml         # Qdrant local instance
├── Makefile                   # setup, lint, test, eval targets
└── pyproject.toml             # Dependencies + tool config
```

## Known limitations

- Re-ingestion takes ~60 minutes on CPU 
  (bge-m3 is a 2.27GB model)

- 0.6% of chunks are under 20 tokens 
  (code-heavy Markdown files) — fix implemented 
  in chunker.py but full re-ingestion pending

- Similarity threshold (0.525) tuned on 20-query 
  set; may not generalise to all query types

- Google Gemini free tier limits full evaluation 
  to ~20 queries/day

- RLHF training examples in Llama 2 paper cause 
  occasional false positives for human preference 
  queries (pizza scoring 0.585 > threshold 0.525)

## Environment variables

| Variable | Description | Default |
| `GOOGLE_API_KEY` | Google AI Studio key | required |
| `QDRANT_URL` | Qdrant server URL | `http://localhost:6333` |
| `QDRANT_COLLECTION_NAME` | Collection name | `research_navigator` |
| `EMBEDDING_MODEL` | HuggingFace model ID | `BAAI/bge-m3` |
| `GENERATION_MODEL` | Gemini model name | `gemini-2.0-flash-lite` |
| `CHUNK_SIZE` | Max tokens per chunk | `512` |
| `CHUNK_OVERLAP` | Overlap tokens | `64` |
| `TOP_K` | Chunks retrieved per query | `6` |
| `SIMILARITY_THRESHOLD` | Refusal threshold | `0.525` |