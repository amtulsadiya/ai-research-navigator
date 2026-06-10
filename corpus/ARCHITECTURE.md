# Architecture

## System Overview

AI Research Navigator is a Retrieval-Augmented Generation (RAG) system with an agentic routing layer. It answers questions about AI/ML research by retrieving relevant chunks from a curated corpus, generating grounded answers, and attaching verifiable citations.

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                  LangGraph Agent (M3)                │
│                                                      │
│  Router ──► concept_explanation ──► M2 Pipeline      │
│         ──► paper_deep_dive     ──► M2 Pipeline      │
│         ──► compare_approaches  ──► M2 Pipeline      │
│         ──► recent_developments ──► M2 Pipeline      │
│         ──► find_papers         ──► Metadata Lookup  │
│         ──► fallback            ──► Decline message  │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                   M2 Query Pipeline                  │
│                                                      │
│  Query Understanding ──► extract metadata filters    │
│  Hybrid Retrieval    ──► dense + sparse in Qdrant    │
│  Confidence Check    ──► refusal if score < 0.525    │
│  Generation          ──► Gemini with citation prompt │
│  Citation Builder    ──► deduplicate + format        │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                    Qdrant (M1)                       │
│                                                      │
│  5,026 chunks from 50 documents                      │
│  Dense vectors (1024-dim, bge-m3)                    │
│  Sparse vectors (BM25)                               │
│  Payload: full metadata per chunk                    │
└─────────────────────────────────────────────────────┘
```

## Component Design

### M1 — Ingestion Pipeline

**Parser** (`ingest/parser.py`): Two parsers — PDF (pymupdf) for arXiv papers and regex-based Markdown for course chapters and blog posts. PDF parsing uses `sort=True` for multi-column layout handling. Markdown parsing splits at `##` headings. References sections are detected and excluded from retrieval chunks.

**Chunker** (`ingest/chunker.py`): Token-based splitting (tiktoken cl100k_base) with 512-token max and 64-token overlap. Abstracts get dedicated chunks. Code blocks are atomic (never split mid-block). Survey blogs use 128-token overlap due to heavy cross-referencing.

**Embedder** (`ingest/embedder.py`): BAAI/bge-m3 via sentence-transformers, running locally. 1024-dimensional output vectors, L2-normalized for cosine similarity.

**Qdrant Store** (`ingest/qdrant_store.py`): Collection with dense (cosine, 1024-dim) and sparse (BM25) vectors. Payload indexing on `content_type`, `year`, `tags`, `primary_category`, `is_foundational`. Idempotent upserts keyed on deterministic chunk IDs: `sha256(doc_id::chunk_index::content_hash)`.

### M2 — Query Pipeline

**Query Understanding** (`retrieve/query_understanding.py`): Gemini extracts structured metadata filters from natural language queries. Falls back to empty filters on failure — never blocks retrieval.

**Retriever** (`retrieve/retriever.py`): Hybrid search using Qdrant's native RRF (Reciprocal Rank Fusion) combining dense semantic and sparse BM25 results. Separate dense-only search provides real cosine similarity scores for confidence thresholding (RRF scores are rank-based and not comparable across queries).

**Citation Builder** (`generate/citation_builder.py`): Deduplicates chunks from the same document into single citation entries. Formats author lists per academic convention (et al. for 3+ authors). Builds numbered context blocks sent to the LLM.

**Generator** (`generate/generator.py`): Calls Gemini with a strict citation prompt instructing every factual claim to be tagged with `[N]`. Refuses with a graceful message if top cosine similarity < 0.525 (data-driven threshold from 20-query evaluation).

### M3 — LangGraph Agent

**State** (`agents/state.py`): `AgentState` TypedDict — explicit, serialisable, total=False so nodes only write fields they produce.

**Router** (`agents/nodes.py`): Gemini classifies query into one of 6 routes with temperature=0 for determinism. Falls back to `concept_explanation` on API failure.

**Tools** (`agents/tools.py`): Two LangChain tools — `corpus_metadata_lookup` (Qdrant scroll by metadata, used by FindPapers) and `get_current_year` (date math for RecentDevelopments recency filter).

**Graph** (`agents/graph.py`): `StateGraph` with START → router → conditional edges → 6 handler nodes → END. Two hops maximum per query.

### M4 — Evaluation Harness

**Golden Set** (`eval/golden_set.json`): 40 hand-curated questions spanning all 6 routes with expected `doc_ids` and `should_refuse` flags.

**Metrics**: Retrieval precision@k and recall@k against expected doc_ids. Citation faithfulness via LLM-as-judge (0-4 rubric, normalised to 0-1). Refusal correctness (false positive and false negative rates). Latency p50/p95. Two configurations compared: hybrid+filters vs hybrid-only.

## Data Flow

### Ingestion time
```
PDF/MD file → parser → ParsedDocument (sections)
           → chunker → [Chunk] (text + metadata)
           → embedder → [vector] (1024-dim)
           → qdrant_store → Qdrant point (vector + payload)
```

### Query time
```
query → embed (bge-m3) → dense vector
      → extract filters (Gemini) → QueryFilters
      → Qdrant hybrid search → [RetrievedChunk]
      → confidence check → refuse if score < 0.525
      → build context block → numbered [1][2] chunks
      → Gemini generation → cited answer
      → citation builder → formatted source list
```

## Key Design Decisions

See `docs/adr/` for full decision records. Summary:

| Decision | Choice | Rationale |
|---|---|---|
| Embedding model | BAAI/bge-m3 (local) | Free, SOTA quality, multilingual, supports sparse vectors |
| Vector dimensions | 1024 | bge-m3 native output |
| Chunk size | 512 tokens | Sweet spot for retrieval quality |
| Hybrid fusion | RRF | No tuning required, industry standard |
| Refusal threshold | 0.525 | Data-driven from 20-query evaluation (90% accuracy) |
| LLM | Gemini 2.0 Flash Lite | Free tier sufficient, fast |
| Agent framework | LangGraph | Explicit state, visualisable, serialisable |

## Corpus

50 documents across 4 content types:

| Type | Count | Format | Source |
|---|---|---|---|
| arxiv_paper | 30 | PDF | arXiv.org |
| course_chapter | 12 | Markdown | Hugging Face Learn |
| survey_blog | 5 | Markdown | Lil'Log (Lilian Weng) |
| lab_blog_post | 3 | Markdown | Anthropic, OpenAI, DeepMind |

Total: 5,026 chunks in Qdrant after ingestion.

## Engineering Standards (M5)

- **Typing**: mypy strict mode on all src/ code
- **Linting**: ruff (lint + format) on every commit
- **Config**: pydantic-settings, zero hardcoded values
- **Logging**: structlog structured JSON throughout
- **Testing**: 75 unit tests across M1-M4
- **Reproducibility**: uv.lock committed, Docker Compose for Qdrant
- **Pre-commit**: ruff + mypy + gitleaks on every commit

## Limitations and Known Issues

1. **Tiny chunks**: 0.525% of chunks ≤20 tokens from code-heavy Markdown. Fix implemented in chunker but full re-ingestion pending.
2. **PDF parsing noise**: Heuristic section heading detection misses some references sections. Some table cells parsed as section titles.
3. **Threshold generalisation**: Refusal threshold tuned on 20-query set. May not hold across all query types.
4. **False positives**: Human preference examples in RLHF papers cause food/lifestyle queries to score anomalously high. Mitigated by M3 router classification.
5. **API quota**: Gemini free tier (20 req/day) limits full evaluation runs.
