# ADR-001: Embedding Model — BAAI/bge-m3

## Status
Accepted

## Date
2026-06-04

## Context
We need an embedding model to convert text chunks into vectors for semantic search in Qdrant. The model runs at two points:
1. Ingestion time — embed 5,026 chunks once
2. Query time — embed every user query before retrieval

The assignment says "use off-the-shelf models throughout" and "OSS-first".

## Options considered

| Option | Dims | Cost | Quality | Speed | Notes |
|---|---|---|---|---|---|
| BAAI/bge-m3 (local) | 1024 | Free | Excellent | Slow (CPU) | SOTA on MTEB benchmark |
| Google text-embedding-004 | 768 | API | Good | Fast | Deprecated Jan 2026 |
| gemini-embedding-001 | 3072 | API | Excellent | Fast | 3x storage cost |
| BAAI/bge-small-en (local) | 384 | Free | Good | Fast | English only |
| OpenAI text-embedding-3-small | 1536 | API | Good | Fast | Requires OpenAI key |

## Decision
**BAAI/bge-m3 running locally via sentence-transformers.**

## Rationale
- Fully local — no API cost, no rate limits, no vendor dependency.
- State of the art on MTEB retrieval benchmark — better than paid alternatives.
- 1024 dimensions — good quality without 3072-dim storage overhead.
- Supports 100+ languages — covers bonus track B4 (Hindi/Tamil queries) for free.
- Supports both dense AND sparse vectors natively — one model for hybrid retrieval.
- OSS-first as required by assignment ground rules.
- One-time download (~2.27GB), cached permanently after first run.

## Consequences
- Initial ingestion takes ~60 minutes on CPU (acceptable for one-time setup)
- Query embedding takes ~0.3 seconds per query (acceptable latency)
- No ongoing API cost or quota constraints for embeddings
- Same model used at ingestion and query time (required for vector space consistency)
