# ADR-004: Hybrid Retrieval — Dense + Sparse with RRF Fusion

## Status
Accepted

## Date
2026-06-05

## Context
Retrieval is the core of RAG quality. We need to find the most relevant chunks for a user query from 5,026 stored chunks. Two fundamentally different retrieval paradigms exist:

**Dense (semantic)**: Embeds query and chunks into vector space. Finds conceptually related content even without exact keyword matches. Good for "what is attention?" → finds chunks about "attention mechanism" even if those exact words aren't used.

**Sparse (keyword/BM25)**: Finds exact term matches using inverted index. Good for technical terms: "FlashAttention-3 memory complexity" → finds chunks containing exact "FlashAttention-3".

Neither alone is sufficient. Dense misses exact technical terms. Sparse misses semantic relationships.

## Decision
**Hybrid retrieval combining dense (bge-m3) and sparse (BM25) vectors, fused using Reciprocal Rank Fusion (RRF) within Qdrant.**

## Why RRF over weighted score fusion

| Fusion method | Tuning needed | Robustness | Implementation |
|---|---|---|---|
| RRF | None | High | Simple |
| Weighted score | Manual weight tuning | Depends on weights | Simple |
| Learned fusion | Training data needed | High if trained well | Complex |

RRF formula: `score = Σ 1/(k + rank_i)` where k=60 is a smoothing constant.

RRF advantages:
- No weights to tune — robust out of the box
- Dense and sparse scores are on completely different scales — weighted fusion requires normalisation
- Industry standard: used by Elasticsearch, Qdrant, and major search systems
- Chunks ranking high in BOTH dense and sparse get the highest combined score

## The RRF score problem and fix
RRF scores are rank-based (`1/(60+rank)`), always producing 0.5, 0.333, 0.25... regardless of actual similarity. This makes RRF scores useless for confidence thresholding.

**Fix**: Run a separate dense-only search to get real cosine similarity for the confidence check. Use RRF results for ranking. The confidence score comes from true cosine similarity, not RRF rank.

```python
# Get real cosine similarity for threshold check
confidence_results = client.query_points(query=dense_vector, using="dense", limit=1)
confidence_score = confidence_results[0].score  # real cosine similarity

# Use RRF for actual retrieval ranking
rrf_results = client.query_points(prefetch=[dense, sparse], query=FusionQuery(RRF))
```

## Metadata filtering
Filters applied INSIDE Qdrant queries, not post-hoc in Python. This ensures Qdrant returns exactly `top_k` filtered+relevant chunks rather than filtering a larger result set.

Payload indexes created on: `content_type`, `year`, `tags`, `primary_category`, `is_foundational`.

## Consequences
- Two Qdrant calls per query (confidence + hybrid) — acceptable latency overhead (~50ms)
- Significantly better recall for technical term queries vs dense-only
- No tuning required for fusion weights
- Metadata filtering pushes relevance filtering into the database layer
