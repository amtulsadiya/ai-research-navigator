"""
Retriever — hybrid search combining dense vectors + sparse BM25 in Qdrant.

Why hybrid retrieval?
- Dense (semantic): finds conceptually related content even without exact keywords
  e.g. "how does self-attention work" → finds chunks about attention mechanism
- Sparse (BM25): finds exact keyword matches
  e.g. "FlashAttention-2 memory complexity" → finds exact term "FlashAttention-2"
- Together: catches both semantic AND keyword queries

Fusion strategy: Reciprocal Rank Fusion (RRF)
Why RRF over weighted score fusion?
- RRF doesn't require tuning weights — robust out of the box
- Works well when dense and sparse scores are on different scales
- Industry standard for hybrid retrieval (used by Elasticsearch, Qdrant)
- Formula: score = Σ 1/(k + rank_i) where k=60 is a smoothing constant

Metadata filtering:
Filters applied INSIDE Qdrant query, not post-hoc in Python.
Why? Post-hoc filtering means: retrieve 100 chunks, then filter to 6.
In-query filtering means: retrieve exactly 6 relevant+filtered chunks.
Much more efficient and accurate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from research_navigator.config import get_settings
from research_navigator.ingest.embedder import embed_single
from research_navigator.ingest.qdrant_store import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    get_qdrant_client,
)
from research_navigator.retrieve.query_understanding import QueryFilters

logger = structlog.get_logger(__name__)


@dataclass
class RetrievedChunk:
    """
    A chunk returned from Qdrant search with its similarity score.
    Wraps the raw Qdrant result into a clean typed object.
    """

    chunk_id: str
    doc_id: str
    chunk_index: int
    section_title: str
    section_index: int
    text: str
    score: float
    is_abstract: bool

    # Document metadata for citation building
    title: str
    authors: list[str]
    year: int
    content_type: str
    source_url: str
    primary_category: str
    tags: list[str]
    is_foundational: bool
    token_count: int

    @classmethod
    def from_qdrant_result(cls, result: Any) -> RetrievedChunk:
        """Build a RetrievedChunk from a raw Qdrant ScoredPoint."""
        p = result.payload
        return cls(
            chunk_id=p.get("chunk_id", ""),
            doc_id=p.get("doc_id", ""),
            chunk_index=p.get("chunk_index", 0),
            section_title=p.get("section_title", ""),
            section_index=p.get("section_index", 0),
            text=p.get("text", ""),
            score=result.score,
            is_abstract=p.get("is_abstract", False),
            title=p.get("title", ""),
            authors=p.get("authors", []),
            year=p.get("year", 0),
            content_type=p.get("content_type", ""),
            source_url=p.get("source_url", ""),
            primary_category=p.get("primary_category", ""),
            tags=p.get("tags", []),
            is_foundational=p.get("is_foundational", False),
            token_count=p.get("token_count", 0),
        )


def _build_qdrant_filter(filters: QueryFilters) -> qmodels.Filter | None:
    """
    Convert QueryFilters into a Qdrant Filter object.

    Qdrant filters use a must/should/must_not structure similar to
    Elasticsearch. We use 'must' for all our filters (AND logic).

    Why build filters here and not in query_understanding.py?
    Separation of concerns:
    - query_understanding.py: understands what the user wants (domain logic)
    - retriever.py: knows how to express that in Qdrant (infrastructure)
    """
    if not filters.has_filters():
        return None

    conditions: list[qmodels.Condition] = []

    if filters.content_type:
        conditions.append(
            qmodels.FieldCondition(
                key="content_type",
                match=qmodels.MatchValue(value=filters.content_type),
            )
        )

    if filters.year_gte is not None:
        conditions.append(
            qmodels.FieldCondition(
                key="year",
                range=qmodels.Range(gte=filters.year_gte),
            )
        )

    if filters.year_lte is not None:
        conditions.append(
            qmodels.FieldCondition(
                key="year",
                range=qmodels.Range(lte=filters.year_lte),
            )
        )

    if filters.tags:
        # Match ANY of the specified tags (OR logic within tags)
        # e.g. tags=["RLHF", "alignment"] matches chunks with either tag
        conditions.append(
            qmodels.FieldCondition(
                key="tags",
                match=qmodels.MatchAny(any=filters.tags),
            )
        )

    if filters.is_foundational is not None:
        conditions.append(
            qmodels.FieldCondition(
                key="is_foundational",
                match=qmodels.MatchValue(value=filters.is_foundational),
            )
        )

    return qmodels.Filter(must=conditions)


def _build_sparse_vector(query: str) -> qmodels.SparseVector:
    """
    Build a sparse BM25 vector for the query.

    Qdrant's built-in sparse vectors use a simple term-frequency approach.
    We encode the query terms as sparse indices and weights.

    For BM25, we use Qdrant's native sparse vector support which handles
    the IDF weighting internally during search.
    """
    # Simple term-frequency sparse encoding
    # Qdrant handles BM25 scoring internally
    terms = query.lower().split()
    term_freq: dict[int, float] = {}

    for term in terms:
        # Use hash of term as sparse index
        idx = abs(hash(term)) % 100000
        term_freq[idx] = term_freq.get(idx, 0) + 1.0

    indices = list(term_freq.keys())
    values = list(term_freq.values())

    return qmodels.SparseVector(indices=indices, values=values)


# def retrieve(
#     query: str,
#     filters: Optional[QueryFilters] = None,
#     top_k: Optional[int] = None,
#     client: Optional[QdrantClient] = None,
# ) -> list[RetrievedChunk]:
#     """
#     Main retrieval function — hybrid dense + sparse search with optional filters.

#     Steps:
#     1. Embed query using bge-m3 (dense vector)
#     2. Build sparse vector from query terms
#     3. Run hybrid search in Qdrant with RRF fusion
#     4. Apply metadata filters inside Qdrant
#     5. Return top-k RetrievedChunk objects

#     Why pass client as optional parameter?
#     Testing — in unit tests we can pass a mock client.
#     In production the function creates its own client.
#     """
#     settings = get_settings()
#     top_k = top_k or settings.top_k
#     qdrant_filter = _build_qdrant_filter(filters) if filters else None

#     if client is None:
#         client = get_qdrant_client()

#     log = logger.bind(query=query[:100], top_k=top_k)
#     log.info("retrieval_start", has_filters=filters.has_filters() if filters else False)

#     # Step 1: Embed query (dense)
#     dense_vector = embed_single(query)

#     # Step 2: Build sparse vector
#     sparse_vector = _build_sparse_vector(query)

#     # Step 3: Hybrid search with RRF fusion
#     try:
#         results = client.query_points(
#             collection_name=settings.qdrant_collection_name,
#             prefetch=[
#                 # Dense search
#                 qmodels.Prefetch(
#                     query=dense_vector,
#                     using=DENSE_VECTOR_NAME,
#                     limit=top_k * 3,  # Fetch 3x for fusion
#                     filter=qdrant_filter,
#                 ),
#                 # Sparse search
#                 qmodels.Prefetch(
#                     query=sparse_vector,
#                     using=SPARSE_VECTOR_NAME,
#                     limit=top_k * 3,
#                     filter=qdrant_filter,
#                 ),
#             ],
#             # RRF fusion
#             query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
#             limit=top_k,
#             with_payload=True,
#             with_vectors=False,
#         ).points

#     except Exception as e:
#         log.error("hybrid_search_failed", error=str(e))
#         # Fallback to dense-only search
#         log.info("falling_back_to_dense_search")
#         results = client.query_points(
#             collection_name=settings.qdrant_collection_name,
#             query=dense_vector,
#             using=DENSE_VECTOR_NAME,
#             query_filter=qdrant_filter,
#             limit=top_k,
#             with_payload=True,
#             with_vectors=False,
#         ).points
#     chunks = [RetrievedChunk.from_qdrant_result(r) for r in results]

#     log.info(
#         "retrieval_done",
#         results=len(chunks),
#         top_score=chunks[0].score if chunks else 0,
#     )


#     return chunks
def retrieve(
    query: str,
    filters: QueryFilters | None = None,
    top_k: int | None = None,
    client: QdrantClient | None = None,
) -> list[RetrievedChunk]:
    settings = get_settings()
    top_k = top_k or settings.top_k
    qdrant_filter = _build_qdrant_filter(filters) if filters else None

    if client is None:
        client = get_qdrant_client()

    log = logger.bind(query=query[:100])
    log.info(
        "retrieval_start",
        has_filters=filters.has_filters() if filters else False,
        top_k=top_k,
    )

    # Embed query
    dense_vector = embed_single(query)
    sparse_vector = _build_sparse_vector(query)

    # Step 1: Get actual cosine similarity score (for confidence/refusal check)
    # This is separate from RRF — we need real similarity, not rank-based score
    confidence_results = client.query_points(
        collection_name=settings.qdrant_collection_name,
        query=dense_vector,
        using=DENSE_VECTOR_NAME,
        query_filter=qdrant_filter,
        limit=1,
        with_payload=False,
        with_vectors=False,
    ).points

    confidence_score = confidence_results[0].score if confidence_results else 0.0

    # Step 2: Hybrid search with RRF for actual retrieval ranking
    try:
        results = client.query_points(
            collection_name=settings.qdrant_collection_name,
            prefetch=[
                qmodels.Prefetch(
                    query=dense_vector,
                    using=DENSE_VECTOR_NAME,
                    limit=top_k * 3,
                    filter=qdrant_filter,
                ),
                qmodels.Prefetch(
                    query=sparse_vector,
                    using=SPARSE_VECTOR_NAME,
                    limit=top_k * 3,
                    filter=qdrant_filter,
                ),
            ],
            query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        ).points

    except Exception as e:
        log.error("hybrid_search_failed", error=str(e))
        results = client.query_points(
            collection_name=settings.qdrant_collection_name,
            query=dense_vector,
            using=DENSE_VECTOR_NAME,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        ).points

    # Override top score with real cosine similarity (not RRF rank score)
    chunks = []
    for i, r in enumerate(results):
        chunk = RetrievedChunk.from_qdrant_result(r)
        # Use real cosine score for rank 0, proportional for others
        chunk.score = confidence_score if i == 0 else r.score
        chunks.append(chunk)

    log.info(
        "retrieval_done",
        results=len(chunks),
        top_score=confidence_score,
    )

    return chunks
