"""
Qdrant store — collection creation, payload indexing, and idempotent upserts.

Design decisions (ADR-004):
- Collection uses both dense vectors (semantic) and sparse vectors (BM25 keyword)
  for hybrid retrieval in M2. We set up both here even though M1 only ingests.
- Payload indexing on: content_type, year, tags, primary_category, is_foundational
  These are the fields we filter on at query time. Without indexing,
  Qdrant does a full collection scan for every filter — O(n) instead of O(log n).
- Idempotent upserts via deterministic chunk_id:
  Same chunk_id → Qdrant updates in place (no duplicate)
  New chunk_id → Qdrant inserts fresh point
- Vector dimension: 768 (Google text-embedding-004 output dimension)
"""

from __future__ import annotations

from typing import Any

import structlog
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from research_navigator.config import get_settings
from research_navigator.ingest.models import ChunkPayload

logger = structlog.get_logger(__name__)

# Google text-embedding-004 produces 768-dimensional vectors
DENSE_VECTOR_DIM = 1024

# Sparse vector name — used for BM25 hybrid retrieval in M2
SPARSE_VECTOR_NAME = "bm25"
DENSE_VECTOR_NAME = "dense"


def get_qdrant_client() -> QdrantClient:
    """Create and return a Qdrant client using settings from .env"""
    settings = get_settings()
    return QdrantClient(url=settings.qdrant_url)


def create_collection_if_not_exists(client: QdrantClient, collection_name: str) -> None:
    """
    Create the Qdrant collection with our schema if it doesn't exist.

    Collection schema:
    - dense vectors: 768-dim, cosine similarity (semantic search)
    - sparse vectors: BM25 (keyword/lexical search)
    - payload fields: all metadata from ChunkPayload

    Why cosine similarity?
    For normalized embeddings (which text-embedding-004 produces),
    cosine similarity is equivalent to dot product and is the standard
    for semantic similarity. It measures the angle between vectors —
    chunks about similar topics cluster together regardless of length.

    Why both dense AND sparse?
    Dense (embedding): "what is the attention mechanism?" → finds semantically
    related content even if exact words don't match
    Sparse (BM25): "FlashAttention-3" → finds exact term matches
    Together: hybrid retrieval catches both semantic AND keyword queries.
    """
    # settings = get_settings()

    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        logger.info("collection_already_exists", collection=collection_name)
        return

    logger.info("creating_collection", collection=collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            DENSE_VECTOR_NAME: qmodels.VectorParams(
                size=DENSE_VECTOR_DIM,
                distance=qmodels.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: qmodels.SparseVectorParams(
                index=qmodels.SparseIndexParams(on_disk=False),
            ),
        },
    )

    # ── Payload indexing ────────────────────────────────────────────────────
    # These are the fields we filter on at query time.
    # Without indexes, Qdrant scans ALL points for every filter.
    # With indexes, it uses an inverted index — much faster.
    #
    # Field types matter:
    # - KEYWORD: exact match strings (content_type, primary_category)
    # - INTEGER: range queries (year)
    # - BOOL: boolean filters (is_foundational)
    # - KEYWORD for tags: because tags is a list[str] and we need
    #   $in / $contains type queries

    _create_payload_index(
        client, collection_name, "content_type", qmodels.PayloadSchemaType.KEYWORD
    )
    _create_payload_index(
        client, collection_name, "year", qmodels.PayloadSchemaType.INTEGER
    )
    _create_payload_index(
        client, collection_name, "tags", qmodels.PayloadSchemaType.KEYWORD
    )
    _create_payload_index(
        client, collection_name, "primary_category", qmodels.PayloadSchemaType.KEYWORD
    )
    _create_payload_index(
        client, collection_name, "is_foundational", qmodels.PayloadSchemaType.BOOL
    )

    logger.info("collection_created_with_indexes", collection=collection_name)


def _create_payload_index(
    client: QdrantClient,
    collection_name: str,
    field_name: str,
    field_type: qmodels.PayloadSchemaType,
) -> None:
    """Create a payload index on a specific field."""
    client.create_payload_index(
        collection_name=collection_name,
        field_name=field_name,
        field_schema=field_type,
    )
    logger.debug("payload_index_created", field=field_name, type=str(field_type))


def upsert_chunks(
    client: QdrantClient,
    collection_name: str,
    payloads: list[ChunkPayload],
    dense_vectors: list[list[float]],
) -> dict[str, int]:
    """
    Upsert chunks into Qdrant with their vectors and payloads.

    Why upsert and not insert?
    Upsert = insert if not exists, update if exists (keyed on point ID).
    Our point ID = chunk_id = hash(doc_id + chunk_index + content_hash).
    If we re-ingest an unchanged document, the same chunk_ids are generated
    and Qdrant updates them in place (which is a no-op if nothing changed).
    If a document's text changed, its content_hash changes → new chunk_id
    → Qdrant inserts the updated chunk.

    This gives us idempotent ingestion: safe to re-run anytime.

    Batching:
    We upsert in batches of 100 to avoid memory issues with large corpora
    and to stay within Qdrant's recommended request sizes.

    Returns: {"upserted": N, "batches": N}
    """
    if not payloads:
        return {"upserted": 0, "batches": 0}

    assert len(payloads) == len(
        dense_vectors
    ), f"Mismatch: {len(payloads)} payloads but {len(dense_vectors)} vectors"

    BATCH_SIZE = 100
    total_upserted = 0
    batch_count = 0

    for i in range(0, len(payloads), BATCH_SIZE):
        batch_payloads = payloads[i : i + BATCH_SIZE]
        batch_vectors = dense_vectors[i : i + BATCH_SIZE]

        points = [
            qmodels.PointStruct(
                id=_chunk_id_to_qdrant_id(payload.chunk_id),
                vector={DENSE_VECTOR_NAME: vector},
                payload=payload.model_dump(),
            )
            for payload, vector in zip(batch_payloads, batch_vectors, strict=False)
        ]

        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,  # Wait for indexing to complete before returning
        )

        total_upserted += len(points)
        batch_count += 1

        logger.debug(
            "batch_upserted",
            batch=batch_count,
            batch_size=len(points),
            total_so_far=total_upserted,
        )

    logger.info("upsert_complete", total=total_upserted, batches=batch_count)
    return {"upserted": total_upserted, "batches": batch_count}


def _chunk_id_to_qdrant_id(chunk_id: str) -> str:
    """
    Qdrant point IDs must be either unsigned integers or UUID strings.
    Our chunk_id is a hex SHA256 string — we format it as a UUID-like string
    by taking the first 32 hex chars and formatting as UUID.

    This is a deterministic mapping: same chunk_id always → same Qdrant ID.
    """
    hex_32 = chunk_id[:32]
    return (
        f"{hex_32[:8]}-{hex_32[8:12]}-{hex_32[12:16]}-{hex_32[16:20]}-{hex_32[20:32]}"
    )


def get_collection_stats(client: QdrantClient, collection_name: str) -> dict[str, Any]:
    """
    Return stats about the collection — used by the CLI stats command.
    """
    try:
        info = client.get_collection(collection_name)
        return {
            "collection": collection_name,
            "total_points": info.points_count,
            "status": str(info.status),
            "vector_size": DENSE_VECTOR_DIM,
        }
    except Exception as e:
        logger.error("stats_failed", error=str(e))
        return {"error": str(e)}


def delete_document_chunks(
    client: QdrantClient,
    collection_name: str,
    doc_id: str,
) -> int:
    """
    Delete all chunks belonging to a specific document.
    Used when a document is updated — delete old chunks, insert new ones.
    Returns the number of deleted points.
    """
    client.delete(
        collection_name=collection_name,
        points_selector=qmodels.FilterSelector(
            filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="doc_id",
                        match=qmodels.MatchValue(value=doc_id),
                    )
                ]
            )
        ),
        wait=True,
    )
    logger.info("deleted_document_chunks", doc_id=doc_id)
    return 0  # Qdrant delete doesn't return count directly


def get_document_chunk_count(
    client: QdrantClient,
    collection_name: str,
    doc_id: str,
) -> int:
    """Get how many chunks exist for a document in Qdrant."""
    result = client.count(
        collection_name=collection_name,
        count_filter=qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="doc_id",
                    match=qmodels.MatchValue(value=doc_id),
                )
            ]
        ),
        exact=True,
    )
    return int(result.count)
