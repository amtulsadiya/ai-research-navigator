"""
Ingestion pipeline — orchestrates the full flow from manifest to Qdrant.
Flow:
manifest.json → DocumentMeta → ParsedDocument → [Chunk] → [ChunkPayload + vector] → Qdrant
This module is the glue. It calls parser, chunker, embedder, and qdrant_store
in the right order and handles errors gracefully.
Idempotency:
Re-running the pipeline on an unchanged corpus = zero new writes.
This is guaranteed by deterministic chunk_ids (hash of content).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from qdrant_client import QdrantClient


from research_navigator.config import get_settings
from research_navigator.ingest.chunker import chunk_document
from research_navigator.ingest.embedder import get_embeddings
from research_navigator.ingest.models import (
    ChunkPayload,
    DocumentMeta,
)
from research_navigator.ingest.parser import parse_document
from research_navigator.ingest.qdrant_store import (
    create_collection_if_not_exists,
    get_collection_stats,
    get_qdrant_client,
    upsert_chunks,
)

logger = structlog.get_logger(__name__)


def load_manifest(manifest_path: Path) -> list[DocumentMeta]:
    """
    Load and parse manifest.json into DocumentMeta objects.

    Validates every field via Pydantic — if the manifest has a typo
    or wrong type, you get a clear ValidationError here, not a cryptic
    KeyError 5 steps later.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found at {manifest_path}")

    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents = raw.get("documents", [])

    metas: list[DocumentMeta] = []
    for entry in documents:
        try:
            metas.append(DocumentMeta(**entry))
        except Exception as e:
            logger.error(
                "manifest_entry_invalid", doc_id=entry.get("doc_id", "?"), error=str(e)
            )
            raise

    logger.info("manifest_loaded", document_count=len(metas))
    return metas


def ingest_document(
    meta: DocumentMeta,
    client: QdrantClient,
    collection_name: str,
    dry_run: bool = False,
) -> dict[str, int | str]:
    """
    Full ingestion flow for a single document.

    Idempotency logic:
    - If stored chunk count == current chunk count AND all chunk_ids exist → skip
    - If counts differ (chunks added/removed by filter changes) → delete all old chunks, re-embed all current chunks
    - This correctly handles: unchanged docs (skip), edited docs (re-embed), filter changes (clean re-ingest)
    """
    log = logger.bind(doc_id=meta.doc_id, content_type=meta.content_type.value)

    # Step 1: Parse
    try:
        parsed = parse_document(meta)
    except FileNotFoundError as e:
        log.error("parse_failed_file_not_found", error=str(e))
        return {"chunks": 0, "upserted": 0, "error": str(e)}
    except Exception as e:
        log.error("parse_failed", error=str(e))
        return {"chunks": 0, "upserted": 0, "error": str(e)}

    # Step 2: Chunk
    chunks = chunk_document(parsed)
    log.info("chunks_produced", count=len(chunks))

    if not chunks:
        log.warning("no_chunks_produced")
        return {"chunks": 0, "upserted": 0}

    if dry_run:
        log.info("dry_run_skipping_embed_and_upsert")
        return {"chunks": len(chunks), "upserted": 0}

    # Step 3: Compare stored vs current chunk count
    from research_navigator.ingest.qdrant_store import (
        _chunk_id_to_qdrant_id,
        delete_document_chunks,
        get_document_chunk_count,
    )

    stored_count = get_document_chunk_count(client, collection_name, meta.doc_id)
    all_point_ids = [_chunk_id_to_qdrant_id(c.chunk_id) for c in chunks]

    # If stored count matches current count, check if all chunk_ids exist
    if stored_count == len(chunks):
        try:
            existing_points = client.retrieve(
                collection_name=collection_name,
                ids=all_point_ids,
                with_payload=False,
                with_vectors=False,
            )
        except Exception:
            existing_points = []

        if len(existing_points) == len(chunks):
            log.info("document_unchanged_skipping", chunks=len(chunks))
            return {"chunks": len(chunks), "upserted": 0}

    # Counts differ or chunk_ids don't match → clean re-ingest
    # Delete ALL old chunks for this doc first
    log.info(
        "document_changed_reinserting",
        stored=stored_count,
        current=len(chunks),
    )
    delete_document_chunks(client, collection_name, meta.doc_id)

    # Step 4: Embed all current chunks
    texts = [c.text for c in chunks]
    try:
        vectors = get_embeddings(texts)
    except Exception as e:
        log.error("embedding_failed", error=str(e))
        return {"chunks": len(chunks), "upserted": 0, "error": str(e)}

    # Step 5: Build payloads and upsert
    payloads = [ChunkPayload.from_chunk_and_meta(chunk, meta) for chunk in chunks]

    result = upsert_chunks(
        client=client,
        collection_name=collection_name,
        payloads=payloads,
        dense_vectors=vectors,
    )

    log.info("document_ingested", chunks=len(chunks), upserted=result["upserted"])
    return {"chunks": len(chunks), "upserted": result["upserted"]}


def run_ingestion(
    manifest_path: Path | None = None,
    doc_ids: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    """
    Run ingestion for all documents (or a subset by doc_id).

    manifest_path: defaults to corpus/manifest.json
    doc_ids: if provided, only ingest these documents
    dry_run: parse and chunk but don't embed or write to Qdrant

    Returns a summary report.
    """
    settings = get_settings()

    if manifest_path is None:
        manifest_path = Path("corpus") / "manifest.json"

    metas = load_manifest(manifest_path)

    # Filter to specific doc_ids if requested
    if doc_ids:
        metas = [m for m in metas if m.doc_id in doc_ids]
        logger.info("filtered_to_doc_ids", count=len(metas), doc_ids=doc_ids)

    # Replace with:
    client: QdrantClient | None = None
    if not dry_run:
        client = get_qdrant_client()
        create_collection_if_not_exists(client, settings.qdrant_collection_name)

    results_total = len(metas)
    results_succeeded = 0
    results_failed = 0
    results_total_chunks = 0
    results_errors: list[dict[str, str]] = []

    for i, meta in enumerate(metas, 1):
        logger.info(
            "ingesting_document",
            progress=f"{i}/{len(metas)}",
            doc_id=meta.doc_id,
        )

        stats = ingest_document(
            meta=meta,
            client=client,
            collection_name=settings.qdrant_collection_name,
            dry_run=dry_run,
        )

        if "error" in stats:
            results_failed += 1
            results_errors.append({"doc_id": meta.doc_id, "error": str(stats["error"])})
        else:
            results_succeeded += 1
            results_total_chunks += int(stats["chunks"])

    # Replace with:
    logger.info(
        "ingestion_complete",
        total=results_total,
        succeeded=results_succeeded,
        failed=results_failed,
        total_chunks=results_total_chunks,
    )

    return {
        "total": results_total,
        "succeeded": results_succeeded,
        "failed": results_failed,
        "total_chunks": results_total_chunks,
        "errors": results_errors,
    }


def run_stats() -> dict[str, object]:
    """
    Return statistics about the current Qdrant collection.
    Used by the CLI stats command.
    """
    settings = get_settings()
    client = get_qdrant_client()
    return get_collection_stats(client, settings.qdrant_collection_name)
