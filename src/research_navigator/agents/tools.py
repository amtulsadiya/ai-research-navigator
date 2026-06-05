"""
Tools used by agent nodes.

The assignment requires at least one tool call within an agent node.
We implement two tools:

1. corpus_metadata_lookup — structured search of Qdrant by metadata
   Used by: FindPapers node
   Returns: list of documents matching filters (no embedding needed)

2. get_current_year — date math helper
   Used by: RecentDevelopments node
   Returns: current year for computing "last 12 months" filter

Why tools instead of direct function calls?
Tools are the standard LangGraph pattern for agent-external interactions.
They're inspectable, loggable, and can be swapped without changing node logic.
"""

from __future__ import annotations

import datetime
from typing import Any

import structlog
from langchain_core.tools import tool
from qdrant_client.http import models as qmodels

from research_navigator.config import get_settings
from research_navigator.ingest.qdrant_store import get_qdrant_client

logger = structlog.get_logger(__name__)


@tool  # type: ignore[misc]
def corpus_metadata_lookup(
    tags: list[str] | None = None,
    is_foundational: bool | None = None,
    year_gte: int | None = None,
    content_type: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """
    Look up documents in the corpus by metadata filters.
    Returns document-level info (not chunks) for building reading lists.

    Used by FindPapers node to recommend papers by topic and level
    without doing semantic search — purely metadata-driven.
    """
    settings = get_settings()
    client = get_qdrant_client()

    conditions: list[qmodels.Condition] = []

    if tags:
        conditions.append(
            qmodels.FieldCondition(
                key="tags",
                match=qmodels.MatchAny(any=tags),
            )
        )
    if is_foundational is not None:
        conditions.append(
            qmodels.FieldCondition(
                key="is_foundational",
                match=qmodels.MatchValue(value=is_foundational),
            )
        )
    if year_gte is not None:
        conditions.append(
            qmodels.FieldCondition(
                key="year",
                range=qmodels.Range(gte=year_gte),
            )
        )
    if content_type:
        conditions.append(
            qmodels.FieldCondition(
                key="content_type",
                match=qmodels.MatchValue(value=content_type),
            )
        )

    qdrant_filter = qmodels.Filter(must=conditions) if conditions else None

    # Scroll to get matching chunks, deduplicate by doc_id
    results, _ = client.scroll(
        collection_name=settings.qdrant_collection_name,
        scroll_filter=qdrant_filter,
        limit=limit * 5,  # Over-fetch to allow deduplication
        with_payload=True,
        with_vectors=False,
    )

    # Deduplicate by doc_id — return one entry per document
    seen_docs: dict[str, dict[str, Any]] = {}
    for point in results:
        p = point.payload or {}
        doc_id = p.get("doc_id", "")
        if doc_id and doc_id not in seen_docs:
            seen_docs[doc_id] = {
                "doc_id": doc_id,
                "title": p.get("title", ""),
                "authors": p.get("authors", []),
                "year": p.get("year", 0),
                "tags": p.get("tags", []),
                "is_foundational": p.get("is_foundational", False),
                "citation_count": p.get("citation_count"),
                "source_url": p.get("source_url", ""),
                "content_type": p.get("content_type", ""),
            }
        if len(seen_docs) >= limit:
            break

    docs = list(seen_docs.values())
    logger.info(
        "corpus_metadata_lookup",
        results=len(docs),
        filters={"tags": tags, "is_foundational": is_foundational},
    )
    return docs


@tool  # type: ignore[misc]
def get_current_year() -> dict[str, int]:
    """
    Get the current year and compute the cutoff year for recent developments.
    Used by RecentDevelopments node for date math.

    Returns: {"current_year": 2025, "recent_cutoff_year": 2024}
    """
    now = datetime.datetime.now()
    return {
        "current_year": now.year,
        "recent_cutoff_year": now.year - 1,
    }
