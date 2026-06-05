"""
Query pipeline — orchestrates the full M2 query flow.

This is the single entry point for M3 agent nodes to call.
It connects: query understanding → retrieval → generation → cited answer.
"""

from __future__ import annotations

import structlog

from research_navigator.config import get_settings
from research_navigator.generate.generator import GeneratedAnswer, generate_answer
from research_navigator.retrieve.query_understanding import extract_filters
from research_navigator.retrieve.retriever import retrieve

logger = structlog.get_logger(__name__)


def query(
    user_query: str,
    top_k: int | None = None,
    similarity_threshold: float | None = None,
    extract_query_filters: bool = True,
) -> GeneratedAnswer:
    """
    Full query pipeline: understand → retrieve → generate.

    Args:
        user_query: the user's natural language question
        top_k: number of chunks to retrieve (default from settings)
        similarity_threshold: refusal threshold (default from settings)
        extract_query_filters: whether to extract metadata filters from query
                               Set False for simple queries or when filters
                               are passed explicitly by the agent

    Returns:
        GeneratedAnswer with answer text, citations, and metadata
    """
    settings = get_settings()
    log = logger.bind(query=user_query[:100])
    log.info("query_pipeline_start")

    # Step 1: Extract metadata filters from query
    filters = None
    if extract_query_filters:
        filters = extract_filters(user_query)
        if filters.has_filters():
            log.info("filters_applied", filters=str(filters))

    # Step 2: Retrieve relevant chunks
    chunks = retrieve(
        query=user_query,
        filters=filters,
        top_k=top_k or settings.top_k,
    )

    log.info("chunks_retrieved", count=len(chunks))

    # Step 3: Generate cited answer
    answer = generate_answer(
        query=user_query,
        chunks=chunks,
        similarity_threshold=similarity_threshold or settings.similarity_threshold,
    )

    log.info(
        "query_pipeline_done",
        was_refused=answer.was_refused,
        top_score=answer.top_score,
        citations=len(answer.citations),
    )

    return answer
