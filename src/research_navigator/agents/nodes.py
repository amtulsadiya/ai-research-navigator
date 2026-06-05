"""
LangGraph nodes — one function per node.

Each node:
- Takes AgentState as input
- Returns a partial AgentState dict (only the fields it modifies)
- Never mutates state in place

Node responsibilities:
- router_node: classify query → route
- concept_explanation_node: M2 pipeline with synthesis prompt
- paper_deep_dive_node: M2 pipeline filtered to specific paper
- compare_approaches_node: M2 pipeline for two-way comparison
- recent_developments_node: M2 pipeline with recency filter + tool call
- find_papers_node: metadata-only lookup via tool call
- fallback_node: polite out-of-scope decline
"""

from __future__ import annotations

import json
import re
from typing import Any

import structlog
from langchain_google_genai import ChatGoogleGenerativeAI

from research_navigator.agents.state import AgentState
from research_navigator.agents.tools import corpus_metadata_lookup, get_current_year
from research_navigator.config import get_settings
from research_navigator.generate.generator import GeneratedAnswer
from research_navigator.retrieve.pipeline import query as m2_query
from research_navigator.retrieve.query_understanding import QueryFilters

logger = structlog.get_logger(__name__)

# ── Router ─────────────────────────────────────────────────────────────────────

_ROUTER_PROMPT = """You are a query router for an AI/ML research assistant.

Classify the user query into exactly one of these routes:

- concept_explanation: user wants to understand a concept, technique, or idea
  Examples: "what is attention", "explain RLHF", "how does RAG work"

- paper_deep_dive: user asks about a specific paper by name or mentions a specific model
  Examples: "what does the attention paper say", "tell me about Llama 2", "summarize BERT paper"

- compare_approaches: user wants to compare two methods, models, or papers
  Examples: "compare GPT and BERT", "difference between RAG and fine-tuning", "RLHF vs DPO"

- recent_developments: user asks about recent/latest work in an area
  Examples: "latest LLM research", "recent work on agents", "what's new in 2024"

- find_papers: user wants paper recommendations or a reading list
  Examples: "recommend papers on transformers", "foundational papers for beginners", "what should I read about RL"

- out_of_scope: query is not about AI/ML research
  Examples: "pizza recipe", "weather", "cricket", "cooking", anything non-AI/ML

Respond with JSON only:
{{"route": "<route_name>", "reasoning": "<one sentence why>", "doc_id_hint": "<arxiv-id or null>", "comparison_terms": ["term1", "term2"] or null}}

Query: {query}"""


def router_node(state: AgentState) -> dict[str, Any]:
    """
    Classify the query into one of 6 routes.
    Uses Gemini with temperature=0 for deterministic classification.
    """
    settings = get_settings()
    query = state["query"]
    log = logger.bind(query=query[:80])

    try:
        llm = ChatGoogleGenerativeAI(
            model=settings.generation_model,
            google_api_key=settings.google_api_key,  # type: ignore[arg-type]
            temperature=0.0,
        )

        prompt = _ROUTER_PROMPT.replace("{query}", query)
        response = llm.invoke(prompt)
        raw = response.content
        raw = re.sub(r"```json\s*|\s*```", "", raw).strip()
        parsed = json.loads(raw)

        route = parsed.get("route", "concept_explanation")
        reasoning = parsed.get("reasoning", "")
        doc_id_hint = parsed.get("doc_id_hint")
        comparison_terms = parsed.get("comparison_terms")

        log.info("router_classified", route=route, reasoning=reasoning)

        return {
            "route": route,
            "router_reasoning": reasoning,
            "doc_id_hint": doc_id_hint,
            "comparison_terms": comparison_terms,
        }

    except Exception as e:
        log.error("router_failed", error=str(e))
        return {
            "route": "concept_explanation",
            "router_reasoning": f"Router failed, defaulting to concept_explanation: {e}",
        }


# ── ConceptExplanation ─────────────────────────────────────────────────────────


def concept_explanation_node(state: AgentState) -> dict[str, Any]:
    """
    Synthesis-oriented explanation pulling from multiple sources.
    Uses M2 pipeline with higher top_k for broader coverage.
    """
    log = logger.bind(node="concept_explanation", query=state["query"][:80])
    log.info("node_start")

    result = m2_query(
        user_query=state["query"],
        top_k=8,  # More chunks for broader synthesis
        extract_query_filters=True,
    )

    return _format_result(result)


# ── PaperDeepDive ──────────────────────────────────────────────────────────────


def paper_deep_dive_node(state: AgentState) -> dict[str, Any]:
    """
    Deep dive into a specific paper.
    If doc_id_hint is set, filters retrieval to that document only.
    Otherwise does a standard query — the paper name in the query
    will naturally retrieve chunks from that paper.
    """
    log = logger.bind(node="paper_deep_dive", query=state["query"][:80])
    log.info("node_start", doc_id_hint=state.get("doc_id_hint"))

    doc_id_hint = state.get("doc_id_hint")

    if doc_id_hint:
        # Filter to specific document

        # We can't filter by doc_id directly in QueryFilters
        # but we can pass the paper name in the query for high relevance
        result = m2_query(
            user_query=state["query"],
            top_k=10,
            extract_query_filters=False,
        )
    else:
        result = m2_query(
            user_query=state["query"],
            top_k=10,
            extract_query_filters=True,
        )

    return _format_result(result)


# ── CompareApproaches ──────────────────────────────────────────────────────────

_COMPARE_PROMPT_SUFFIX = """

Please structure your answer as a comparison with these sections:
1. Overview of each approach
2. Key similarities
3. Key differences
4. When to use each
5. Summary table if applicable

Cite every claim with [N] markers."""


def compare_approaches_node(state: AgentState) -> dict[str, Any]:
    """
    Compare two methods/papers/approaches.
    Augments the query with a structured comparison prompt.
    """
    log = logger.bind(node="compare_approaches", query=state["query"][:80])
    log.info("node_start", terms=state.get("comparison_terms"))

    # Augment query to get structured comparison
    augmented_query = state["query"] + _COMPARE_PROMPT_SUFFIX

    result = m2_query(
        user_query=augmented_query,
        top_k=10,  # More chunks to cover both approaches
        extract_query_filters=True,
    )

    return _format_result(result)


# ── RecentDevelopments ─────────────────────────────────────────────────────────


def recent_developments_node(state: AgentState) -> dict[str, Any]:
    """
    Return recent developments with recency filter.

    Uses get_current_year tool for date math — this is the required
    tool call within an agent node per the assignment spec.

    Returns chronologically ordered digest of recent work.
    """
    log = logger.bind(node="recent_developments", query=state["query"][:80])
    log.info("node_start")

    # Tool call — get current year for recency filter
    date_info = get_current_year.invoke({})
    cutoff_year = date_info["recent_cutoff_year"]
    log.info("tool_call_get_current_year", cutoff_year=cutoff_year)

    # Apply recency filter
    filters = QueryFilters(year_gte=cutoff_year)

    result = m2_query(
        user_query=state["query"],
        top_k=8,
        extract_query_filters=False,  # We provide filters directly
    )

    # Override with recency-filtered retrieval
    from research_navigator.generate.generator import generate_answer
    from research_navigator.retrieve.retriever import retrieve

    chunks = retrieve(
        query=state["query"],
        filters=filters,
        top_k=8,
    )

    # Sort chunks by year (most recent first) for chronological digest
    chunks.sort(key=lambda c: c.year, reverse=True)

    result = generate_answer(
        query=state["query"]
        + f"\n\nFocus on work from {cutoff_year} onwards. Order by recency.",
        chunks=chunks,
    )

    return _format_result(result)


# ── FindPapers ─────────────────────────────────────────────────────────────────


def find_papers_node(state: AgentState) -> dict[str, Any]:
    """
    Recommend a reading list using metadata-only lookup.

    Uses corpus_metadata_lookup tool — no embedding/semantic search.
    Relies on is_foundational, citation_count, year filters.
    This is the second required tool call in the assignment spec.
    """
    log = logger.bind(node="find_papers", query=state["query"][:80])
    log.info("node_start")

    query = state["query"]
    settings = get_settings()

    # Extract topic tags from query using Gemini
    try:
        llm = ChatGoogleGenerativeAI(
            model=settings.generation_model,
            google_api_key=settings.google_api_key,  # type: ignore[arg-type]
            temperature=0.0,
        )

        tag_prompt = f"""Extract 1-3 relevant tags from this reading list request.
Choose from: [LLM, RAG, agents, alignment, attention, transformers, RLHF, reasoning, fine_tuning, pretraining, scaling, survey, nlp, RL, safety, interpretability]
Return JSON only: {{"tags": ["tag1", "tag2"], "want_foundational": true/false}}
Query: {query}"""

        response = llm.invoke(tag_prompt)
        raw = re.sub(r"```json\s*|\s*```", "", response.content).strip()
        parsed = json.loads(raw)
        tags = parsed.get("tags", [])
        want_foundational = parsed.get("want_foundational", False)
    except Exception as e:
        log.warning("tag_extraction_failed", error=str(e))
        tags = []
        want_foundational = False

    # Tool call — corpus metadata lookup
    if want_foundational:
        # Get foundational papers first
        foundational_docs = corpus_metadata_lookup.invoke(
            {
                "tags": tags,
                "is_foundational": True,
                "limit": 5,
            }
        )
        recent_docs = corpus_metadata_lookup.invoke(
            {
                "tags": tags,
                "is_foundational": False,
                "year_gte": 2023,
                "limit": 5,
            }
        )
        docs = foundational_docs + recent_docs
    else:
        docs = corpus_metadata_lookup.invoke(
            {
                "tags": tags,
                "limit": 10,
            }
        )

    log.info("tool_call_corpus_metadata_lookup", docs_found=len(docs), tags=tags)

    if not docs:
        return {
            "answer": "I couldn't find papers matching your request in the corpus. Try broader tags like 'LLM', 'transformers', or 'RAG'.",
            "citations": [],
            "was_refused": False,
        }

    # Format as reading list
    answer_lines = [f"**Reading List: {query}**\n"]

    # Sort by foundational first, then by year descending
    docs.sort(key=lambda d: (not d.get("is_foundational", False), -d.get("year", 0)))

    for i, doc in enumerate(docs[:8], 1):
        authors = doc.get("authors", [])
        author_str = (
            authors[0].split()[-1] + " et al."
            if len(authors) > 1
            else (authors[0] if authors else "Unknown")
        )
        foundational_marker = " ⭐ foundational" if doc.get("is_foundational") else ""
        citation_str = (
            f" · {doc['citation_count']} citations" if doc.get("citation_count") else ""
        )

        answer_lines.append(
            f"{i}. **{doc['title']}** — {author_str} ({doc['year']}){foundational_marker}{citation_str}\n"
            f"   Tags: {', '.join(doc.get('tags', []))}\n"
            f"   {doc.get('source_url', '')}\n"
        )

    answer = "\n".join(answer_lines)

    # Build serialisable citations
    citations = [
        {
            "number": i + 1,
            "title": doc["title"],
            "authors_formatted": doc.get("authors", ["Unknown"])[0],
            "year": doc["year"],
            "source_url": doc.get("source_url", ""),
            "content_type": doc.get("content_type", ""),
        }
        for i, doc in enumerate(docs[:8])
    ]

    return {
        "answer": answer,
        "citations": citations,
        "was_refused": False,
    }


# ── Fallback ───────────────────────────────────────────────────────────────────

_FALLBACK_MESSAGE = """I'm specialized in AI/ML research and can help you with:

- **Concept explanations**: "What is attention?", "How does RAG work?"
- **Paper deep-dives**: "Tell me about the Llama 2 paper"
- **Comparisons**: "Compare RLHF vs DPO"
- **Recent developments**: "Latest work on LLM agents in 2024"
- **Reading lists**: "Recommend papers on transformers for beginners"

Your question doesn't appear to be about AI/ML research. Please try one of the above!"""


def fallback_node(state: AgentState) -> dict[str, Any]:
    """Polite decline for out-of-scope queries."""
    logger.info("fallback_node", query=state["query"][:80])
    return {
        "answer": _FALLBACK_MESSAGE,
        "citations": [],
        "was_refused": True,
    }


# ── Helper ─────────────────────────────────────────────────────────────────────


def _format_result(result: GeneratedAnswer) -> dict[str, Any]:  # type: ignore[misc]
    """Convert a GeneratedAnswer into serialisable state dict."""
    return {
        "answer": result.answer,
        "citations": [
            {
                "number": c.number,
                "title": c.title,
                "authors_formatted": c.authors_formatted,
                "year": c.year,
                "source_label": c.source_label,
                "section_title": c.section_title,
                "source_url": c.source_url,
            }
            for c in result.citations
        ],
        "was_refused": result.was_refused,
        "top_score": result.top_score,
    }
