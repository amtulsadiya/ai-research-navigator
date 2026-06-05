"""
Query understanding — extracts intent and metadata filters from a user query.

Why do we need this?
A raw query like "recent papers on RLHF from 2024" contains implicit filters:
  - year >= 2024
  - tags contains "RLHF"
  - content_type = "arxiv_paper"

Without extracting these, we'd do a pure semantic search and might return
a 2019 blog post about RLHF when the user clearly wanted recent papers.

By extracting filters and applying them in Qdrant at retrieval time,
we get both semantic relevance AND structured filtering — the best of both worlds.

Design decision:
We use Gemini to extract filters via structured JSON output.
Alternative: regex/rule-based extraction — simpler but misses nuanced queries.
Gemini handles "what did the attention paper say about positional encoding" →
correctly identifies this as a paper_deep_dive about arxiv-1706.03762.
"""

from __future__ import annotations

import json
import re

from langchain_google_genai import ChatGoogleGenerativeAI

from research_navigator.config import get_settings
from research_navigator.logger import get_logger

logger = get_logger(__name__)

# ── Filter extraction prompt ───────────────────────────────────────────────────

_FILTER_EXTRACTION_PROMPT = """You are a filter extraction system for an AI/ML research database.

Given a user query, extract structured metadata filters.

Available filters:
- content_type: one of ["arxiv_paper", "course_chapter", "survey_blog", "lab_blog_post"]
- year_gte: minimum year (integer)
- year_lte: maximum year (integer)  
- tags: list of relevant tags from: [LLM, RAG, agents, alignment, attention, architecture, benchmark, chain_of_thought, efficiency, evaluation, few_shot, fine_tuning, inference_optimization, instruction_tuning, interpretability, long_context, MoE, multimodal, nlp, open_models, preference_optimization, pretraining, prompting, quantization, reasoning, retrieval, RL, RLAIF, RLHF, safety, scaling, survey, tokenization, tool_use, transformers, data]
- is_foundational: true only if user explicitly asks for foundational/classic papers

Rules:
- Only include filters you are confident about
- "recent" means year_gte: 2024
- "last year" means year_gte: 2024
- "foundational" or "classic" or "original" means is_foundational: true
- If no filters apply, return empty object {}
- Return ONLY valid JSON, no explanation, no markdown

Examples:
Query: "recent work on RLHF"
Output: {"year_gte": 2024, "tags": ["RLHF"]}

Query: "foundational transformer paper"
Output: {"is_foundational": true, "tags": ["transformers", "attention"]}

Query: "how does RAG work"
Output: {"tags": ["RAG", "retrieval"]}

Query: "explain attention mechanism"
Output: {"tags": ["attention", "transformers"]}

Query: "latest LLM benchmarks"
Output: {"year_gte": 2024, "tags": ["benchmark", "LLM"]}

Now extract filters for this query:
Query: "{query}"
Output:"""


class QueryFilters:
    """
    Structured filters extracted from a user query.
    Used to filter Qdrant results at retrieval time.
    """

    def __init__(
        self,
        content_type: str | None = None,
        year_gte: int | None = None,
        year_lte: int | None = None,
        tags: list[str] | None = None,
        is_foundational: bool | None = None,
    ) -> None:
        self.content_type = content_type
        self.year_gte = year_gte
        self.year_lte = year_lte
        self.tags = tags or []
        self.is_foundational = is_foundational

    def has_filters(self) -> bool:
        """Returns True if any filter is set."""
        return any(
            [
                self.content_type,
                self.year_gte,
                self.year_lte,
                self.tags,
                self.is_foundational is not None,
            ]
        )

    def __repr__(self) -> str:
        parts = []
        if self.content_type:
            parts.append(f"content_type={self.content_type}")
        if self.year_gte:
            parts.append(f"year>={self.year_gte}")
        if self.year_lte:
            parts.append(f"year<={self.year_lte}")
        if self.tags:
            parts.append(f"tags={self.tags}")
        if self.is_foundational is not None:
            parts.append(f"is_foundational={self.is_foundational}")
        return f"QueryFilters({', '.join(parts)})"


def extract_filters(query: str) -> QueryFilters:
    """
    Extract metadata filters from a user query using Gemini.

    Returns a QueryFilters object with extracted filters.
    Falls back to empty filters if extraction fails — we never
    let filter extraction failure block the retrieval pipeline.
    """
    settings = get_settings()
    log = logger.bind(query=query[:100])

    try:
        llm = ChatGoogleGenerativeAI(
            model=settings.generation_model,
            google_api_key=settings.google_api_key,  # type: ignore[arg-type]
            temperature=0.0,  # Deterministic for filter extraction
        )

        prompt = _FILTER_EXTRACTION_PROMPT.replace("{query}", query)
        response = llm.invoke(prompt)
        raw = response.content

        # Strip markdown code fences if present
        raw = re.sub(r"```json\s*|\s*```", "", raw).strip()

        filters_dict = json.loads(raw)
        log.info("filters_extracted", filters=filters_dict)

        return QueryFilters(
            content_type=filters_dict.get("content_type"),
            year_gte=filters_dict.get("year_gte"),
            year_lte=filters_dict.get("year_lte"),
            tags=filters_dict.get("tags", []),
            is_foundational=filters_dict.get("is_foundational"),
        )

    except json.JSONDecodeError as e:
        log.warning("filter_extraction_json_error", error=str(e), raw=raw)
        return QueryFilters()
    except Exception as e:
        log.warning("filter_extraction_failed", error=str(e))
        return QueryFilters()
