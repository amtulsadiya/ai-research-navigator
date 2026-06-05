"""
Unit tests for M2 — citation building, filter extraction, refusal logic.

We test the pure logic components that don't need external services:
- Citation deduplication
- Author formatting
- Source label formatting
- Refusal threshold logic
- Filter building

We do NOT unit test:
- Gemini API calls (integration test territory)
- Qdrant search (integration test territory)
- bge-m3 embedding (tested implicitly via retrieval)
"""

from __future__ import annotations

from unittest.mock import patch

from research_navigator.generate.citation_builder import (
    _format_authors,
    _format_source_label,
    build_citations,
    build_context_block,
    format_citation_block,
)
from research_navigator.generate.generator import (
    GeneratedAnswer,
    generate_answer,
)
from research_navigator.retrieve.query_understanding import QueryFilters
from research_navigator.retrieve.retriever import RetrievedChunk

# ── Fixtures ───────────────────────────────────────────────────────────────────


def make_chunk(
    doc_id: str = "arxiv-1706.03762",
    chunk_id: str = "abc123",
    section_title: str = "Abstract",
    score: float = 0.85,
    content_type: str = "arxiv_paper",
    authors: list[str] | None = None,
    year: int = 2017,
    title: str = "Attention Is All You Need",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        chunk_index=0,
        section_title=section_title,
        section_index=0,
        text="The transformer architecture uses self-attention mechanisms.",
        score=score,
        is_abstract=True,
        title=title,
        authors=authors or ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
        year=year,
        content_type=content_type,
        source_url="https://arxiv.org/abs/1706.03762",
        primary_category="cs.CL",
        tags=["transformers", "attention"],
        is_foundational=True,
        token_count=150,
    )


# ── Author formatting tests ────────────────────────────────────────────────────


def test_format_authors_single() -> None:
    result = _format_authors(["Ashish Vaswani"], "arxiv_paper")
    assert result == "Vaswani"


def test_format_authors_two() -> None:
    result = _format_authors(["Ashish Vaswani", "Noam Shazeer"], "arxiv_paper")
    assert result == "Vaswani & Shazeer"


def test_format_authors_three_or_more() -> None:
    result = _format_authors(
        ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"], "arxiv_paper"
    )
    assert result == "Vaswani et al."


def test_format_authors_non_paper() -> None:
    result = _format_authors(["Hugging Face"], "course_chapter")
    assert result == "Hugging Face"


def test_format_authors_empty() -> None:
    result = _format_authors([], "arxiv_paper")
    assert result == "Unknown"


# ── Source label tests ─────────────────────────────────────────────────────────


def test_source_label_arxiv() -> None:
    chunk = make_chunk(doc_id="arxiv-1706.03762", content_type="arxiv_paper")
    label = _format_source_label(chunk)
    assert label == "arXiv:1706.03762"


def test_source_label_hf_course() -> None:
    chunk = make_chunk(
        doc_id="hf-nlp-ch01",
        content_type="course_chapter",
        title="HF NLP Course",
        authors=["Hugging Face"],
    )
    label = _format_source_label(chunk)
    assert label == "Hugging Face Learn"


def test_source_label_lillog() -> None:
    chunk = make_chunk(
        doc_id="lillog-llm-agents-2023-06",
        content_type="survey_blog",
        title="LLM Powered Agents",
        authors=["Lilian Weng"],
    )
    label = _format_source_label(chunk)
    assert "Lil'Log" in label


def test_source_label_anthropic_blog() -> None:
    chunk = make_chunk(
        doc_id="anthropic-mapping-mind-2024-05",
        content_type="lab_blog_post",
        title="Mapping the Mind",
        authors=["Anthropic"],
    )
    label = _format_source_label(chunk)
    assert label == "Anthropic Blog"


# ── Citation deduplication tests ───────────────────────────────────────────────


def test_citation_deduplication_same_doc() -> None:
    """Multiple chunks from same document → single citation."""
    chunks = [
        make_chunk(
            doc_id="arxiv-1706.03762", chunk_id="chunk1", section_title="Abstract"
        ),
        make_chunk(
            doc_id="arxiv-1706.03762", chunk_id="chunk2", section_title="Section 3"
        ),
        make_chunk(
            doc_id="arxiv-1706.03762", chunk_id="chunk3", section_title="Section 4"
        ),
    ]
    citations, mapping = build_citations(chunks)
    # Should produce only 1 citation
    assert len(citations) == 1
    assert citations[0].number == 1
    # All chunks should map to citation 1
    assert mapping["chunk1"] == 1
    assert mapping["chunk2"] == 1
    assert mapping["chunk3"] == 1


def test_citation_different_docs() -> None:
    """Chunks from different documents → separate citations."""
    chunks = [
        make_chunk(doc_id="arxiv-1706.03762", chunk_id="chunk1"),
        make_chunk(
            doc_id="arxiv-1810.04805",
            chunk_id="chunk2",
            title="BERT",
            authors=["Jacob Devlin"],
        ),
    ]
    citations, mapping = build_citations(chunks)
    assert len(citations) == 2
    assert citations[0].number == 1
    assert citations[1].number == 2
    assert mapping["chunk1"] == 1
    assert mapping["chunk2"] == 2


def test_citation_ordering_preserved() -> None:
    """Citation numbers follow retrieval order (highest score first)."""
    chunks = [
        make_chunk(doc_id="arxiv-1706.03762", chunk_id="c1", score=0.95),
        make_chunk(
            doc_id="hf-nlp-ch01",
            chunk_id="c2",
            score=0.80,
            content_type="course_chapter",
            title="HF Course",
            authors=["Hugging Face"],
        ),
    ]
    citations, _ = build_citations(chunks)
    # [1] should be the highest scoring doc
    assert citations[0].doc_id == "arxiv-1706.03762"
    assert citations[1].doc_id == "hf-nlp-ch01"


# ── Context block tests ────────────────────────────────────────────────────────


def test_context_block_contains_citation_numbers() -> None:
    """Context block should have [N] markers for each chunk."""
    chunks = [
        make_chunk(chunk_id="c1"),
        make_chunk(chunk_id="c2", doc_id="arxiv-1810.04805"),
    ]
    _, mapping = build_citations(chunks)
    context = build_context_block(chunks, mapping)
    assert "[1]" in context
    assert "[2]" in context


def test_context_block_contains_text() -> None:
    """Context block should contain the actual chunk text."""
    chunks = [make_chunk(chunk_id="c1")]
    _, mapping = build_citations(chunks)
    context = build_context_block(chunks, mapping)
    assert "self-attention mechanisms" in context


# ── Citation block formatting tests ───────────────────────────────────────────


def test_format_citation_block_empty() -> None:
    result = format_citation_block([])
    assert result == ""


def test_format_citation_block_contains_title() -> None:
    chunks = [make_chunk(chunk_id="c1")]
    citations, _ = build_citations(chunks)
    block = format_citation_block(citations)
    assert "Attention Is All You Need" in block
    assert "arXiv:1706.03762" in block
    assert "https://arxiv.org/abs/1706.03762" in block


# ── Query filters tests ────────────────────────────────────────────────────────


def test_query_filters_has_filters_empty() -> None:
    f = QueryFilters()
    assert f.has_filters() is False


def test_query_filters_has_filters_with_tags() -> None:
    f = QueryFilters(tags=["RLHF", "alignment"])
    assert f.has_filters() is True


def test_query_filters_has_filters_with_year() -> None:
    f = QueryFilters(year_gte=2024)
    assert f.has_filters() is True


def test_query_filters_repr() -> None:
    f = QueryFilters(year_gte=2024, tags=["RAG"])
    r = repr(f)
    assert "2024" in r
    assert "RAG" in r


# ── Refusal logic tests ────────────────────────────────────────────────────────


def test_refusal_no_chunks() -> None:
    """Empty chunks should always refuse."""
    answer = generate_answer(query="What is attention?", chunks=[])
    assert answer.was_refused is True
    assert "don't have enough" in answer.answer.lower() or "don't have" in answer.answer


def test_refusal_low_score() -> None:
    """Chunks with score below threshold should refuse."""
    low_score_chunks = [make_chunk(score=0.05)]
    answer = generate_answer(
        query="What is attention?",
        chunks=low_score_chunks,
        similarity_threshold=0.3,
    )
    assert answer.was_refused is True


def test_no_refusal_high_score() -> None:
    """Chunks with high score should NOT refuse (mocks Gemini call)."""
    high_score_chunks = [make_chunk(score=0.95)]

    with patch(
        "research_navigator.generate.generator.ChatGoogleGenerativeAI"
    ) as mock_llm:
        mock_instance = mock_llm.return_value
        mock_instance.invoke.return_value.content = (
            "The transformer uses self-attention [1]."
        )
        answer = generate_answer(
            query="What is attention?",
            chunks=high_score_chunks,
            similarity_threshold=0.3,
        )

    assert answer.was_refused is False
    assert len(answer.citations) > 0


def test_generated_answer_format_full_response() -> None:
    """format_full_response should include answer and citation block."""
    chunks = [make_chunk(chunk_id="c1")]
    citations, _ = build_citations(chunks)
    answer = GeneratedAnswer(
        answer="The transformer uses attention [1].",
        citations=citations,
        retrieved_chunks=chunks,
        was_refused=False,
        top_score=0.9,
    )
    full = answer.format_full_response()
    assert "transformer uses attention" in full
    assert "Sources:" in full
    assert "Attention Is All You Need" in full
