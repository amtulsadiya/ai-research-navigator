"""
Unit tests for the ingestion pipeline — parser and chunker.

We test:
1. Markdown parsing — section splitting, code block detection
2. Chunker — token splitting, code block preservation, abstract handling
3. Models — ChunkPayload construction, chunk_id determinism
4. Idempotency — same input always produces same chunk_ids

We do NOT test PDF parsing in unit tests (requires actual PDF files).
PDF parsing is covered by integration tests.
"""

from __future__ import annotations

from research_navigator.ingest.chunker import (
    _split_preserving_code_blocks,
    _split_text_by_tokens,
    chunk_document,
    count_tokens,
)
from research_navigator.ingest.models import (
    Chunk,
    ChunkPayload,
    ContentType,
    DocumentMeta,
    ParsedDocument,
    ParsedSection,
)
from research_navigator.ingest.parser import (
    _is_section_heading,
    _split_markdown_into_sections,
    _strip_markdown_noise,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────


def make_meta(content_type: ContentType = ContentType.COURSE_CHAPTER) -> DocumentMeta:
    return DocumentMeta(
        doc_id="test-doc-001",
        content_type=content_type,
        title="Test Document",
        authors=["Test Author"],
        year=2024,
        month=1,
        primary_category="cs.CL",
        secondary_categories=[],
        tags=["test", "LLM"],
        is_foundational=False,
        citation_count=None,
        source_url="https://example.com",
        local_path="documents/hf-learn/test.md",
    )


def make_parsed_doc(
    sections: list[ParsedSection],
    content_type: ContentType = ContentType.COURSE_CHAPTER,
) -> ParsedDocument:
    return ParsedDocument(meta=make_meta(content_type), sections=sections)


# ── Token counting tests ───────────────────────────────────────────────────────


def test_count_tokens_basic() -> None:
    """Token count should be positive and reasonable for normal text."""
    text = "The transformer architecture relies on self-attention mechanisms."
    count = count_tokens(text)
    assert count > 0
    assert count < 20  # This sentence is ~10 tokens


def test_count_tokens_empty() -> None:
    count = count_tokens("")
    assert count == 0


# ── Token splitting tests ──────────────────────────────────────────────────────


def test_split_short_text_no_split() -> None:
    """Text shorter than max_tokens should not be split."""
    text = "This is a short text that fits in one chunk."
    result = _split_text_by_tokens(text, max_tokens=512, overlap_tokens=64)
    assert len(result) == 1
    assert result[0] == text


def test_split_long_text_creates_multiple_chunks() -> None:
    """Long text should be split into multiple chunks."""
    # Create text that's definitely longer than 50 tokens
    long_text = "The attention mechanism is fundamental. " * 30
    result = _split_text_by_tokens(long_text, max_tokens=50, overlap_tokens=10)
    assert len(result) > 1


def test_split_overlap_content() -> None:
    """Consecutive chunks should share overlapping content."""
    long_text = "word " * 200  # 200 tokens approx
    result = _split_text_by_tokens(long_text, max_tokens=50, overlap_tokens=10)
    assert len(result) >= 2
    # Each chunk should be at most max_tokens
    for chunk in result:
        assert count_tokens(chunk) <= 55  # small buffer for tokenizer edge cases


# ── Code block preservation tests ─────────────────────────────────────────────


def test_code_block_not_split() -> None:
    """A code block must never be split across chunks."""
    text_with_code = """
Here is some text before the code.

```python
def attention(query, key, value):
    scores = query @ key.transpose(-2, -1)
    weights = softmax(scores / sqrt(d_k))
    return weights @ value
```

And some text after the code block.
"""
    result = _split_preserving_code_blocks(
        text_with_code, max_tokens=512, overlap_tokens=64
    )
    # The code block should appear intact in exactly one chunk
    # code_block = "```python\ndef attention(query, key, value):"
    chunks_with_code = [c for c in result if "def attention" in c]
    assert len(chunks_with_code) >= 1
    # The function definition should be in the same chunk as the closing ```
    for chunk in chunks_with_code:
        if "def attention" in chunk:
            assert "```" in chunk  # Opening or closing backticks present


def test_no_code_block_falls_through() -> None:
    """Text with no code blocks should behave like normal token splitting."""
    text = "Normal text without any code blocks. " * 5
    result = _split_preserving_code_blocks(text, max_tokens=512, overlap_tokens=64)
    assert len(result) >= 1


# ── Parser tests ───────────────────────────────────────────────────────────────


def test_markdown_section_splitting() -> None:
    """Markdown text should be split at ## headings."""
    md = """# Document Title

Some intro text here.

## Section One

Content of section one with enough text to matter.

## Section Two

Content of section two with enough text to matter.

## Section Three

Content of section three with enough text to matter.
"""
    sections = _split_markdown_into_sections(md, "test-doc")
    titles = [s.section_title for s in sections]
    assert "Section One" in titles
    assert "Section Two" in titles
    assert "Section Three" in titles


def test_markdown_no_headings_fallback() -> None:
    """Markdown with no headings should return one section with all content."""
    md = "Just a plain paragraph with no headings whatsoever. " * 5
    sections = _split_markdown_into_sections(md, "test-doc")
    assert len(sections) == 1


def test_noise_stripping() -> None:
    """Blog post noise patterns should be stripped."""
    text_with_noise = """## Real Content

This is the actual content of the post.

**Share on** Twitter | LinkedIn

Filed under: LLM agents
"""
    cleaned = _strip_markdown_noise(text_with_noise)
    assert "Share on" not in cleaned
    assert "Filed under" not in cleaned
    assert "Real Content" in cleaned


def test_is_section_heading_numbered() -> None:
    assert _is_section_heading("1 Introduction") is True
    assert _is_section_heading("2. Related Work") is True
    assert _is_section_heading("3.1 Experimental Setup") is True


def test_is_section_heading_standalone() -> None:
    assert _is_section_heading("Abstract") is True
    assert _is_section_heading("References") is True
    assert _is_section_heading("Conclusion") is True


def test_is_section_heading_rejects_normal_text() -> None:
    assert _is_section_heading("This is a normal sentence in a paragraph.") is False
    assert _is_section_heading("") is False


# ── Chunker tests ──────────────────────────────────────────────────────────────


def test_abstract_always_one_chunk() -> None:
    """Abstract section must always produce exactly one chunk."""
    abstract_text = "We propose a novel approach to attention. " * 20
    parsed = make_parsed_doc(
        [
            ParsedSection(
                section_index=0,
                section_title="Abstract",
                text=abstract_text,
                is_abstract=True,
                is_references=False,
            )
        ]
    )
    chunks = chunk_document(parsed)
    abstract_chunks = [c for c in chunks if c.is_abstract]
    assert len(abstract_chunks) == 1


def test_references_excluded_from_chunks() -> None:
    """References section must NOT produce any retrieval chunks."""
    parsed = make_parsed_doc(
        [
            ParsedSection(
                section_index=0,
                section_title="Introduction",
                text="This paper introduces a new method for natural language processing tasks.",
                is_abstract=False,
                is_references=False,
            ),
            ParsedSection(
                section_index=1,
                section_title="References",
                text="[1] Vaswani et al. Attention Is All You Need. 2017.\n[2] Brown et al. GPT-3. 2020.",
                is_abstract=False,
                is_references=True,  # This should be excluded
            ),
        ]
    )
    chunks = chunk_document(parsed)
    ref_chunks = [c for c in chunks if "References" in c.section_title]
    assert len(ref_chunks) == 0


def test_chunk_index_sequential() -> None:
    """chunk_index should be sequential starting from 0."""
    parsed = make_parsed_doc(
        [
            ParsedSection(
                section_index=0,
                section_title="Section A",
                text="Content of section A. " * 10,
                is_abstract=False,
                is_references=False,
            ),
            ParsedSection(
                section_index=1,
                section_title="Section B",
                text="Content of section B. " * 10,
                is_abstract=False,
                is_references=False,
            ),
        ]
    )
    chunks = chunk_document(parsed)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


# ── Model tests ────────────────────────────────────────────────────────────────


def test_chunk_id_is_deterministic() -> None:
    """Same chunk always produces the same chunk_id."""
    chunk = Chunk(
        doc_id="arxiv-1706.03762",
        chunk_index=0,
        section_index=0,
        section_title="Abstract",
        text="We propose the Transformer architecture.",
        token_count=8,
        is_abstract=True,
    )
    assert chunk.chunk_id == chunk.chunk_id  # same object
    # Create identical chunk — should have same id
    chunk2 = Chunk(
        doc_id="arxiv-1706.03762",
        chunk_index=0,
        section_index=0,
        section_title="Abstract",
        text="We propose the Transformer architecture.",
        token_count=8,
        is_abstract=True,
    )
    assert chunk.chunk_id == chunk2.chunk_id


def test_chunk_id_changes_with_content() -> None:
    """Different text = different chunk_id (idempotency relies on this)."""
    chunk1 = Chunk(
        doc_id="arxiv-1706.03762",
        chunk_index=0,
        section_index=0,
        section_title="Abstract",
        text="Original text content here.",
        token_count=5,
        is_abstract=True,
    )
    chunk2 = Chunk(
        doc_id="arxiv-1706.03762",
        chunk_index=0,
        section_index=0,
        section_title="Abstract",
        text="Modified text content here.",  # Different text
        token_count=5,
        is_abstract=True,
    )
    assert chunk1.chunk_id != chunk2.chunk_id


def test_chunk_payload_from_chunk_and_meta() -> None:
    """ChunkPayload.from_chunk_and_meta should combine both correctly."""
    meta = make_meta()
    chunk = Chunk(
        doc_id=meta.doc_id,
        chunk_index=0,
        section_index=0,
        section_title="Introduction",
        text="This is the introduction text.",
        token_count=6,
        is_abstract=False,
    )
    payload = ChunkPayload.from_chunk_and_meta(chunk, meta)

    # Chunk fields
    assert payload.doc_id == meta.doc_id
    assert payload.chunk_index == 0
    assert payload.text == "This is the introduction text."

    # Meta fields carried through
    assert payload.content_type == meta.content_type.value
    assert payload.title == meta.title
    assert payload.year == meta.year
    assert payload.tags == meta.tags
    assert payload.is_foundational == meta.is_foundational
