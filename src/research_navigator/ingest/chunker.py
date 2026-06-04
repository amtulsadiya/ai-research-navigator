"""
Chunking strategies — converts ParsedSections into Chunks ready for embedding.

Design decisions (ADR-003):
- Chunk size: 512 tokens (sweet spot for text-embedding-004 retrieval quality)
- Overlap: 64 tokens for papers/blogs, 128 for survey blogs (heavy cross-refs)
- Code blocks: atomic units, never split mid-block
- Abstract: always its own chunk regardless of length
- References: excluded from chunking entirely (handled in pipeline)
- Tokenizer: tiktoken cl100k_base (same family as OpenAI/Google embedding models)

Why token-based over character-based splitting?
Characters vary wildly — "a" and "transformer" are both 1 character but very
different in token count. Token-based splitting ensures consistent chunk sizes
as seen by the embedding model, which matters for retrieval quality.
"""

from __future__ import annotations

import re

import structlog
import tiktoken

from research_navigator.ingest.models import (
    Chunk,
    ContentType,
    ParsedDocument,
)

logger = structlog.get_logger(__name__)

# ── Tokenizer setup ────────────────────────────────────────────────────────────

# cl100k_base is used by GPT-4, text-embedding-3, and is compatible with
# Google's embedding models for size estimation purposes.
# We use it for counting tokens, not for the actual embedding call.
_TOKENIZER = tiktoken.get_encoding("cl100k_base")

# Code block pattern — we must never split inside these
_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)


# ── Chunking parameters per content type ──────────────────────────────────────

_CHUNK_PARAMS: dict[ContentType, dict[str, int]] = {
    ContentType.ARXIV_PAPER: {
        "max_tokens": 512,
        "overlap_tokens": 64,
    },
    ContentType.COURSE_CHAPTER: {
        "max_tokens": 512,
        "overlap_tokens": 64,
    },
    ContentType.SURVEY_BLOG: {
        # Larger overlap because Lil'Log posts have heavy cross-references
        # between sections — we want boundary chunks to carry more context
        "max_tokens": 512,
        "overlap_tokens": 128,
    },
    ContentType.LAB_BLOG_POST: {
        "max_tokens": 512,
        "overlap_tokens": 64,
    },
}


# ── Token utilities ────────────────────────────────────────────────────────────


def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string."""
    # return len(_TOKENIZER.encode(text))
    return len(_TOKENIZER.encode(text, disallowed_special=()))


def _split_text_by_tokens(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """
    Split text into chunks of at most max_tokens tokens, with overlap.

    Algorithm:
    1. Encode the full text into token IDs
    2. Slide a window of max_tokens over the token IDs
    3. Each window advances by (max_tokens - overlap_tokens) positions
    4. Decode each window back to text

    The overlap ensures sentences at chunk boundaries aren't completely lost.
    Example with max=512, overlap=64:
    - Chunk 1: tokens 0..511
    - Chunk 2: tokens 448..959  (448 = 512 - 64)
    - Chunk 3: tokens 896..1407
    """
    if not text.strip():
        return []

    # token_ids = _TOKENIZER.encode(text)
    token_ids = _TOKENIZER.encode(text, disallowed_special=())

    if len(token_ids) <= max_tokens:
        return [text]  # Fits in one chunk — no splitting needed

    chunks: list[str] = []
    step = max_tokens - overlap_tokens
    start = 0

    while start < len(token_ids):
        end = min(start + max_tokens, len(token_ids))
        chunk_tokens = token_ids[start:end]
        chunk_text = _TOKENIZER.decode(chunk_tokens)
        chunks.append(chunk_text)

        if end == len(token_ids):
            break
        start += step

    return chunks


def _split_preserving_code_blocks(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """
    Split text while preserving code blocks as atomic units.

    Why this matters:
    A code block like:
    ```python
    def attention(q, k, v):
        scores = q @ k.T / sqrt(d_k)
        return softmax(scores) @ v
    ```
    Must never be split in the middle. If we token-split this naively,
    we might get "def attention(q, k, v):\n    scores = q @" as one chunk
    and "k.T / sqrt(d_k)..." as another — neither is meaningful alone.

    Algorithm:
    1. Find all code block positions using regex
    2. Split the text into: [text_segment, code_block, text_segment, ...]
    3. Token-split each text segment normally
    4. Keep each code block as its own atomic chunk (if it fits in max_tokens)
       or as a standalone oversized chunk (if it's too long — rare but possible)
    5. Reassemble in order
    """
    # Find all code blocks and their positions
    code_block_spans = [
        (m.start(), m.end(), m.group()) for m in _CODE_BLOCK_RE.finditer(text)
    ]

    if not code_block_spans:
        # No code blocks — just do normal token splitting
        return _split_text_by_tokens(text, max_tokens, overlap_tokens)

    result_chunks: list[str] = []
    last_end = 0

    for start, end, code_block in code_block_spans:
        # Split the text segment BEFORE this code block
        text_before = text[last_end:start]
        if text_before.strip():
            result_chunks.extend(
                _split_text_by_tokens(text_before, max_tokens, overlap_tokens)
            )

        # Add the code block as an atomic chunk
        # If it's longer than max_tokens, we keep it whole anyway
        # (an oversized code chunk is better than a broken one)
        if code_block.strip():
            result_chunks.append(code_block)

        last_end = end

    # Handle text after the last code block
    text_after = text[last_end:]
    if text_after.strip():
        result_chunks.extend(
            _split_text_by_tokens(text_after, max_tokens, overlap_tokens)
        )

    return result_chunks


# ── Main chunking function ─────────────────────────────────────────────────────


def chunk_document(parsed: ParsedDocument) -> list[Chunk]:
    """
    Convert a ParsedDocument into a list of Chunks ready for embedding.

    This is the main entry point called by the pipeline.

    Per-type strategy:
    - arxiv_paper: abstract gets 1 dedicated chunk; other sections token-split
    - course_chapter: section-boundary chunking with code block preservation
    - survey_blog: same as course but with larger overlap (128 tokens)
    - lab_blog_post: same as course_chapter

    References sections are SKIPPED — they were stored in ParsedDocument
    but must not become retrieval chunks (they'd pollute search results
    with citation strings instead of content).
    """
    content_type = parsed.meta.content_type
    params = _CHUNK_PARAMS[content_type]
    max_tokens = params["max_tokens"]
    overlap_tokens = params["overlap_tokens"]

    log = logger.bind(doc_id=parsed.meta.doc_id, content_type=content_type.value)

    chunks: list[Chunk] = []
    chunk_index = 0

    for section in parsed.sections:
        # Skip references — not for retrieval
        if section.is_references:
            log.debug(
                "skipping_references_section", section_title=section.section_title
            )
            continue

        if not section.text.strip():
            continue

        # Abstract: always one dedicated chunk, regardless of length
        # Why? The abstract is a self-contained summary. It's the most
        # important chunk for "what is this paper about?" queries.
        if section.is_abstract:
            chunks.append(
                Chunk(
                    doc_id=parsed.meta.doc_id,
                    chunk_index=chunk_index,
                    section_index=section.section_index,
                    section_title=section.section_title,
                    text=section.text.strip(),
                    token_count=count_tokens(section.text),
                    is_abstract=True,
                )
            )
            chunk_index += 1
            continue

        # All other sections: split preserving code blocks
        text_splits = _split_preserving_code_blocks(
            section.text,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )

        for split_text in text_splits:
            if not split_text.strip():
                continue
            if count_tokens(split_text.strip()) < 20:  # Skip chunks under 20 tokens
                continue

            chunks.append(
                Chunk(
                    doc_id=parsed.meta.doc_id,
                    chunk_index=chunk_index,
                    section_index=section.section_index,
                    section_title=section.section_title,
                    text=split_text.strip(),
                    token_count=count_tokens(split_text),
                    is_abstract=False,
                )
            )
            chunk_index += 1

    log.info(
        "chunking_done",
        chunk_count=len(chunks),
        section_count=len(parsed.sections),
    )
    return chunks
