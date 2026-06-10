# ADR-003: Chunking Strategy — Section-Aware Token-Based Splitting

## Status
Accepted

## Date
2026-06-04

## Context
Chunking is the most consequential design decision in the RAG pipeline. Chunk size and boundaries directly determine retrieval quality — too large and chunks contain irrelevant content, too small and chunks lose meaning.

We have 4 content types with different structural characteristics, warranting different strategies.

## Options considered

| Strategy | Pros | Cons |
|---|---|---|
| Fixed character split | Simple | Cuts mid-sentence, mid-word |
| Fixed token split | Consistent model input size | Ignores document structure |
| Sentence-based split | Natural boundaries | Variable chunk sizes, hard to control |
| Section-boundary split | Preserves semantic units | Sections vary wildly in length |
| Semantic chunking | Intelligent topic boundaries | 30x slower, needs embedding during ingest |
| **Section-aware token split** | Structure + consistent size | More complex to implement |

## Decision
**Section-aware token-based splitting with per-content-type parameters.**

Abstracts always get a dedicated chunk regardless of length.
Code blocks are never split mid-block.
References sections are excluded from retrieval entirely.

## Parameters by content type

| Content Type | Max Tokens | Overlap | Rationale |
|---|---|---|---|
| arxiv_paper | 512 | 64 | Standard RAG sweet spot |
| course_chapter | 512 | 64 | Self-contained sections |
| survey_blog | 512 | 128 | Heavy cross-references need larger overlap |
| lab_blog_post | 512 | 64 | Shorter, less cross-referential |

## Why 512 tokens
- Established by RAG research as the sweet spot for retrieval quality
- Large enough to contain a complete thought or explanation
- Small enough to be topically specific
- Compatible with bge-m3 (which handles up to 8192 tokens but performs best on shorter inputs)

## Why abstract gets a dedicated chunk
The abstract is a self-contained summary of the entire paper. Queries like "what is this paper about?" retrieve the abstract perfectly. Mixing the abstract into the first section destroys this retrieval advantage.

## Why 64-token overlap
Sentences at chunk boundaries are cut in half without overlap. 64 tokens (approximately 2-3 sentences) ensures no sentence is completely lost at a boundary. The 128-token overlap for survey blogs handles their heavier cross-referential structure.

## Why code blocks are atomic
A split code block is semantically meaningless:
```python
# Chunk 1 ends with:
def attention(query, key,

# Chunk 2 starts with:
value):
```
Neither chunk is useful in isolation. Code blocks are preserved as complete units even if they exceed max_tokens.

## Tokenizer
tiktoken cl100k_base — same family as the embedding model. Ensures our token counts accurately reflect what the model sees. The `disallowed_special=()` flag is required to handle educational content containing special tokens like `<|endoftext|>`.

## Known limitations
- Minimum 20-token filter removes tiny chunks but 0.6% remain after current re-ingestion run
- Some Markdown course chapters produce single-line code chunks that slip through
