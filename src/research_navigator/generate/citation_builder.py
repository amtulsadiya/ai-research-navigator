"""
Citation builder — converts retrieved chunks into formatted citation blocks.

This module handles:
1. Deduplication: multiple chunks from same document → single citation entry
2. Formatting: builds Perplexity-style citation blocks
3. Context building: numbered chunk context for the LLM prompt

Why deduplicate?
If we retrieve chunks [0,1,2] all from "Attention Is All You Need",
the answer shouldn't have [1][2][3] all pointing to the same paper.
It should have [1] pointing to that paper, referencing the most relevant section.

Citation format (Perplexity-style):
[1] Vaswani et al. "Attention Is All You Need" (2017)
    arXiv:1706.03762 · Section: 3.2 Scaled Dot-Product Attention
    https://arxiv.org/abs/1706.03762
"""

from __future__ import annotations

from dataclasses import dataclass

from research_navigator.retrieve.retriever import RetrievedChunk


@dataclass
class Citation:
    """
    A single citation entry in the answer's reference list.
    One Citation can represent multiple chunks from the same document
    (after deduplication).
    """

    number: int  # [1], [2], [3]...
    doc_id: str
    title: str
    authors_formatted: str  # "Vaswani et al." or "Vaswani & Shazeer" for 2 authors
    year: int
    source_label: str  # "arXiv:1706.03762", "Lil'Log", "Hugging Face Learn"
    section_title: str  # Most relevant section from retrieved chunks
    source_url: str
    content_type: str

    def format_inline(self) -> str:
        """Returns the inline marker: [1]"""
        return f"[{self.number}]"

    def format_full(self) -> str:
        """
        Returns the full citation block entry.

        Example:
        [1] Vaswani et al. "Attention Is All You Need" (2017)
            arXiv:1706.03762 · Section: 3.2 Scaled Dot-Product Attention
            https://arxiv.org/abs/1706.03762
        """
        lines = [
            f'[{self.number}] {self.authors_formatted} "{self.title}" ({self.year})',
            f"    {self.source_label} · Section: {self.section_title}",
            f"    {self.source_url}",
        ]
        return "\n".join(lines)


def _format_authors(authors: list[str], content_type: str) -> str:
    """
    Format author list following citation conventions.

    Rules:
    - 1 author: "Vaswani"
    - 2 authors: "Vaswani & Shazeer"
    - 3+ authors: "Vaswani et al."
    - Non-paper content: use as-is (e.g. "Hugging Face", "Lilian Weng")
    """
    if not authors:
        return "Unknown"

    if content_type in ("course_chapter", "survey_blog", "lab_blog_post"):
        return authors[0] if authors else "Unknown"

    if len(authors) == 1:
        # Last name only
        return authors[0].split()[-1]
    elif len(authors) == 2:
        return f"{authors[0].split()[-1]} & {authors[1].split()[-1]}"
    else:
        return f"{authors[0].split()[-1]} et al."


def _format_source_label(chunk: RetrievedChunk) -> str:
    """
    Build the source label for the citation.

    Examples:
    - arxiv_paper: "arXiv:1706.03762"
    - course_chapter: "Hugging Face Learn"
    - survey_blog: "Lil'Log"
    - lab_blog_post: "Anthropic Blog" / "OpenAI Blog" / "DeepMind Blog"
    """
    if chunk.content_type == "arxiv_paper":
        arxiv_id = chunk.doc_id.removeprefix("arxiv-")
        return f"arXiv:{arxiv_id}"
    elif chunk.content_type == "course_chapter":
        return "Hugging Face Learn"
    elif chunk.content_type == "survey_blog":
        return "Lil'Log (Lilian Weng)"
    elif chunk.content_type == "lab_blog_post":
        # Derive lab from doc_id prefix
        if chunk.doc_id.startswith("anthropic"):
            return "Anthropic Blog"
        elif chunk.doc_id.startswith("openai"):
            return "OpenAI Blog"
        elif chunk.doc_id.startswith("deepmind"):
            return "DeepMind Blog"
        else:
            return "Lab Blog"
    return "Unknown Source"


def build_citations(
    chunks: list[RetrievedChunk],
) -> tuple[list[Citation], dict[str, int]]:
    """
    Build citation list from retrieved chunks with deduplication.

    Deduplication logic:
    - Group chunks by doc_id
    - Each unique document gets ONE citation number
    - The section_title used is from the highest-scoring chunk for that doc
    - Returns: (citations list, chunk_id → citation_number mapping)

    The chunk_id → citation_number mapping is used by the generator
    to know which [N] to assign to each chunk in the prompt.

    Example:
    Retrieved: [chunk_A from paper1, chunk_B from paper1, chunk_C from paper2]
    Result citations: [[1] paper1, [2] paper2]
    Mapping: {chunk_A.id: 1, chunk_B.id: 1, chunk_C.id: 2}
    """
    citations: list[Citation] = []
    chunk_to_citation: dict[str, int] = {}

    # Track which doc_ids we've already created citations for
    doc_id_to_citation_num: dict[str, int] = {}
    citation_num = 1

    for chunk in chunks:
        if chunk.doc_id not in doc_id_to_citation_num:
            # New document — create a citation entry
            citation = Citation(
                number=citation_num,
                doc_id=chunk.doc_id,
                title=chunk.title,
                authors_formatted=_format_authors(chunk.authors, chunk.content_type),
                year=chunk.year,
                source_label=_format_source_label(chunk),
                section_title=chunk.section_title,  # Most relevant (highest score) section
                source_url=chunk.source_url,
                content_type=chunk.content_type,
            )
            citations.append(citation)
            doc_id_to_citation_num[chunk.doc_id] = citation_num
            citation_num += 1

        # Map this chunk to its citation number
        chunk_to_citation[chunk.chunk_id] = doc_id_to_citation_num[chunk.doc_id]

    return citations, chunk_to_citation


def build_context_block(
    chunks: list[RetrievedChunk],
    chunk_to_citation: dict[str, int],
) -> str:
    """
    Build the numbered context block sent to the LLM.

    Format:
    [1] (from "Attention Is All You Need", Section: 3.2)
    <chunk text here>

    [1] (from "Attention Is All You Need", Section: Abstract)
    <another chunk from same paper>

    [2] (from "HF NLP Course Chapter 1", Section: Self-Attention)
    <chunk text here>

    Why include the citation number AND source in context?
    The LLM needs to know which [N] to use when citing a claim.
    Including the source helps the LLM understand context.
    """
    blocks: list[str] = []

    for chunk in chunks:
        cite_num = chunk_to_citation.get(chunk.chunk_id, 0)
        header = f'[{cite_num}] (from "{chunk.title}", Section: {chunk.section_title})'
        blocks.append(f"{header}\n{chunk.text}")

    return "\n\n".join(blocks)


def format_citation_block(citations: list[Citation]) -> str:
    """
    Format the full citation block appended to the answer.

    Example output:
    ---
    Sources:
    [1] Vaswani et al. "Attention Is All You Need" (2017)
        arXiv:1706.03762 · Section: 3.2 Scaled Dot-Product Attention
        https://arxiv.org/abs/1706.03762

    [2] Hugging Face "HF NLP Course — Chapter 1" (2023)
        Hugging Face Learn · Section: Self-Attention
        https://huggingface.co/learn/nlp-course/chapter1
    """
    if not citations:
        return ""

    lines = ["---", "**Sources:**"]
    for citation in citations:
        lines.append("")
        lines.append(citation.format_full())

    return "\n".join(lines)
