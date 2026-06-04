"""
Document parsers — one per content type.
Responsibility: take a raw file (PDF or Markdown) and return a ParsedDocument
with sections extracted, headings identified, and references isolated.
Design decisions documented here (ADR-002):
1. pymupdf for PDFs — fastest, good quality, handles most arXiv layouts
2. Regex-based Markdown splitter — no heavy library needed, headings are the only
   structure we care about
3. References section excluded from retrieval chunks but preserved in payload
4. Abstract always gets its own dedicated section (section_index=0)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from research_navigator.ingest.models import (
    ContentType,
    DocumentMeta,
    ParsedDocument,
    ParsedSection,
)

if TYPE_CHECKING:
    import fitz

logger = structlog.get_logger(__name__)
# ── Heading patterns ───────────────────────────────────────────────────────────

# Matches common academic paper section headings:
# "1 Introduction", "2. Related Work", "A Appendix", "Abstract"
_ARXIV_HEADING_RE = re.compile(
    r"^(?:\d+\.?\s+|[A-Z]\.\s+)?([A-Z][A-Za-z\s\-:]{2,60})$",
    re.MULTILINE,
)

# Matches Markdown headings at ## or ### level
# We use ## as primary split boundary, ### as secondary
_MD_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

# Common reference section title variants
_REFERENCES_TITLES = {
    "references",
    "bibliography",
    "works cited",
    "reference",
    "citations",
    "literature cited",
}

# Noise patterns in blog posts (share buttons, navigation, footer links)
_NOISE_PATTERNS = [
    re.compile(r"\*\*Share\s+on\*\*.*", re.IGNORECASE | re.DOTALL),
    re.compile(r"Filed under:.*", re.IGNORECASE | re.DOTALL),
    re.compile(r"Tags:.*", re.IGNORECASE | re.DOTALL),
    re.compile(r"\[Tweet\].*", re.IGNORECASE | re.DOTALL),
    re.compile(r"← Previous.*", re.IGNORECASE | re.DOTALL),
    re.compile(r"Next →.*", re.IGNORECASE | re.DOTALL),
    re.compile(r"©\s+\d{4}.*", re.IGNORECASE | re.DOTALL),
]


# ── PDF Parser ─────────────────────────────────────────────────────────────────


def parse_pdf(meta: DocumentMeta) -> ParsedDocument:
    """
    Parse an arXiv PDF into sections using pymupdf.

    Strategy:
    1. Extract all text using pymupdf with sort=True (handles multi-column)
    2. Detect the abstract — always the first section
    3. Split remaining text at section headings using regex
    4. Detect and isolate the references section
    5. Return ParsedDocument with ordered sections

    Known limitations (documented in ADR-002):
    - Multi-column PDFs: sort=True helps but isn't perfect for all layouts
    - Figures/tables: text inside them is skipped (pymupdf reads text, not images)
    - Footnotes: small font footnotes sometimes merge with main body text
    - Equations: rendered as LaTeX-like text, often garbled
    We accept these limitations because:
    a) The semantic content is still retrievable from surrounding text
    b) The alternative (unstructured) adds significant complexity and system deps
    """
    try:
        import fitz  # pymupdf
    except ImportError as e:
        raise ImportError(
            "pymupdf is required for PDF parsing. Run: uv add pymupdf"
        ) from e

    corpus_root = Path("corpus")
    pdf_path = corpus_root / meta.local_path

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    log = logger.bind(doc_id=meta.doc_id, path=str(pdf_path))
    log.info("parsing_pdf_start")

    doc = fitz.open(str(pdf_path))
    full_text = _extract_pdf_text(doc)
    doc.close()

    if not full_text.strip():
        log.warning("pdf_empty_text")
        return ParsedDocument(meta=meta, sections=[], references_text=None)

    sections, references_text = _split_pdf_into_sections(full_text, meta.doc_id)

    log.info(
        "parsing_pdf_done",
        section_count=len(sections),
        has_references=references_text is not None,
    )
    return ParsedDocument(meta=meta, sections=sections, references_text=references_text)


def _extract_pdf_text(doc: fitz.Document) -> str:
    """
    Extract text from all pages.

    sort=True tells pymupdf to sort text blocks by their (y, x) position
    before reading. This is critical for multi-column layouts — without it,
    text from column 2 gets interleaved with column 1 mid-sentence.

    We still get some noise from headers/footers (page numbers, running titles)
    but they're short and won't significantly affect chunking quality.
    """
    pages: list[str] = []
    for page in doc:
        # sort=True: critical for multi-column layouts
        text = page.get_text("text", sort=True)
        if text.strip():
            pages.append(text)
    return "\n".join(pages)


def _split_pdf_into_sections(
    text: str, doc_id: str
) -> tuple[list[ParsedSection], str | None]:
    """
    Split extracted PDF text into sections.

    We detect sections by:
    1. Looking for "Abstract" near the top → dedicated abstract section
    2. Looking for numbered headings like "1 Introduction", "2 Related Work"
    3. Detecting the references section by its title

    This is heuristic — academic PDFs have no standard heading format.
    We handle the most common patterns from arXiv cs.CL/cs.LG papers.
    """
    sections: list[ParsedSection] = []
    references_text: str | None = None

    # --- Step 1: Extract abstract ---
    abstract_text, remaining_text = _extract_abstract(text)
    if abstract_text:
        sections.append(
            ParsedSection(
                section_index=0,
                section_title="Abstract",
                text=abstract_text.strip(),
                is_abstract=True,
                is_references=False,
            )
        )

    # --- Step 2: Split remaining text at heading boundaries ---
    # Find all lines that look like section headings
    lines = remaining_text.split("\n")
    current_title = "Introduction"
    current_lines: list[str] = []
    section_idx = 1 if abstract_text else 0

    for line in lines:
        stripped = line.strip()
        if _is_section_heading(stripped):
            # Save accumulated text as a section
            body = "\n".join(current_lines).strip()
            if body and len(body) > 50:  # skip tiny sections (page headers etc)
                # Check if this is the references section
                if current_title.lower() in _REFERENCES_TITLES:
                    references_text = body
                else:
                    sections.append(
                        ParsedSection(
                            section_index=section_idx,
                            section_title=current_title,
                            text=body,
                            is_abstract=False,
                            is_references=False,
                        )
                    )
                    section_idx += 1
            current_title = stripped
            current_lines = []
        else:
            current_lines.append(line)

    # Don't forget the last accumulated section
    body = "\n".join(current_lines).strip()
    if body and len(body) > 50:
        if current_title.lower() in _REFERENCES_TITLES:
            references_text = body
        else:
            sections.append(
                ParsedSection(
                    section_index=section_idx,
                    section_title=current_title,
                    text=body,
                    is_abstract=False,
                    is_references=False,
                )
            )

    # Fallback: if no sections detected, treat entire text as one section
    if not sections:
        logger.warning("pdf_no_sections_detected", doc_id=doc_id)
        sections.append(
            ParsedSection(
                section_index=0,
                section_title="Full Text",
                text=remaining_text.strip(),
                is_abstract=False,
                is_references=False,
            )
        )

    return sections, references_text


def _extract_abstract(text: str) -> tuple[str, str]:
    """
    Extract the abstract from PDF text.
    Returns (abstract_text, remaining_text).

    Looks for "Abstract" keyword followed by the abstract body.
    The abstract ends at the first numbered section heading.
    """
    # Find "Abstract" (case-insensitive) near the start of the document
    abstract_match = re.search(
        r"\bAbstract\b[.\s—–-]*\n?(.*?)(?=\n\s*(?:\d+\.?\s+[A-Z]|1\s+Introduction|Introduction\s*\n))",
        text[:5000],  # Only look in the first 5000 chars
        re.IGNORECASE | re.DOTALL,
    )
    if abstract_match:
        abstract_text = abstract_match.group(1).strip()
        # Remove the abstract from the remaining text
        remaining = text[abstract_match.end() :]
        return abstract_text, remaining

    return "", text


def _is_section_heading(line: str) -> bool:
    if not line or len(line) > 80:
        return False

    # All-caps headings like "REFERENCES", "INTRODUCTION"
    if line.isupper() and 3 < len(line) < 50:
        return True

    """
    Heuristic: is this line a section heading?

    Checks for patterns like:
    - "1 Introduction"
    - "2. Related Work"
    - "3.1 Experimental Setup"
    - "A Appendix"
    - "References"
    - "Conclusion"

    We're conservative — better to miss a heading than to split mid-paragraph.
    """
    if not line or len(line) > 80:  # headings are short
        return False

    # Numbered section: "1 Introduction", "2.1 Methods"
    if re.match(r"^\d+\.?\d*\s+[A-Z]", line):
        return True

    # Appendix: "A Additional Results"
    if re.match(r"^[A-Z]\.\s+[A-Z]", line):
        return True

    # Known standalone headings
    standalone = {
        "abstract",
        "introduction",
        "conclusion",
        "conclusions",
        "references",
        "bibliography",
        "acknowledgements",
        "acknowledgments",
        "related work",
        "background",
        "methodology",
        "experiments",
        "results",
        "discussion",
        "appendix",
        # Add these:
        "references.",
        "bibliography.",
        "works cited",
        "literature cited",
        "citations",
    }
    # if line.lower().strip(".") in standalone:
    #     return True

    # return False
    return line.lower().strip(".") in standalone


# ── Markdown Parser ────────────────────────────────────────────────────────────


def parse_markdown(meta: DocumentMeta) -> ParsedDocument:
    """
    Parse a Markdown file into sections.

    Works for: course_chapter, survey_blog, lab_blog_post

    Strategy:
    1. Read the file
    2. Strip noise (navigation/footer cruft in blog posts)
    3. Split at ## headings — each ## becomes a section
    4. Preserve code blocks as atomic units (never split them)
    5. Return ParsedDocument

    Why split at ## and not #?
    The # heading is the document title — splitting there gives us one huge
    section. ## is the first meaningful structural boundary.
    """
    corpus_root = Path("corpus")
    md_path = corpus_root / meta.local_path

    if not md_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {md_path}")

    log = logger.bind(doc_id=meta.doc_id, path=str(md_path))
    log.info("parsing_markdown_start")

    text = md_path.read_text(encoding="utf-8")

    # Strip noise from blog posts
    if meta.content_type in (ContentType.SURVEY_BLOG, ContentType.LAB_BLOG_POST):
        text = _strip_markdown_noise(text)

    sections = _split_markdown_into_sections(text, meta.doc_id)

    log.info("parsing_markdown_done", section_count=len(sections))
    return ParsedDocument(meta=meta, sections=sections, references_text=None)


def _strip_markdown_noise(text: str) -> str:
    """
    Remove footer/navigation noise from blog posts.

    The README warns: Lil'Log posts have share buttons, tag links,
    and navigation cruft at the end. We strip these because:
    - They're not content — they're website chrome
    - If chunked, they'd produce irrelevant retrieval results
    - Example noise: "Share on Twitter | Filed under: LLM, agents"

    Strategy: apply regex patterns that match known noise markers.
    We strip from the first noise match to the end of the file.
    """
    for pattern in _NOISE_PATTERNS:
        match = pattern.search(text)
        if match:
            text = text[: match.start()].rstrip()
            break  # Stop at first noise marker — rest is also noise
    return text


def _split_markdown_into_sections(text: str, doc_id: str) -> list[ParsedSection]:
    """
    Split Markdown text at ## headings.

    Algorithm:
    1. Find all ## heading positions using regex
    2. Slice the text between consecutive headings
    3. Each slice = one section

    Code blocks (``` ... ```) are never split — we treat them as atomic.
    The token-based splitting in chunker.py respects code block boundaries.

    Why not split at ### as well?
    ### headings are often very short subsections (single paragraphs).
    Splitting there would create too many tiny chunks that lose context.
    We use ## as the primary boundary and let the chunker handle long sections.
    """
    sections: list[ParsedSection] = []

    # Find all heading positions: (position_in_text, level, heading_text)
    heading_matches = list(_MD_HEADING_RE.finditer(text))

    if not heading_matches:
        # No headings found — treat entire text as one section
        logger.warning("markdown_no_headings", doc_id=doc_id)
        cleaned = text.strip()
        if cleaned:
            sections.append(
                ParsedSection(
                    section_index=0,
                    section_title="Content",
                    text=cleaned,
                    is_abstract=False,
                    is_references=False,
                )
            )
        return sections

    # Extract text before the first heading (preamble/intro text)
    first_heading_pos = heading_matches[0].start()
    preamble = text[:first_heading_pos].strip()
    section_idx = 0

    if preamble and len(preamble) > 50:
        sections.append(
            ParsedSection(
                section_index=section_idx,
                section_title="Introduction",
                text=preamble,
                is_abstract=False,
                is_references=False,
            )
        )
        section_idx += 1

    # Slice between consecutive headings
    for i, match in enumerate(heading_matches):
        heading_level = len(match.group(1))  # number of # chars
        heading_text = match.group(2).strip()

        # Only use ## level as primary section boundaries
        # ### headings stay inside their parent section
        if heading_level > 2:
            continue

        # Text for this section = from end of this heading to start of next ## heading
        start = match.end()
        # Find the next ## heading (not ###)
        end = len(text)
        for next_match in heading_matches[i + 1 :]:
            if len(next_match.group(1)) <= 2:  # ## or #
                end = next_match.start()
                break

        section_text = text[start:end].strip()

        if not section_text or len(section_text) < 20:
            continue  # Skip empty or near-empty sections

        sections.append(
            ParsedSection(
                section_index=section_idx,
                section_title=heading_text,
                text=section_text,
                is_abstract=False,
                is_references=False,
            )
        )
        section_idx += 1

    # Fallback
    if not sections:
        logger.warning("markdown_no_sections_extracted", doc_id=doc_id)
        sections.append(
            ParsedSection(
                section_index=0,
                section_title="Content",
                text=text.strip(),
                is_abstract=False,
                is_references=False,
            )
        )

    return sections


# ── Public dispatcher ──────────────────────────────────────────────────────────


def parse_document(meta: DocumentMeta) -> ParsedDocument:
    """
    Route to the correct parser based on content_type.
    This is the single entry point used by the pipeline.
    """
    if meta.content_type == ContentType.ARXIV_PAPER:
        return parse_pdf(meta)
    elif meta.content_type in (
        ContentType.COURSE_CHAPTER,
        ContentType.SURVEY_BLOG,
        ContentType.LAB_BLOG_POST,
    ):
        return parse_markdown(meta)
    else:
        raise ValueError(f"Unknown content type: {meta.content_type}")
