# ADR-002: PDF Parsing Strategy — pymupdf with Heuristic Section Detection

## Status
Accepted

## Date
2026-06-04

## Context
30 of our 50 corpus documents are arXiv PDFs. We need to extract clean text and section structure from them. Academic PDFs have well-known practical difficulties:
- Multi-column layouts (text from column 2 interleaves with column 1)
- References sections (must be excluded from retrieval)
- Figures and tables (not extractable as meaningful text)
- Footnotes (appear mid-sentence in text stream)
- Equations (rendered as garbled text)
- Non-standard heading formats (no single standard exists)

## Options considered

| Library | Multi-column | Quality | Speed | System deps |
|---|---|---|---|---|
| pymupdf | Handles with sort=True | Good | Very fast | None |
| pdfplumber | Better | Very good | Slow | None |
| unstructured | Excellent | Excellent | Very slow | Many |
| pdfminer | Poor | Good | Slow | None |
| pypdf2 | Poor | Poor | Fast | None |

## Decision
**pymupdf with `sort=True` flag and heuristic section heading detection.**

## Rationale
- pymupdf is the fastest PDF library in Python — processes 30 papers in seconds
- `sort=True` sorts text blocks by (y, x) position before reading, which handles most multi-column layouts correctly
- No system dependencies — works on all platforms including Windows
- Already widely used for academic PDF processing

## Implementation details
- Abstract extracted separately using regex pattern matching near document start — always becomes chunk_index=0
- Section headings detected by pattern: numbered ("1 Introduction"), appendix ("A Appendix"), or known standalone names ("Abstract", "References", "Conclusion")
- References section detected by title and excluded from retrieval chunks — stored separately in `ParsedDocument.references_text` for citation lookup
- Figures/tables: skipped silently (text inside them is not extracted)
- Footnotes: mixed into body text — accepted as known limitation

## Known limitations
- Some references sections not detected when using non-standard titles or no title
- Table cells sometimes parsed as section titles (e.g. "0.45 Med wrt 20 samples")
- Multi-column handling is imperfect for unusual layouts
- These are documented as known limitations in eval/report.md

## Consequences
- Fast ingestion (seconds per PDF)
- Some references leak into retrieval chunks (~5% of paper chunks)
- Table content occasionally appears as section headings
- Acceptable for assignment scope; production would use unstructured
