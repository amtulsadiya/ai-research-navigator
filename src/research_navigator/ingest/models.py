"""
Data models for the ingestion pipeline.
Every piece of data flowing through the system is typed here.
These models act as contracts between parser → chunker → embedder → qdrant.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum

from pydantic import BaseModel, Field, computed_field


class ContentType(StrEnum):
    ARXIV_PAPER = "arxiv_paper"
    COURSE_CHAPTER = "course_chapter"
    SURVEY_BLOG = "survey_blog"
    LAB_BLOG_POST = "lab_blog_post"


class DocumentMeta(BaseModel):
    doc_id: str = Field(..., description="Unique document identifier")
    content_type: ContentType
    title: str
    authors: list[str]
    year: int
    month: int | None = None
    primary_category: str
    secondary_categories: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    is_foundational: bool = False
    citation_count: int | None = None
    source_url: str
    local_path: str


class ParsedSection(BaseModel):
    section_index: int
    section_title: str
    text: str
    is_abstract: bool = False
    is_references: bool = False


class ParsedDocument(BaseModel):
    meta: DocumentMeta
    sections: list[ParsedSection]
    references_text: str | None = None


class Chunk(BaseModel):
    """
    One chunk = one unit that will be embedded and stored in Qdrant.
    """

    doc_id: str
    chunk_index: int
    section_index: int
    section_title: str
    text: str
    token_count: int
    is_abstract: bool = False

    @computed_field  # type: ignore[misc,prop-decorator]
    @property
    def content_hash(self) -> str:
        """SHA256 hash of the chunk text — used for change detection."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @computed_field  # type: ignore[misc,prop-decorator]
    @property
    def chunk_id(self) -> str:
        """
        Deterministic unique ID: hash(doc_id + chunk_index + content_hash).
        Same chunk → same ID → Qdrant upsert is a no-op.
        Changed text → new hash → new ID → Qdrant updates only that chunk.
        """
        raw = f"{self.doc_id}::{self.chunk_index}::{self.content_hash}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ChunkPayload(BaseModel):
    """
    Full payload stored alongside each vector in Qdrant.
    All chunk fields + all document metadata, denormalized so
    Qdrant can filter on any field without a separate lookup.
    """

    # Chunk identity
    chunk_id: str
    doc_id: str
    chunk_index: int
    section_index: int
    section_title: str
    content_hash: str
    token_count: int
    is_abstract: bool
    text: str

    # Document metadata from manifest
    content_type: str
    title: str
    authors: list[str]
    year: int
    month: int | None
    primary_category: str
    secondary_categories: list[str]
    tags: list[str]
    is_foundational: bool
    citation_count: int | None
    source_url: str

    @classmethod
    def from_chunk_and_meta(cls, chunk: Chunk, meta: DocumentMeta) -> ChunkPayload:
        return cls(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            chunk_index=chunk.chunk_index,
            section_index=chunk.section_index,
            section_title=chunk.section_title,
            content_hash=chunk.content_hash,
            token_count=chunk.token_count,
            is_abstract=chunk.is_abstract,
            text=chunk.text,
            content_type=meta.content_type.value,
            title=meta.title,
            authors=meta.authors,
            year=meta.year,
            month=meta.month,
            primary_category=meta.primary_category,
            secondary_categories=meta.secondary_categories,
            tags=meta.tags,
            is_foundational=meta.is_foundational,
            citation_count=meta.citation_count,
            source_url=meta.source_url,
        )
