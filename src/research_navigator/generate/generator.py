"""
Generator — produces cited answers using Gemini with retrieved context.

Key responsibilities:
1. Refusal: if top similarity score below threshold → refuse gracefully
2. Prompt construction: numbered context blocks with citation instructions
3. Generation: call Gemini with the citation prompt
4. Post-processing: ensure citations are present and valid

Why refuse instead of hallucinate?
The assignment explicitly requires: "refuse gracefully when retrieval confidence
is low, rather than hallucinating an answer or fabricating a citation."
A system that admits uncertainty is more trustworthy than one that confabulates.

Citation prompt design:
We tell Gemini exactly:
- Which context blocks to use [1], [2], etc.
- To cite EVERY factual claim
- To NEVER make claims without a citation
- Format requirements

This is prompt engineering — the quality of citations depends heavily
on how clearly we instruct the model.
"""

from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI

from research_navigator.config import get_settings
from research_navigator.generate.citation_builder import (
    Citation,
    build_citations,
    build_context_block,
    format_citation_block,
)
from research_navigator.logger import get_logger
from research_navigator.retrieve.retriever import RetrievedChunk

logger = get_logger(__name__)

# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an AI/ML research assistant that answers questions using ONLY the provided source documents.

CRITICAL RULES:
1. Every factual claim MUST be followed by an inline citation marker like [1], [2], etc.
2. NEVER make a claim without citing a source from the provided context.
3. If the context doesn't contain enough information to answer, say so honestly.
4. Do NOT use prior knowledge — only use what's in the provided context.
5. Be precise and technical — the audience is AI/ML learners.
6. Write in clear, educational prose.

Citation format: Place [N] immediately after the claim it supports.
Example: "The transformer uses multi-head attention [1], which allows the model to attend to different positions [2]."

If multiple chunks support the same claim, cite all relevant ones: [1][2]
"""

_ANSWER_PROMPT_TEMPLATE = """Answer the following question using ONLY the context provided below.
Cite every factual claim with [N] markers corresponding to the source numbers.

Question: {query}

Context:
{context}

Answer (with inline citations):"""

_REFUSAL_MESSAGE = """I don't have enough relevant material in my corpus to answer this question confidently.

This could be because:
- The topic isn't covered in the current document set (50 AI/ML papers, courses, and blogs)
- The question requires very specific information not present in the retrieved chunks
- The similarity scores between your query and available content are too low

Try rephrasing your question or asking about a related AI/ML topic that might be covered."""


class GeneratedAnswer:
    """
    The complete response from the generator.
    Contains the answer text, citations, and metadata.
    """

    def __init__(
        self,
        answer: str,
        citations: list[Citation],
        retrieved_chunks: list[RetrievedChunk],
        was_refused: bool = False,
        top_score: float = 0.0,
    ) -> None:
        self.answer = answer
        self.citations = citations
        self.retrieved_chunks = retrieved_chunks
        self.was_refused = was_refused
        self.top_score = top_score

    def format_full_response(self) -> str:
        """
        Returns the complete formatted response with answer + citation block.
        """
        if self.was_refused:
            return self.answer

        citation_block = format_citation_block(self.citations)
        return f"{self.answer}\n\n{citation_block}"


def generate_answer(
    query: str,
    chunks: list[RetrievedChunk],
    similarity_threshold: float | None = None,
) -> GeneratedAnswer:
    """
    Generate a cited answer from retrieved chunks.

    Steps:
    1. Check if top similarity score is above threshold
    2. If below → return graceful refusal
    3. Build numbered context block
    4. Call Gemini with citation prompt
    5. Append citation block to answer

    similarity_threshold: minimum score to attempt generation.
    Defaults to settings.similarity_threshold (0.3).
    Tune this value based on M4 evaluation results.
    """
    settings = get_settings()
    threshold = similarity_threshold or settings.similarity_threshold
    log = logger.bind(query=query[:100])

    # Step 1: Check confidence
    if not chunks:
        log.info("refusal_no_chunks")
        return GeneratedAnswer(
            answer=_REFUSAL_MESSAGE,
            citations=[],
            retrieved_chunks=[],
            was_refused=True,
            top_score=0.0,
        )

    top_score = chunks[0].score
    log.info(
        "generation_start", top_score=top_score, threshold=threshold, chunks=len(chunks)
    )

    # Step 2: Refusal path
    if top_score < threshold:
        log.info("refusal_low_confidence", top_score=top_score, threshold=threshold)
        return GeneratedAnswer(
            answer=_REFUSAL_MESSAGE,
            citations=[],
            retrieved_chunks=chunks,
            was_refused=True,
            top_score=top_score,
        )

    # Step 3: Build citations and context
    citations, chunk_to_citation = build_citations(chunks)
    context_block = build_context_block(chunks, chunk_to_citation)

    # Step 4: Call Gemini
    try:
        llm = ChatGoogleGenerativeAI(
            model=settings.generation_model,
            google_api_key=settings.google_api_key,  # type: ignore[arg-type]
            temperature=0.1,  # Slightly creative but mostly factual
        )

        prompt = _ANSWER_PROMPT_TEMPLATE.format(
            query=query,
            context=context_block,
        )

        messages = [
            ("system", _SYSTEM_PROMPT),
            ("human", prompt),
        ]

        response = llm.invoke(messages)
        answer_text = response.content

        log.info(
            "generation_done", answer_length=len(answer_text), citations=len(citations)
        )

        return GeneratedAnswer(
            answer=answer_text,
            citations=citations,
            retrieved_chunks=chunks,
            was_refused=False,
            top_score=top_score,
        )

    except Exception as e:
        log.error("generation_failed", error=str(e))
        error_message = (
            "Generation temporarily unavailable. "
            "The AI service returned an error — please try again in a moment.\n\n"
            f"Technical details: {str(e)[:200]}"
        )
    return GeneratedAnswer(
        answer=error_message,
        citations=[],
        retrieved_chunks=chunks,
        was_refused=True,
        top_score=top_score,
    )
