"""
Embedder — runs BAAI/bge-m3 locally via sentence-transformers.

No API needed. Model downloads once (~2GB) on first run,
then runs entirely on your machine.

Why bge-m3?
- State of the art on MTEB benchmark
- 1024 dimensions (good quality vs storage balance)
- Supports 100+ languages (covers bonus track B4)
- Supports both dense AND sparse vectors (one model for hybrid retrieval)
- Fully local — no rate limits, no cost, no API key
- OSS-first as required by assignment ground rules
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from research_navigator.logger import get_logger

logger = get_logger(__name__)

# Model name — pulled from HuggingFace Hub on first run, cached locally after
BGE_MODEL_NAME = "BAAI/bge-m3"

# Singleton — load once, reuse across all calls
# Loading a transformer model takes ~5 seconds, so we load it once
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """
    Load the bge-m3 model (once) and return it.
    Uses a module-level singleton so we don't reload on every call.
    """
    global _model
    if _model is None:
        logger.info("loading_embedding_model", model=BGE_MODEL_NAME)
        _model = SentenceTransformer(BGE_MODEL_NAME)
        logger.info("embedding_model_loaded")
    return _model


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts using bge-m3 locally.

    Returns a list of 1024-dimensional float vectors.
    Order is preserved — texts[i] → vectors[i].

    batch_size=32: process 32 chunks at a time to manage memory.
    show_progress_bar=True: shows a progress bar for long ingestion runs.
    normalize_embeddings=True: L2-normalize vectors so cosine similarity
    equals dot product — required for Qdrant's cosine distance metric.
    """
    if not texts:
        return []

    model = _get_model()

    logger.info("embedding_texts", count=len(texts))

    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,  # Required for cosine similarity
        convert_to_numpy=True,
    )

    # Convert numpy array to list of lists for Qdrant
    result = [v.tolist() for v in vectors]

    logger.info("embedding_complete", count=len(result), dims=len(result[0]))
    return result


def embed_single(text: str) -> list[float]:
    """
    Embed a single text — used at query time in M2.
    Returns a single 1024-dimensional vector.
    """
    return get_embeddings([text])[0]
