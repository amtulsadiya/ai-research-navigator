from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Google AI
    google_api_key: str = Field(default="", description="Google AI Studio API key")

    # Qdrant
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_collection_name: str = Field(default="research_navigator")

    # Models
    embedding_model: str = Field(default="models/text-embedding-004")
    generation_model: str = Field(default="gemini-1.5-flash")

    # Logging
    log_level: str = Field(default="INFO")

    # Chunking
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=64)

    # Retrieval
    # Retrieval
    top_k: int = Field(default=6)

    similarity_threshold: float = Field(
        default=0.525,
        description=(
            "Minimum cosine similarity for generation. "
            "Tuned on 20-query dataset (10 valid, 10 invalid): 90% accuracy. "
            "Re-tune in M4 evaluation with full 40-query golden set."
        ),
    )


def get_settings() -> Settings:
    return Settings()
