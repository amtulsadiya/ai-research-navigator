from unittest.mock import patch


def test_settings_loads() -> None:
    with patch.dict(
        "os.environ",
        {"GOOGLE_API_KEY": "test-key-123"},
    ):
        from research_navigator.config import get_settings

        settings = get_settings()
        assert settings.google_api_key == "test-key-123"
        assert settings.qdrant_url == "http://localhost:6333"
        assert settings.chunk_size == 512
        assert settings.top_k == 6


def test_similarity_threshold_default() -> None:
    with patch.dict(
        "os.environ",
        {"GOOGLE_API_KEY": "test-key-123"},
    ):
        from research_navigator.config import get_settings

        settings = get_settings()
        assert 0.0 < settings.similarity_threshold < 1.0
