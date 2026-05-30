"""Config validation tests for RFC-089 / ADR-096 DGX fields."""

from __future__ import annotations

import pytest

from podcast_scraper import Config


def test_dgx_requires_host() -> None:
    with pytest.raises(ValueError, match="dgx_tailnet_host"):
        Config.model_validate(
            {
                "rss_url": "https://example.com/feed.xml",
                "transcription_provider": "tailnet_dgx_whisper",
                "transcription_fallback_provider": "openai",
                "openai_api_key": "sk-test",
            }
        )


def test_gemini_fallback_accepts_with_api_key() -> None:
    cfg = Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "transcription_provider": "tailnet_dgx_whisper",
            "transcription_fallback_provider": "gemini",
            "dgx_tailnet_host": "dgx-llm-1.tail-test.ts.net",
            "gemini_api_key": "g-test",
        }
    )
    assert cfg.transcription_fallback_provider == "gemini"


def test_mistral_fallback_accepts_with_api_key() -> None:
    cfg = Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "transcription_provider": "tailnet_dgx_whisper",
            "transcription_fallback_provider": "mistral",
            "dgx_tailnet_host": "dgx-llm-1.tail-test.ts.net",
            "mistral_api_key": "m-test",
        }
    )
    assert cfg.transcription_fallback_provider == "mistral"


def test_vector_embedding_endpoint_round_trip() -> None:
    cfg = Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "vector_embedding_endpoint": "http://dgx:8001/embed",
        }
    )
    assert cfg.vector_embedding_endpoint == "http://dgx:8001/embed"
