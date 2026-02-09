#!/usr/bin/env python3
"""Integration tests for Ollama E2E server mock endpoints.

These tests verify that Ollama providers correctly use the E2E server's
Ollama mock endpoints. Ollama uses OpenAI-compatible API format via httpx,
so it uses the same endpoints as OpenAI:
- /v1/chat/completions: For summarization and speaker detection

Note: Ollama does NOT support transcription (no audio API).

These tests verify component interactions with infrastructure, not complete
user workflows.
"""

import os
import sys
from unittest.mock import Mock, patch

import pytest

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from podcast_scraper import config
from podcast_scraper.speaker_detectors.factory import create_speaker_detector
from podcast_scraper.summarization.factory import create_summarization_provider


@pytest.mark.integration
@pytest.mark.llm
@pytest.mark.ollama
class TestOllamaE2EServerIntegration:
    """Test that Ollama providers correctly use E2E server endpoints.

    Note: Ollama provider uses httpx directly for health checks and model validation.
    These tests mock httpx to simulate E2E server usage.
    """

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    def test_ollama_summarization_provider_uses_e2e_server(self, mock_httpx, e2e_server):
        """Test that Ollama summarization provider can be used with E2E server."""
        # Mock httpx responses for health check and model validation
        mock_health_response = Mock()
        mock_health_response.raise_for_status = Mock()
        mock_models_response = Mock()
        mock_models_response.raise_for_status = Mock()
        mock_models_response.json.return_value = {
            "models": [
                {"name": "llama3.3:latest"},
                {"name": "llama3.2:latest"},
                {"name": "llama3.1:8b"},
            ]
        }
        # Return same mock for all httpx.get calls
        mock_httpx.get.return_value = mock_models_response

        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            ollama_api_base=e2e_server.urls.ollama_api_base(),  # Use E2E server
            summary_provider="ollama",
            generate_summaries=True,
            generate_metadata=True,
        )

        provider = create_summarization_provider(cfg)
        provider.initialize()

        # Verify provider is configured to use E2E server
        # Ollama uses OpenAI SDK with custom base_url
        assert str(provider.client.base_url).rstrip(
            "/"
        ) == e2e_server.urls.ollama_api_base().rstrip(
            "/"
        ), "Provider should use E2E server base URL"

        # Summarize should use E2E server endpoints (real HTTP request via OpenAI SDK)
        result = provider.summarize(
            text="This is a long transcript that needs to be summarized. " * 10,
            episode_title="Test Episode",
        )

        # Verify summary is returned (from E2E server mock)
        assert "summary" in result
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0
        assert "test summary" in result["summary"].lower()

        # Clean up provider
        provider.cleanup()

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    def test_ollama_speaker_detector_uses_e2e_server(self, mock_httpx, e2e_server):
        """Test that Ollama speaker detector can be used with E2E server."""
        # Mock httpx responses for health check and model validation
        mock_health_response = Mock()
        mock_health_response.raise_for_status = Mock()
        mock_models_response = Mock()
        mock_models_response.raise_for_status = Mock()
        mock_models_response.json.return_value = {
            "models": [
                {"name": "llama3.3:latest"},
                {"name": "llama3.2:latest"},
                {"name": "llama3.1:8b"},
            ]
        }
        # Return same mock for all httpx.get calls
        mock_httpx.get.return_value = mock_models_response

        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            ollama_api_base=e2e_server.urls.ollama_api_base(),  # Use E2E server
            speaker_detector_provider="ollama",
            auto_speakers=True,
        )

        detector = create_speaker_detector(cfg)
        detector.initialize()

        # Verify detector is configured to use E2E server
        # Ollama uses OpenAI SDK with custom base_url
        assert str(detector.client.base_url).rstrip(
            "/"
        ) == e2e_server.urls.ollama_api_base().rstrip(
            "/"
        ), "Detector should use E2E server base URL"

        # Detect speakers should use E2E server endpoints (real HTTP request via OpenAI SDK)
        speakers, detected_hosts, success = detector.detect_speakers(
            episode_title="Test Episode with Alice and Bob",
            episode_description="Alice interviews Bob about their work",
            known_hosts={"Alice"},
        )

        # Verify speakers are returned (from E2E server mock)
        assert isinstance(speakers, list)
        assert len(speakers) > 0
        assert isinstance(detected_hosts, set)
        assert isinstance(success, bool)

        # Clean up detector
        detector.cleanup()
