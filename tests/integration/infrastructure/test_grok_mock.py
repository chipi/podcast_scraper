#!/usr/bin/env python3
"""Integration tests for Grok E2E server mock endpoints.

These tests verify that Grok providers correctly use the E2E server's
Grok mock endpoints via HTTP requests. Grok uses OpenAI-compatible API format,
so it uses the same endpoints as OpenAI:
- /v1/chat/completions: For summarization and speaker detection

These tests verify component interactions with infrastructure, not complete
user workflows.
"""

import os
import sys

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
@pytest.mark.grok
class TestGrokE2EServerIntegration:
    """Test that Grok providers correctly use E2E server endpoints."""

    def test_grok_summarization_provider_uses_e2e_server(self, e2e_server):
        """Test that Grok summarization provider uses E2E server endpoints."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            grok_api_key="test-api-key-123",
            grok_api_base=e2e_server.urls.grok_api_base(),  # Use E2E server
            summary_provider="grok",
            generate_summaries=True,
            generate_metadata=True,
        )

        provider = create_summarization_provider(cfg)
        provider.initialize()

        # Verify provider is configured to use E2E server
        assert str(provider.client.base_url).rstrip("/") == e2e_server.urls.grok_api_base().rstrip(
            "/"
        ), "Provider should use E2E server base URL"

        # Summarize should use E2E server endpoints (real HTTP request)
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

    def test_grok_speaker_detector_uses_e2e_server(self, e2e_server):
        """Test that Grok speaker detector uses E2E server endpoints."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            grok_api_key="test-api-key-123",
            grok_api_base=e2e_server.urls.grok_api_base(),  # Use E2E server
            speaker_detector_provider="grok",
            auto_speakers=True,
        )

        detector = create_speaker_detector(cfg)
        detector.initialize()

        # Verify detector is configured to use E2E server
        assert str(detector.client.base_url).rstrip("/") == e2e_server.urls.grok_api_base().rstrip(
            "/"
        ), "Detector should use E2E server base URL"

        # Detect speakers should use E2E server endpoints (real HTTP request)
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
