#!/usr/bin/env python3
"""Integration tests for Anthropic E2E server mock endpoints.

These tests verify that Anthropic providers correctly use the E2E server's
Anthropic mock endpoints. These tests mock the Anthropic SDK calls to simulate
E2E server usage.

These tests verify component interactions with infrastructure, not complete
user workflows.
"""

import json
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
@pytest.mark.anthropic
class TestAnthropicE2EServerIntegration:
    """Test that Anthropic providers correctly use E2E server endpoints.

    These tests mock the Anthropic SDK calls to simulate E2E server usage.
    """

    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    def test_anthropic_summarization_provider_uses_e2e_server(self, mock_anthropic, e2e_server):
        """Test that Anthropic summarization provider can be used with E2E server."""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            anthropic_api_key="test-api-key-123",
            anthropic_api_base=e2e_server.urls.anthropic_api_base(),  # Use E2E server
            summary_provider="anthropic",
            generate_summaries=True,
            generate_metadata=True,
        )

        # Mock Anthropic SDK response
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "This is a test summary from Anthropic E2E server."
        mock_response.usage = Mock()
        mock_response.usage.input_tokens = 200
        mock_response.usage.output_tokens = 100

        mock_client.messages.create.return_value = mock_response

        provider = create_summarization_provider(cfg)
        provider.initialize()

        # Summarize should work with mocked SDK
        result = provider.summarize(
            text="This is a long transcript that needs to be summarized. " * 10,
            episode_title="Test Episode",
        )

        # Verify summary is returned
        assert "summary" in result
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0
        assert (
            "test summary" in result["summary"].lower() or "anthropic" in result["summary"].lower()
        )

        # Clean up provider
        provider.cleanup()

    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    def test_anthropic_speaker_detector_uses_e2e_server(self, mock_anthropic, e2e_server):
        """Test that Anthropic speaker detector can be used with E2E server."""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            anthropic_api_key="test-api-key-123",
            anthropic_api_base=e2e_server.urls.anthropic_api_base(),  # Use E2E server
            speaker_detector_provider="anthropic",
            auto_speakers=True,
        )

        # Mock Anthropic SDK response
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = json.dumps(
            {
                "speakers": ["Alice", "Bob"],
                "hosts": ["Alice"],
                "guests": ["Bob"],
            }
        )
        mock_response.usage = Mock()
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50

        mock_client.messages.create.return_value = mock_response

        detector = create_speaker_detector(cfg)
        detector.initialize()

        # Detect speakers should work with mocked SDK
        speakers, detected_hosts, success = detector.detect_speakers(
            episode_title="Test Episode with Alice and Bob",
            episode_description="Alice interviews Bob about their work",
            known_hosts={"Alice"},
        )

        # Verify speakers are returned
        assert isinstance(speakers, list)
        assert len(speakers) > 0
        assert isinstance(detected_hosts, set)
        assert isinstance(success, bool)

        # Clean up detector
        detector.cleanup()
