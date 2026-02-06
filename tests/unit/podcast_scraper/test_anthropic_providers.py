#!/usr/bin/env python3
"""Unit tests for Anthropic providers (Issue #106).

These tests verify Anthropic provider implementations with mocked API calls.
"""

import json
import os
import unittest
from unittest.mock import Mock, patch

import pytest

from podcast_scraper import config
from podcast_scraper.speaker_detectors.factory import create_speaker_detector
from podcast_scraper.summarization.factory import create_summarization_provider

pytestmark = [pytest.mark.unit, pytest.mark.module_anthropic_providers]


@pytest.mark.llm
@pytest.mark.anthropic
class TestAnthropicSpeakerDetector(unittest.TestCase):
    """Test Anthropic speaker detection provider."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="anthropic",
            anthropic_api_key="sk-ant-test123",
            auto_speakers=True,
        )

    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    def test_provider_initialization(self, mock_anthropic_class):
        """Test that Anthropic speaker detector initializes correctly via factory."""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        detector = create_speaker_detector(self.cfg)
        detector.initialize()

        # Verify client was created with API key
        mock_anthropic_class.assert_called_once()
        self.assertTrue(detector._speaker_detection_initialized)

    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_detect_speakers_success(self, mock_render_prompt, mock_anthropic_class):
        """Test successful speaker detection via Anthropic API via factory."""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Mock Anthropic API response
        mock_response = Mock()
        mock_response.content = [
            Mock(
                text=json.dumps(
                    {
                        "speakers": ["John Doe"],
                        "hosts": ["John Doe"],
                        "guests": [],
                    }
                )
            )
        ]
        mock_response.usage = Mock()
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_client.messages.create.return_value = mock_response

        # Mock prompt rendering
        mock_render_prompt.side_effect = ["System prompt", "User prompt"]

        detector = create_speaker_detector(self.cfg)
        detector.initialize()

        speakers, detected_hosts, success = detector.detect_speakers(
            episode_title="Test Episode with John Doe",
            episode_description="A test episode",
            known_hosts={"John Doe"},
        )

        self.assertEqual(detected_hosts, {"John Doe"})
        self.assertTrue(success)

    def test_detect_speakers_missing_api_key(self):
        """Test that missing API key raises ValidationError."""
        from pydantic import ValidationError

        # Unset environment variable to ensure it's not loaded
        original_key = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            with self.assertRaises(ValidationError) as cm:
                config.Config(
                    rss_url="https://example.com/feed.xml",
                    speaker_detector_provider="anthropic",
                    auto_speakers=True,
                )
            self.assertIn("Anthropic API key required", str(cm.exception))
        finally:
            # Restore original environment variable if it existed
            if original_key is not None:
                os.environ["ANTHROPIC_API_KEY"] = original_key

    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    def test_factory_creates_anthropic_detector(self, mock_anthropic):
        """Test that factory creates Anthropic speaker detector."""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        detector = create_speaker_detector(self.cfg)
        # Factory returns unified AnthropicProvider
        self.assertEqual(detector.__class__.__name__, "AnthropicProvider")


@pytest.mark.llm
@pytest.mark.anthropic
class TestAnthropicSummarizationProvider(unittest.TestCase):
    """Test Anthropic summarization provider."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="anthropic",
            anthropic_api_key="sk-ant-test123",
            generate_summaries=True,
            generate_metadata=True,
        )

    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    def test_provider_initialization(self, mock_anthropic_class):
        """Test that Anthropic summarization provider initializes correctly via factory."""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        provider = create_summarization_provider(self.cfg)
        provider.initialize()

        # Verify client was created
        mock_anthropic_class.assert_called_once()
        self.assertTrue(provider._summarization_initialized)

    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_summarize_success(self, mock_render_prompt, mock_anthropic_class):
        """Test successful summarization via Anthropic API via factory."""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Mock Anthropic API response
        mock_response = Mock()
        mock_response.content = [Mock(text="This is a summary of the transcript.")]
        mock_response.usage = Mock()
        mock_response.usage.input_tokens = 1000
        mock_response.usage.output_tokens = 200
        mock_client.messages.create.return_value = mock_response

        # Mock prompt rendering
        mock_render_prompt.side_effect = ["System prompt", "User prompt"]

        provider = create_summarization_provider(self.cfg)
        provider.initialize()

        result = provider.summarize("This is a long transcript text that needs summarization.")

        self.assertEqual(result["summary"], "This is a summary of the transcript.")
        self.assertIn("metadata", result)
        self.assertEqual(result["metadata"]["provider"], "anthropic")

    def test_summarize_missing_api_key(self):
        """Test that missing API key raises ValidationError."""
        from pydantic import ValidationError

        # Unset environment variable to ensure it's not loaded
        original_key = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            with self.assertRaises(ValidationError) as cm:
                config.Config(
                    rss_url="https://example.com/feed.xml",
                    summary_provider="anthropic",
                    generate_summaries=True,
                    generate_metadata=True,
                )
            self.assertIn("Anthropic API key required", str(cm.exception))
        finally:
            # Restore original environment variable if it existed
            if original_key is not None:
                os.environ["ANTHROPIC_API_KEY"] = original_key

    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    def test_factory_creates_anthropic_provider(self, mock_anthropic):
        """Test that factory creates Anthropic summarization provider."""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        provider = create_summarization_provider(self.cfg)
        # Factory returns unified AnthropicProvider
        self.assertEqual(provider.__class__.__name__, "AnthropicProvider")
