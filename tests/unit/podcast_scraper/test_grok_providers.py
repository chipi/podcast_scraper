#!/usr/bin/env python3
"""Unit tests for Grok providers (Issue #1095).

These tests verify Grok provider implementations with mocked API calls.
"""

import json
import os
import unittest
from unittest.mock import Mock, patch

import pytest

from podcast_scraper import config
from podcast_scraper.speaker_detectors.factory import create_speaker_detector
from podcast_scraper.summarization.factory import create_summarization_provider

pytestmark = [pytest.mark.unit, pytest.mark.module_grok_providers]


@pytest.mark.llm
@pytest.mark.grok
class TestGrokSpeakerDetector(unittest.TestCase):
    """Test Grok speaker detection provider."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="grok",
            grok_api_key="test-api-key-123",
            auto_speakers=True,
        )

    @patch("podcast_scraper.providers.grok.grok_provider.OpenAI")
    def test_provider_initialization(self, mock_openai):
        """Test that Grok speaker detector initializes correctly via factory."""
        provider = create_speaker_detector(self.cfg)
        provider.initialize()

        # Verify OpenAI client was created
        mock_openai.assert_called_once()
        self.assertTrue(provider._speaker_detection_initialized)

    @patch("podcast_scraper.providers.grok.grok_provider.OpenAI")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_detect_speakers_success(self, mock_render, mock_openai):
        """Test successful speaker detection via Grok API via factory."""
        # Mock OpenAI client and response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "speakers": ["John Doe", "Jane Smith"],
                "hosts": ["John Doe"],
                "guests": ["Jane Smith"],
            }
        )
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        mock_render.side_effect = lambda name, **kwargs: "test prompt"

        provider = create_speaker_detector(self.cfg)
        provider.initialize()

        speakers, hosts, success = provider.detect_speakers(
            episode_title="Test Episode",
            episode_description="Test description",
            known_hosts={"John Doe"},
        )

        self.assertEqual(speakers, ["John Doe", "Jane Smith"])
        self.assertEqual(hosts, {"John Doe"})
        self.assertTrue(success)

    def test_detect_speakers_missing_api_key(self):
        """Test that missing API key raises ValidationError."""
        from pydantic import ValidationError

        # Unset environment variable to ensure it's not loaded
        original_key = os.environ.pop("GROK_API_KEY", None)
        try:
            with self.assertRaises(ValidationError) as cm:
                config.Config(
                    rss_url="https://example.com/feed.xml", speaker_detector_provider="grok"
                )
            self.assertIn("Grok API key required", str(cm.exception))
        finally:
            # Restore original environment variable if it existed
            if original_key is not None:
                os.environ["GROK_API_KEY"] = original_key

    @patch("podcast_scraper.providers.grok.grok_provider.OpenAI")
    def test_factory_creates_grok_provider(self, mock_openai):
        """Test that factory creates Grok speaker detector."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        provider = create_speaker_detector(self.cfg)
        # Factory returns unified GrokProvider
        self.assertEqual(provider.__class__.__name__, "GrokProvider")


@pytest.mark.llm
@pytest.mark.grok
class TestGrokSummarizationProvider(unittest.TestCase):
    """Test Grok summarization provider."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="grok",
            grok_api_key="test-api-key-123",
            generate_summaries=True,
            generate_metadata=True,
        )

    @patch("podcast_scraper.providers.grok.grok_provider.OpenAI")
    def test_provider_initialization(self, mock_openai):
        """Test that Grok summarization provider initializes correctly via factory."""
        provider = create_summarization_provider(self.cfg)
        provider.initialize()

        # Verify OpenAI client was created
        mock_openai.assert_called_once()
        self.assertTrue(provider._summarization_initialized)

    @patch("podcast_scraper.providers.grok.grok_provider.OpenAI")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_summarize_success(self, mock_render, mock_openai):
        """Test successful summarization via Grok API via factory."""
        # Mock OpenAI client and response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "This is a summary."
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 1000
        mock_response.usage.completion_tokens = 200
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        mock_render.side_effect = lambda name, **kwargs: "test prompt"

        provider = create_summarization_provider(self.cfg)
        provider.initialize()

        result = provider.summarize("Long transcript text here...")

        self.assertEqual(result["summary"], "This is a summary.")
        self.assertEqual(result["metadata"]["provider"], "grok")

    def test_summarize_missing_api_key(self):
        """Test that missing API key raises ValidationError."""
        from pydantic import ValidationError

        # Unset environment variable to ensure it's not loaded
        original_key = os.environ.pop("GROK_API_KEY", None)
        try:
            with self.assertRaises(ValidationError) as cm:
                config.Config(
                    rss_url="https://example.com/feed.xml",
                    summary_provider="grok",
                    generate_summaries=True,
                    generate_metadata=True,
                )
            self.assertIn("Grok API key required", str(cm.exception))
        finally:
            # Restore original environment variable if it existed
            if original_key is not None:
                os.environ["GROK_API_KEY"] = original_key

    @patch("podcast_scraper.providers.grok.grok_provider.OpenAI")
    def test_factory_creates_grok_provider(self, mock_openai):
        """Test that factory creates Grok summarization provider."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        provider = create_summarization_provider(self.cfg)
        # Factory returns unified GrokProvider
        self.assertEqual(provider.__class__.__name__, "GrokProvider")
