#!/usr/bin/env python3
"""Unit tests for Mistral providers.

These tests verify Mistral provider implementations with mocked API calls.
"""

import json
import os
import unittest
from unittest.mock import Mock, patch

import pytest

from podcast_scraper import config
from podcast_scraper.speaker_detectors.factory import create_speaker_detector
from podcast_scraper.summarization.factory import create_summarization_provider

pytestmark = [pytest.mark.unit, pytest.mark.module_mistral_providers]


@pytest.mark.llm
@pytest.mark.mistral
class TestMistralSpeakerDetector(unittest.TestCase):
    """Test Mistral speaker detection provider."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="mistral",
            mistral_api_key="test-api-key-123",
            auto_speakers=True,
        )

    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    def test_provider_initialization(self, mock_mistral_class):
        """Test that Mistral speaker detector initializes correctly via factory."""
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client

        detector = create_speaker_detector(self.cfg)
        detector.initialize()

        # Verify client was created with API key
        mock_mistral_class.assert_called_once()
        self.assertTrue(detector._speaker_detection_initialized)

    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_detect_speakers_success(self, mock_render_prompt, mock_mistral_class):
        """Test successful speaker detection via Mistral API via factory."""
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client

        # Mock Mistral API response
        # Mistral uses client.chat.complete() not completions.create()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = json.dumps(
            {
                "speakers": ["John Doe"],
                "hosts": ["John Doe"],
                "guests": [],
            }
        )
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_client.chat.complete.return_value = mock_response

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
        original_key = os.environ.pop("MISTRAL_API_KEY", None)
        try:
            with self.assertRaises(ValidationError) as cm:
                config.Config(
                    rss_url="https://example.com/feed.xml",
                    speaker_detector_provider="mistral",
                    auto_speakers=True,
                )
            self.assertIn("Mistral API key required", str(cm.exception))
        finally:
            # Restore original environment variable if it existed
            if original_key is not None:
                os.environ["MISTRAL_API_KEY"] = original_key

    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    def test_factory_creates_mistral_detector(self, mock_mistral):
        """Test that factory creates Mistral speaker detector."""
        mock_client = Mock()
        mock_mistral.return_value = mock_client
        detector = create_speaker_detector(self.cfg)
        # Factory returns unified MistralProvider
        self.assertEqual(detector.__class__.__name__, "MistralProvider")


@pytest.mark.llm
@pytest.mark.mistral
class TestMistralSummarizationProvider(unittest.TestCase):
    """Test Mistral summarization provider."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="mistral",
            mistral_api_key="test-api-key-123",
            generate_summaries=True,
            generate_metadata=True,
        )

    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    def test_provider_initialization(self, mock_mistral_class):
        """Test that Mistral summarization provider initializes correctly via factory."""
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client

        provider = create_summarization_provider(self.cfg)
        provider.initialize()

        # Verify client was created
        mock_mistral_class.assert_called_once()
        self.assertTrue(provider._summarization_initialized)

    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_summarize_success(self, mock_render_prompt, mock_mistral_class):
        """Test successful summarization via Mistral API via factory."""
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client

        # Mock Mistral API response
        # Mistral uses client.chat.complete() not completions.create()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "This is a summary of the transcript."
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 1000
        mock_response.usage.completion_tokens = 200
        mock_client.chat.complete.return_value = mock_response

        # Mock prompt rendering
        mock_render_prompt.side_effect = ["System prompt", "User prompt"]

        provider = create_summarization_provider(self.cfg)
        provider.initialize()

        result = provider.summarize("This is a long transcript text that needs summarization.")

        self.assertEqual(result["summary"], "This is a summary of the transcript.")
        self.assertIn("metadata", result)
        self.assertEqual(result["metadata"]["provider"], "mistral")

    def test_summarize_missing_api_key(self):
        """Test that missing API key raises ValidationError."""
        from pydantic import ValidationError

        # Unset environment variable to ensure it's not loaded
        original_key = os.environ.pop("MISTRAL_API_KEY", None)
        try:
            with self.assertRaises(ValidationError) as cm:
                config.Config(
                    rss_url="https://example.com/feed.xml",
                    summary_provider="mistral",
                    generate_summaries=True,
                    generate_metadata=True,
                )
            self.assertIn("Mistral API key required", str(cm.exception))
        finally:
            # Restore original environment variable if it existed
            if original_key is not None:
                os.environ["MISTRAL_API_KEY"] = original_key

    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    def test_factory_creates_mistral_provider(self, mock_mistral):
        """Test that factory creates Mistral summarization provider."""
        mock_client = Mock()
        mock_mistral.return_value = mock_client
        provider = create_summarization_provider(self.cfg)
        # Factory returns unified MistralProvider
        self.assertEqual(provider.__class__.__name__, "MistralProvider")
