#!/usr/bin/env python3
"""Unit tests for DeepSeek providers via factory (Issue #107).

These tests verify DeepSeek provider implementations with mocked API calls,
using factory functions to create providers (tests factory integration).

For standalone provider tests, see test_deepseek_provider.py.
"""

import unittest
from unittest.mock import MagicMock, Mock, patch

import pytest

# Mock openai before importing modules that require it
# Unit tests run without openai package installed
# Use patch.dict without 'with' to avoid context manager conflicts with @patch decorators
mock_openai = MagicMock()
mock_openai.OpenAI = Mock()
_patch_openai = patch.dict(
    "sys.modules",
    {
        "openai": mock_openai,
    },
)
_patch_openai.start()

from podcast_scraper import config
from podcast_scraper.speaker_detectors.factory import create_speaker_detector
from podcast_scraper.summarization.factory import create_summarization_provider

pytestmark = [pytest.mark.unit, pytest.mark.module_deepseek_providers]


@pytest.mark.llm
@pytest.mark.deepseek
class TestDeepSeekSpeakerDetectionProviderFactory(unittest.TestCase):
    """Test DeepSeek speaker detection provider via factory."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="deepseek",
            deepseek_api_key="test-api-key-123",
        )

    @patch("podcast_scraper.providers.deepseek.deepseek_provider.OpenAI")
    def test_provider_initialization(self, mock_openai_class):
        """Test that DeepSeek speaker detection provider initializes correctly via factory."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        provider = create_speaker_detector(self.cfg)
        provider.initialize()

        # Verify OpenAI client was created
        mock_openai_class.assert_called_once()
        self.assertTrue(provider._speaker_detection_initialized)

    @patch("podcast_scraper.providers.deepseek.deepseek_provider.OpenAI")
    def test_detect_speakers_success(self, mock_openai_class):
        """Test successful speaker detection via DeepSeek API via factory."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = (
            '{"speakers": ["Alice", "Bob"], "hosts": ["Alice"], "guests": ["Bob"]}'
        )
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50

        mock_client.chat.completions.create.return_value = mock_response

        provider = create_speaker_detector(self.cfg)
        provider.initialize()

        speakers, hosts, success = provider.detect_speakers(
            episode_title="Alice interviews Bob",
            episode_description="A great conversation",
            known_hosts={"Alice"},
        )

        self.assertTrue(success)
        self.assertIn("Alice", speakers)


@pytest.mark.llm
@pytest.mark.deepseek
class TestDeepSeekSummarizationProviderFactory(unittest.TestCase):
    """Test DeepSeek summarization provider via factory."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="deepseek",
            deepseek_api_key="test-api-key-123",
            generate_summaries=True,
            generate_metadata=True,  # Required when generate_summaries=True
        )

    @patch("podcast_scraper.providers.deepseek.deepseek_provider.OpenAI")
    def test_provider_initialization(self, mock_openai_class):
        """Test that DeepSeek summarization provider initializes correctly via factory."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        provider = create_summarization_provider(self.cfg)
        provider.initialize()

        # Verify OpenAI client was created
        mock_openai_class.assert_called_once()
        self.assertTrue(provider._summarization_initialized)

    @patch("podcast_scraper.providers.deepseek.deepseek_provider.OpenAI")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_summarize_success(self, mock_render_prompt, mock_openai_class):
        """Test successful summarization via DeepSeek API via factory."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock render_prompt to return prompts
        mock_render_prompt.side_effect = ["System Prompt", "User Prompt"]

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "This is a summary."
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 1000
        mock_response.usage.completion_tokens = 200

        mock_client.chat.completions.create.return_value = mock_response

        provider = create_summarization_provider(self.cfg)
        provider.initialize()

        result = provider.summarize("This is a long transcript text.")

        self.assertEqual(result["summary"], "This is a summary.")
        self.assertIn("metadata", result)
