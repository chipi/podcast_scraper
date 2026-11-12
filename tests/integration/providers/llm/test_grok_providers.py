"""Integration tests for Grok providers.

These tests verify Grok provider implementations with mocked API calls.
"""

import os
import unittest
from unittest.mock import Mock, patch

import pytest

from podcast_scraper import config
from podcast_scraper.summarization.factory import create_summarization_provider

pytestmark = [pytest.mark.integration, pytest.mark.module_grok_providers]


@pytest.mark.integration
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

    def test_detect_speakers_missing_api_key(self):
        """Test that missing API key raises ValidationError (caught by config validator)."""
        from pydantic import ValidationError

        # Unset environment variable to ensure it's not loaded
        original_key = os.environ.pop("GROK_API_KEY", None)
        try:
            with self.assertRaises(ValidationError) as cm:
                config.Config(
                    rss_url="https://example.com/feed.xml",
                    speaker_detector_provider="grok",
                    auto_speakers=True,
                )
            self.assertIn("Grok API key required", str(cm.exception))
        finally:
            # Restore original environment variable if it existed
            if original_key is not None:
                os.environ["GROK_API_KEY"] = original_key


@pytest.mark.integration
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
            generate_metadata=True,
            generate_summaries=True,
        )

    @patch("podcast_scraper.providers.grok.grok_provider.OpenAI")
    def test_provider_initialization(self, mock_openai_class):
        """Test that Grok summarization provider initializes correctly via factory."""
        provider = create_summarization_provider(self.cfg)
        provider.initialize()

        # Verify client was created with API key and Grok base URL
        mock_openai_class.assert_called_once()
        call_kwargs = mock_openai_class.call_args[1]
        self.assertEqual(call_kwargs["api_key"], "test-api-key-123")
        self.assertEqual(call_kwargs["base_url"], "https://api.x.ai/v1")
        self.assertTrue(provider._summarization_initialized)

    @patch("podcast_scraper.providers.grok.grok_provider.OpenAI")
    @patch("podcast_scraper.prompts.store.render_prompt")
    @patch("podcast_scraper.prompts.store.get_prompt_metadata")
    @patch("podcast_scraper.providers.grok.grok_provider.GrokProvider._build_summarization_prompts")
    def test_summarize_success(
        self, mock_build_prompts, mock_get_metadata, mock_render_prompt, mock_openai_class
    ):
        """Test successful summarization via Grok API via factory."""
        # Mock OpenAI client and response
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        # _build_summarization_prompts returns (system_prompt, user_prompt, system_prompt_name, user_prompt_name, paragraphs_min, paragraphs_max)
        mock_build_prompts.return_value = (
            "System Prompt",
            "User Prompt",
            "summarization/system_v1",
            "summarization/user_v1",
            1,
            5,
        )
        mock_render_prompt.return_value = "Rendered Prompt"
        mock_get_metadata.return_value = {"version": "v1"}

        # Mock API response
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content="This is a test summary of the transcript."))
        ]
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 1000
        mock_response.usage.completion_tokens = 200
        mock_client.chat.completions.create.return_value = mock_response

        provider = create_summarization_provider(self.cfg)
        provider.client = mock_client
        provider.initialize()

        result = provider.summarize(
            text="This is a long transcript that needs to be summarized.",
            episode_title="Test Episode",
        )

        self.assertIn("summary", result)
        self.assertEqual(result["summary"], "This is a test summary of the transcript.")
        self.assertIn("metadata", result)
        self.assertEqual(result["metadata"]["model"], provider.summary_model)

    def test_summarize_missing_api_key(self):
        """Test that missing API key raises ValidationError (caught by config validator)."""
        from pydantic import ValidationError

        # Unset environment variable to ensure it's not loaded
        original_key = os.environ.pop("GROK_API_KEY", None)
        try:
            with self.assertRaises(ValidationError) as cm:
                config.Config(
                    rss_url="https://example.com/feed.xml",
                    summary_provider="grok",
                    generate_metadata=True,
                    generate_summaries=True,
                )
            self.assertIn("Grok API key required", str(cm.exception))
        finally:
            # Restore original environment variable if it existed
            if original_key is not None:
                os.environ["GROK_API_KEY"] = original_key

    def test_factory_creates_grok_provider(self):
        """Test that factory creates unified Grok provider."""
        provider = create_summarization_provider(self.cfg)
        # Verify it's the unified Grok provider
        self.assertEqual(provider.__class__.__name__, "GrokProvider")
        # Verify protocol compliance
        self.assertTrue(hasattr(provider, "summarize"))
        self.assertTrue(hasattr(provider, "initialize"))
        self.assertTrue(hasattr(provider, "cleanup"))
