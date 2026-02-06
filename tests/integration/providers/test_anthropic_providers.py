"""Integration tests for Anthropic providers.

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

pytestmark = [pytest.mark.integration, pytest.mark.module_anthropic_providers]


@pytest.mark.integration
@pytest.mark.slow
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
        detector = create_speaker_detector(self.cfg)
        detector.initialize()

        # Verify client was created with API key
        mock_anthropic_class.assert_called_once()
        self.assertTrue(detector._speaker_detection_initialized)

    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    @patch("podcast_scraper.prompts.store.render_prompt")
    @patch(
        "podcast_scraper.providers.anthropic.anthropic_provider.AnthropicProvider._build_speaker_detection_prompt"
    )
    @patch(
        "podcast_scraper.providers.anthropic.anthropic_provider.AnthropicProvider._parse_speakers_from_response"
    )
    def test_detect_speakers_success(
        self, mock_parse, mock_build_prompt, mock_render_prompt, mock_anthropic_class
    ):
        """Test successful speaker detection via Anthropic API via factory."""
        # Mock Anthropic client and response
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_build_prompt.return_value = "User Prompt"
        mock_render_prompt.return_value = "System Prompt"

        # Mock API response
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = json.dumps(
            {
                "speakers": ["John Doe", "Jane Smith"],
                "hosts": ["John Doe"],
                "guests": ["Jane Smith"],
            }
        )
        mock_response.usage = Mock()
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_client.messages.create.return_value = mock_response
        # _parse_speakers_from_response returns (speaker_names_list, detected_hosts_set, detection_succeeded)
        mock_parse.return_value = (["John Doe", "Jane Smith"], {"John Doe"}, True)

        detector = create_speaker_detector(self.cfg)
        detector.client = mock_client
        detector.initialize()

        speakers, detected_hosts, success = detector.detect_speakers(
            episode_title="Test Episode with John Doe and Jane Smith",
            episode_description="A test episode",
            known_hosts={"John Doe"},
        )

        self.assertEqual(len(speakers), 2)
        self.assertIn("John Doe", speakers)
        self.assertIn("Jane Smith", speakers)
        self.assertEqual(detected_hosts, {"John Doe"})
        self.assertTrue(success)

    def test_detect_speakers_missing_api_key(self):
        """Test that missing API key raises ValidationError (caught by config validator)."""
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

    def test_factory_creates_anthropic_detector(self):
        """Test that factory creates unified Anthropic provider."""
        detector = create_speaker_detector(self.cfg)
        # Verify it's the unified Anthropic provider
        self.assertEqual(detector.__class__.__name__, "AnthropicProvider")
        # Verify protocol compliance
        self.assertTrue(hasattr(detector, "detect_speakers"))
        self.assertTrue(hasattr(detector, "detect_hosts"))
        self.assertTrue(hasattr(detector, "analyze_patterns"))
        self.assertTrue(hasattr(detector, "clear_cache"))


@pytest.mark.integration
@pytest.mark.slow
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
            generate_metadata=True,
            generate_summaries=True,
        )

    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    def test_provider_initialization(self, mock_anthropic_class):
        """Test that Anthropic summarization provider initializes correctly via factory."""
        provider = create_summarization_provider(self.cfg)
        provider.initialize()

        # Verify client was created with API key
        mock_anthropic_class.assert_called_once()
        self.assertTrue(provider._summarization_initialized)

    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    @patch("podcast_scraper.prompts.store.render_prompt")
    @patch("podcast_scraper.prompts.store.get_prompt_metadata")
    @patch(
        "podcast_scraper.providers.anthropic.anthropic_provider.AnthropicProvider._build_summarization_prompts"
    )
    def test_summarize_success(
        self, mock_build_prompts, mock_get_metadata, mock_render_prompt, mock_anthropic_class
    ):
        """Test successful summarization via Anthropic API via factory."""
        # Mock Anthropic client and response
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
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
        mock_response.content = [Mock()]
        mock_response.content[0].text = "This is a test summary of the transcript."
        mock_response.usage = Mock()
        mock_response.usage.input_tokens = 1000
        mock_response.usage.output_tokens = 200
        mock_client.messages.create.return_value = mock_response

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
        original_key = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            with self.assertRaises(ValidationError) as cm:
                config.Config(
                    rss_url="https://example.com/feed.xml",
                    summary_provider="anthropic",
                    generate_metadata=True,
                    generate_summaries=True,
                )
            self.assertIn("Anthropic API key required", str(cm.exception))
        finally:
            # Restore original environment variable if it existed
            if original_key is not None:
                os.environ["ANTHROPIC_API_KEY"] = original_key

    def test_factory_creates_anthropic_provider(self):
        """Test that factory creates unified Anthropic provider."""
        provider = create_summarization_provider(self.cfg)
        # Verify it's the unified Anthropic provider
        self.assertEqual(provider.__class__.__name__, "AnthropicProvider")
        # Verify protocol compliance
        self.assertTrue(hasattr(provider, "summarize"))
        self.assertTrue(hasattr(provider, "initialize"))
        self.assertTrue(hasattr(provider, "cleanup"))


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.llm
@pytest.mark.anthropic
class TestAnthropicProviderErrorHandling(unittest.TestCase):
    """Test error handling for Anthropic providers."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="anthropic",
            anthropic_api_key="sk-ant-test123",
            auto_speakers=True,
        )

    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_detect_speakers_api_error(self, mock_render_prompt, mock_anthropic_class):
        """Test that API errors are handled gracefully via factory."""
        # Mock Anthropic client to raise exception
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("API Error")
        mock_render_prompt.return_value = "System Prompt"

        detector = create_speaker_detector(self.cfg)
        detector.client = mock_client
        detector.initialize()

        from podcast_scraper.exceptions import ProviderRuntimeError

        with self.assertRaises(ProviderRuntimeError) as cm:
            detector.detect_speakers(
                episode_title="Test Episode",
                episode_description="A test episode",
                known_hosts=set(),
            )
        self.assertIn("Anthropic speaker detection failed", str(cm.exception))
