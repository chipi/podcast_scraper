"""Integration tests for DeepSeek providers.

These tests verify DeepSeek provider implementations with mocked API calls.
"""

import json
import os
import unittest
from unittest.mock import Mock, patch

import pytest

from podcast_scraper import config
from podcast_scraper.speaker_detectors.factory import create_speaker_detector
from podcast_scraper.summarization.factory import create_summarization_provider

pytestmark = [pytest.mark.integration, pytest.mark.module_deepseek_providers]


@pytest.mark.integration
@pytest.mark.llm
@pytest.mark.deepseek
class TestDeepSeekSpeakerDetector(unittest.TestCase):
    """Test DeepSeek speaker detection provider."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="deepseek",
            deepseek_api_key="sk-test123",
            auto_speakers=True,
        )

    @patch("podcast_scraper.providers.deepseek.deepseek_provider.OpenAI")
    def test_provider_initialization(self, mock_openai_class):
        """Test that DeepSeek speaker detector initializes correctly via factory."""
        detector = create_speaker_detector(self.cfg)
        detector.initialize()

        # Verify client was created with API key and DeepSeek base URL
        mock_openai_class.assert_called_once()
        call_kwargs = mock_openai_class.call_args[1]
        self.assertEqual(call_kwargs["api_key"], "sk-test123")
        self.assertEqual(call_kwargs["base_url"], "https://api.deepseek.com")
        self.assertTrue(detector._speaker_detection_initialized)

    @patch("podcast_scraper.providers.deepseek.deepseek_provider.OpenAI")
    @patch("podcast_scraper.prompts.store.render_prompt")
    @patch(
        "podcast_scraper.providers.deepseek.deepseek_provider.DeepSeekProvider._build_speaker_detection_prompt"
    )
    @patch(
        "podcast_scraper.providers.deepseek.deepseek_provider.DeepSeekProvider._parse_speakers_from_response"
    )
    def test_detect_speakers_success(
        self, mock_parse, mock_build_prompt, mock_render_prompt, mock_openai_class
    ):
        """Test successful speaker detection via DeepSeek API via factory."""
        # Mock OpenAI client and response
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_build_prompt.return_value = "User Prompt"
        mock_render_prompt.return_value = "System Prompt"

        # Mock API response
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
        original_key = os.environ.pop("DEEPSEEK_API_KEY", None)
        try:
            with self.assertRaises(ValidationError) as cm:
                config.Config(
                    rss_url="https://example.com/feed.xml",
                    speaker_detector_provider="deepseek",
                    auto_speakers=True,
                )
            self.assertIn("DeepSeek API key required", str(cm.exception))
        finally:
            # Restore original environment variable if it existed
            if original_key is not None:
                os.environ["DEEPSEEK_API_KEY"] = original_key

    def test_factory_creates_deepseek_detector(self):
        """Test that factory creates unified DeepSeek provider."""
        detector = create_speaker_detector(self.cfg)
        # Verify it's the unified DeepSeek provider
        self.assertEqual(detector.__class__.__name__, "DeepSeekProvider")
        # Verify protocol compliance
        self.assertTrue(hasattr(detector, "detect_speakers"))
        self.assertTrue(hasattr(detector, "detect_hosts"))
        self.assertTrue(hasattr(detector, "analyze_patterns"))
        self.assertTrue(hasattr(detector, "clear_cache"))


@pytest.mark.integration
@pytest.mark.llm
@pytest.mark.deepseek
class TestDeepSeekSummarizationProvider(unittest.TestCase):
    """Test DeepSeek summarization provider."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="deepseek",
            deepseek_api_key="sk-test123",
            generate_metadata=True,
            generate_summaries=True,
        )

    @patch("podcast_scraper.providers.deepseek.deepseek_provider.OpenAI")
    def test_provider_initialization(self, mock_openai_class):
        """Test that DeepSeek summarization provider initializes correctly via factory."""
        provider = create_summarization_provider(self.cfg)
        provider.initialize()

        # Verify client was created with API key and DeepSeek base URL
        mock_openai_class.assert_called_once()
        call_kwargs = mock_openai_class.call_args[1]
        self.assertEqual(call_kwargs["api_key"], "sk-test123")
        self.assertEqual(call_kwargs["base_url"], "https://api.deepseek.com")
        self.assertTrue(provider._summarization_initialized)

    @patch("podcast_scraper.providers.deepseek.deepseek_provider.OpenAI")
    @patch("podcast_scraper.prompts.store.render_prompt")
    @patch("podcast_scraper.prompts.store.get_prompt_metadata")
    @patch(
        "podcast_scraper.providers.deepseek.deepseek_provider.DeepSeekProvider._build_summarization_prompts"
    )
    def test_summarize_success(
        self, mock_build_prompts, mock_get_metadata, mock_render_prompt, mock_openai_class
    ):
        """Test successful summarization via DeepSeek API via factory."""
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
        original_key = os.environ.pop("DEEPSEEK_API_KEY", None)
        try:
            with self.assertRaises(ValidationError) as cm:
                config.Config(
                    rss_url="https://example.com/feed.xml",
                    summary_provider="deepseek",
                    generate_metadata=True,
                    generate_summaries=True,
                )
            self.assertIn("DeepSeek API key required", str(cm.exception))
        finally:
            # Restore original environment variable if it existed
            if original_key is not None:
                os.environ["DEEPSEEK_API_KEY"] = original_key

    def test_factory_creates_deepseek_provider(self):
        """Test that factory creates unified DeepSeek provider."""
        provider = create_summarization_provider(self.cfg)
        # Verify it's the unified DeepSeek provider
        self.assertEqual(provider.__class__.__name__, "DeepSeekProvider")
        # Verify protocol compliance
        self.assertTrue(hasattr(provider, "summarize"))
        self.assertTrue(hasattr(provider, "initialize"))
        self.assertTrue(hasattr(provider, "cleanup"))


@pytest.mark.integration
@pytest.mark.llm
@pytest.mark.deepseek
class TestDeepSeekProviderErrorHandling(unittest.TestCase):
    """Test error handling for DeepSeek providers."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="deepseek",
            deepseek_api_key="sk-test123",
            auto_speakers=True,
        )

    @patch("podcast_scraper.providers.deepseek.deepseek_provider.OpenAI")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_detect_speakers_api_error(self, mock_render_prompt, mock_openai_class):
        """Test that API errors are handled gracefully via factory."""
        # Mock OpenAI client to raise exception
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API Error")
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
        self.assertIn("DeepSeek speaker detection failed", str(cm.exception))
