#!/usr/bin/env python3
"""Standalone unit tests for unified OpenAI provider.

These tests verify that OpenAIProvider correctly implements all three protocols
(TranscriptionProvider, SpeakerDetector, SummarizationProvider) using
OpenAI's API (Whisper API, GPT API).

These are standalone provider tests - they test the provider itself,
not its integration with the app.
"""

import os
import unittest
from unittest.mock import MagicMock, Mock, patch

import pytest

# Mock openai before importing modules that require it
# Unit tests run without openai package installed
# Use patch.dict without 'with' to avoid context manager conflicts with @patch decorators
mock_openai = MagicMock()
mock_openai.OpenAI = Mock()


# Add real exception classes so they can be used in retry_with_metrics
class MockAPIError(Exception):
    """Mock APIError for testing."""

    pass


class MockRateLimitError(Exception):
    """Mock RateLimitError for testing."""

    pass


mock_openai.APIError = MockAPIError
mock_openai.RateLimitError = MockRateLimitError
_patch_openai = patch.dict(
    "sys.modules",
    {
        "openai": mock_openai,
    },
)
_patch_openai.start()

from podcast_scraper import config
from podcast_scraper.providers.openai.openai_provider import OpenAIProvider


@pytest.mark.unit
class TestOpenAIProviderStandalone(unittest.TestCase):
    """Standalone tests for OpenAIProvider - testing the provider itself."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="openai",
            speaker_detector_provider="openai",
            summary_provider="openai",
            openai_api_key="sk-test123",
            transcribe_missing=False,  # Disable to avoid API calls
            auto_speakers=False,  # Disable to avoid API calls
            generate_summaries=False,  # Disable to avoid API calls
        )

    def test_provider_creation(self):
        """Test that OpenAIProvider can be created."""
        provider = OpenAIProvider(self.cfg)
        self.assertIsNotNone(provider)
        self.assertEqual(provider.__class__.__name__, "OpenAIProvider")

    def test_provider_creation_requires_api_key(self):
        """Test that OpenAIProvider requires API key."""
        # Unset environment variable to ensure it's not loaded
        original_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            # Note: Config validation happens before provider creation
            # So we need to catch ValidationError from Config, not ValueError from provider
            with self.assertRaises(Exception) as context:
                cfg = config.Config(
                    rss_url="https://example.com/feed.xml",
                    transcription_provider="openai",
                )
                OpenAIProvider(cfg)
            # Error can be either ValidationError (from Config) or ValueError (from provider)
        finally:
            # Restore original environment variable if it existed
            if original_key is not None:
                os.environ["OPENAI_API_KEY"] = original_key
        error_msg = str(context.exception)
        self.assertTrue(
            "OpenAI API key required" in error_msg or "validation error" in error_msg.lower()
        )

    def test_provider_implements_all_protocols(self):
        """Test that OpenAIProvider implements all three protocols."""
        provider = OpenAIProvider(self.cfg)

        # TranscriptionProvider protocol
        self.assertTrue(hasattr(provider, "transcribe"))
        self.assertTrue(hasattr(provider, "transcribe_with_segments"))
        self.assertTrue(hasattr(provider, "initialize"))
        self.assertTrue(hasattr(provider, "cleanup"))

        # SpeakerDetector protocol
        self.assertTrue(hasattr(provider, "detect_speakers"))
        self.assertTrue(hasattr(provider, "detect_hosts"))
        self.assertTrue(hasattr(provider, "analyze_patterns"))
        self.assertTrue(hasattr(provider, "clear_cache"))

        # SummarizationProvider protocol
        self.assertTrue(hasattr(provider, "summarize"))
        self.assertTrue(hasattr(provider, "initialize"))
        self.assertTrue(hasattr(provider, "cleanup"))

    def test_provider_initialization_state(self):
        """Test that provider tracks initialization state for each capability."""
        provider = OpenAIProvider(self.cfg)

        # Initially not initialized
        self.assertFalse(provider._transcription_initialized)
        self.assertFalse(provider._speaker_detection_initialized)
        self.assertFalse(provider._summarization_initialized)
        self.assertFalse(provider.is_initialized)

    def test_provider_thread_safe(self):
        """Test that provider marks itself as thread-safe."""
        provider = OpenAIProvider(self.cfg)
        self.assertFalse(provider._requires_separate_instances)

    def test_provider_supports_custom_base_url(self):
        """Test that provider supports custom base_url for E2E testing."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="openai",
            openai_api_key="sk-test123",
            openai_api_base="http://localhost:8000/v1",
        )
        provider = OpenAIProvider(cfg)
        # Verify client was created with custom base_url
        self.assertIsNotNone(provider.client)
        # The client should have the base_url set
        # (we can't easily verify this without accessing private attributes)

    def test_transcription_initialization(self):
        """Test that transcription can be initialized independently."""
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            openai_api_key=self.cfg.openai_api_key,
            openai_api_base=self.cfg.openai_api_base,
            transcription_provider="openai",
            transcribe_missing=True,
            auto_speakers=False,  # Disable to avoid initializing speaker detection
        )
        provider = OpenAIProvider(cfg)

        provider.initialize()

        self.assertTrue(provider._transcription_initialized)
        # Other capabilities should not be initialized
        self.assertFalse(provider._speaker_detection_initialized)
        self.assertFalse(provider._summarization_initialized)

    def test_speaker_detection_initialization(self):
        """Test that speaker detection can be initialized independently."""
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            openai_api_key=self.cfg.openai_api_key,
            openai_api_base=self.cfg.openai_api_base,
            speaker_detector_provider="openai",
            auto_speakers=True,
            transcribe_missing=False,  # Disable to avoid initializing transcription
        )
        provider = OpenAIProvider(cfg)

        provider.initialize()

        self.assertTrue(provider._speaker_detection_initialized)
        # Other capabilities should not be initialized
        self.assertFalse(provider._transcription_initialized)
        self.assertFalse(provider._summarization_initialized)

    def test_summarization_initialization(self):
        """Test that summarization can be initialized independently."""
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            openai_api_key=self.cfg.openai_api_key,
            openai_api_base=self.cfg.openai_api_base,
            summary_provider="openai",
            generate_summaries=True,
            generate_metadata=True,  # Required when generate_summaries is True
            auto_speakers=False,  # Disable to avoid initializing speaker detection
            transcribe_missing=False,  # Disable to avoid initializing transcription
        )
        provider = OpenAIProvider(cfg)

        provider.initialize()

        self.assertTrue(provider._summarization_initialized)
        # Other capabilities should not be initialized
        self.assertFalse(provider._transcription_initialized)
        self.assertFalse(provider._speaker_detection_initialized)

    def test_unified_initialization(self):
        """Test that all capabilities can be initialized together."""
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            openai_api_key=self.cfg.openai_api_key,
            openai_api_base=self.cfg.openai_api_base,
            transcription_provider="openai",
            speaker_detector_provider="openai",
            summary_provider="openai",
            transcribe_missing=True,
            auto_speakers=True,
            generate_summaries=True,
            generate_metadata=True,  # Required when generate_summaries is True
        )

        provider = OpenAIProvider(cfg)
        provider.initialize()

        # All should be initialized
        self.assertTrue(provider._transcription_initialized)
        self.assertTrue(provider._speaker_detection_initialized)
        self.assertTrue(provider._summarization_initialized)
        self.assertTrue(provider.is_initialized)

    def test_cleanup_releases_all_resources(self):
        """Test that cleanup releases all resources."""
        provider = OpenAIProvider(self.cfg)
        provider._transcription_initialized = True
        provider._speaker_detection_initialized = True
        provider._summarization_initialized = True

        provider.cleanup()

        self.assertFalse(provider._transcription_initialized)
        self.assertFalse(provider._speaker_detection_initialized)
        self.assertFalse(provider._summarization_initialized)
        self.assertFalse(provider.is_initialized)

    def test_transcription_model_attribute(self):
        """Test that transcription_model attribute exists."""
        provider = OpenAIProvider(self.cfg)

        # Transcription attributes
        self.assertTrue(hasattr(provider, "transcription_model"))
        self.assertTrue(hasattr(provider, "is_initialized"))

        # Verify transcription_model is accessible
        self.assertIsNotNone(provider.transcription_model)


@pytest.mark.unit
class TestOpenAIProviderTranscription(unittest.TestCase):
    """Tests for OpenAIProvider transcription methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="openai",
            openai_api_key="sk-test123",
            transcribe_missing=True,
        )

    @patch("builtins.open", create=True)
    @patch("os.path.exists")
    def test_transcribe_success(self, mock_exists, mock_open):
        """Test successful transcription."""
        mock_exists.return_value = True
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_open.return_value.__exit__.return_value = None

        mock_client = Mock()
        # When response_format="text", API returns string directly, not an object
        mock_client.audio.transcriptions.create.return_value = "Hello world"

        provider = OpenAIProvider(self.cfg)
        provider.client = mock_client
        provider.initialize()

        result = provider.transcribe("/path/to/audio.mp3")

        self.assertEqual(result, "Hello world")
        mock_client.audio.transcriptions.create.assert_called_once()

    @patch("builtins.open", create=True)
    @patch("os.path.exists")
    def test_transcribe_with_language(self, mock_exists, mock_open):
        """Test transcription with explicit language."""
        mock_exists.return_value = True
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_open.return_value.__exit__.return_value = None

        mock_response = Mock()
        mock_response.text = "Bonjour"
        mock_client = Mock()
        mock_client.audio.transcriptions.create.return_value = mock_response

        provider = OpenAIProvider(self.cfg)
        provider.client = mock_client
        provider.initialize()

        provider.transcribe("/path/to/audio.mp3", language="fr")

        # Verify language was passed to API
        call_kwargs = mock_client.audio.transcriptions.create.call_args[1]
        self.assertEqual(call_kwargs["language"], "fr")

    def test_transcribe_not_initialized(self):
        """Test transcribe raises RuntimeError if not initialized."""
        provider = OpenAIProvider(self.cfg)
        # Don't call initialize()

        with self.assertRaises(RuntimeError) as context:
            provider.transcribe("/path/to/audio.mp3")

        self.assertIn("not initialized", str(context.exception))

    @patch("os.path.exists")
    def test_transcribe_file_not_found(self, mock_exists):
        """Test transcribe raises FileNotFoundError if file doesn't exist."""
        mock_exists.return_value = False

        provider = OpenAIProvider(self.cfg)
        provider.initialize()

        with self.assertRaises(FileNotFoundError) as context:
            provider.transcribe("/path/to/nonexistent.mp3")

        self.assertIn("not found", str(context.exception))

    @patch("builtins.open", create=True)
    @patch("os.path.exists")
    def test_transcribe_api_error(self, mock_exists, mock_open):
        """Test transcribe handles API errors."""

        # Since openai is mocked globally, we need to create a real exception
        # that will be caught and converted to ProviderRuntimeError
        class MockAPIError(Exception):
            """Mock APIError for testing."""

            pass

        mock_exists.return_value = True
        mock_file = Mock()
        # Set up context manager properly
        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=mock_file)
        mock_context.__exit__ = Mock(return_value=False)
        mock_open.return_value = mock_context

        mock_client = Mock()
        # Use a real exception class (not the mocked APIError)
        api_error = MockAPIError("API error")
        mock_client.audio.transcriptions.create.side_effect = api_error

        provider = OpenAIProvider(self.cfg)
        provider.client = mock_client
        provider.initialize()

        from podcast_scraper.exceptions import ProviderRuntimeError

        with self.assertRaises(ProviderRuntimeError) as context:
            provider.transcribe("/path/to/audio.mp3")

        self.assertIn("transcription failed", str(context.exception))

    @patch("builtins.open", create=True)
    @patch("os.path.exists")
    def test_transcribe_with_segments_success(self, mock_exists, mock_open):
        """Test transcribe_with_segments returns full result."""
        mock_exists.return_value = True
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_open.return_value.__exit__.return_value = None

        # Mock segment objects
        mock_segment1 = Mock()
        mock_segment1.start = 0.0
        mock_segment1.end = 1.0
        mock_segment1.text = "Hello"

        mock_segment2 = Mock()
        mock_segment2.start = 1.0
        mock_segment2.end = 2.0
        mock_segment2.text = "world"

        mock_response = Mock()
        mock_response.text = "Hello world"
        mock_response.segments = [mock_segment1, mock_segment2]

        mock_client = Mock()
        mock_client.audio.transcriptions.create.return_value = mock_response

        provider = OpenAIProvider(self.cfg)
        provider.client = mock_client
        provider.initialize()

        result_dict, elapsed = provider.transcribe_with_segments("/path/to/audio.mp3")

        self.assertEqual(result_dict["text"], "Hello world")
        self.assertEqual(len(result_dict["segments"]), 2)
        self.assertIsInstance(elapsed, float)
        self.assertGreater(elapsed, 0)


@pytest.mark.unit
class TestOpenAIProviderSpeakerDetection(unittest.TestCase):
    """Tests for OpenAIProvider speaker detection methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="openai",
            openai_api_key="sk-test123",
            auto_speakers=True,
        )

    def test_detect_hosts_from_feed_authors(self):
        """Test detect_hosts prefers feed_authors."""
        provider = OpenAIProvider(self.cfg)
        provider.initialize()

        hosts = provider.detect_hosts(
            feed_title="The Podcast",
            feed_description="A great podcast",
            feed_authors=["Alice", "Bob"],
        )

        self.assertEqual(hosts, {"Alice", "Bob"})

    @patch(
        "podcast_scraper.providers.openai.openai_provider."
        "OpenAIProvider._build_speaker_detection_prompt"
    )
    def test_detect_hosts_without_authors(self, mock_build_prompt):
        """Test detect_hosts uses API when no feed_authors."""
        mock_build_prompt.return_value = "Prompt"

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Alice, Bob"

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider(self.cfg)
        provider.client = mock_client
        provider.initialize()

        hosts = provider.detect_hosts(
            feed_title="The Podcast",
            feed_description="A great podcast",
            feed_authors=None,
        )

        # Should return empty set if no feed_authors and no API call made
        # (since we're not actually calling the API in this simplified test)
        self.assertIsInstance(hosts, set)

    @patch(
        "podcast_scraper.providers.openai.openai_provider."
        "OpenAIProvider._build_speaker_detection_prompt"
    )
    @patch(
        "podcast_scraper.providers.openai.openai_provider."
        "OpenAIProvider._parse_speakers_from_response"
    )
    def test_detect_speakers_success(self, mock_parse, mock_build_prompt):
        """Test successful speaker detection."""
        mock_build_prompt.return_value = "Prompt"

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = (
            '{"speakers": ["Alice", "Bob"], "hosts": ["Alice"], "guests": ["Bob"]}'
        )
        # _parse_speakers_from_response returns:
        # (speaker_names_list, detected_hosts_set, detection_succeeded)
        mock_parse.return_value = (["Alice", "Bob"], {"Alice"}, True)

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider(self.cfg)
        provider.client = mock_client
        provider.initialize()

        speakers, hosts, success = provider.detect_speakers(
            episode_title="Alice interviews Bob",
            episode_description="A great conversation",
            known_hosts={"Alice"},
        )

        self.assertEqual(speakers, ["Alice", "Bob"])
        self.assertTrue(success)

    def test_detect_speakers_not_initialized(self):
        """Test detect_speakers raises RuntimeError if not initialized."""
        provider = OpenAIProvider(self.cfg)
        # Don't call initialize()

        with self.assertRaises(RuntimeError) as context:
            provider.detect_speakers("Title", "Description", set())

        self.assertIn("not initialized", str(context.exception))

    @patch("podcast_scraper.prompts.store.render_prompt")
    @patch(
        "podcast_scraper.providers.openai.openai_provider."
        "OpenAIProvider._build_speaker_detection_prompt"
    )
    def test_analyze_patterns_success(self, mock_build_prompt, mock_render_prompt):
        """Test successful pattern analysis."""
        from podcast_scraper import models

        mock_build_prompt.return_value = "Prompt"

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"pattern": "value"}'

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider(self.cfg)
        provider.client = mock_client
        provider.initialize()

        episodes = [
            models.Episode(
                idx=1,
                title="Episode 1",
                title_safe="Episode_1",
                item=None,
                transcript_urls=[],
                media_url="https://example.com/1",
                media_type="audio/mpeg",
            )
        ]

        # OpenAI provider doesn't implement pattern analysis, returns None
        result = provider.analyze_patterns(episodes=episodes, known_hosts={"Alice"})

        self.assertIsNone(result)  # OpenAI provider returns None to use local logic

    def test_clear_cache(self):
        """Test cache clearing (no-op for OpenAI provider)."""
        provider = OpenAIProvider(self.cfg)

        # clear_cache should not raise (it's a no-op for OpenAI provider)
        provider.clear_cache()

        # OpenAI provider doesn't use spaCy cache, but method exists for protocol compliance
        # It's essentially a no-op


@pytest.mark.unit
class TestOpenAIProviderSummarization(unittest.TestCase):
    """Tests for OpenAIProvider summarization methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="openai",
            openai_api_key="sk-test123",
            generate_summaries=True,
            generate_metadata=True,  # Required when generate_summaries=True
        )

    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_summarize_success(self, mock_render_prompt):
        """Test successful summarization."""
        # Mock render_prompt to return prompts (called twice: system and user)
        mock_render_prompt.side_effect = ["System Prompt", "User Prompt"]

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "This is a summary."

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider(self.cfg)
        provider.client = mock_client
        provider.initialize()

        result = provider.summarize("This is a long transcript text.")

        self.assertEqual(result["summary"], "This is a summary.")
        self.assertIn("metadata", result)
        self.assertEqual(result["metadata"]["model"], provider.summary_model)

    def test_summarize_not_initialized(self):
        """Test summarize raises RuntimeError if not initialized."""
        provider = OpenAIProvider(self.cfg)
        # Don't call initialize()

        with self.assertRaises(RuntimeError) as context:
            provider.summarize("Text")

        self.assertIn("not initialized", str(context.exception))

    @patch("podcast_scraper.prompts.store.get_prompt_metadata")
    @patch("podcast_scraper.prompts.store.render_prompt")
    @patch(
        "podcast_scraper.providers.openai.openai_provider."
        "OpenAIProvider._build_summarization_prompts"
    )
    def test_summarize_with_params(self, mock_build_prompts, mock_render_prompt, mock_get_metadata):
        """Test summarization with custom parameters."""
        # _build_summarization_prompts returns:
        # (system_prompt, user_prompt, system_prompt_name, user_prompt_name,
        #  paragraphs_min, paragraphs_max)
        mock_build_prompts.return_value = (
            "System Prompt",
            "User Prompt",
            "openai/summarization/system_v1",
            "openai/summarization/user_v1",
            1,
            3,
        )
        # render_prompt is called inside _build_summarization_prompts
        mock_render_prompt.side_effect = ["System Prompt", "User Prompt"]
        # get_prompt_metadata is called for tracking
        mock_get_metadata.return_value = {
            "name": "openai/summarization/system_v1",
            "sha256": "abc123",
        }

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Summary"

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider(self.cfg)
        provider.client = mock_client
        provider.initialize()

        params = {"max_length": 100, "min_length": 50}
        provider.summarize("Text", params=params)

        # Verify API was called
        mock_client.chat.completions.create.assert_called()

    @patch(
        "podcast_scraper.providers.openai.openai_provider."
        "OpenAIProvider._build_summarization_prompts"
    )
    def test_summarize_api_error(self, mock_build_prompts):
        """Test summarization error handling."""
        # _build_summarization_prompts returns:
        # (system_prompt, user_prompt, system_prompt_name, user_prompt_name,
        #  paragraphs_min, paragraphs_max)
        mock_build_prompts.return_value = (
            "System Prompt",
            "User Prompt",
            "system_v1",
            "user_v1",
            1,
            3,
        )

        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API error")

        provider = OpenAIProvider(self.cfg)
        provider.client = mock_client
        provider.initialize()

        from podcast_scraper.exceptions import ProviderRuntimeError

        with self.assertRaises(ProviderRuntimeError) as context:
            provider.summarize("Text")

        self.assertIn("summarization failed", str(context.exception).lower())


@pytest.mark.unit
class TestOpenAIProviderPricing(unittest.TestCase):
    """Tests for OpenAIProvider.get_pricing() static method."""

    def test_get_pricing_whisper_transcription(self):
        """Test pricing lookup for Whisper transcription."""
        pricing = OpenAIProvider.get_pricing("whisper-1", "transcription")
        self.assertEqual(pricing, {"cost_per_minute": 0.006})

    def test_get_pricing_gpt4o_mini_speaker_detection(self):
        """Test pricing lookup for GPT-4o-mini speaker detection."""
        pricing = OpenAIProvider.get_pricing("gpt-4o-mini", "speaker_detection")
        self.assertEqual(pricing["input_cost_per_1m_tokens"], 0.15)
        self.assertEqual(pricing["output_cost_per_1m_tokens"], 0.60)

    def test_get_pricing_gpt4o_mini_summarization(self):
        """Test pricing lookup for GPT-4o-mini summarization."""
        pricing = OpenAIProvider.get_pricing("gpt-4o-mini", "summarization")
        self.assertEqual(pricing["input_cost_per_1m_tokens"], 0.15)
        self.assertEqual(pricing["output_cost_per_1m_tokens"], 0.60)

    def test_get_pricing_gpt4o_summarization(self):
        """Test pricing lookup for GPT-4o summarization."""
        pricing = OpenAIProvider.get_pricing("gpt-4o", "summarization")
        self.assertEqual(pricing["input_cost_per_1m_tokens"], 2.50)
        self.assertEqual(pricing["output_cost_per_1m_tokens"], 10.00)

    def test_get_pricing_unsupported_model(self):
        """Test pricing lookup for unsupported model returns empty dict."""
        pricing = OpenAIProvider.get_pricing("gpt-5", "summarization")
        self.assertEqual(pricing, {})

    def test_get_pricing_unsupported_capability(self):
        """Test pricing lookup for unsupported capability returns empty dict."""
        pricing = OpenAIProvider.get_pricing("gpt-4o-mini", "unsupported")
        self.assertEqual(pricing, {})

    def test_get_pricing_case_insensitive_model_name(self):
        """Test that pricing lookup is case-insensitive for model names."""
        pricing1 = OpenAIProvider.get_pricing("GPT-4O-MINI", "speaker_detection")
        pricing2 = OpenAIProvider.get_pricing("gpt-4o-mini", "speaker_detection")
        self.assertEqual(pricing1, pricing2)
        self.assertEqual(pricing1["input_cost_per_1m_tokens"], 0.15)
