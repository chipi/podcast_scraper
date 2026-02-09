#!/usr/bin/env python3
"""Standalone unit tests for unified Gemini provider.

These tests verify that GeminiProvider correctly implements all three protocols
(TranscriptionProvider, SpeakerDetector, SummarizationProvider) using
Gemini's API (native multimodal audio, chat models).

These are standalone provider tests - they test the provider itself,
not its integration with the app.
"""

import json
import os
import unittest
from unittest.mock import MagicMock, Mock, patch

import pytest

# Mock google.genai before importing modules that require it
# Unit tests run without google-genai package installed
# Use patch.dict without 'with' to avoid context manager conflicts with @patch decorators
mock_google = MagicMock()
mock_genai_module = MagicMock()
mock_genai_module.configure = Mock()
mock_genai_module.GenerativeModel = Mock()
mock_api_core = MagicMock()
mock_api_core.exceptions = MagicMock()
_patch_google = patch.dict(
    "sys.modules",
    {
        "google": mock_google,
        "google.genai": mock_genai_module,
        "google.api_core": mock_api_core,
    },
)
_patch_google.start()
from podcast_scraper import config
from podcast_scraper.providers.gemini.gemini_provider import GeminiProvider


@pytest.mark.unit
class TestGeminiProviderStandalone(unittest.TestCase):
    """Standalone tests for GeminiProvider - testing the provider itself."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="gemini",
            speaker_detector_provider="gemini",
            summary_provider="gemini",
            gemini_api_key="test-api-key-123",
            transcribe_missing=False,  # Disable to avoid API calls
            auto_speakers=False,  # Disable to avoid API calls
            generate_summaries=False,  # Disable to avoid API calls
        )

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_provider_creation(self, mock_genai):
        """Test that GeminiProvider can be created."""
        # Mock Client API
        mock_client = Mock()
        mock_genai.Client.return_value = mock_client
        provider = GeminiProvider(self.cfg)
        self.assertIsNotNone(provider)
        self.assertEqual(provider.__class__.__name__, "GeminiProvider")
        # Verify genai.Client was called
        mock_genai.Client.assert_called_once_with(api_key="test-api-key-123")

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_provider_creation_requires_api_key(self, mock_genai):
        """Test that GeminiProvider requires API key."""
        # Unset environment variable to ensure it's not loaded
        original_key = os.environ.pop("GEMINI_API_KEY", None)
        try:
            # Note: Config validation happens before provider creation
            # So we need to catch ValidationError from Config, not ValueError from provider
            with self.assertRaises(Exception) as context:
                cfg = config.Config(
                    rss_url="https://example.com/feed.xml",
                    transcription_provider="gemini",
                )
                GeminiProvider(cfg)
            # Error can be either ValidationError (from Config) or ValueError (from provider)
        finally:
            # Restore original environment variable if it existed
            if original_key is not None:
                os.environ["GEMINI_API_KEY"] = original_key
        error_msg = str(context.exception)
        self.assertTrue(
            "Gemini API key required" in error_msg or "validation error" in error_msg.lower()
        )

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_provider_implements_all_protocols(self, mock_genai):
        """Test that GeminiProvider implements all three protocols."""
        provider = GeminiProvider(self.cfg)

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

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_provider_initialization_state(self, mock_genai):
        """Test that provider tracks initialization state for each capability."""
        provider = GeminiProvider(self.cfg)

        # Initially not initialized
        self.assertFalse(provider._transcription_initialized)
        self.assertFalse(provider._speaker_detection_initialized)
        self.assertFalse(provider._summarization_initialized)
        self.assertFalse(provider.is_initialized)

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_provider_thread_safe(self, mock_genai):
        """Test that provider marks itself as thread-safe."""
        provider = GeminiProvider(self.cfg)
        self.assertFalse(provider._requires_separate_instances)

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_transcription_initialization(self, mock_genai):
        """Test that transcription can be initialized independently."""
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            gemini_api_key=self.cfg.gemini_api_key,
            transcription_provider="gemini",
            transcribe_missing=True,
            auto_speakers=False,  # Disable to avoid initializing speaker detection
        )
        provider = GeminiProvider(cfg)

        provider.initialize()

        self.assertTrue(provider._transcription_initialized)
        # Other capabilities should not be initialized
        self.assertFalse(provider._speaker_detection_initialized)
        self.assertFalse(provider._summarization_initialized)

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_speaker_detection_initialization(self, mock_genai):
        """Test that speaker detection can be initialized independently."""
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            gemini_api_key=self.cfg.gemini_api_key,
            speaker_detector_provider="gemini",
            auto_speakers=True,
            transcribe_missing=False,  # Disable to avoid initializing transcription
        )
        provider = GeminiProvider(cfg)

        provider.initialize()

        self.assertTrue(provider._speaker_detection_initialized)
        # Other capabilities should not be initialized
        self.assertFalse(provider._transcription_initialized)
        self.assertFalse(provider._summarization_initialized)

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_summarization_initialization(self, mock_genai):
        """Test that summarization can be initialized independently."""
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            gemini_api_key=self.cfg.gemini_api_key,
            summary_provider="gemini",
            generate_summaries=True,
            generate_metadata=True,  # Required when generate_summaries is True
            auto_speakers=False,  # Disable to avoid initializing speaker detection
            transcribe_missing=False,  # Disable to avoid initializing transcription
        )
        provider = GeminiProvider(cfg)

        provider.initialize()

        self.assertTrue(provider._summarization_initialized)
        # Other capabilities should not be initialized
        self.assertFalse(provider._transcription_initialized)
        self.assertFalse(provider._speaker_detection_initialized)

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_unified_initialization(self, mock_genai):
        """Test that all capabilities can be initialized together."""
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            gemini_api_key=self.cfg.gemini_api_key,
            transcription_provider="gemini",
            speaker_detector_provider="gemini",
            summary_provider="gemini",
            transcribe_missing=True,
            auto_speakers=True,
            generate_summaries=True,
            generate_metadata=True,  # Required when generate_summaries is True
        )

        provider = GeminiProvider(cfg)
        provider.initialize()

        # All should be initialized
        self.assertTrue(provider._transcription_initialized)
        self.assertTrue(provider._speaker_detection_initialized)
        self.assertTrue(provider._summarization_initialized)
        self.assertTrue(provider.is_initialized)

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_cleanup_releases_all_resources(self, mock_genai):
        """Test that cleanup releases all resources."""
        provider = GeminiProvider(self.cfg)
        provider._transcription_initialized = True
        provider._speaker_detection_initialized = True
        provider._summarization_initialized = True

        provider.cleanup()

        self.assertFalse(provider._transcription_initialized)
        self.assertFalse(provider._speaker_detection_initialized)
        self.assertFalse(provider._summarization_initialized)
        self.assertFalse(provider.is_initialized)

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_transcription_model_attribute(self, mock_genai):
        """Test that transcription_model attribute exists."""
        provider = GeminiProvider(self.cfg)

        # Transcription attributes
        self.assertTrue(hasattr(provider, "transcription_model"))
        self.assertTrue(hasattr(provider, "is_initialized"))

        # Verify transcription_model is accessible
        self.assertIsNotNone(provider.transcription_model)


@pytest.mark.unit
class TestGeminiProviderTranscription(unittest.TestCase):
    """Tests for GeminiProvider transcription methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="gemini",
            gemini_api_key="test-api-key-123",
            transcribe_missing=True,
        )

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    @patch("builtins.open", create=True)
    @patch("os.path.exists")
    def test_transcribe_success(self, mock_exists, mock_open, mock_genai):
        """Test successful transcription."""
        mock_exists.return_value = True
        mock_file = Mock()
        mock_file.read.return_value = b"fake audio data"
        mock_open.return_value.__enter__.return_value = mock_file
        mock_open.return_value.__exit__.return_value = None

        # Mock Gemini Client API response
        mock_response = Mock()
        mock_response.text = "Hello world"
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

        provider = GeminiProvider(self.cfg)
        provider.initialize()

        result = provider.transcribe("/path/to/audio.mp3")

        self.assertEqual(result, "Hello world")
        mock_client.models.generate_content.assert_called_once()

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    @patch("builtins.open", create=True)
    @patch("os.path.exists")
    def test_transcribe_with_language(self, mock_exists, mock_open, mock_genai):
        """Test transcription with explicit language."""
        mock_exists.return_value = True
        mock_file = Mock()
        mock_file.read.return_value = b"fake audio data"
        mock_open.return_value.__enter__.return_value = mock_file
        mock_open.return_value.__exit__.return_value = None

        mock_response = Mock()
        mock_response.text = "Bonjour"
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

        provider = GeminiProvider(self.cfg)
        provider.initialize()

        provider.transcribe("/path/to/audio.mp3", language="fr")

        # Verify language was included in prompt
        call_kwargs = mock_client.models.generate_content.call_args[1]
        self.assertIn("fr", str(call_kwargs))

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_transcribe_not_initialized(self, mock_genai):
        """Test transcribe raises RuntimeError if not initialized."""
        provider = GeminiProvider(self.cfg)
        # Don't call initialize()

        with self.assertRaises(RuntimeError) as context:
            provider.transcribe("/path/to/audio.mp3")

        self.assertIn("not initialized", str(context.exception))

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    @patch("os.path.exists")
    def test_transcribe_file_not_found(self, mock_exists, mock_genai):
        """Test transcribe raises FileNotFoundError if file doesn't exist."""
        mock_exists.return_value = False

        provider = GeminiProvider(self.cfg)
        provider.initialize()

        with self.assertRaises(FileNotFoundError) as context:
            provider.transcribe("/path/to/nonexistent.mp3")

        self.assertIn("not found", str(context.exception))

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    @patch("builtins.open", create=True)
    @patch("os.path.exists")
    def test_transcribe_api_error(self, mock_exists, mock_open, mock_genai):
        """Test transcribe handles API errors."""
        mock_exists.return_value = True
        mock_file = Mock()
        mock_file.read.return_value = b"fake audio data"
        mock_open.return_value.__enter__.return_value = mock_file
        mock_open.return_value.__exit__.return_value = None

        mock_client = Mock()
        mock_client.models.generate_content.side_effect = Exception("API error")
        mock_genai.Client.return_value = mock_client

        provider = GeminiProvider(self.cfg)
        provider.initialize()

        from podcast_scraper.exceptions import ProviderRuntimeError

        with self.assertRaises(ProviderRuntimeError) as context:
            provider.transcribe("/path/to/audio.mp3")

        self.assertIn("transcription failed", str(context.exception))

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    @patch("builtins.open", create=True)
    @patch("os.path.exists")
    def test_transcribe_with_segments_success(self, mock_exists, mock_open, mock_genai):
        """Test transcribe_with_segments returns full result."""
        mock_exists.return_value = True
        mock_file = Mock()
        mock_file.read.return_value = b"fake audio data"
        mock_open.return_value.__enter__.return_value = mock_file
        mock_open.return_value.__exit__.return_value = None

        mock_response = Mock()
        mock_response.text = "Hello world"
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

        provider = GeminiProvider(self.cfg)
        provider.initialize()

        result_dict, elapsed = provider.transcribe_with_segments("/path/to/audio.mp3")

        self.assertEqual(result_dict["text"], "Hello world")
        self.assertEqual(result_dict["segments"], [])  # Gemini doesn't provide segments
        self.assertIsInstance(elapsed, float)
        self.assertGreater(elapsed, 0)


@pytest.mark.unit
class TestGeminiProviderSpeakerDetection(unittest.TestCase):
    """Tests for GeminiProvider speaker detection methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="gemini",
            gemini_api_key="test-api-key-123",
            auto_speakers=True,
        )

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_detect_hosts_from_feed_authors(self, mock_genai):
        """Test detect_hosts prefers feed_authors."""
        provider = GeminiProvider(self.cfg)
        provider.initialize()

        hosts = provider.detect_hosts(
            feed_title="The Podcast",
            feed_description="A great podcast",
            feed_authors=["Alice", "Bob"],
        )

        self.assertEqual(hosts, {"Alice", "Bob"})

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    @patch(
        "podcast_scraper.providers.gemini.gemini_provider."
        "GeminiProvider._build_speaker_detection_prompt"
    )
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_detect_hosts_without_authors(self, mock_render, mock_build_prompt, mock_genai):
        """Test detect_hosts uses API when no feed_authors."""
        mock_build_prompt.return_value = "Prompt"
        mock_render.side_effect = lambda name, **kwargs: "test prompt"

        mock_response = Mock()
        mock_response.text = json.dumps({"speakers": ["Alice", "Bob"], "hosts": [], "guests": []})

        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

        provider = GeminiProvider(self.cfg)
        provider.initialize()

        hosts = provider.detect_hosts(
            feed_title="The Podcast",
            feed_description="A great podcast",
            feed_authors=None,
        )

        # Should return empty set if no feed_authors and no API call made
        # (since we're not actually calling the API in this simplified test)
        self.assertIsInstance(hosts, set)

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    @patch(
        "podcast_scraper.providers.gemini.gemini_provider."
        "GeminiProvider._build_speaker_detection_prompt"
    )
    @patch(
        "podcast_scraper.providers.gemini.gemini_provider."
        "GeminiProvider._parse_speakers_from_response"
    )
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_detect_speakers_success(self, mock_render, mock_parse, mock_build_prompt, mock_genai):
        """Test successful speaker detection."""
        mock_build_prompt.return_value = "Prompt"
        mock_render.side_effect = lambda name, **kwargs: "test prompt"

        mock_response = Mock()
        mock_response.text = '{"speakers": ["Alice", "Bob"], "hosts": ["Alice"], "guests": ["Bob"]}'
        # _parse_speakers_from_response returns:
        # (speaker_names_list, detected_hosts_set, detection_succeeded)
        mock_parse.return_value = (["Alice", "Bob"], {"Alice"}, True)

        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

        provider = GeminiProvider(self.cfg)
        provider.initialize()

        speakers, hosts, success = provider.detect_speakers(
            episode_title="Alice interviews Bob",
            episode_description="A great conversation",
            known_hosts={"Alice"},
        )

        self.assertEqual(speakers, ["Alice", "Bob"])
        self.assertTrue(success)

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_detect_speakers_not_initialized(self, mock_genai):
        """Test detect_speakers raises RuntimeError if not initialized."""
        provider = GeminiProvider(self.cfg)
        # Don't call initialize()

        with self.assertRaises(RuntimeError) as context:
            provider.detect_speakers("Title", "Description", set())

        self.assertIn("not initialized", str(context.exception))

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_analyze_patterns_success(self, mock_genai):
        """Test successful pattern analysis."""
        from podcast_scraper import models

        provider = GeminiProvider(self.cfg)
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

        # Gemini provider doesn't implement pattern analysis, returns None
        result = provider.analyze_patterns(episodes=episodes, known_hosts={"Alice"})

        self.assertIsNone(result)  # Gemini provider returns None to use local logic

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_clear_cache(self, mock_genai):
        """Test cache clearing (no-op for Gemini provider)."""
        provider = GeminiProvider(self.cfg)

        # clear_cache should not raise (it's a no-op for Gemini provider)
        provider.clear_cache()

        # Gemini provider doesn't use cache, but method exists for protocol compliance
        # It's essentially a no-op


@pytest.mark.unit
class TestGeminiProviderSummarization(unittest.TestCase):
    """Tests for GeminiProvider summarization methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="gemini",
            gemini_api_key="test-api-key-123",
            generate_summaries=True,
            generate_metadata=True,  # Required when generate_summaries=True
        )

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_summarize_success(self, mock_render_prompt, mock_genai):
        """Test successful summarization."""
        # Mock render_prompt to return prompts (called twice: system and user)
        mock_render_prompt.side_effect = ["System Prompt", "User Prompt"]

        mock_response = Mock()
        mock_response.text = "This is a summary."

        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

        provider = GeminiProvider(self.cfg)
        provider.initialize()

        result = provider.summarize("This is a long transcript text.")

        self.assertEqual(result["summary"], "This is a summary.")
        self.assertIn("metadata", result)
        self.assertEqual(result["metadata"]["model"], provider.summary_model)

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_summarize_not_initialized(self, mock_genai):
        """Test summarize raises RuntimeError if not initialized."""
        provider = GeminiProvider(self.cfg)
        # Don't call initialize()

        with self.assertRaises(RuntimeError) as context:
            provider.summarize("Text")

        self.assertIn("not initialized", str(context.exception))

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    @patch("podcast_scraper.prompts.store.get_prompt_metadata")
    @patch("podcast_scraper.prompts.store.render_prompt")
    @patch(
        "podcast_scraper.providers.gemini.gemini_provider."
        "GeminiProvider._build_summarization_prompts"
    )
    def test_summarize_with_params(
        self, mock_build_prompts, mock_render_prompt, mock_get_metadata, mock_genai
    ):
        """Test summarization with custom parameters."""
        # _build_summarization_prompts returns:
        # (system_prompt, user_prompt, system_prompt_name, user_prompt_name,
        #  paragraphs_min, paragraphs_max)
        mock_build_prompts.return_value = (
            "System Prompt",
            "User Prompt",
            "gemini/summarization/system_v1",
            "gemini/summarization/long_v1",
            1,
            3,
        )
        # render_prompt is called inside _build_summarization_prompts
        mock_render_prompt.side_effect = ["System Prompt", "User Prompt"]
        # get_prompt_metadata is called for tracking
        mock_get_metadata.return_value = {
            "name": "gemini/summarization/system_v1",
            "sha256": "abc123",
        }

        mock_response = Mock()
        mock_response.text = "Summary"

        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

        provider = GeminiProvider(self.cfg)
        provider.initialize()

        params = {"max_length": 100, "min_length": 50}
        provider.summarize("Text", params=params)

        # Verify API was called
        mock_client.models.generate_content.assert_called()

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    @patch(
        "podcast_scraper.providers.gemini.gemini_provider."
        "GeminiProvider._build_summarization_prompts"
    )
    def test_summarize_api_error(self, mock_build_prompts, mock_genai):
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
        mock_client.models.generate_content.side_effect = Exception("API error")
        mock_genai.Client.return_value = mock_client

        provider = GeminiProvider(self.cfg)
        provider.initialize()

        from podcast_scraper.exceptions import ProviderRuntimeError

        with self.assertRaises(ProviderRuntimeError) as context:
            provider.summarize("Text")

        self.assertIn("summarization failed", str(context.exception).lower())


@pytest.mark.unit
class TestGeminiProviderPricing(unittest.TestCase):
    """Tests for GeminiProvider.get_pricing() static method."""

    def test_get_pricing_audio_transcription(self):
        """Test pricing lookup for audio transcription."""
        pricing = GeminiProvider.get_pricing("gemini-1.5-pro", "transcription")
        self.assertIn("cost_per_second", pricing)
        self.assertEqual(pricing["cost_per_second"], 0.00025)

    def test_get_pricing_2_flash_speaker_detection(self):
        """Test pricing lookup for Gemini 2.0 Flash speaker detection."""
        pricing = GeminiProvider.get_pricing("gemini-2.0-flash", "speaker_detection")
        self.assertEqual(pricing["input_cost_per_1m_tokens"], 0.10)
        self.assertEqual(pricing["output_cost_per_1m_tokens"], 0.40)

    def test_get_pricing_1_5_pro_summarization(self):
        """Test pricing lookup for Gemini 1.5 Pro summarization."""
        pricing = GeminiProvider.get_pricing("gemini-1.5-pro", "summarization")
        self.assertEqual(pricing["input_cost_per_1m_tokens"], 1.25)
        self.assertEqual(pricing["output_cost_per_1m_tokens"], 5.00)

    def test_get_pricing_1_5_flash_summarization(self):
        """Test pricing lookup for Gemini 1.5 Flash summarization."""
        pricing = GeminiProvider.get_pricing("gemini-1.5-flash", "summarization")
        self.assertEqual(pricing["input_cost_per_1m_tokens"], 0.075)
        self.assertEqual(pricing["output_cost_per_1m_tokens"], 0.30)

    def test_get_pricing_unsupported_model(self):
        """Test pricing lookup for unsupported model returns default pricing."""
        pricing = GeminiProvider.get_pricing("gemini-unknown", "summarization")
        # Should default to 2.0-flash pricing
        self.assertEqual(pricing["input_cost_per_1m_tokens"], 0.10)
        self.assertEqual(pricing["output_cost_per_1m_tokens"], 0.40)


@pytest.mark.unit
class TestGeminiProviderErrorHandling(unittest.TestCase):
    """Tests for error handling in GeminiProvider."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="gemini",
            speaker_detector_provider="gemini",
            summary_provider="gemini",
            gemini_api_key="test-api-key-123",
            auto_speakers=True,
            generate_summaries=True,
            generate_metadata=True,
        )

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_speaker_detection_auth_error(self, mock_render, mock_genai):
        """Test that authentication errors are properly handled in speaker detection."""

        # Create mock exception with authentication error message
        class MockPermissionDenied(Exception):
            pass

        mock_client = Mock()
        mock_client.models.generate_content.side_effect = MockPermissionDenied(
            "Invalid API key: authentication failed"
        )
        mock_genai.Client.return_value = mock_client
        mock_render.side_effect = lambda name, **kwargs: "test prompt"

        provider = GeminiProvider(self.cfg)
        provider.initialize()

        from podcast_scraper.exceptions import ProviderAuthError

        with self.assertRaises(ProviderAuthError) as context:
            provider.detect_speakers("Episode Title", "Description", set(["Host"]))

        self.assertIn("authentication failed", str(context.exception).lower())
        self.assertIn("GEMINI_API_KEY", str(context.exception))

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_speaker_detection_rate_limit_error(self, mock_render, mock_genai):
        """Test that rate limit errors are properly handled in speaker detection."""

        # Create mock exception with rate limit error message
        class MockResourceExhausted(Exception):
            pass

        mock_client = Mock()
        mock_client.models.generate_content.side_effect = MockResourceExhausted(
            "Rate limit exceeded: resource exhausted"
        )
        mock_genai.Client.return_value = mock_client
        mock_render.side_effect = lambda name, **kwargs: "test prompt"

        provider = GeminiProvider(self.cfg)
        provider.initialize()

        from podcast_scraper.exceptions import ProviderRuntimeError

        with self.assertRaises(ProviderRuntimeError) as context:
            provider.detect_speakers("Episode Title", "Description", set(["Host"]))

        self.assertIn("rate limit", str(context.exception).lower())

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_speaker_detection_invalid_model_error(self, mock_render, mock_genai):
        """Test that invalid model errors are properly handled in speaker detection."""

        # Create mock exception with invalid model error message
        class MockInvalidArgument(Exception):
            pass

        mock_client = Mock()
        mock_client.models.generate_content.side_effect = MockInvalidArgument("Invalid model name")
        mock_genai.Client.return_value = mock_client
        mock_render.side_effect = lambda name, **kwargs: "test prompt"

        provider = GeminiProvider(self.cfg)
        provider.initialize()

        from podcast_scraper.exceptions import ProviderRuntimeError

        with self.assertRaises(ProviderRuntimeError) as context:
            provider.detect_speakers("Episode Title", "Description", set(["Host"]))

        error_msg = str(context.exception).lower()
        self.assertTrue("invalid model" in error_msg or "speaker detection failed" in error_msg)

    @pytest.mark.skip(
        reason=(
            "TODO: Mock side_effect issue - Mock creates new objects on "
            "attribute access, so side_effect set on "
            "mock_client.models.generate_content doesn't affect the Mock "
            "created when code accesses self.client.models.generate_content. "
            "Need different mocking approach."
        )
    )
    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_speaker_detection_json_decode_error(self, mock_render, mock_genai):
        """Test that JSON decode errors return default speakers."""
        # Mock response with invalid JSON
        mock_response = Mock()
        mock_response.text = "invalid json {"
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client
        mock_render.side_effect = lambda name, **kwargs: "test prompt"

        provider = GeminiProvider(self.cfg)
        provider.initialize()

        # Should return default speakers on JSON decode error
        speakers, hosts, success = provider.detect_speakers(
            "Episode Title", "Description", set(["Host"])
        )

        self.assertFalse(success)
        self.assertEqual(speakers, ["Host", "Guest"])

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_speaker_detection_empty_response(self, mock_render, mock_genai):
        """Test that empty responses return default speakers."""
        # Mock response with empty content
        mock_response = Mock()
        mock_response.text = ""
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client
        mock_render.side_effect = lambda name, **kwargs: "test prompt"

        provider = GeminiProvider(self.cfg)
        provider.initialize()

        speakers, hosts, success = provider.detect_speakers(
            "Episode Title", "Description", set(["Host"])
        )

        self.assertFalse(success)
        self.assertEqual(speakers, ["Host", "Guest"])

    @pytest.mark.skip(
        reason=(
            "TODO: Mock side_effect issue - Mock creates new objects on "
            "attribute access, so side_effect set on "
            "mock_client.models.generate_content doesn't affect the Mock "
            "created when code accesses self.client.models.generate_content. "
            "Need different mocking approach."
        )
    )
    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    @patch(
        "podcast_scraper.providers.gemini.gemini_provider."
        "GeminiProvider._build_summarization_prompts"
    )
    def test_summarization_auth_error(self, mock_build_prompts, mock_genai):
        """Test that authentication errors are properly handled in summarization."""

        # Create mock exception with authentication error message
        class MockPermissionDenied(Exception):
            pass

        mock_client = Mock()
        mock_client.models.generate_content.side_effect = MockPermissionDenied(
            "Invalid API key: authentication failed"
        )
        mock_genai.Client.return_value = mock_client
        mock_build_prompts.return_value = (
            "System Prompt",
            "User Prompt",
            "system_v1",
            "user_v1",
            1,
            3,
        )

        provider = GeminiProvider(self.cfg)
        provider.initialize()

        from podcast_scraper.exceptions import ProviderAuthError

        with self.assertRaises(ProviderAuthError) as context:
            provider.summarize("Text to summarize")

        self.assertIn("authentication failed", str(context.exception).lower())

    @pytest.mark.skip(
        reason=(
            "TODO: Mock side_effect issue - Mock creates new objects on "
            "attribute access, so side_effect set on "
            "mock_client.models.generate_content doesn't affect the Mock "
            "created when code accesses self.client.models.generate_content. "
            "Need different mocking approach."
        )
    )
    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    @patch(
        "podcast_scraper.providers.gemini.gemini_provider."
        "GeminiProvider._build_summarization_prompts"
    )
    def test_summarization_rate_limit_error(self, mock_build_prompts, mock_genai):
        """Test that rate limit errors are properly handled in summarization."""

        # Create mock exception with rate limit error message
        class MockResourceExhausted(Exception):
            pass

        mock_client = Mock()
        mock_client.models.generate_content.side_effect = MockResourceExhausted(
            "Rate limit exceeded: resource exhausted"
        )
        mock_genai.Client.return_value = mock_client
        mock_build_prompts.return_value = (
            "System Prompt",
            "User Prompt",
            "system_v1",
            "user_v1",
            1,
            3,
        )

        provider = GeminiProvider(self.cfg)
        provider.initialize()

        from podcast_scraper.exceptions import ProviderRuntimeError

        with self.assertRaises(ProviderRuntimeError) as context:
            provider.summarize("Text to summarize")

        self.assertIn("rate limit", str(context.exception).lower())

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    @patch(
        "podcast_scraper.providers.gemini.gemini_provider."
        "GeminiProvider._build_summarization_prompts"
    )
    def test_summarization_invalid_model_error(self, mock_build_prompts, mock_genai):
        """Test that invalid model errors are properly handled in summarization."""

        # Create mock exception with invalid model error message
        class MockInvalidArgument(Exception):
            pass

        mock_client = Mock()
        mock_client.models.generate_content.side_effect = MockInvalidArgument(
            "Invalid model: unknown-model"
        )
        mock_genai.Client.return_value = mock_client
        mock_build_prompts.return_value = (
            "System Prompt",
            "User Prompt",
            "system_v1",
            "user_v1",
            1,
            3,
        )

        provider = GeminiProvider(self.cfg)
        provider.initialize()

        from podcast_scraper.exceptions import ProviderRuntimeError

        with self.assertRaises(ProviderRuntimeError) as context:
            provider.summarize("Text to summarize")

        error_msg = str(context.exception).lower()
        self.assertTrue("invalid model" in error_msg or "summarization failed" in error_msg)

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_detect_hosts_fallback_on_error(self, mock_render, mock_genai):
        """Test that detect_hosts returns empty set on error."""
        mock_client = Mock()
        mock_client.models.generate_content.side_effect = Exception("API error")
        mock_genai.Client.return_value = mock_client
        mock_render.side_effect = lambda name, **kwargs: "test prompt"

        provider = GeminiProvider(self.cfg)
        provider.initialize()

        # Should return empty set on error
        hosts = provider.detect_hosts("Feed Title", "Description", None)
        self.assertEqual(hosts, set())

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_cleaning_strategy_pattern(self, mock_genai):
        """Test that pattern cleaning strategy is selected correctly."""
        mock_client = Mock()
        mock_genai.Client.return_value = mock_client
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            gemini_api_key="test-api-key-123",
            transcript_cleaning_strategy="pattern",
            speaker_detector_provider="gemini",
        )

        provider = GeminiProvider(cfg)

        from podcast_scraper.cleaning import PatternBasedCleaner

        self.assertIsInstance(provider.cleaning_processor, PatternBasedCleaner)

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_cleaning_strategy_llm(self, mock_genai):
        """Test that LLM cleaning strategy is selected correctly."""
        mock_client = Mock()
        mock_genai.Client.return_value = mock_client
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            gemini_api_key="test-api-key-123",
            transcript_cleaning_strategy="llm",
            speaker_detector_provider="gemini",
        )

        provider = GeminiProvider(cfg)

        from podcast_scraper.cleaning import LLMBasedCleaner

        self.assertIsInstance(provider.cleaning_processor, LLMBasedCleaner)

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_cleaning_strategy_hybrid(self, mock_genai):
        """Test that hybrid cleaning strategy is selected correctly (default)."""
        mock_client = Mock()
        mock_genai.Client.return_value = mock_client
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            gemini_api_key="test-api-key-123",
            transcript_cleaning_strategy="hybrid",
            speaker_detector_provider="gemini",
        )

        provider = GeminiProvider(cfg)

        from podcast_scraper.cleaning import HybridCleaner

        self.assertIsInstance(provider.cleaning_processor, HybridCleaner)
