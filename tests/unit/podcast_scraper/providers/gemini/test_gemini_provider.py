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

# Mock google.genai for imports; unit-only pytest process (``make test-ci-fast``).
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
from podcast_scraper.providers.ml import speaker_detection


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

        speakers, hosts, success, _ = provider.detect_speakers(
            episode_title="Alice interviews Bob",
            episode_description="A great conversation",
            known_hosts={"Alice"},
        )

        self.assertEqual(speakers, ["Alice", "Bob"])
        self.assertTrue(success)

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
    def test_detect_speakers_25_flash_merges_thinking_in_config(
        self, mock_render, mock_parse, mock_build_prompt, mock_genai
    ):
        mock_build_prompt.return_value = "Prompt"
        mock_render.side_effect = lambda name, **kwargs: "test prompt"
        mock_parse.return_value = (["A"], {"A"}, True)
        mock_response = Mock()
        mock_response.text = "{}"
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="gemini",
            gemini_api_key="test-api-key-123",
            auto_speakers=True,
            gemini_speaker_model="gemini-2.5-flash",
        )
        provider = GeminiProvider(cfg)
        provider.initialize()
        provider.detect_speakers("T", "D", {"A"})

        call_kw = mock_client.models.generate_content.call_args[1]
        self.assertEqual(call_kw["config"]["thinking_config"]["thinking_budget"], 0)

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
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_summarize_25_flash_merges_thinking_in_config(self, mock_render_prompt, mock_genai):
        """generate_content config sets thinking_budget=0 for gemini-2.5-flash."""
        mock_render_prompt.side_effect = ["System Prompt", "User Prompt"]
        mock_response = Mock()
        mock_response.text = "Summary."
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="gemini",
            gemini_api_key="test-api-key-123",
            generate_summaries=True,
            generate_metadata=True,
            gemini_summary_model="gemini-2.5-flash",
        )
        provider = GeminiProvider(cfg)
        provider.initialize()
        provider.summarize("Long transcript text.")

        call_kw = mock_client.models.generate_content.call_args[1]
        self.assertEqual(call_kw["config"]["thinking_config"]["thinking_budget"], 0)

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_summarize_default_lite_omits_thinking_config(self, mock_render_prompt, mock_genai):
        """Default test summary model (lite) must not add thinking_config."""
        mock_render_prompt.side_effect = ["System Prompt", "User Prompt"]
        mock_response = Mock()
        mock_response.text = "Summary."
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

        provider = GeminiProvider(self.cfg)
        provider.initialize()
        provider.summarize("Text.")

        call_kw = mock_client.models.generate_content.call_args[1]
        self.assertNotIn("thinking_config", call_kw["config"])

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

    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("podcast_scraper.prompts.store.render_prompt", return_value="clean me")
    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_clean_transcript_success(self, mock_genai, mock_render, mock_retry):
        """clean_transcript calls generate_content with max_output_tokens in config."""
        mock_retry.side_effect = lambda fn, **kwargs: fn()
        mock_resp = Mock()
        mock_resp.text = "cleaned body"
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_resp
        mock_genai.Client.return_value = mock_client

        provider = GeminiProvider(self.cfg)
        provider.initialize()
        out = provider.clean_transcript("word " * 25)
        self.assertEqual(out, "cleaned body")
        mock_client.models.generate_content.assert_called_once()
        call_kw = mock_client.models.generate_content.call_args[1]
        self.assertIn("config", call_kw)
        self.assertIn("max_output_tokens", call_kw["config"])

    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("podcast_scraper.prompts.store.render_prompt", return_value="clean me")
    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_clean_transcript_25_flash_merges_thinking_in_config(
        self, mock_genai, mock_render, mock_retry
    ):
        mock_retry.side_effect = lambda fn, **kwargs: fn()
        mock_resp = Mock()
        mock_resp.text = "cleaned"
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_resp
        mock_genai.Client.return_value = mock_client

        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="gemini",
            gemini_api_key="test-api-key-123",
            generate_summaries=True,
            generate_metadata=True,
            gemini_cleaning_model="gemini-2.5-flash",
        )
        provider = GeminiProvider(cfg)
        provider.initialize()
        provider.clean_transcript("word " * 25)
        call_kw = mock_client.models.generate_content.call_args[1]
        self.assertEqual(call_kw["config"]["thinking_config"]["thinking_budget"], 0)

    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("podcast_scraper.prompts.store.render_prompt", return_value="clean me")
    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_clean_transcript_auth_error(self, mock_genai, mock_render, mock_retry):
        """clean_transcript maps API key errors to ProviderAuthError."""
        mock_retry.side_effect = lambda fn, **kwargs: fn()
        mock_client = Mock()
        mock_client.models.generate_content.side_effect = Exception(
            "Request failed: invalid API key authentication"
        )
        mock_genai.Client.return_value = mock_client

        provider = GeminiProvider(self.cfg)
        provider.initialize()

        from podcast_scraper.exceptions import ProviderAuthError

        with self.assertRaises(ProviderAuthError):
            provider.clean_transcript("some transcript text")


@pytest.mark.unit
class TestGeminiProviderGIL(unittest.TestCase):
    """GIL: generate_insights, extract_quotes, score_entailment."""

    def setUp(self):
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="gemini",
            gemini_api_key="test-api-key-123",
            generate_summaries=True,
            generate_metadata=True,
        )

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    @patch("podcast_scraper.prompts.store.render_prompt", return_value="prompt")
    def test_generate_insights_success(self, mock_render, mock_genai):
        mock_resp = Mock()
        mock_resp.text = "First line insight\nSecond line insight"
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_resp
        mock_genai.Client.return_value = mock_client
        provider = GeminiProvider(self.cfg)
        provider.initialize()
        out = provider.generate_insights("transcript", max_insights=5)
        self.assertGreaterEqual(len(out), 1)
        mock_client.models.generate_content.assert_called_once()

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    @patch("podcast_scraper.prompts.store.render_prompt", return_value="prompt")
    def test_generate_insights_25_flash_merges_thinking_in_config(self, mock_render, mock_genai):
        mock_resp = Mock()
        mock_resp.text = "One insight"
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_resp
        mock_genai.Client.return_value = mock_client
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="gemini",
            gemini_api_key="test-api-key-123",
            generate_summaries=True,
            generate_metadata=True,
            gemini_summary_model="gemini-2.5-flash",
        )
        provider = GeminiProvider(cfg)
        provider.initialize()
        provider.generate_insights("transcript", max_insights=3)
        call_kw = mock_client.models.generate_content.call_args[1]
        self.assertEqual(call_kw["config"]["thinking_config"]["thinking_budget"], 0)

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    @patch("podcast_scraper.prompts.store.render_prompt", return_value="p")
    def test_generate_insights_error_returns_empty(self, mock_render, mock_genai):
        mock_client = Mock()
        mock_client.models.generate_content.side_effect = Exception("API error")
        mock_genai.Client.return_value = mock_client
        provider = GeminiProvider(self.cfg)
        provider.initialize()
        self.assertEqual(provider.generate_insights("t"), [])

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_generate_insights_not_initialized_returns_empty(self, mock_genai):
        mock_genai.Client.return_value = Mock()
        provider = GeminiProvider(self.cfg)
        self.assertEqual(provider.generate_insights("t"), [])

    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_extract_quotes_success(self, mock_genai, mock_retry):
        mock_retry.side_effect = lambda fn, **kwargs: fn()
        mock_resp = Mock()
        mock_resp.text = '{"quote_text": "evidence here"}'
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_resp
        mock_genai.Client.return_value = mock_client
        provider = GeminiProvider(self.cfg)
        provider.initialize()
        from podcast_scraper.gi.grounding import QuoteCandidate

        r = provider.extract_quotes("We have evidence here in the text.", "i")
        self.assertEqual(len(r), 1)
        self.assertIsInstance(r[0], QuoteCandidate)

    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_extract_quotes_25_flash_merges_thinking_in_config(self, mock_genai, mock_retry):
        mock_retry.side_effect = lambda fn, **kwargs: fn()
        mock_resp = Mock()
        mock_resp.text = '{"quote_text": "evidence"}'
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_resp
        mock_genai.Client.return_value = mock_client
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="gemini",
            gemini_api_key="test-api-key-123",
            generate_summaries=True,
            generate_metadata=True,
            gemini_summary_model="gemini-2.5-flash",
        )
        provider = GeminiProvider(cfg)
        provider.initialize()
        provider.extract_quotes("We have evidence in the text.", "i")
        call_kw = mock_client.models.generate_content.call_args[1]
        self.assertEqual(call_kw["config"]["thinking_config"]["thinking_budget"], 0)

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_extract_quotes_not_initialized_returns_empty(self, mock_genai):
        mock_genai.Client.return_value = Mock()
        provider = GeminiProvider(self.cfg)
        self.assertEqual(provider.extract_quotes("a", "b"), [])

    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_extract_quotes_bad_json_returns_empty(self, mock_genai, mock_retry):
        mock_retry.side_effect = lambda fn, **kwargs: fn()
        mock_resp = Mock()
        mock_resp.text = "not json"
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_resp
        mock_genai.Client.return_value = mock_client
        provider = GeminiProvider(self.cfg)
        provider.initialize()
        self.assertEqual(provider.extract_quotes("t", "i"), [])

    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_score_entailment_success(self, mock_genai, mock_retry):
        mock_retry.side_effect = lambda fn, **kwargs: fn()
        mock_resp = Mock()
        mock_resp.text = "0.88"
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_resp
        mock_genai.Client.return_value = mock_client
        provider = GeminiProvider(self.cfg)
        provider.initialize()
        self.assertEqual(provider.score_entailment("p", "h"), 0.88)

    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_score_entailment_25_flash_merges_thinking_in_config(self, mock_genai, mock_retry):
        mock_retry.side_effect = lambda fn, **kwargs: fn()
        mock_resp = Mock()
        mock_resp.text = "0.5"
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_resp
        mock_genai.Client.return_value = mock_client
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="gemini",
            gemini_api_key="test-api-key-123",
            generate_summaries=True,
            generate_metadata=True,
            gemini_summary_model="gemini-2.5-flash",
        )
        provider = GeminiProvider(cfg)
        provider.initialize()
        self.assertEqual(provider.score_entailment("premise text", "hypothesis text"), 0.5)
        call_kw = mock_client.models.generate_content.call_args[1]
        self.assertEqual(call_kw["config"]["thinking_config"]["thinking_budget"], 0)

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_score_entailment_not_initialized_returns_zero(self, mock_genai):
        mock_genai.Client.return_value = Mock()
        provider = GeminiProvider(self.cfg)
        self.assertEqual(provider.score_entailment("a", "b"), 0.0)

    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_score_entailment_exception_returns_zero(self, mock_genai, mock_retry):
        mock_retry.side_effect = lambda fn, **kwargs: fn()
        mock_client = Mock()
        mock_client.models.generate_content.side_effect = Exception("fail")
        mock_genai.Client.return_value = mock_client
        provider = GeminiProvider(self.cfg)
        provider.initialize()
        self.assertEqual(provider.score_entailment("p", "h"), 0.0)

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    @patch("podcast_scraper.prompts.store.render_prompt", return_value="p")
    def test_generate_insights_truncates_long_transcript(self, mock_render, mock_genai):
        mock_resp = Mock()
        mock_resp.text = "Insight"
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_resp
        mock_genai.Client.return_value = mock_client
        provider = GeminiProvider(self.cfg)
        provider.initialize()
        provider.generate_insights("g" * 120_001, max_insights=3)
        self.assertIn("[Transcript truncated.]", mock_render.call_args[1]["transcript"])

    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_score_entailment_no_numeric_token_returns_zero(self, mock_genai, mock_retry):
        mock_retry.side_effect = lambda fn, **kwargs: fn()
        mock_resp = Mock()
        mock_resp.text = "not numeric"
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_resp
        mock_genai.Client.return_value = mock_client
        provider = GeminiProvider(self.cfg)
        provider.initialize()
        self.assertEqual(provider.score_entailment("p", "h"), 0.0)

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_extract_quotes_empty_inputs_returns_empty(self, mock_genai):
        mock_genai.Client.return_value = Mock()
        provider = GeminiProvider(self.cfg)
        provider._summarization_initialized = True
        self.assertEqual(provider.extract_quotes("", "i"), [])
        self.assertEqual(provider.extract_quotes("t", ""), [])


@pytest.mark.unit
class TestGeminiProviderKG(unittest.TestCase):
    """KG: extract_kg_graph, extract_kg_from_summary_bullets (generate_content)."""

    _KG_JSON = '{"topics": [{"label": "Asia"}], "entities": []}'

    def setUp(self):
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="gemini",
            gemini_api_key="test-api-key-123",
            generate_summaries=True,
            generate_metadata=True,
        )

    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_extract_kg_graph_success(self, mock_genai, mock_retry):
        mock_retry.side_effect = lambda fn, **kwargs: fn()
        mock_resp = Mock()
        mock_resp.text = self._KG_JSON
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_resp
        mock_genai.Client.return_value = mock_client
        provider = GeminiProvider(self.cfg)
        provider.initialize()
        out = provider.extract_kg_graph("Asian economies transcript.")
        self.assertIsNotNone(out)
        self.assertEqual(out["topics"][0]["label"], "Asia")

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_extract_kg_graph_not_initialized_returns_none(self, mock_genai):
        mock_genai.Client.return_value = Mock()
        provider = GeminiProvider(self.cfg)
        self.assertIsNone(provider.extract_kg_graph("t"))

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_extract_kg_graph_empty_text_returns_none(self, mock_genai):
        mock_genai.Client.return_value = Mock()
        provider = GeminiProvider(self.cfg)
        provider._summarization_initialized = True
        self.assertIsNone(provider.extract_kg_graph("   "))

    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_extract_kg_graph_api_error_returns_none(self, mock_genai, mock_retry):
        mock_retry.side_effect = lambda fn, **kwargs: fn()
        mock_client = Mock()
        mock_client.models.generate_content.side_effect = RuntimeError("quota")
        mock_genai.Client.return_value = mock_client
        provider = GeminiProvider(self.cfg)
        provider.initialize()
        self.assertIsNone(provider.extract_kg_graph("content"))

    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_extract_kg_graph_params_model_override(self, mock_genai, mock_retry):
        mock_retry.side_effect = lambda fn, **kwargs: fn()
        mock_resp = Mock()
        mock_resp.text = self._KG_JSON
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_resp
        mock_genai.Client.return_value = mock_client
        provider = GeminiProvider(self.cfg)
        provider.initialize()
        provider.extract_kg_graph("z", params={"kg_extraction_model": "gemini-2.0-flash-kg"})
        self.assertEqual(
            mock_client.models.generate_content.call_args[1]["model"],
            "gemini-2.0-flash-kg",
        )

    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_extract_kg_graph_25_flash_params_merges_thinking_in_config(
        self, mock_genai, mock_retry
    ):
        mock_retry.side_effect = lambda fn, **kwargs: fn()
        mock_resp = Mock()
        mock_resp.text = self._KG_JSON
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_resp
        mock_genai.Client.return_value = mock_client
        provider = GeminiProvider(self.cfg)
        provider.initialize()
        provider.extract_kg_graph(
            "Asian economies transcript.",
            params={"kg_extraction_model": "gemini-2.5-flash"},
        )
        call_kw = mock_client.models.generate_content.call_args[1]
        self.assertEqual(call_kw["model"], "gemini-2.5-flash")
        self.assertEqual(call_kw["config"]["thinking_config"]["thinking_budget"], 0)

    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_extract_kg_from_summary_bullets_25_flash_params_merges_thinking(
        self, mock_genai, mock_retry
    ):
        mock_retry.side_effect = lambda fn, **kwargs: fn()
        mock_resp = Mock()
        mock_resp.text = self._KG_JSON
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_resp
        mock_genai.Client.return_value = mock_client
        provider = GeminiProvider(self.cfg)
        provider.initialize()
        provider.extract_kg_from_summary_bullets(
            ["Point"],
            params={"kg_extraction_model": "gemini-2.5-flash"},
        )
        call_kw = mock_client.models.generate_content.call_args[1]
        self.assertEqual(call_kw["config"]["thinking_config"]["thinking_budget"], 0)

    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_extract_kg_from_summary_bullets_success(self, mock_genai, mock_retry):
        mock_retry.side_effect = lambda fn, **kwargs: fn()
        mock_resp = Mock()
        mock_resp.text = self._KG_JSON
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_resp
        mock_genai.Client.return_value = mock_client
        provider = GeminiProvider(self.cfg)
        provider.initialize()
        out = provider.extract_kg_from_summary_bullets(["Point"], episode_title="Ep")
        self.assertIsNotNone(out)

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_extract_kg_from_summary_bullets_not_initialized(self, mock_genai):
        mock_genai.Client.return_value = Mock()
        provider = GeminiProvider(self.cfg)
        self.assertIsNone(provider.extract_kg_from_summary_bullets(["a"]))

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_extract_kg_from_summary_bullets_empty(self, mock_genai):
        mock_genai.Client.return_value = Mock()
        provider = GeminiProvider(self.cfg)
        provider._summarization_initialized = True
        self.assertIsNone(provider.extract_kg_from_summary_bullets([]))


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

    def test_get_pricing_2_5_flash_lite_speaker_detection(self):
        """Test pricing lookup for default Gemini 2.5 Flash-Lite (flash-tier estimate)."""
        pricing = GeminiProvider.get_pricing("gemini-2.5-flash-lite", "speaker_detection")
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

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_speaker_detection_json_decode_error(self, mock_render, mock_genai):
        """Test that JSON decode errors return default speakers."""
        # Mock response with invalid JSON (must start with "{" to return default speakers)
        mock_response = Mock()
        mock_response.text = "{ invalid"
        mock_client = Mock()
        gen_mock = Mock(return_value=mock_response)
        mock_client.models.generate_content = gen_mock
        mock_genai.Client.return_value = mock_client
        mock_render.side_effect = lambda name, **kwargs: "test prompt"

        provider = GeminiProvider(self.cfg)
        provider.initialize()

        # Should return default speakers on JSON decode error
        speakers, hosts, success, _ = provider.detect_speakers(
            "Episode Title", "Description", set(["Host"])
        )

        self.assertFalse(success)
        self.assertEqual(speakers, speaker_detection.DEFAULT_SPEAKER_NAMES)

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

        speakers, hosts, success, _ = provider.detect_speakers(
            "Episode Title", "Description", set(["Host"])
        )

        self.assertFalse(success)
        self.assertEqual(speakers, speaker_detection.DEFAULT_SPEAKER_NAMES)

    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    @patch(
        "podcast_scraper.providers.gemini.gemini_provider."
        "GeminiProvider._build_summarization_prompts"
    )
    def test_summarization_auth_error(self, mock_build_prompts, mock_genai, mock_retry):
        """Test that authentication errors are properly handled in summarization."""

        # Bypass retry so exception propagates (avoids google.api_core in retryable_exceptions)
        def _call_once(func, **kwargs):
            return func()

        mock_retry.side_effect = _call_once

        # Create mock exception with authentication error message
        class MockPermissionDenied(Exception):
            pass

        mock_client = Mock()
        gen_mock = Mock(side_effect=MockPermissionDenied("Invalid API key: authentication failed"))
        mock_client.models.generate_content = gen_mock
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

    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    @patch(
        "podcast_scraper.providers.gemini.gemini_provider."
        "GeminiProvider._build_summarization_prompts"
    )
    def test_summarization_rate_limit_error(self, mock_build_prompts, mock_genai, mock_retry):
        """Test that rate limit errors are properly handled in summarization."""

        # Bypass retry so exception propagates (avoids google.api_core in retryable_exceptions)
        def _call_once(func, **kwargs):
            return func()

        mock_retry.side_effect = _call_once

        # Create mock exception with rate limit error message
        class MockResourceExhausted(Exception):
            pass

        mock_client = Mock()
        gen_mock = Mock(
            side_effect=MockResourceExhausted("Rate limit exceeded: resource exhausted")
        )
        mock_client.models.generate_content = gen_mock
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


@pytest.mark.unit
class TestGeminiSummarizeBundled(unittest.TestCase):
    """Unit tests for summarize_bundled() (Issue #477)."""

    def setUp(self):
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="gemini",
            speaker_detector_provider="gemini",
            summary_provider="gemini",
            gemini_api_key="test-api-key-123",
            transcribe_missing=False,
            auto_speakers=False,
            generate_summaries=False,
            llm_pipeline_mode="bundled",
        )
        self.valid_json = (
            '{"title": "Test Title", '
            '"summary": "A detailed prose summary.", '
            '"bullets": ["Point one.", "Point two."]}'
        )

    def _make_provider(self):
        provider = GeminiProvider(self.cfg)
        provider._summarization_initialized = True
        mock_client = Mock()
        provider.client = mock_client
        return provider

    def _mock_response(self, content, pt=100, ct=50):
        resp = Mock()
        resp.text = content
        usage = Mock()
        usage.prompt_token_count = pt
        usage.candidates_token_count = ct
        resp.usage_metadata = usage
        return resp

    @patch("podcast_scraper.prompts.store.get_prompt_metadata")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_bundled_success_returns_expected_shape(self, mock_render, mock_meta):
        mock_render.side_effect = ["System", "User"]
        mock_meta.return_value = {"name": "test", "sha256": "abc"}
        provider = self._make_provider()
        provider.client.models.generate_content.return_value = self._mock_response(self.valid_json)

        result = provider.summarize_bundled("transcript text")

        self.assertIn("summary", result)
        self.assertIn("metadata", result)
        self.assertTrue(result["metadata"]["bundled"])
        self.assertNotIn("bundled_cleaned_transcript", result)

    @patch("podcast_scraper.prompts.store.get_prompt_metadata")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_bundled_json_contains_title_summary_bullets(self, mock_render, mock_meta):
        mock_render.side_effect = ["System", "User"]
        mock_meta.return_value = {"name": "test", "sha256": "abc"}
        provider = self._make_provider()
        provider.client.models.generate_content.return_value = self._mock_response(self.valid_json)

        result = provider.summarize_bundled("transcript text")
        parsed = json.loads(result["summary"])
        self.assertEqual(parsed["title"], "Test Title")
        self.assertEqual(parsed["summary"], "A detailed prose summary.")
        self.assertEqual(len(parsed["bullets"]), 2)

    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_bundled_rejects_missing_summary_field(self, mock_render):
        mock_render.side_effect = ["System", "User"]
        provider = self._make_provider()
        bad_json = '{"title": "T", "bullets": ["b"]}'
        provider.client.models.generate_content.return_value = self._mock_response(bad_json)

        with self.assertRaises(ValueError) as ctx:
            provider.summarize_bundled("text")
        self.assertIn("summary", str(ctx.exception))

    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_bundled_rejects_missing_bullets(self, mock_render):
        mock_render.side_effect = ["System", "User"]
        provider = self._make_provider()
        bad_json = '{"title": "T", "summary": "s"}'
        provider.client.models.generate_content.return_value = self._mock_response(bad_json)

        with self.assertRaises(ValueError) as ctx:
            provider.summarize_bundled("text")
        self.assertIn("bullets", str(ctx.exception))

    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_bundled_rejects_invalid_json(self, mock_render):
        mock_render.side_effect = ["System", "User"]
        provider = self._make_provider()
        provider.client.models.generate_content.return_value = self._mock_response("not json")

        with self.assertRaises(ValueError) as ctx:
            provider.summarize_bundled("text")
        self.assertIn("JSON", str(ctx.exception))

    @patch("podcast_scraper.prompts.store.get_prompt_metadata")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_bundled_uses_provider_prefixed_prompt_names(self, mock_render, mock_meta):
        mock_render.side_effect = ["System", "User"]
        mock_meta.return_value = {"name": "test", "sha256": "abc"}
        provider = self._make_provider()
        provider.client.models.generate_content.return_value = self._mock_response(self.valid_json)

        provider.summarize_bundled("text")

        calls = [c[0][0] for c in mock_render.call_args_list]
        self.assertTrue(
            any("gemini/summarization/bundled" in c for c in calls),
            f"Expected gemini-prefixed prompt name, got: {calls}",
        )

    @patch("podcast_scraper.prompts.store.get_prompt_metadata")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_bundled_25_flash_merges_thinking_in_config(self, mock_render, mock_meta):
        mock_render.side_effect = ["System", "User"]
        mock_meta.return_value = {"name": "test", "sha256": "abc"}
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="gemini",
            speaker_detector_provider="gemini",
            summary_provider="gemini",
            gemini_api_key="test-api-key-123",
            transcribe_missing=False,
            auto_speakers=False,
            generate_summaries=False,
            llm_pipeline_mode="bundled",
            gemini_summary_model="gemini-2.5-flash",
        )
        provider = GeminiProvider(cfg)
        provider._summarization_initialized = True
        mock_client = Mock()
        provider.client = mock_client
        mock_client.models.generate_content.return_value = self._mock_response(self.valid_json)
        provider.summarize_bundled("transcript text")
        call_kw = mock_client.models.generate_content.call_args[1]
        self.assertEqual(call_kw["config"]["thinking_config"]["thinking_budget"], 0)


@pytest.mark.unit
class TestGeminiThinkingConfigMerge(unittest.TestCase):
    """Issue #572: thinking_budget=0 for gemini-2.5-flash (non-lite)."""

    def test_should_disable_gate(self):
        from podcast_scraper.providers.gemini.gemini_provider import (
            _should_disable_thinking_for_model,
        )

        self.assertTrue(_should_disable_thinking_for_model("gemini-2.5-flash"))
        self.assertTrue(_should_disable_thinking_for_model("models/gemini-2.5-flash-preview"))
        self.assertFalse(_should_disable_thinking_for_model("gemini-2.5-flash-lite"))
        self.assertFalse(_should_disable_thinking_for_model("gemini-2.0-flash"))
        self.assertFalse(_should_disable_thinking_for_model("gemini-2.5-pro"))

    def test_merge_injects_thinking_budget(self):
        from podcast_scraper.providers.gemini.gemini_provider import (
            _merge_generate_content_config,
        )

        cfg = _merge_generate_content_config(
            "gemini-2.5-flash",
            {"temperature": 0.2, "max_output_tokens": 100},
        )
        self.assertEqual(cfg["thinking_config"]["thinking_budget"], 0)
        self.assertEqual(cfg["temperature"], 0.2)

    def test_merge_skips_lite(self):
        from podcast_scraper.providers.gemini.gemini_provider import (
            _merge_generate_content_config,
        )

        cfg = _merge_generate_content_config(
            "gemini-2.5-flash-lite",
            {"temperature": 0.2},
        )
        self.assertNotIn("thinking_config", cfg)

    def test_merge_preserves_explicit_thinking_config(self):
        from podcast_scraper.providers.gemini.gemini_provider import (
            _merge_generate_content_config,
        )

        existing = {"thinking_config": {"thinking_budget": 128}}
        cfg = _merge_generate_content_config("gemini-2.5-flash", existing)
        self.assertEqual(cfg["thinking_config"]["thinking_budget"], 128)

    def test_merge_treats_explicit_none_thinking_config_as_absent(self):
        from podcast_scraper.providers.gemini.gemini_provider import (
            _merge_generate_content_config,
        )

        cfg = _merge_generate_content_config(
            "gemini-2.5-flash",
            {"thinking_config": None, "temperature": 0.1},
        )
        self.assertEqual(cfg["thinking_config"]["thinking_budget"], 0)
        self.assertEqual(cfg["temperature"], 0.1)

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    @patch("builtins.open", create=True)
    @patch("os.path.exists")
    def test_transcribe_25_flash_passes_thinking_disabled_config(
        self, mock_exists, mock_open, mock_genai
    ):
        mock_exists.return_value = True
        mock_file = Mock()
        mock_file.read.return_value = b"\x00audio"
        mock_open.return_value.__enter__.return_value = mock_file
        mock_open.return_value.__exit__.return_value = None

        mock_response = Mock()
        mock_response.text = "ok"
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="gemini",
            gemini_api_key="test-api-key-123",
            transcribe_missing=True,
            gemini_transcription_model="gemini-2.5-flash",
        )
        provider = GeminiProvider(cfg)
        provider.initialize()
        provider.transcribe("/tmp/sample.mp3")

        kw = mock_client.models.generate_content.call_args[1]
        self.assertIn("config", kw)
        self.assertEqual(kw["config"]["thinking_config"]["thinking_budget"], 0)
