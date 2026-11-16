#!/usr/bin/env python3
"""Standalone unit tests for unified Mistral provider.

These tests verify that MistralProvider correctly implements all three protocols
(TranscriptionProvider, SpeakerDetector, SummarizationProvider) using
Mistral's API (Voxtral for transcription, chat models for speaker detection and summarization).

These are standalone provider tests - they test the provider itself,
not its integration with the app.
"""

import importlib
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from podcast_scraper import config
from podcast_scraper.providers.mistral import mistral_provider as mistral_provider_mod
from podcast_scraper.providers.mistral.mistral_provider import MistralProvider


class _DummyMistralUploadFile:
    """Stand-in for mistralai ``File`` when the SDK is absent or unpatchable in unit tests."""

    def __init__(self, file_name: str = "", content: bytes = b""):
        self.file_name = file_name
        self.content = content


# Preserve PR-time coverage post #678 PR-A1 (moved from ``tests/unit/``).
pytestmark = pytest.mark.critical_path


@pytest.mark.integration
class TestMistralProviderStandalone(unittest.TestCase):
    """Standalone tests for MistralProvider - testing the provider itself."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="mistral",
            speaker_detector_provider="mistral",
            summary_provider="mistral",
            mistral_api_key="test-api-key-123",
            transcribe_missing=False,  # Disable to avoid API calls
            auto_speakers=False,  # Disable to avoid API calls
            generate_summaries=False,  # Disable to avoid API calls
        )

    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    def test_provider_creation(self, mock_mistral_class):
        """Test that MistralProvider can be created."""
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client

        provider = MistralProvider(self.cfg)
        self.assertIsNotNone(provider)
        self.assertEqual(provider.__class__.__name__, "MistralProvider")
        # Verify Mistral client was created with API key
        # Note: timeout is not passed to Mistral SDK (not supported)
        mock_mistral_class.assert_called_once()
        call_kwargs = mock_mistral_class.call_args[1]
        self.assertEqual(call_kwargs["api_key"], "test-api-key-123")
        # Timeout is not included (Mistral SDK doesn't support it)
        self.assertNotIn("timeout", call_kwargs)

    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    def test_provider_creation_with_custom_base_uses_server_url(self, mock_mistral_class):
        """Custom base URL must be passed as server_url (not server) for E2E fixtures."""
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="mistral",
            speaker_detector_provider="mistral",
            summary_provider="mistral",
            mistral_api_key="test-api-key-123",
            mistral_api_base="http://127.0.0.1:63436/v1",
            transcribe_missing=False,
            auto_speakers=False,
            generate_summaries=False,
        )
        MistralProvider(cfg)
        call_kwargs = mock_mistral_class.call_args[1]
        self.assertEqual(call_kwargs.get("server_url"), "http://127.0.0.1:63436/v1")
        self.assertNotIn("server", call_kwargs)

    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    def test_provider_creation_requires_api_key(self, mock_mistral_class):
        """Test that MistralProvider requires API key."""
        # Unset environment variable to ensure it's not loaded
        original_key = os.environ.pop("MISTRAL_API_KEY", None)
        try:
            # Note: Config validation happens before provider creation
            # So we need to catch ValidationError from Config, not ValueError from provider
            with self.assertRaises(Exception) as context:
                cfg = config.Config(
                    rss_url="https://example.com/feed.xml",
                    transcription_provider="mistral",
                )
                MistralProvider(cfg)
            # Error can be either ValidationError (from Config) or ValueError (from provider)
        finally:
            # Restore original environment variable if it existed
            if original_key is not None:
                os.environ["MISTRAL_API_KEY"] = original_key
        error_msg = str(context.exception)
        self.assertTrue(
            "Mistral API key required" in error_msg or "validation error" in error_msg.lower()
        )

    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    def test_provider_initialization_state(self, mock_mistral_class):
        """Test that provider tracks initialization state for each capability."""
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client

        provider = MistralProvider(self.cfg)

        # Initially not initialized
        self.assertFalse(provider._transcription_initialized)
        self.assertFalse(provider._speaker_detection_initialized)
        self.assertFalse(provider._summarization_initialized)
        self.assertFalse(provider.is_initialized)

    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    def test_provider_thread_safe(self, mock_mistral_class):
        """Test that provider marks itself as thread-safe."""
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client

        provider = MistralProvider(self.cfg)
        self.assertFalse(provider._requires_separate_instances)

    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    def test_transcription_initialization(self, mock_mistral_class):
        """Test that transcription can be initialized independently."""
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client

        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            mistral_api_key=self.cfg.mistral_api_key,
            transcription_provider="mistral",
            transcribe_missing=True,
            auto_speakers=False,  # Disable to avoid initializing speaker detection
        )
        provider = MistralProvider(cfg)

        provider.initialize()

        self.assertTrue(provider._transcription_initialized)
        # Other capabilities should not be initialized
        self.assertFalse(provider._speaker_detection_initialized)
        # #1720: the LLM capability is ALWAYS readied by initialize() — it also serves
        # extract_kg_graph/GI/cleaning, not just the summary stage.
        self.assertTrue(provider._summarization_initialized)

    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    def test_speaker_detection_initialization(self, mock_mistral_class):
        """Test that speaker detection can be initialized independently."""
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client

        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            mistral_api_key=self.cfg.mistral_api_key,
            speaker_detector_provider="mistral",
            auto_speakers=True,
            transcribe_missing=False,  # Disable to avoid initializing transcription
        )
        provider = MistralProvider(cfg)

        provider.initialize()

        self.assertTrue(provider._speaker_detection_initialized)
        # Other capabilities should not be initialized
        self.assertFalse(provider._transcription_initialized)
        # #1720: the LLM capability is ALWAYS readied by initialize() — it also serves
        # extract_kg_graph/GI/cleaning, not just the summary stage.
        self.assertTrue(provider._summarization_initialized)

    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    def test_summarization_initialization(self, mock_mistral_class):
        """Test that summarization can be initialized independently."""
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client

        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            mistral_api_key=self.cfg.mistral_api_key,
            summary_provider="mistral",
            generate_summaries=True,
            generate_metadata=True,  # Required when generate_summaries is True
            auto_speakers=False,  # Disable to avoid initializing speaker detection
            transcribe_missing=False,  # Disable to avoid initializing transcription
        )
        provider = MistralProvider(cfg)

        provider.initialize()

        self.assertTrue(provider._summarization_initialized)
        # Other capabilities should not be initialized
        self.assertFalse(provider._transcription_initialized)
        self.assertFalse(provider._speaker_detection_initialized)

    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    def test_unified_initialization(self, mock_mistral_class):
        """Test that all capabilities can be initialized together."""
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client

        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            mistral_api_key=self.cfg.mistral_api_key,
            transcription_provider="mistral",
            speaker_detector_provider="mistral",
            summary_provider="mistral",
            transcribe_missing=True,
            auto_speakers=True,
            generate_summaries=True,
            generate_metadata=True,  # Required when generate_summaries is True
        )

        provider = MistralProvider(cfg)
        provider.initialize()

        # All should be initialized
        self.assertTrue(provider._transcription_initialized)
        self.assertTrue(provider._speaker_detection_initialized)
        self.assertTrue(provider._summarization_initialized)
        self.assertTrue(provider.is_initialized)

    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    def test_cleanup_releases_all_resources(self, mock_mistral_class):
        """Test that cleanup releases all resources."""
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client

        provider = MistralProvider(self.cfg)
        provider._transcription_initialized = True
        provider._speaker_detection_initialized = True
        provider._summarization_initialized = True

        provider.cleanup()

        self.assertFalse(provider._transcription_initialized)
        self.assertFalse(provider._speaker_detection_initialized)
        self.assertFalse(provider._summarization_initialized)
        self.assertFalse(provider.is_initialized)

    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    def test_transcription_model_attribute(self, mock_mistral_class):
        """Test that transcription_model attribute exists."""
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client

        provider = MistralProvider(self.cfg)

        # Transcription attributes
        self.assertTrue(hasattr(provider, "transcription_model"))
        self.assertTrue(hasattr(provider, "is_initialized"))

        # Verify transcription_model is accessible
        self.assertIsNotNone(provider.transcription_model)


@pytest.mark.integration
class TestMistralProviderTranscription(unittest.TestCase):
    """Tests for MistralProvider transcription methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="mistral",
            mistral_api_key="test-api-key-123",
            transcribe_missing=True,
        )

    @patch("podcast_scraper.providers.mistral.mistral_provider._mistral_file_class")
    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    @patch("builtins.open", create=True)
    @patch("os.path.exists")
    def test_transcribe_success(self, mock_exists, mock_open, mock_mistral_class, mock_file_cls):
        """Test successful transcription."""
        mock_file_cls.return_value = _DummyMistralUploadFile
        mock_exists.return_value = True
        mock_file = Mock()
        mock_file.read.return_value = b"fake audio data"
        mock_open.return_value.__enter__.return_value = mock_file
        mock_open.return_value.__exit__.return_value = None

        # Mock Mistral API response
        mock_transcription = Mock()
        mock_transcription.text = "Hello world"
        mock_client = Mock()
        # Mistral uses 'complete' method, not 'create'
        mock_client.audio.transcriptions.complete.return_value = mock_transcription
        mock_mistral_class.return_value = mock_client

        provider = MistralProvider(self.cfg)
        provider.initialize()

        result = provider.transcribe("/path/to/audio.mp3")

        self.assertEqual(result, "Hello world")
        mock_client.audio.transcriptions.complete.assert_called_once()

    @patch("podcast_scraper.providers.mistral.mistral_provider._mistral_file_class")
    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    @patch("builtins.open", create=True)
    @patch("os.path.exists")
    def test_transcribe_with_language(
        self, mock_exists, mock_open, mock_mistral_class, mock_file_cls
    ):
        """Test transcription with explicit language."""
        mock_file_cls.return_value = _DummyMistralUploadFile
        mock_exists.return_value = True
        mock_file = Mock()
        mock_file.read.return_value = b"fake audio data"
        mock_open.return_value.__enter__.return_value = mock_file
        mock_open.return_value.__exit__.return_value = None

        mock_transcription = Mock()
        mock_transcription.text = "Bonjour"
        mock_client = Mock()
        mock_client.audio.transcriptions.complete.return_value = mock_transcription
        mock_mistral_class.return_value = mock_client

        provider = MistralProvider(self.cfg)
        provider.initialize()

        provider.transcribe("/path/to/audio.mp3", language="fr")

        # Verify language was passed to API
        call_kwargs = mock_client.audio.transcriptions.complete.call_args[1]
        self.assertEqual(call_kwargs.get("language"), "fr")

    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    def test_transcribe_not_initialized(self, mock_mistral_class):
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client

        provider = MistralProvider(self.cfg)
        # Don't call initialize()

        with self.assertRaises(RuntimeError) as context:
            provider.transcribe("/path/to/audio.mp3")

        self.assertIn("not initialized", str(context.exception))

    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    @patch("os.path.exists")
    def test_transcribe_file_not_found(self, mock_exists, mock_mistral_class):
        """Test transcribe raises FileNotFoundError if file doesn't exist."""
        mock_exists.return_value = False
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client

        provider = MistralProvider(self.cfg)
        provider.initialize()

        with self.assertRaises(FileNotFoundError) as context:
            provider.transcribe("/path/to/nonexistent.mp3")

        self.assertIn("not found", str(context.exception))

    @patch("podcast_scraper.providers.mistral.mistral_provider._mistral_file_class")
    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    @patch("builtins.open", create=True)
    @patch("os.path.exists")
    def test_transcribe_api_error(self, mock_exists, mock_open, mock_mistral_class, mock_file_cls):
        """Test transcribe handles API errors."""
        mock_file_cls.return_value = _DummyMistralUploadFile
        mock_exists.return_value = True
        mock_file = Mock()
        mock_file.read.return_value = b"fake audio data"
        mock_open.return_value.__enter__.return_value = mock_file
        mock_open.return_value.__exit__.return_value = None

        mock_client = Mock()
        mock_client.audio.transcriptions.complete.side_effect = Exception("API error")
        mock_mistral_class.return_value = mock_client

        provider = MistralProvider(self.cfg)
        provider.initialize()

        from podcast_scraper.exceptions import ProviderRuntimeError

        with self.assertRaises(ProviderRuntimeError) as context:
            provider.transcribe("/path/to/audio.mp3")

        self.assertIn("transcription failed", str(context.exception))

    @patch("podcast_scraper.providers.mistral.mistral_provider._mistral_file_class")
    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    @patch("builtins.open", create=True)
    @patch("os.path.exists")
    def test_transcribe_with_segments_success(
        self, mock_exists, mock_open, mock_mistral_class, mock_file_cls
    ):
        """Test transcribe_with_segments returns full result."""
        mock_file_cls.return_value = _DummyMistralUploadFile
        mock_exists.return_value = True
        mock_file = Mock()
        mock_file.read.return_value = b"fake audio data"
        mock_open.return_value.__enter__.return_value = mock_file
        mock_open.return_value.__exit__.return_value = None

        mock_transcription = Mock()
        mock_transcription.text = "Hello world"
        mock_client = Mock()
        mock_client.audio.transcriptions.complete.return_value = mock_transcription
        mock_mistral_class.return_value = mock_client

        provider = MistralProvider(self.cfg)
        provider.initialize()

        result_dict, elapsed = provider.transcribe_with_segments("/path/to/audio.mp3")

        self.assertEqual(result_dict["text"], "Hello world")
        self.assertEqual(result_dict["segments"], [])  # Mistral may not provide segments
        self.assertIsInstance(elapsed, float)
        self.assertGreater(elapsed, 0)

    @patch("podcast_scraper.providers.mistral.mistral_provider._mistral_file_class")
    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    @patch("os.path.getsize")
    @patch("builtins.open", create=True)
    @patch("os.path.exists")
    def test_transcribe_with_segments_file_size_fallback_scales_by_bitrate(
        self, mock_exists, mock_open, mock_getsize, mock_mistral_class, mock_file_cls
    ):
        """#650 Finding 18 symmetry — file-size→minutes fallback on Mistral must
        scale by configured bitrate when episode_duration_seconds is unset.
        """
        mock_file_cls.return_value = _DummyMistralUploadFile
        mock_exists.return_value = True
        mock_file = Mock()
        mock_file.read.return_value = b"fake audio data"
        mock_open.return_value.__enter__.return_value = mock_file
        mock_open.return_value.__exit__.return_value = None
        mock_getsize.return_value = 3 * 1024 * 1024  # 3 MB

        mock_transcription = Mock()
        mock_transcription.text = "fallback transcript"
        mock_client = Mock()
        mock_client.audio.transcriptions.complete.return_value = mock_transcription
        mock_mistral_class.return_value = mock_client

        cfg = config.Config(
            transcription_provider="mistral",
            mistral_api_key="test-api-key-123",
            transcribe_missing=True,
            preprocessing_mp3_bitrate_kbps=32,
        )
        provider = MistralProvider(cfg)
        provider.initialize()

        pm = Mock()
        provider.transcribe_with_segments("/path/to/audio.mp3", pipeline_metrics=pm)

        # Expected: 3 MB × (128/32) = 12.0 minutes.
        self.assertTrue(pm.record_llm_transcription_call.called)
        recorded_minutes = pm.record_llm_transcription_call.call_args[0][0]
        self.assertAlmostEqual(recorded_minutes, 12.0, places=4)
        cost_kwarg = pm.record_llm_transcription_call.call_args.kwargs.get("cost_usd")
        self.assertIsNotNone(cost_kwarg)

    @patch("podcast_scraper.providers.mistral.mistral_provider._mistral_file_class")
    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    @patch("os.path.getsize")
    @patch("builtins.open", create=True)
    @patch("os.path.exists")
    def test_transcribe_with_segments_file_size_fallback_default_bitrate(
        self, mock_exists, mock_open, mock_getsize, mock_mistral_class, mock_file_cls
    ):
        """Finding 18 — unset bitrate falls back to 128 kbps so pre-fix behaviour
        is preserved for untuned audio."""
        mock_file_cls.return_value = _DummyMistralUploadFile
        mock_exists.return_value = True
        mock_file = Mock()
        mock_file.read.return_value = b"fake audio data"
        mock_open.return_value.__enter__.return_value = mock_file
        mock_open.return_value.__exit__.return_value = None
        mock_getsize.return_value = 5 * 1024 * 1024  # 5 MB

        mock_transcription = Mock()
        mock_transcription.text = "default-bitrate transcript"
        mock_client = Mock()
        mock_client.audio.transcriptions.complete.return_value = mock_transcription
        mock_mistral_class.return_value = mock_client

        cfg = config.Config(
            transcription_provider="mistral",
            mistral_api_key="test-api-key-123",
            transcribe_missing=True,
        )
        provider = MistralProvider(cfg)
        provider.initialize()

        pm = Mock()
        provider.transcribe_with_segments("/path/to/audio.mp3", pipeline_metrics=pm)

        # Expected: 5 MB × (128/128) = 5.0 minutes.
        recorded_minutes = pm.record_llm_transcription_call.call_args[0][0]
        self.assertAlmostEqual(recorded_minutes, 5.0, places=4)


@pytest.mark.integration
class TestMistralProviderSpeakerDetection(unittest.TestCase):
    """Tests for MistralProvider speaker detection methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="mistral",
            mistral_api_key="test-api-key-123",
            auto_speakers=True,
        )

    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    def test_analyze_patterns_success(self, mock_mistral_class):
        """Test successful pattern analysis."""
        from podcast_scraper import models

        mock_client = Mock()
        mock_mistral_class.return_value = mock_client

        provider = MistralProvider(self.cfg)
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

        # Mistral provider doesn't implement pattern analysis, returns None
        result = provider.analyze_patterns(episodes=episodes, known_hosts={"Alice"})

        self.assertIsNone(result)  # Mistral provider returns None to use local logic

    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    def test_clear_cache(self, mock_mistral_class):
        """Test cache clearing (no-op for Mistral provider)."""
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client

        provider = MistralProvider(self.cfg)

        # clear_cache should not raise (it's a no-op for Mistral provider)
        provider.clear_cache()

        # Mistral provider doesn't use cache, but method exists for protocol compliance
        # It's essentially a no-op


@pytest.mark.integration
class TestMistralProviderSummarization(unittest.TestCase):
    """Tests for MistralProvider summarization methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="mistral",
            mistral_api_key="test-api-key-123",
            generate_summaries=True,
            generate_metadata=True,  # Required when generate_summaries=True
        )

    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_summarize_success(self, mock_render_prompt, mock_mistral_class):
        """Test successful summarization."""
        # Mock render_prompt to return prompts (called twice: system and user)
        mock_render_prompt.side_effect = ["System Prompt", "User Prompt"]

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "This is a summary."

        mock_client = Mock()
        # Mistral uses 'complete' method, not 'create'
        mock_client.chat.complete.return_value = mock_response
        mock_mistral_class.return_value = mock_client

        provider = MistralProvider(self.cfg)
        provider.initialize()

        result = provider.summarize("This is a long transcript text.")

        self.assertEqual(result["summary"], "This is a summary.")
        self.assertIn("metadata", result)
        self.assertEqual(result["metadata"]["model"], provider.summary_model)

    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    def test_summarize_not_initialized(self, mock_mistral_class):
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client

        provider = MistralProvider(self.cfg)
        # Don't call initialize()

        with self.assertRaises(RuntimeError) as context:
            provider.summarize("Text")

        self.assertIn("not initialized", str(context.exception))

    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    @patch("podcast_scraper.prompts.store.get_prompt_metadata")
    @patch("podcast_scraper.prompts.store.render_prompt")
    @patch(
        "podcast_scraper.providers.mistral.mistral_provider."
        "MistralProvider._build_summarization_prompts"
    )
    def test_summarize_with_params(
        self, mock_build_prompts, mock_render_prompt, mock_get_metadata, mock_mistral_class
    ):
        """Test summarization with custom parameters."""
        # _build_summarization_prompts returns:
        # (system_prompt, user_prompt, system_prompt_name, user_prompt_name,
        #  paragraphs_min, paragraphs_max)
        mock_build_prompts.return_value = (
            "System Prompt",
            "User Prompt",
            "mistral/summarization/system_v1",
            "mistral/summarization/long_v1",
            1,
            3,
        )
        # render_prompt is called inside _build_summarization_prompts
        mock_render_prompt.side_effect = ["System Prompt", "User Prompt"]
        # get_prompt_metadata is called for tracking
        mock_get_metadata.return_value = {
            "name": "mistral/summarization/system_v1",
            "sha256": "abc123",
        }

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Summary"

        mock_client = Mock()
        # Mistral uses 'complete' method, not 'create'
        mock_client.chat.complete.return_value = mock_response
        mock_mistral_class.return_value = mock_client

        provider = MistralProvider(self.cfg)
        provider.initialize()

        params = {"max_length": 100, "min_length": 50}
        provider.summarize("Text", params=params)

        # Verify API was called
        mock_client.chat.complete.assert_called()

    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    @patch(
        "podcast_scraper.providers.mistral.mistral_provider."
        "MistralProvider._build_summarization_prompts"
    )
    def test_summarize_api_error(self, mock_build_prompts, mock_mistral_class):
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
        # Mistral uses 'complete' method, not 'create'
        mock_client.chat.complete.side_effect = Exception("API error")
        mock_mistral_class.return_value = mock_client

        provider = MistralProvider(self.cfg)
        provider.initialize()

        from podcast_scraper.exceptions import ProviderRuntimeError

        with self.assertRaises(ProviderRuntimeError) as context:
            provider.summarize("Text")

        self.assertIn("summarization failed", str(context.exception).lower())


@pytest.mark.integration
class TestMistralProviderKG(unittest.TestCase):
    """KG: extract_kg_graph, extract_kg_from_summary_bullets (Mistral chat.complete)."""

    _KG_JSON = '{"topics": [{"label": "EU"}], "entities": []}'

    def setUp(self):
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="mistral",
            mistral_api_key="test-api-key-123",
            generate_summaries=True,
            generate_metadata=True,
            auto_speakers=False,
            transcribe_missing=False,
        )

    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    def test_extract_kg_graph_success(self, mock_mistral_class, mock_retry):
        mock_retry.side_effect = lambda fn, **kwargs: fn()
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client
        mock_resp = Mock()
        mock_resp.choices = [Mock(message=Mock(content=self._KG_JSON))]
        mock_client.chat.complete.return_value = mock_resp
        provider = MistralProvider(self.cfg)
        provider.initialize()
        out = provider.extract_kg_graph("European markets overview.")
        self.assertIsNotNone(out)
        self.assertEqual(out["topics"][0]["label"], "EU")

    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    def test_extract_kg_graph_not_initialized_raises(self, mock_mistral_class):
        """#34: an unusable provider must SAY SO, not fake a well-formed result.

        Post-#1657 an empty/zero/None result is a LEGAL outcome, so it no longer
        distinguishes "the model found nothing" from "never initialized".
        """
        mock_mistral_class.return_value = Mock()
        provider = MistralProvider(self.cfg)
        with self.assertRaises(RuntimeError):
            provider.extract_kg_graph("t")

    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    def test_extract_kg_graph_empty_text_returns_none(self, mock_mistral_class):
        mock_mistral_class.return_value = Mock()
        provider = MistralProvider(self.cfg)
        provider._summarization_initialized = True
        self.assertIsNone(provider.extract_kg_graph("   "))

    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    def test_extract_kg_graph_api_error_returns_none(self, mock_mistral_class, mock_retry):
        mock_retry.side_effect = lambda fn, **kwargs: fn()
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client
        mock_client.chat.complete.side_effect = RuntimeError("down")
        provider = MistralProvider(self.cfg)
        provider.initialize()
        self.assertIsNone(provider.extract_kg_graph("text"))

    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    def test_extract_kg_graph_params_model_override(self, mock_mistral_class, mock_retry):
        mock_retry.side_effect = lambda fn, **kwargs: fn()
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client
        mock_resp = Mock()
        mock_resp.choices = [Mock(message=Mock(content=self._KG_JSON))]
        mock_client.chat.complete.return_value = mock_resp
        provider = MistralProvider(self.cfg)
        provider.initialize()
        provider.extract_kg_graph("x", params={"kg_extraction_model": "mistral-large-x"})
        self.assertEqual(
            mock_client.chat.complete.call_args[1]["model"],
            "mistral-large-x",
        )


@pytest.mark.integration
class TestMistralSdkPathResolution(unittest.TestCase):
    """mistralai 1.x vs 2.x import paths (_load_mistral_sdk / _mistral_file_class)."""

    def test_load_mistral_sdk_falls_back_to_package_root(self) -> None:
        real_import = importlib.import_module
        sentinel_mistral = object()

        def fake_import(name: str, package: str | None = None):
            if name in ("mistralai.client.sdk", "mistralai.client.errors"):
                raise ImportError("no 2.x layout")
            if name == "mistralai":
                root = SimpleNamespace(Mistral=sentinel_mistral, SDKError=ValueError)
                return root
            return real_import(name, package)

        with patch.object(importlib, "import_module", side_effect=fake_import):
            Mistral, SDKError = mistral_provider_mod._load_mistral_sdk()
        self.assertIs(Mistral, sentinel_mistral)
        self.assertIs(SDKError, ValueError)

    def test_mistral_file_class_second_module_path(self) -> None:
        file_cls = object()

        def fake_import(name: str, package: str | None = None):
            if name == "mistralai.client.models":
                raise ImportError("missing")
            if name == "mistralai.models.file":
                return SimpleNamespace(File=file_cls)
            raise AssertionError(f"unexpected import {name!r}")

        with patch.object(importlib, "import_module", side_effect=fake_import):
            got = mistral_provider_mod._mistral_file_class()
        self.assertIs(got, file_cls)

    def test_mistral_file_class_raises_when_unresolved(self) -> None:
        def fake_import(name: str, package: str | None = None):
            raise ImportError("missing")

        with patch.object(importlib, "import_module", side_effect=fake_import):
            with self.assertRaisesRegex(ImportError, "File model"):
                mistral_provider_mod._mistral_file_class()


@pytest.mark.integration
class TestMistralProviderPricing(unittest.TestCase):
    """Tests for MistralProvider.get_pricing() static method."""

    def test_get_pricing_audio_transcription(self):
        """Test pricing lookup for audio transcription (YAML-backed post-#651)."""
        pricing = MistralProvider.get_pricing("voxtral-mini-latest", "transcription")
        self.assertIn("cost_per_minute", pricing)
        self.assertEqual(pricing["cost_per_minute"], 0.001)

    def test_get_pricing_small_speaker_detection(self):
        """Test pricing lookup for mistral-small speaker detection (YAML-backed)."""
        pricing = MistralProvider.get_pricing("mistral-small-latest", "speaker_detection")
        self.assertEqual(pricing["input_cost_per_1m_tokens"], 0.20)
        self.assertEqual(pricing["output_cost_per_1m_tokens"], 0.60)

    def test_get_pricing_large_summarization(self):
        """Test pricing lookup for mistral-large summarization (YAML-backed)."""
        pricing = MistralProvider.get_pricing("mistral-large-latest", "summarization")
        self.assertEqual(pricing["input_cost_per_1m_tokens"], 2.00)
        self.assertEqual(pricing["output_cost_per_1m_tokens"], 6.00)

    def test_get_pricing_unsupported_model(self):
        """Unsupported model falls back to YAML ``default`` row."""
        pricing = MistralProvider.get_pricing("mistral-unknown", "summarization")
        # YAML default: input=0.20, output=0.60
        self.assertEqual(pricing["input_cost_per_1m_tokens"], 0.20)
        self.assertEqual(pricing["output_cost_per_1m_tokens"], 0.60)
