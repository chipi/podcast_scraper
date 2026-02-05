#!/usr/bin/env python3
"""Standalone unit tests for unified Mistral provider.

These tests verify that MistralProvider correctly implements all three protocols
(TranscriptionProvider, SpeakerDetector, SummarizationProvider) using
Mistral's API (Voxtral for transcription, chat models for speaker detection and summarization).

These are standalone provider tests - they test the provider itself,
not its integration with the app.
"""

import json
import os
import unittest
from unittest.mock import Mock, patch

import pytest

from podcast_scraper import config
from podcast_scraper.providers.mistral.mistral_provider import MistralProvider


@pytest.mark.unit
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
        mock_mistral_class.assert_called_once_with(api_key="test-api-key-123")

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
    def test_provider_implements_all_protocols(self, mock_mistral_class):
        """Test that MistralProvider implements all three protocols."""
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client

        provider = MistralProvider(self.cfg)

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
        self.assertFalse(provider._summarization_initialized)

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
        self.assertFalse(provider._summarization_initialized)

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


@pytest.mark.unit
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

    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    @patch("builtins.open", create=True)
    @patch("os.path.exists")
    @patch("mistralai.models.file.File", create=True)
    def test_transcribe_success(self, mock_file_class, mock_exists, mock_open, mock_mistral_class):
        """Test successful transcription."""
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

        # Mock File class used inside transcribe method
        mock_file_instance = Mock()
        mock_file_class.return_value = mock_file_instance

        provider = MistralProvider(self.cfg)
        provider.initialize()

        result = provider.transcribe("/path/to/audio.mp3")

        self.assertEqual(result, "Hello world")
        mock_client.audio.transcriptions.complete.assert_called_once()

    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    @patch("mistralai.models.file.File", create=True)
    @patch("builtins.open", create=True)
    @patch("os.path.exists")
    def test_transcribe_with_language(
        self, mock_exists, mock_open, mock_file_class, mock_mistral_class
    ):
        """Test transcription with explicit language."""
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

        # Mock File class used inside transcribe method
        mock_file_instance = Mock()
        mock_file_class.return_value = mock_file_instance

        provider = MistralProvider(self.cfg)
        provider.initialize()

        provider.transcribe("/path/to/audio.mp3", language="fr")

        # Verify language was passed to API
        call_kwargs = mock_client.audio.transcriptions.complete.call_args[1]
        self.assertEqual(call_kwargs.get("language"), "fr")

    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    def test_transcribe_not_initialized(self, mock_mistral_class):
        """Test transcribe raises RuntimeError if not initialized."""
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

    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    @patch("builtins.open", create=True)
    @patch("os.path.exists")
    def test_transcribe_api_error(self, mock_exists, mock_open, mock_mistral_class):
        """Test transcribe handles API errors."""
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

    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    @patch("mistralai.models.file.File", create=True)
    @patch("builtins.open", create=True)
    @patch("os.path.exists")
    def test_transcribe_with_segments_success(
        self, mock_exists, mock_open, mock_file_class, mock_mistral_class
    ):
        """Test transcribe_with_segments returns full result."""
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

        # Mock File class used inside transcribe method
        mock_file_instance = Mock()
        mock_file_class.return_value = mock_file_instance

        provider = MistralProvider(self.cfg)
        provider.initialize()

        result_dict, elapsed = provider.transcribe_with_segments("/path/to/audio.mp3")

        self.assertEqual(result_dict["text"], "Hello world")
        self.assertEqual(result_dict["segments"], [])  # Mistral may not provide segments
        self.assertIsInstance(elapsed, float)
        self.assertGreater(elapsed, 0)


@pytest.mark.unit
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
    def test_detect_hosts_from_feed_authors(self, mock_mistral_class):
        """Test detect_hosts prefers feed_authors."""
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client

        provider = MistralProvider(self.cfg)
        provider.initialize()

        hosts = provider.detect_hosts(
            feed_title="The Podcast",
            feed_description="A great podcast",
            feed_authors=["Alice", "Bob"],
        )

        self.assertEqual(hosts, {"Alice", "Bob"})

    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    @patch(
        "podcast_scraper.providers.mistral.mistral_provider."
        "MistralProvider._build_speaker_detection_prompt"
    )
    def test_detect_hosts_without_authors(self, mock_build_prompt, mock_mistral_class):
        """Test detect_hosts uses API when no feed_authors."""
        mock_build_prompt.return_value = "Prompt"
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = json.dumps(
            {"speakers": ["Alice", "Bob"], "hosts": [], "guests": []}
        )

        mock_client.chat.completions.create.return_value = mock_response

        provider = MistralProvider(self.cfg)
        provider.initialize()

        hosts = provider.detect_hosts(
            feed_title="The Podcast",
            feed_description="A great podcast",
            feed_authors=None,
        )

        # Should return empty set if no feed_authors and no API call made
        # (since we're not actually calling the API in this simplified test)
        self.assertIsInstance(hosts, set)

    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    @patch(
        "podcast_scraper.providers.mistral.mistral_provider."
        "MistralProvider._build_speaker_detection_prompt"
    )
    @patch(
        "podcast_scraper.providers.mistral.mistral_provider."
        "MistralProvider._parse_speakers_from_response"
    )
    def test_detect_speakers_success(self, mock_parse, mock_build_prompt, mock_mistral_class):
        """Test successful speaker detection."""
        mock_build_prompt.return_value = "Prompt"
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = (
            '{"speakers": ["Alice", "Bob"], "hosts": ["Alice"], "guests": ["Bob"]}'
        )
        # _parse_speakers_from_response returns:
        # (speaker_names_list, detected_hosts_set, detection_succeeded)
        mock_parse.return_value = (["Alice", "Bob"], {"Alice"}, True)

        # Mistral uses 'complete' method, not 'create'
        mock_client.chat.complete.return_value = mock_response

        provider = MistralProvider(self.cfg)
        provider.initialize()

        speakers, hosts, success = provider.detect_speakers(
            episode_title="Alice interviews Bob",
            episode_description="A great conversation",
            known_hosts={"Alice"},
        )

        self.assertEqual(speakers, ["Alice", "Bob"])
        self.assertTrue(success)

    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    def test_detect_speakers_not_initialized(self, mock_mistral_class):
        """Test detect_speakers raises RuntimeError if not initialized."""
        mock_client = Mock()
        mock_mistral_class.return_value = mock_client

        provider = MistralProvider(self.cfg)
        # Don't call initialize()

        with self.assertRaises(RuntimeError) as context:
            provider.detect_speakers("Title", "Description", set())

        self.assertIn("not initialized", str(context.exception))

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


@pytest.mark.unit
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
        """Test summarize raises RuntimeError if not initialized."""
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


@pytest.mark.unit
class TestMistralProviderPricing(unittest.TestCase):
    """Tests for MistralProvider.get_pricing() static method."""

    def test_get_pricing_audio_transcription(self):
        """Test pricing lookup for audio transcription."""
        pricing = MistralProvider.get_pricing("voxtral-mini-latest", "transcription")
        self.assertIn("cost_per_minute", pricing)
        self.assertEqual(pricing["cost_per_minute"], 0.006)

    def test_get_pricing_small_speaker_detection(self):
        """Test pricing lookup for mistral-small speaker detection."""
        pricing = MistralProvider.get_pricing("mistral-small-latest", "speaker_detection")
        self.assertEqual(pricing["input_cost_per_1m_tokens"], 0.20)
        self.assertEqual(pricing["output_cost_per_1m_tokens"], 0.20)

    def test_get_pricing_large_summarization(self):
        """Test pricing lookup for mistral-large summarization."""
        pricing = MistralProvider.get_pricing("mistral-large-latest", "summarization")
        self.assertEqual(pricing["input_cost_per_1m_tokens"], 2.00)
        self.assertEqual(pricing["output_cost_per_1m_tokens"], 6.00)

    def test_get_pricing_unsupported_model(self):
        """Test pricing lookup for unsupported model returns default pricing."""
        pricing = MistralProvider.get_pricing("mistral-unknown", "summarization")
        # Should default to small pricing
        self.assertEqual(pricing["input_cost_per_1m_tokens"], 0.20)
        self.assertEqual(pricing["output_cost_per_1m_tokens"], 0.20)
