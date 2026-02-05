#!/usr/bin/env python3
"""Standalone unit tests for unified Anthropic provider.

These tests verify that AnthropicProvider correctly implements all three protocols
(TranscriptionProvider, SpeakerDetector, SummarizationProvider) using
Anthropic's API (Claude chat models).

Note: Anthropic does NOT support native audio transcription, so transcription
methods raise NotImplementedError.

These are standalone provider tests - they test the provider itself,
not its integration with the app.
"""

import json
import os
import unittest
from unittest.mock import Mock, patch

import pytest

from podcast_scraper import config
from podcast_scraper.providers.anthropic.anthropic_provider import AnthropicProvider


@pytest.mark.unit
class TestAnthropicProviderStandalone(unittest.TestCase):
    """Standalone tests for AnthropicProvider - testing the provider itself."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="whisper",  # Anthropic doesn't support transcription
            speaker_detector_provider="anthropic",
            summary_provider="anthropic",
            anthropic_api_key="test-api-key-123",
            transcribe_missing=False,  # Disable to avoid API calls
            auto_speakers=False,  # Disable to avoid API calls
            generate_summaries=False,  # Disable to avoid API calls
        )

    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    def test_provider_creation(self, mock_anthropic):
        """Test that AnthropicProvider can be created."""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        provider = AnthropicProvider(self.cfg)
        self.assertIsNotNone(provider)
        self.assertEqual(provider.__class__.__name__, "AnthropicProvider")
        # Verify Anthropic client was created
        mock_anthropic.assert_called_once_with(api_key="test-api-key-123")

    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    def test_provider_creation_requires_api_key(self, mock_anthropic):
        """Test that AnthropicProvider requires API key."""
        # Unset environment variable to ensure it's not loaded
        original_key = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            # Note: Config validation happens before provider creation
            # So we need to catch ValidationError from Config, not ValueError from provider
            with self.assertRaises(Exception) as context:
                cfg = config.Config(
                    rss_url="https://example.com/feed.xml",
                    speaker_detector_provider="anthropic",
                )
                AnthropicProvider(cfg)
            # Error can be either ValidationError (from Config) or ValueError (from provider)
        finally:
            # Restore original environment variable if it existed
            if original_key is not None:
                os.environ["ANTHROPIC_API_KEY"] = original_key
        error_msg = str(context.exception)
        self.assertTrue(
            "Anthropic API key required" in error_msg or "validation error" in error_msg.lower()
        )

    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    def test_provider_implements_all_protocols(self, mock_anthropic):
        """Test that AnthropicProvider implements all three protocols."""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        provider = AnthropicProvider(self.cfg)

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

    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    def test_provider_initialization_state(self, mock_anthropic):
        """Test that provider tracks initialization state for each capability."""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        provider = AnthropicProvider(self.cfg)

        # Initially not initialized
        self.assertFalse(provider._transcription_initialized)
        self.assertFalse(provider._speaker_detection_initialized)
        self.assertFalse(provider._summarization_initialized)
        self.assertFalse(provider.is_initialized)

    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    def test_provider_thread_safe(self, mock_anthropic):
        """Test that provider marks itself as thread-safe."""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        provider = AnthropicProvider(self.cfg)
        self.assertFalse(provider._requires_separate_instances)

    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    def test_transcription_initialization(self, mock_anthropic):
        """Test that transcription can be initialized independently."""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            anthropic_api_key=self.cfg.anthropic_api_key,
            transcription_provider="whisper",  # Anthropic doesn't support transcription
            transcribe_missing=True,
            auto_speakers=False,  # Disable to avoid initializing speaker detection
        )
        provider = AnthropicProvider(cfg)

        provider.initialize()

        self.assertTrue(provider._transcription_initialized)
        # Other capabilities should not be initialized
        self.assertFalse(provider._speaker_detection_initialized)
        self.assertFalse(provider._summarization_initialized)

    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    def test_speaker_detection_initialization(self, mock_anthropic):
        """Test that speaker detection can be initialized independently."""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            anthropic_api_key=self.cfg.anthropic_api_key,
            speaker_detector_provider="anthropic",
            auto_speakers=True,
            transcribe_missing=False,  # Disable to avoid initializing transcription
        )
        provider = AnthropicProvider(cfg)

        provider.initialize()

        self.assertTrue(provider._speaker_detection_initialized)
        # Other capabilities should not be initialized
        self.assertFalse(provider._transcription_initialized)
        self.assertFalse(provider._summarization_initialized)

    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    def test_summarization_initialization(self, mock_anthropic):
        """Test that summarization can be initialized independently."""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            anthropic_api_key=self.cfg.anthropic_api_key,
            summary_provider="anthropic",
            generate_summaries=True,
            generate_metadata=True,  # Required when generate_summaries is True
            auto_speakers=False,  # Disable to avoid initializing speaker detection
            transcribe_missing=False,  # Disable to avoid initializing transcription
        )
        provider = AnthropicProvider(cfg)

        provider.initialize()

        self.assertTrue(provider._summarization_initialized)
        # Other capabilities should not be initialized
        self.assertFalse(provider._transcription_initialized)
        self.assertFalse(provider._speaker_detection_initialized)

    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    def test_unified_initialization(self, mock_anthropic):
        """Test that all capabilities can be initialized together."""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            anthropic_api_key=self.cfg.anthropic_api_key,
            transcription_provider="whisper",  # Anthropic doesn't support transcription
            speaker_detector_provider="anthropic",
            summary_provider="anthropic",
            transcribe_missing=True,
            auto_speakers=True,
            generate_summaries=True,
            generate_metadata=True,  # Required when generate_summaries is True
        )

        provider = AnthropicProvider(cfg)
        provider.initialize()

        # All should be initialized
        self.assertTrue(provider._transcription_initialized)
        self.assertTrue(provider._speaker_detection_initialized)
        self.assertTrue(provider._summarization_initialized)
        self.assertTrue(provider.is_initialized)

    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    def test_cleanup_releases_all_resources(self, mock_anthropic):
        """Test that cleanup releases all resources."""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        provider = AnthropicProvider(self.cfg)
        provider._transcription_initialized = True
        provider._speaker_detection_initialized = True
        provider._summarization_initialized = True

        provider.cleanup()

        self.assertFalse(provider._transcription_initialized)
        self.assertFalse(provider._speaker_detection_initialized)
        self.assertFalse(provider._summarization_initialized)
        self.assertFalse(provider.is_initialized)

    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    def test_transcription_model_attribute(self, mock_anthropic):
        """Test that transcription_model attribute exists."""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        provider = AnthropicProvider(self.cfg)

        # Transcription attributes
        self.assertTrue(hasattr(provider, "transcription_model"))
        self.assertTrue(hasattr(provider, "is_initialized"))

        # Verify transcription_model is accessible
        self.assertIsNotNone(provider.transcription_model)


@pytest.mark.unit
class TestAnthropicProviderTranscription(unittest.TestCase):
    """Tests for AnthropicProvider transcription methods.

    Note: Anthropic does NOT support native audio transcription.
    These tests verify that transcription methods correctly raise NotImplementedError.
    """

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="whisper",  # Anthropic doesn't support transcription
            anthropic_api_key="test-api-key-123",
            transcribe_missing=True,
        )

    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    def test_transcribe_raises_not_implemented(self, mock_anthropic):
        """Test that transcribe raises NotImplementedError."""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        provider = AnthropicProvider(self.cfg)
        provider.initialize()

        with self.assertRaises(NotImplementedError) as context:
            provider.transcribe("/path/to/audio.mp3")

        self.assertIn(
            "Anthropic doesn't support native audio transcription", str(context.exception)
        )

    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    def test_transcribe_not_initialized(self, mock_anthropic):
        """Test transcribe raises RuntimeError if not initialized."""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        provider = AnthropicProvider(self.cfg)
        # Don't call initialize()

        with self.assertRaises(RuntimeError) as context:
            provider.transcribe("/path/to/audio.mp3")

        self.assertIn("not initialized", str(context.exception))

    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    def test_transcribe_with_segments_raises_not_implemented(self, mock_anthropic):
        """Test that transcribe_with_segments raises NotImplementedError."""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        provider = AnthropicProvider(self.cfg)
        provider.initialize()

        with self.assertRaises(NotImplementedError) as context:
            provider.transcribe_with_segments("/path/to/audio.mp3")

        self.assertIn(
            "Anthropic doesn't support native audio transcription", str(context.exception)
        )


@pytest.mark.unit
class TestAnthropicProviderSpeakerDetection(unittest.TestCase):
    """Tests for AnthropicProvider speaker detection methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="anthropic",
            anthropic_api_key="test-api-key-123",
            auto_speakers=True,
        )

    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    def test_detect_hosts_from_feed_authors(self, mock_anthropic):
        """Test detect_hosts prefers feed_authors."""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        provider = AnthropicProvider(self.cfg)
        provider.initialize()

        hosts = provider.detect_hosts(
            feed_title="The Podcast",
            feed_description="A great podcast",
            feed_authors=["Alice", "Bob"],
        )

        self.assertEqual(hosts, {"Alice", "Bob"})

    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    @patch(
        "podcast_scraper.providers.anthropic.anthropic_provider."
        "AnthropicProvider._build_speaker_detection_prompt"
    )
    def test_detect_hosts_without_authors(self, mock_build_prompt, mock_anthropic):
        """Test detect_hosts uses API when no feed_authors."""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        mock_build_prompt.return_value = "Prompt"

        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = json.dumps(
            {"speakers": ["Alice", "Bob"], "hosts": [], "guests": []}
        )

        mock_client.messages.create.return_value = mock_response

        provider = AnthropicProvider(self.cfg)
        provider.initialize()

        hosts = provider.detect_hosts(
            feed_title="The Podcast",
            feed_description="A great podcast",
            feed_authors=None,
        )

        # Should return empty set if no feed_authors and no API call made
        # (since we're not actually calling the API in this simplified test)
        self.assertIsInstance(hosts, set)

    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    @patch(
        "podcast_scraper.providers.anthropic.anthropic_provider."
        "AnthropicProvider._build_speaker_detection_prompt"
    )
    @patch(
        "podcast_scraper.providers.anthropic.anthropic_provider."
        "AnthropicProvider._parse_speakers_from_response"
    )
    def test_detect_speakers_success(self, mock_parse, mock_build_prompt, mock_anthropic):
        """Test successful speaker detection."""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        mock_build_prompt.return_value = "Prompt"

        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = (
            '{"speakers": ["Alice", "Bob"], "hosts": ["Alice"], "guests": ["Bob"]}'
        )
        mock_response.usage = Mock()
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50

        # _parse_speakers_from_response returns:
        # (speaker_names_list, detected_hosts_set, detection_succeeded)
        mock_parse.return_value = (["Alice", "Bob"], {"Alice"}, True)

        mock_client.messages.create.return_value = mock_response

        provider = AnthropicProvider(self.cfg)
        provider.initialize()

        speakers, hosts, success = provider.detect_speakers(
            episode_title="Alice interviews Bob",
            episode_description="A great conversation",
            known_hosts={"Alice"},
        )

        self.assertEqual(speakers, ["Alice", "Bob"])
        self.assertTrue(success)

    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    def test_detect_speakers_not_initialized(self, mock_anthropic):
        """Test detect_speakers raises RuntimeError if not initialized."""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        provider = AnthropicProvider(self.cfg)
        # Don't call initialize()

        with self.assertRaises(RuntimeError) as context:
            provider.detect_speakers("Title", "Description", set())

        self.assertIn("not initialized", str(context.exception))

    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    def test_analyze_patterns_success(self, mock_anthropic):
        """Test successful pattern analysis."""
        from podcast_scraper import models

        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        provider = AnthropicProvider(self.cfg)
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

        # Anthropic provider doesn't implement pattern analysis, returns None
        result = provider.analyze_patterns(episodes=episodes, known_hosts={"Alice"})

        self.assertIsNone(result)  # Anthropic provider returns None to use local logic

    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    def test_clear_cache(self, mock_anthropic):
        """Test cache clearing (no-op for Anthropic provider)."""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        provider = AnthropicProvider(self.cfg)

        # clear_cache should not raise (it's a no-op for Anthropic provider)
        provider.clear_cache()

        # Anthropic provider doesn't use cache, but method exists for protocol compliance
        # It's essentially a no-op


@pytest.mark.unit
class TestAnthropicProviderSummarization(unittest.TestCase):
    """Tests for AnthropicProvider summarization methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="anthropic",
            anthropic_api_key="test-api-key-123",
            generate_summaries=True,
            generate_metadata=True,  # Required when generate_summaries=True
        )

    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_summarize_success(self, mock_render_prompt, mock_anthropic):
        """Test successful summarization."""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        # Mock render_prompt to return prompts (called twice: system and user)
        mock_render_prompt.side_effect = ["System Prompt", "User Prompt"]

        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "This is a summary."
        mock_response.usage = Mock()
        mock_response.usage.input_tokens = 200
        mock_response.usage.output_tokens = 100

        mock_client.messages.create.return_value = mock_response

        provider = AnthropicProvider(self.cfg)
        provider.initialize()

        result = provider.summarize("This is a long transcript text.")

        self.assertEqual(result["summary"], "This is a summary.")
        self.assertIn("metadata", result)
        self.assertEqual(result["metadata"]["model"], provider.summary_model)

    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    def test_summarize_not_initialized(self, mock_anthropic):
        """Test summarize raises RuntimeError if not initialized."""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        provider = AnthropicProvider(self.cfg)
        # Don't call initialize()

        with self.assertRaises(RuntimeError) as context:
            provider.summarize("Text")

        self.assertIn("not initialized", str(context.exception))

    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    @patch("podcast_scraper.prompts.store.get_prompt_metadata")
    @patch("podcast_scraper.prompts.store.render_prompt")
    @patch(
        "podcast_scraper.providers.anthropic.anthropic_provider."
        "AnthropicProvider._build_summarization_prompts"
    )
    def test_summarize_with_params(
        self, mock_build_prompts, mock_render_prompt, mock_get_metadata, mock_anthropic
    ):
        """Test summarization with custom parameters."""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        # _build_summarization_prompts returns:
        # (system_prompt, user_prompt, system_prompt_name, user_prompt_name,
        #  paragraphs_min, paragraphs_max)
        mock_build_prompts.return_value = (
            "System Prompt",
            "User Prompt",
            "anthropic/summarization/system_v1",
            "anthropic/summarization/long_v1",
            1,
            3,
        )
        # render_prompt is called inside _build_summarization_prompts
        mock_render_prompt.side_effect = ["System Prompt", "User Prompt"]
        # get_prompt_metadata is called for tracking
        mock_get_metadata.return_value = {
            "name": "anthropic/summarization/system_v1",
            "sha256": "abc123",
        }

        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Summary"
        mock_response.usage = Mock()
        mock_response.usage.input_tokens = 200
        mock_response.usage.output_tokens = 100

        mock_client.messages.create.return_value = mock_response

        provider = AnthropicProvider(self.cfg)
        provider.initialize()

        params = {"max_length": 100, "min_length": 50}
        provider.summarize("Text", params=params)

        # Verify API was called
        mock_client.messages.create.assert_called()

    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    @patch(
        "podcast_scraper.providers.anthropic.anthropic_provider."
        "AnthropicProvider._build_summarization_prompts"
    )
    def test_summarize_api_error(self, mock_build_prompts, mock_anthropic):
        """Test summarization error handling."""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

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

        mock_client.messages.create.side_effect = Exception("API error")

        provider = AnthropicProvider(self.cfg)
        provider.initialize()

        from podcast_scraper.exceptions import ProviderRuntimeError

        with self.assertRaises(ProviderRuntimeError) as context:
            provider.summarize("Text")

        self.assertIn("summarization failed", str(context.exception).lower())


@pytest.mark.unit
class TestAnthropicProviderPricing(unittest.TestCase):
    """Tests for AnthropicProvider.get_pricing() static method."""

    def test_get_pricing_transcription(self):
        """Test pricing lookup for transcription (returns placeholder)."""
        pricing = AnthropicProvider.get_pricing("claude-3-5-sonnet-20241022", "transcription")
        # Anthropic doesn't support transcription, returns placeholder
        self.assertIn("cost_per_second", pricing)
        self.assertEqual(pricing["cost_per_second"], 0.0)

    def test_get_pricing_3_5_sonnet_speaker_detection(self):
        """Test pricing lookup for Claude 3.5 Sonnet speaker detection."""
        pricing = AnthropicProvider.get_pricing("claude-3-5-sonnet-20241022", "speaker_detection")
        self.assertEqual(pricing["input_cost_per_1m_tokens"], 3.00)
        self.assertEqual(pricing["output_cost_per_1m_tokens"], 15.00)

    def test_get_pricing_3_5_sonnet_summarization(self):
        """Test pricing lookup for Claude 3.5 Sonnet summarization."""
        pricing = AnthropicProvider.get_pricing("claude-3-5-sonnet-20241022", "summarization")
        self.assertEqual(pricing["input_cost_per_1m_tokens"], 3.00)
        self.assertEqual(pricing["output_cost_per_1m_tokens"], 15.00)

    def test_get_pricing_3_5_haiku_summarization(self):
        """Test pricing lookup for Claude 3.5 Haiku summarization."""
        pricing = AnthropicProvider.get_pricing("claude-3-5-haiku-20241022", "summarization")
        self.assertEqual(pricing["input_cost_per_1m_tokens"], 0.80)
        self.assertEqual(pricing["output_cost_per_1m_tokens"], 4.00)

    def test_get_pricing_3_opus_summarization(self):
        """Test pricing lookup for Claude 3 Opus summarization."""
        pricing = AnthropicProvider.get_pricing("claude-3-opus-20240229", "summarization")
        self.assertEqual(pricing["input_cost_per_1m_tokens"], 15.00)
        self.assertEqual(pricing["output_cost_per_1m_tokens"], 75.00)

    def test_get_pricing_3_haiku_summarization(self):
        """Test pricing lookup for Claude 3 Haiku summarization."""
        pricing = AnthropicProvider.get_pricing("claude-3-haiku-20240307", "summarization")
        self.assertEqual(pricing["input_cost_per_1m_tokens"], 0.25)
        self.assertEqual(pricing["output_cost_per_1m_tokens"], 1.25)

    def test_get_pricing_unsupported_model(self):
        """Test pricing lookup for unsupported model returns default pricing."""
        pricing = AnthropicProvider.get_pricing("claude-unknown", "summarization")
        # Should default to 3.5-sonnet pricing
        self.assertEqual(pricing["input_cost_per_1m_tokens"], 3.00)
        self.assertEqual(pricing["output_cost_per_1m_tokens"], 15.00)
