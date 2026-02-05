#!/usr/bin/env python3
"""Standalone unit tests for unified DeepSeek provider.

These tests verify that DeepSeekProvider correctly implements two protocols
(SpeakerDetector, SummarizationProvider) using DeepSeek's OpenAI-compatible API.

These are standalone provider tests - they test the provider itself,
not its integration with the app.

Note: DeepSeek does NOT support transcription (no audio API).
"""

import os
import unittest
from unittest.mock import Mock, patch

import pytest

from podcast_scraper import config
from podcast_scraper.providers.deepseek.deepseek_provider import DeepSeekProvider


@pytest.mark.unit
class TestDeepSeekProviderStandalone(unittest.TestCase):
    """Standalone tests for DeepSeekProvider - testing the provider itself."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="deepseek",
            summary_provider="deepseek",
            deepseek_api_key="test-api-key-123",
            auto_speakers=False,  # Disable to avoid API calls
            generate_summaries=False,  # Disable to avoid API calls
        )

    @patch("podcast_scraper.providers.deepseek.deepseek_provider.OpenAI")
    def test_provider_creation(self, mock_openai_class):
        """Test that DeepSeekProvider can be created."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        provider = DeepSeekProvider(self.cfg)
        self.assertIsNotNone(provider)
        self.assertEqual(provider.__class__.__name__, "DeepSeekProvider")

        # Verify OpenAI client was created with correct base_url
        mock_openai_class.assert_called_once()
        call_kwargs = mock_openai_class.call_args[1]
        self.assertEqual(call_kwargs["api_key"], "test-api-key-123")
        self.assertEqual(call_kwargs["base_url"], "https://api.deepseek.com")

    @patch("podcast_scraper.providers.deepseek.deepseek_provider.OpenAI")
    def test_provider_creation_requires_api_key(self, mock_openai_class):
        """Test that DeepSeekProvider requires API key."""
        # Unset environment variable to ensure it's not loaded
        original_key = os.environ.pop("DEEPSEEK_API_KEY", None)
        try:
            # Note: Config validation happens before provider creation
            # So we need to catch ValidationError from Config, not ValueError from provider
            with self.assertRaises(Exception) as context:
                cfg = config.Config(
                    rss_url="https://example.com/feed.xml",
                    speaker_detector_provider="deepseek",
                )
                DeepSeekProvider(cfg)
            # Error can be either ValidationError (from Config) or ValueError (from provider)
        finally:
            # Restore original environment variable if it existed
            if original_key is not None:
                os.environ["DEEPSEEK_API_KEY"] = original_key
        error_msg = str(context.exception)
        self.assertTrue(
            "DeepSeek API key required" in error_msg or "validation error" in error_msg.lower()
        )

    @patch("podcast_scraper.providers.deepseek.deepseek_provider.OpenAI")
    def test_provider_implements_protocols(self, mock_openai_class):
        """Test that DeepSeekProvider implements required protocols."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        provider = DeepSeekProvider(self.cfg)

        # SpeakerDetector protocol
        self.assertTrue(hasattr(provider, "detect_speakers"))
        self.assertTrue(hasattr(provider, "detect_hosts"))
        self.assertTrue(hasattr(provider, "analyze_patterns"))
        self.assertTrue(hasattr(provider, "clear_cache"))

        # SummarizationProvider protocol
        self.assertTrue(hasattr(provider, "summarize"))
        self.assertTrue(hasattr(provider, "initialize"))
        self.assertTrue(hasattr(provider, "cleanup"))

        # Note: DeepSeek does NOT implement TranscriptionProvider

    @patch("podcast_scraper.providers.deepseek.deepseek_provider.OpenAI")
    def test_provider_initialization_state(self, mock_openai_class):
        """Test that provider tracks initialization state for each capability."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        provider = DeepSeekProvider(self.cfg)

        # Initially not initialized
        self.assertFalse(provider._speaker_detection_initialized)
        self.assertFalse(provider._summarization_initialized)
        self.assertFalse(provider.is_initialized)

    @patch("podcast_scraper.providers.deepseek.deepseek_provider.OpenAI")
    def test_provider_thread_safe(self, mock_openai_class):
        """Test that provider marks itself as thread-safe."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        provider = DeepSeekProvider(self.cfg)
        self.assertFalse(provider._requires_separate_instances)

    @patch("podcast_scraper.providers.deepseek.deepseek_provider.OpenAI")
    def test_speaker_detection_initialization(self, mock_openai_class):
        """Test that speaker detection can be initialized independently."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            deepseek_api_key=self.cfg.deepseek_api_key,
            speaker_detector_provider="deepseek",
            auto_speakers=True,
            generate_summaries=False,  # Disable to avoid initializing summarization
        )
        provider = DeepSeekProvider(cfg)

        provider.initialize()

        self.assertTrue(provider._speaker_detection_initialized)
        # Other capability should not be initialized
        self.assertFalse(provider._summarization_initialized)

    @patch("podcast_scraper.providers.deepseek.deepseek_provider.OpenAI")
    def test_summarization_initialization(self, mock_openai_class):
        """Test that summarization can be initialized independently."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            deepseek_api_key=self.cfg.deepseek_api_key,
            summary_provider="deepseek",
            generate_summaries=True,
            generate_metadata=True,  # Required when generate_summaries is True
            auto_speakers=False,  # Disable to avoid initializing speaker detection
        )
        provider = DeepSeekProvider(cfg)

        provider.initialize()

        self.assertTrue(provider._summarization_initialized)
        # Other capability should not be initialized
        self.assertFalse(provider._speaker_detection_initialized)

    @patch("podcast_scraper.providers.deepseek.deepseek_provider.OpenAI")
    def test_unified_initialization(self, mock_openai_class):
        """Test that all capabilities can be initialized together."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            deepseek_api_key=self.cfg.deepseek_api_key,
            speaker_detector_provider="deepseek",
            summary_provider="deepseek",
            auto_speakers=True,
            generate_summaries=True,
            generate_metadata=True,  # Required when generate_summaries is True
        )

        provider = DeepSeekProvider(cfg)
        provider.initialize()

        # All should be initialized
        self.assertTrue(provider._speaker_detection_initialized)
        self.assertTrue(provider._summarization_initialized)
        self.assertTrue(provider.is_initialized)

    @patch("podcast_scraper.providers.deepseek.deepseek_provider.OpenAI")
    def test_cleanup_releases_all_resources(self, mock_openai_class):
        """Test that cleanup releases all resources."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        provider = DeepSeekProvider(self.cfg)
        provider._speaker_detection_initialized = True
        provider._summarization_initialized = True

        provider.cleanup()

        self.assertFalse(provider._speaker_detection_initialized)
        self.assertFalse(provider._summarization_initialized)
        self.assertFalse(provider.is_initialized)

    @patch("podcast_scraper.providers.deepseek.deepseek_provider.OpenAI")
    def test_custom_base_url(self, mock_openai_class):
        """Test that custom base_url is used when provided."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            deepseek_api_key="test-key",
            deepseek_api_base="https://custom.deepseek.com",
            speaker_detector_provider="deepseek",
        )

        DeepSeekProvider(cfg)

        # Verify OpenAI client was created with custom base_url
        call_kwargs = mock_openai_class.call_args[1]
        self.assertEqual(call_kwargs["base_url"], "https://custom.deepseek.com")


@pytest.mark.unit
class TestDeepSeekProviderSpeakerDetection(unittest.TestCase):
    """Tests for DeepSeekProvider speaker detection methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="deepseek",
            deepseek_api_key="test-api-key-123",
            auto_speakers=True,
        )

    @patch("podcast_scraper.providers.deepseek.deepseek_provider.OpenAI")
    def test_detect_hosts_from_feed_authors(self, mock_openai_class):
        """Test detect_hosts prefers feed_authors."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        provider = DeepSeekProvider(self.cfg)
        provider.initialize()

        hosts = provider.detect_hosts(
            feed_title="The Podcast",
            feed_description="A great podcast",
            feed_authors=["Alice", "Bob"],
        )

        self.assertEqual(hosts, {"Alice", "Bob"})

    @patch("podcast_scraper.providers.deepseek.deepseek_provider.OpenAI")
    @patch(
        "podcast_scraper.providers.deepseek.deepseek_provider."
        "DeepSeekProvider._build_speaker_detection_prompt"
    )
    @patch(
        "podcast_scraper.providers.deepseek.deepseek_provider."
        "DeepSeekProvider._parse_speakers_from_response"
    )
    def test_detect_speakers_success(self, mock_parse, mock_build_prompt, mock_openai_class):
        """Test successful speaker detection."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_build_prompt.return_value = "Prompt"

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

        # _parse_speakers_from_response returns:
        # (speaker_names_list, detected_hosts_set, detection_succeeded)
        mock_parse.return_value = (["Alice", "Bob"], {"Alice"}, True)

        provider = DeepSeekProvider(self.cfg)
        provider.initialize()

        speakers, hosts, success = provider.detect_speakers(
            episode_title="Alice interviews Bob",
            episode_description="A great conversation",
            known_hosts={"Alice"},
        )

        self.assertEqual(speakers, ["Alice", "Bob"])
        self.assertTrue(success)

    @patch("podcast_scraper.providers.deepseek.deepseek_provider.OpenAI")
    def test_detect_speakers_not_initialized(self, mock_openai_class):
        """Test detect_speakers raises RuntimeError if not initialized."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        provider = DeepSeekProvider(self.cfg)
        # Don't call initialize()

        with self.assertRaises(RuntimeError) as context:
            provider.detect_speakers("Title", "Description", set())

        self.assertIn("not initialized", str(context.exception))

    @patch("podcast_scraper.providers.deepseek.deepseek_provider.OpenAI")
    def test_analyze_patterns_success(self, mock_openai_class):
        """Test pattern analysis (returns None for DeepSeek)."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        from podcast_scraper import models

        provider = DeepSeekProvider(self.cfg)
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

        # DeepSeek provider doesn't implement pattern analysis, returns None
        result = provider.analyze_patterns(episodes=episodes, known_hosts={"Alice"})

        self.assertIsNone(result)  # DeepSeek provider returns None to use local logic

    @patch("podcast_scraper.providers.deepseek.deepseek_provider.OpenAI")
    def test_clear_cache(self, mock_openai_class):
        """Test cache clearing (no-op for DeepSeek provider)."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        provider = DeepSeekProvider(self.cfg)

        # clear_cache should not raise (it's a no-op for DeepSeek provider)
        provider.clear_cache()

        # DeepSeek provider doesn't use cache, but method exists for protocol compliance
        # It's essentially a no-op


@pytest.mark.unit
class TestDeepSeekProviderSummarization(unittest.TestCase):
    """Tests for DeepSeekProvider summarization methods."""

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
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_summarize_success(self, mock_render_prompt, mock_openai_class):
        """Test successful summarization."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock render_prompt to return prompts (called twice: system and user)
        mock_render_prompt.side_effect = ["System Prompt", "User Prompt"]

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "This is a summary."
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 1000
        mock_response.usage.completion_tokens = 200

        mock_client.chat.completions.create.return_value = mock_response

        provider = DeepSeekProvider(self.cfg)
        provider.initialize()

        result = provider.summarize("This is a long transcript text.")

        self.assertEqual(result["summary"], "This is a summary.")
        self.assertIn("metadata", result)
        self.assertEqual(result["metadata"]["model"], provider.summary_model)

    @patch("podcast_scraper.providers.deepseek.deepseek_provider.OpenAI")
    def test_summarize_not_initialized(self, mock_openai_class):
        """Test summarize raises RuntimeError if not initialized."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        provider = DeepSeekProvider(self.cfg)
        # Don't call initialize()

        with self.assertRaises(RuntimeError) as context:
            provider.summarize("Text")

        self.assertIn("not initialized", str(context.exception))

    @patch("podcast_scraper.providers.deepseek.deepseek_provider.OpenAI")
    @patch("podcast_scraper.prompts.store.get_prompt_metadata")
    @patch("podcast_scraper.prompts.store.render_prompt")
    @patch(
        "podcast_scraper.providers.deepseek.deepseek_provider."
        "DeepSeekProvider._build_summarization_prompts"
    )
    def test_summarize_with_params(
        self, mock_build_prompts, mock_render_prompt, mock_get_metadata, mock_openai_class
    ):
        """Test summarization with custom parameters."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # _build_summarization_prompts returns:
        # (system_prompt, user_prompt, system_prompt_name, user_prompt_name,
        #  paragraphs_min, paragraphs_max)
        mock_build_prompts.return_value = (
            "System Prompt",
            "User Prompt",
            "deepseek/summarization/system_v1",
            "deepseek/summarization/long_v1",
            1,
            3,
        )
        # render_prompt is called inside _build_summarization_prompts
        mock_render_prompt.side_effect = ["System Prompt", "User Prompt"]
        # get_prompt_metadata is called for tracking
        mock_get_metadata.return_value = {
            "name": "deepseek/summarization/system_v1",
            "sha256": "abc123",
        }

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Summary"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 500
        mock_response.usage.completion_tokens = 100

        mock_client.chat.completions.create.return_value = mock_response

        provider = DeepSeekProvider(self.cfg)
        provider.initialize()

        params = {"max_length": 100, "min_length": 50}
        provider.summarize("Text", params=params)

        # Verify API was called
        mock_client.chat.completions.create.assert_called()

    @patch("podcast_scraper.providers.deepseek.deepseek_provider.OpenAI")
    @patch(
        "podcast_scraper.providers.deepseek.deepseek_provider."
        "DeepSeekProvider._build_summarization_prompts"
    )
    def test_summarize_api_error(self, mock_build_prompts, mock_openai_class):
        """Test summarization error handling."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

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

        mock_client.chat.completions.create.side_effect = Exception("API error")

        provider = DeepSeekProvider(self.cfg)
        provider.initialize()

        from podcast_scraper.exceptions import ProviderRuntimeError

        with self.assertRaises(ProviderRuntimeError) as context:
            provider.summarize("Text")

        self.assertIn("summarization failed", str(context.exception).lower())


@pytest.mark.unit
class TestDeepSeekProviderPricing(unittest.TestCase):
    """Tests for DeepSeekProvider.get_pricing() static method."""

    def test_get_pricing_speaker_detection(self):
        """Test pricing lookup for speaker detection."""
        pricing = DeepSeekProvider.get_pricing("deepseek-chat", "speaker_detection")
        self.assertIn("input_cost_per_1m_tokens", pricing)
        self.assertIn("output_cost_per_1m_tokens", pricing)
        self.assertIn("cache_hit_input_cost_per_1m_tokens", pricing)
        self.assertEqual(pricing["input_cost_per_1m_tokens"], 0.28)
        self.assertEqual(pricing["output_cost_per_1m_tokens"], 0.42)
        self.assertEqual(pricing["cache_hit_input_cost_per_1m_tokens"], 0.028)

    def test_get_pricing_summarization(self):
        """Test pricing lookup for summarization."""
        pricing = DeepSeekProvider.get_pricing("deepseek-chat", "summarization")
        self.assertIn("input_cost_per_1m_tokens", pricing)
        self.assertIn("output_cost_per_1m_tokens", pricing)
        self.assertIn("cache_hit_input_cost_per_1m_tokens", pricing)
        self.assertEqual(pricing["input_cost_per_1m_tokens"], 0.28)
        self.assertEqual(pricing["output_cost_per_1m_tokens"], 0.42)
        self.assertEqual(pricing["cache_hit_input_cost_per_1m_tokens"], 0.028)

    def test_get_pricing_reasoner_model(self):
        """Test pricing lookup for reasoner model (same pricing)."""
        pricing = DeepSeekProvider.get_pricing("deepseek-reasoner", "summarization")
        # DeepSeek reasoner uses same pricing as chat
        self.assertEqual(pricing["input_cost_per_1m_tokens"], 0.28)
        self.assertEqual(pricing["output_cost_per_1m_tokens"], 0.42)

    def test_get_pricing_unsupported_capability(self):
        """Test pricing lookup for unsupported capability returns empty dict."""
        pricing = DeepSeekProvider.get_pricing("deepseek-chat", "transcription")
        # DeepSeek doesn't support transcription
        self.assertEqual(pricing, {})
