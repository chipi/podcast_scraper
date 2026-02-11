#!/usr/bin/env python3
"""Standalone unit tests for unified Grok provider.

These tests verify that GrokProvider correctly implements two protocols
(SpeakerDetector, SummarizationProvider) using Grok's API via OpenAI SDK.

These are standalone provider tests - they test the provider itself,
not its integration with the app.

Note: Grok does NOT support transcription (no audio API).
"""

import json
import os
import unittest
from unittest.mock import MagicMock, Mock, patch

import pytest

# Mock openai before importing modules that require it
# Unit tests run without openai package installed
# Use patch.dict without 'with' to avoid context manager conflicts with @patch decorators
mock_openai = MagicMock()
mock_openai.OpenAI = Mock()
_patch_openai = patch.dict(
    "sys.modules",
    {
        "openai": mock_openai,
    },
)
_patch_openai.start()

from podcast_scraper import config
from podcast_scraper.providers.grok.grok_provider import GrokProvider
from podcast_scraper.providers.ml import speaker_detection


@pytest.mark.unit
class TestGrokProviderStandalone(unittest.TestCase):
    """Standalone tests for GrokProvider - testing the provider itself."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="grok",
            summary_provider="grok",
            grok_api_key="test-api-key-123",
            auto_speakers=False,  # Disable to avoid API calls
            generate_summaries=False,  # Disable to avoid API calls
        )

    @patch("podcast_scraper.providers.grok.grok_provider.OpenAI")
    def test_provider_creation(self, mock_openai):
        """Test that GrokProvider can be created."""
        provider = GrokProvider(self.cfg)
        self.assertIsNotNone(provider)
        self.assertEqual(provider.__class__.__name__, "GrokProvider")
        # Verify OpenAI client was created with Grok base URL
        mock_openai.assert_called_once()
        call_kwargs = mock_openai.call_args[1]
        self.assertEqual(call_kwargs["api_key"], "test-api-key-123")
        self.assertEqual(call_kwargs["base_url"], "https://api.x.ai/v1")

    @patch("podcast_scraper.providers.grok.grok_provider.OpenAI")
    def test_provider_creation_requires_api_key(self, mock_openai):
        """Test that GrokProvider requires API key."""
        # Unset environment variable to ensure it's not loaded
        original_key = os.environ.pop("GROK_API_KEY", None)
        try:
            # Note: Config validation happens before provider creation
            with self.assertRaises(Exception) as context:
                cfg = config.Config(
                    rss_url="https://example.com/feed.xml",
                    speaker_detector_provider="grok",
                )
                GrokProvider(cfg)
            # Error can be either ValidationError (from Config) or ValueError (from provider)
        finally:
            # Restore original environment variable if it existed
            if original_key is not None:
                os.environ["GROK_API_KEY"] = original_key
        error_msg = str(context.exception)
        self.assertTrue(
            "Grok API key required" in error_msg or "validation error" in error_msg.lower()
        )

    @patch("podcast_scraper.providers.grok.grok_provider.OpenAI")
    def test_provider_implements_protocols(self, mock_openai):
        """Test that GrokProvider implements required protocols."""
        provider = GrokProvider(self.cfg)

        # SpeakerDetector protocol
        self.assertTrue(hasattr(provider, "detect_speakers"))
        self.assertTrue(hasattr(provider, "detect_hosts"))
        self.assertTrue(hasattr(provider, "analyze_patterns"))
        self.assertTrue(hasattr(provider, "clear_cache"))

        # SummarizationProvider protocol
        self.assertTrue(hasattr(provider, "summarize"))
        self.assertTrue(hasattr(provider, "initialize"))
        self.assertTrue(hasattr(provider, "cleanup"))

        # Note: Grok does NOT implement TranscriptionProvider
        self.assertFalse(hasattr(provider, "transcribe"))

    @patch("podcast_scraper.providers.grok.grok_provider.OpenAI")
    def test_provider_initialization_state(self, mock_openai):
        """Test that provider tracks initialization state for each capability."""
        provider = GrokProvider(self.cfg)

        # Initially not initialized
        self.assertFalse(provider._speaker_detection_initialized)
        self.assertFalse(provider._summarization_initialized)
        self.assertFalse(provider.is_initialized)

    @patch("podcast_scraper.providers.grok.grok_provider.OpenAI")
    def test_provider_thread_safe(self, mock_openai):
        """Test that provider marks itself as thread-safe."""
        provider = GrokProvider(self.cfg)
        self.assertFalse(provider._requires_separate_instances)

    @patch("podcast_scraper.providers.grok.grok_provider.OpenAI")
    def test_speaker_detection_initialization(self, mock_openai):
        """Test that speaker detection can be initialized independently."""
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            grok_api_key=self.cfg.grok_api_key,
            speaker_detector_provider="grok",
            auto_speakers=True,
            generate_summaries=False,  # Disable to avoid initializing summarization
        )
        provider = GrokProvider(cfg)

        provider.initialize()

        self.assertTrue(provider._speaker_detection_initialized)
        # Other capability should not be initialized
        self.assertFalse(provider._summarization_initialized)

    @patch("podcast_scraper.providers.grok.grok_provider.OpenAI")
    def test_summarization_initialization(self, mock_openai):
        """Test that summarization can be initialized independently."""
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            grok_api_key=self.cfg.grok_api_key,
            summary_provider="grok",
            generate_summaries=True,
            generate_metadata=True,  # Required when generate_summaries=True
            auto_speakers=False,  # Disable to avoid initializing speaker detection
        )
        provider = GrokProvider(cfg)

        provider.initialize()

        self.assertTrue(provider._summarization_initialized)
        # Other capability should not be initialized
        self.assertFalse(provider._speaker_detection_initialized)

    @patch("podcast_scraper.providers.grok.grok_provider.OpenAI")
    def test_custom_base_url(self, mock_openai):
        """Test that custom base URL is used for E2E testing."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="grok",
            grok_api_key="test-key",
            grok_api_base="http://localhost:8000/v1",
        )
        GrokProvider(cfg)  # Create provider to test initialization

        # Verify OpenAI client was created with custom base URL
        call_kwargs = mock_openai.call_args[1]
        self.assertEqual(call_kwargs["base_url"], "http://localhost:8000/v1")

    @patch("podcast_scraper.providers.grok.grok_provider.OpenAI")
    def test_cleanup(self, mock_openai):
        """Test that cleanup resets initialization state."""
        provider = GrokProvider(self.cfg)
        provider._speaker_detection_initialized = True
        provider._summarization_initialized = True

        provider.cleanup()

        self.assertFalse(provider._speaker_detection_initialized)
        self.assertFalse(provider._summarization_initialized)
        self.assertFalse(provider.is_initialized)


@pytest.mark.unit
class TestGrokProviderSpeakerDetection(unittest.TestCase):
    """Tests for GrokProvider speaker detection methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="grok",
            grok_api_key="test-api-key-123",
            auto_speakers=True,
        )

    @patch("podcast_scraper.providers.grok.grok_provider.OpenAI")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_detect_speakers_success(self, mock_render, mock_openai):
        """Test successful speaker detection."""
        # Mock OpenAI client
        mock_client = Mock()
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
        mock_openai.return_value = mock_client

        # Mock prompt rendering
        mock_render.side_effect = lambda name, **kwargs: "test prompt"

        provider = GrokProvider(self.cfg)
        provider.initialize()

        speakers, hosts, success = provider.detect_speakers(
            episode_title="Test Episode",
            episode_description="Test description",
            known_hosts={"John Doe"},
        )

        self.assertEqual(speakers, ["John Doe", "Jane Smith"])
        self.assertEqual(hosts, {"John Doe"})
        self.assertTrue(success)

    @patch("podcast_scraper.providers.grok.grok_provider.OpenAI")
    def test_detect_speakers_not_initialized(self, mock_openai):
        """Test detect_speakers raises RuntimeError if not initialized."""
        provider = GrokProvider(self.cfg)
        # Don't call initialize()

        with self.assertRaises(RuntimeError) as context:
            provider.detect_speakers("Test", None, set())

        self.assertIn("not initialized", str(context.exception))

    @patch("podcast_scraper.providers.grok.grok_provider.OpenAI")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_detect_speakers_empty_response(self, mock_render, mock_openai):
        """Test handling of empty API response."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = None
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        mock_render.side_effect = lambda name, **kwargs: "test prompt"

        provider = GrokProvider(self.cfg)
        provider.initialize()

        speakers, hosts, success = provider.detect_speakers("Test", None, set())

        self.assertEqual(speakers, speaker_detection.DEFAULT_SPEAKER_NAMES)
        self.assertEqual(hosts, set())
        self.assertFalse(success)

    @patch("podcast_scraper.providers.grok.grok_provider.OpenAI")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_detect_speakers_api_error(self, mock_render, mock_openai):
        """Test handling of API errors."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client

        mock_render.side_effect = lambda name, **kwargs: "test prompt"

        provider = GrokProvider(self.cfg)
        provider.initialize()

        from podcast_scraper.exceptions import ProviderRuntimeError

        with self.assertRaises(ProviderRuntimeError) as context:
            provider.detect_speakers("Test", None, set())

        self.assertIn("speaker detection failed", str(context.exception).lower())


@pytest.mark.unit
class TestGrokProviderSummarization(unittest.TestCase):
    """Tests for GrokProvider summarization methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="grok",
            grok_api_key="test-api-key-123",
            generate_summaries=True,
            generate_metadata=True,
        )

    @patch("podcast_scraper.providers.grok.grok_provider.OpenAI")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_summarize_success(self, mock_render, mock_openai):
        """Test successful summarization."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "This is a summary."
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 1000
        mock_response.usage.completion_tokens = 200
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        mock_render.side_effect = lambda name, **kwargs: "test prompt"

        provider = GrokProvider(self.cfg)
        provider.initialize()

        result = provider.summarize("Long transcript text here...")

        self.assertEqual(result["summary"], "This is a summary.")
        self.assertIsNone(result["summary_short"])
        self.assertEqual(result["metadata"]["provider"], "grok")

    @patch("podcast_scraper.providers.grok.grok_provider.OpenAI")
    def test_summarize_not_initialized(self, mock_openai):
        """Test summarize raises RuntimeError if not initialized."""
        provider = GrokProvider(self.cfg)
        # Don't call initialize()

        with self.assertRaises(RuntimeError) as context:
            provider.summarize("Text")

        self.assertIn("not initialized", str(context.exception))

    @patch("podcast_scraper.providers.grok.grok_provider.OpenAI")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_summarize_empty_response(self, mock_render, mock_openai):
        """Test handling of empty API response."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = None
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        mock_render.side_effect = lambda name, **kwargs: "test prompt"

        provider = GrokProvider(self.cfg)
        provider.initialize()

        result = provider.summarize("Text")

        self.assertEqual(result["summary"], "")

    @patch("podcast_scraper.providers.grok.grok_provider.OpenAI")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_summarize_api_error(self, mock_render, mock_openai):
        """Test handling of API errors."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client

        mock_render.side_effect = lambda name, **kwargs: "test prompt"

        provider = GrokProvider(self.cfg)
        provider.initialize()

        from podcast_scraper.exceptions import ProviderRuntimeError

        with self.assertRaises(ProviderRuntimeError) as context:
            provider.summarize("Text")

        self.assertIn("summarization failed", str(context.exception).lower())


@pytest.mark.unit
class TestGrokProviderPricing(unittest.TestCase):
    """Tests for GrokProvider.get_pricing() static method."""

    def test_get_pricing_grok_2_speaker_detection(self):
        """Test pricing lookup for Grok-2 speaker detection."""
        pricing = GrokProvider.get_pricing("grok-2", "speaker_detection")
        # Pricing should be set to 0.0 until verified (placeholder)
        self.assertIn("input_cost_per_1m_tokens", pricing)
        self.assertIn("output_cost_per_1m_tokens", pricing)

    def test_get_pricing_grok_beta_summarization(self):
        """Test pricing lookup for Grok-beta summarization."""
        pricing = GrokProvider.get_pricing("grok-beta", "summarization")
        # Pricing should be set to 0.0 until verified (placeholder)
        self.assertIn("input_cost_per_1m_tokens", pricing)
        self.assertIn("output_cost_per_1m_tokens", pricing)

    def test_get_pricing_unsupported_model(self):
        """Test pricing lookup for unsupported model returns default pricing."""
        pricing = GrokProvider.get_pricing("unknown-model", "summarization")
        # Should default to grok-2 pricing (conservative estimate)
        self.assertIn("input_cost_per_1m_tokens", pricing)
        self.assertIn("output_cost_per_1m_tokens", pricing)

    def test_get_pricing_case_insensitive_model_name(self):
        """Test that pricing lookup is case-insensitive."""
        pricing1 = GrokProvider.get_pricing("GROK-2", "summarization")
        pricing2 = GrokProvider.get_pricing("grok-2", "summarization")
        self.assertEqual(pricing1, pricing2)


@pytest.mark.unit
class TestGrokProviderEdgeCases(unittest.TestCase):
    """Tests for GrokProvider edge cases and error handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="grok",
            summary_provider="grok",
            grok_api_key="test-api-key-123",
            auto_speakers=True,
            generate_summaries=True,
            generate_metadata=True,
        )

    @patch("podcast_scraper.providers.grok.grok_provider.OpenAI")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_rate_limit_error(self, mock_render, mock_openai):
        """Test handling of rate limit errors."""

        # Since openai is mocked globally, we need to create a real exception class
        class MockRateLimitError(Exception):
            """Mock RateLimitError for testing."""

            pass

        mock_client = Mock()
        # Use a real exception class so it actually raises
        mock_client.chat.completions.create.side_effect = MockRateLimitError("Rate limit exceeded")
        mock_openai.return_value = mock_client

        mock_render.side_effect = lambda name, **kwargs: "test prompt"

        provider = GrokProvider(self.cfg)
        provider.initialize()

        from podcast_scraper.exceptions import ProviderRuntimeError

        with self.assertRaises(ProviderRuntimeError) as context:
            provider.detect_speakers("Test", None, set())

        self.assertIn("rate limit", str(context.exception).lower())

    @patch("podcast_scraper.providers.grok.grok_provider.OpenAI")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_authentication_error(self, mock_render, mock_openai):
        """Test handling of authentication errors."""

        # Since openai is mocked globally, we need to create a real exception class
        # that will be detected by the string-based error detection
        class MockAuthenticationError(Exception):
            """Mock AuthenticationError for testing."""

            pass

        mock_client = Mock()
        # Use a real exception class so it actually raises
        mock_client.chat.completions.create.side_effect = MockAuthenticationError("Invalid API key")
        mock_openai.return_value = mock_client

        mock_render.side_effect = lambda name, **kwargs: "test prompt"

        provider = GrokProvider(self.cfg)
        provider.initialize()

        from podcast_scraper.exceptions import ProviderAuthError

        with self.assertRaises(ProviderAuthError) as context:
            provider.detect_speakers("Test", None, set())

        self.assertIn("authentication", str(context.exception).lower())

    @patch("podcast_scraper.providers.grok.grok_provider.OpenAI")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_invalid_model_error(self, mock_render, mock_openai):
        """Test handling of invalid model errors."""

        # Since openai is mocked globally, we need to create a real exception class
        class MockBadRequestError(Exception):
            """Mock BadRequestError for testing."""

            pass

        mock_client = Mock()
        # Use a real exception class so it actually raises
        mock_client.chat.completions.create.side_effect = MockBadRequestError(
            "Invalid model: unknown-model"
        )
        mock_openai.return_value = mock_client

        mock_render.side_effect = lambda name, **kwargs: "test prompt"

        provider = GrokProvider(self.cfg)
        provider.initialize()

        from podcast_scraper.exceptions import ProviderRuntimeError

        with self.assertRaises(ProviderRuntimeError) as context:
            provider.detect_speakers("Test", None, set())

        self.assertIn("invalid model", str(context.exception).lower())

    @patch("podcast_scraper.providers.grok.grok_provider.OpenAI")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_json_parse_error_fallback(self, mock_render, mock_openai):
        """Test fallback when JSON parsing fails."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        # Return invalid JSON
        mock_response.choices[0].message.content = "Not valid JSON {"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        mock_render.side_effect = lambda name, **kwargs: "test prompt"

        provider = GrokProvider(self.cfg)
        provider.initialize()

        # Should fallback to text parsing
        speakers, hosts, success = provider.detect_speakers("Test", None, set())

        # Should return defaults or parsed text
        self.assertIsInstance(speakers, list)
        self.assertIsInstance(hosts, set)

    @patch("podcast_scraper.providers.grok.grok_provider.OpenAI")
    def test_detect_hosts_prefers_feed_authors(self, mock_openai):
        """Test that detect_hosts prefers RSS feed authors."""
        provider = GrokProvider(self.cfg)
        provider.initialize()

        # Should return feed_authors without API call
        hosts = provider.detect_hosts(
            feed_title="Test Feed",
            feed_description="Test",
            feed_authors=["John Doe", "Jane Smith"],
        )

        self.assertEqual(hosts, {"John Doe", "Jane Smith"})
        # Verify no API call was made
        mock_openai.return_value.chat.completions.create.assert_not_called()

    @patch("podcast_scraper.providers.grok.grok_provider.OpenAI")
    def test_auto_speakers_disabled_returns_defaults(self, mock_openai):
        """Test that disabled auto_speakers returns defaults without initialization."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="grok",
            grok_api_key="test-key",
            auto_speakers=False,  # Disabled
        )
        provider = GrokProvider(cfg)
        # Don't initialize

        speakers, hosts, success = provider.detect_speakers("Test", None, set())

        self.assertEqual(speakers, speaker_detection.DEFAULT_SPEAKER_NAMES)
        self.assertEqual(hosts, set())
        self.assertFalse(success)
        # Verify no API call was made
        mock_openai.return_value.chat.completions.create.assert_not_called()


@pytest.mark.unit
class TestGrokProviderErrorHandling(unittest.TestCase):
    """Tests for error handling in GrokProvider."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="grok",
            summary_provider="grok",
            grok_api_key="test-api-key-123",
            auto_speakers=True,
            generate_summaries=True,
            generate_metadata=True,
        )

    @patch("podcast_scraper.providers.grok.grok_provider.OpenAI")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_speaker_detection_auth_error(self, mock_render, mock_openai):
        """Test that authentication errors are properly handled in speaker detection."""

        # Use a real Exception subclass so side_effect actually raises (openai is mocked)
        class AuthError(Exception):
            pass

        mock_client = Mock()
        mock_openai.return_value = mock_client
        create_mock = Mock(side_effect=AuthError("Invalid API key: authentication failed"))
        mock_client.chat.completions.create = create_mock
        mock_render.side_effect = lambda name, **kwargs: "test prompt"

        provider = GrokProvider(self.cfg)
        provider.initialize()

        from podcast_scraper.exceptions import ProviderAuthError

        with self.assertRaises(ProviderAuthError) as context:
            provider.detect_speakers("Episode Title", "Description", set(["Host"]))

        self.assertIn("authentication failed", str(context.exception).lower())
        self.assertIn("GROK_API_KEY", str(context.exception))

    @patch("podcast_scraper.providers.grok.grok_provider.OpenAI")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_speaker_detection_rate_limit_error(self, mock_render, mock_openai):
        """Test that rate limit errors are properly handled in speaker detection."""

        # Use a real Exception subclass so side_effect actually raises (openai is mocked)
        class RateLimitError(Exception):
            pass

        mock_client = Mock()
        mock_openai.return_value = mock_client
        create_mock = Mock(side_effect=RateLimitError("Rate limit exceeded: quota exceeded"))
        mock_client.chat.completions.create = create_mock
        mock_render.side_effect = lambda name, **kwargs: "test prompt"

        provider = GrokProvider(self.cfg)
        provider.initialize()

        from podcast_scraper.exceptions import ProviderRuntimeError

        with self.assertRaises(ProviderRuntimeError) as context:
            provider.detect_speakers("Episode Title", "Description", set(["Host"]))

        self.assertIn("rate limit", str(context.exception).lower())

    @patch("podcast_scraper.providers.grok.grok_provider.OpenAI")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_speaker_detection_empty_response(self, mock_render, mock_openai):
        """Test that empty responses return default speakers."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_chat = Mock()
        mock_client.chat.completions.create = mock_chat

        # Mock response with empty content
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=""))]
        mock_chat.return_value = mock_response
        mock_render.side_effect = lambda name, **kwargs: "test prompt"

        provider = GrokProvider(self.cfg)
        provider.initialize()

        speakers, hosts, success = provider.detect_speakers(
            "Episode Title", "Description", set(["Host"])
        )

        self.assertFalse(success)
        self.assertEqual(speakers, speaker_detection.DEFAULT_SPEAKER_NAMES)

    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("podcast_scraper.providers.grok.grok_provider.OpenAI")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_summarization_auth_error(self, mock_render, mock_openai, mock_retry):
        """Test that authentication errors are properly handled in summarization."""
        mock_retry.side_effect = lambda func, **kwargs: func()

        # Use a real Exception subclass so side_effect actually raises (openai is mocked)
        class AuthError(Exception):
            pass

        mock_client = Mock()
        mock_openai.return_value = mock_client
        create_mock = Mock(side_effect=AuthError("Invalid API key: authentication failed"))
        mock_client.chat.completions.create = create_mock
        mock_render.side_effect = lambda name, **kwargs: "test prompt"

        provider = GrokProvider(self.cfg)
        provider.initialize()

        from podcast_scraper.exceptions import ProviderAuthError

        with self.assertRaises(ProviderAuthError) as context:
            provider.summarize("Text to summarize")

        self.assertIn("authentication failed", str(context.exception).lower())

    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("podcast_scraper.providers.grok.grok_provider.OpenAI")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_summarization_rate_limit_error(self, mock_render, mock_openai, mock_retry):
        """Test that rate limit errors are properly handled in summarization."""
        mock_retry.side_effect = lambda func, **kwargs: func()

        # Use a real Exception subclass so side_effect actually raises (openai is mocked)
        class RateLimitError(Exception):
            pass

        mock_client = Mock()
        mock_openai.return_value = mock_client
        create_mock = Mock(side_effect=RateLimitError("Rate limit exceeded: quota exceeded"))
        mock_client.chat.completions.create = create_mock
        mock_render.side_effect = lambda name, **kwargs: "test prompt"

        provider = GrokProvider(self.cfg)
        provider.initialize()

        from podcast_scraper.exceptions import ProviderRuntimeError

        with self.assertRaises(ProviderRuntimeError) as context:
            provider.summarize("Text to summarize")

        self.assertIn("rate limit", str(context.exception).lower())

    @patch("podcast_scraper.providers.grok.grok_provider.OpenAI")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_summarization_invalid_model_error(self, mock_render, mock_openai):
        """Test that invalid model errors are properly handled in summarization."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_chat = Mock()
        mock_client.chat.completions.create = mock_chat
        mock_chat.side_effect = ValueError("Invalid model: unknown-model")
        mock_render.side_effect = lambda name, **kwargs: "test prompt"

        provider = GrokProvider(self.cfg)
        provider.initialize()

        from podcast_scraper.exceptions import ProviderRuntimeError

        with self.assertRaises(ProviderRuntimeError) as context:
            provider.summarize("Text to summarize")

        error_msg = str(context.exception).lower()
        self.assertTrue("invalid model" in error_msg or "summarization failed" in error_msg)

    @patch("podcast_scraper.providers.grok.grok_provider.OpenAI")
    def test_detect_hosts_fallback_on_error(self, mock_openai):
        """Test that detect_hosts returns empty set on error."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        mock_chat = Mock()
        mock_client.chat.completions.create = mock_chat
        mock_chat.side_effect = Exception("API error")

        provider = GrokProvider(self.cfg)
        provider.initialize()

        # Should return empty set on error
        hosts = provider.detect_hosts("Feed Title", "Description", None)
        self.assertEqual(hosts, set())

    @patch("podcast_scraper.providers.grok.grok_provider.OpenAI")
    def test_cleaning_strategy_pattern(self, mock_openai):
        """Test that pattern cleaning strategy is selected correctly."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            grok_api_key="test-api-key-123",
            transcript_cleaning_strategy="pattern",
            speaker_detector_provider="grok",
        )

        mock_client = Mock()
        mock_openai.return_value = mock_client

        provider = GrokProvider(cfg)

        from podcast_scraper.cleaning import PatternBasedCleaner

        self.assertIsInstance(provider.cleaning_processor, PatternBasedCleaner)

    @patch("podcast_scraper.providers.grok.grok_provider.OpenAI")
    def test_cleaning_strategy_llm(self, mock_openai):
        """Test that LLM cleaning strategy is selected correctly."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            grok_api_key="test-api-key-123",
            transcript_cleaning_strategy="llm",
            speaker_detector_provider="grok",
        )

        mock_client = Mock()
        mock_openai.return_value = mock_client

        provider = GrokProvider(cfg)

        from podcast_scraper.cleaning import LLMBasedCleaner

        self.assertIsInstance(provider.cleaning_processor, LLMBasedCleaner)

    @patch("podcast_scraper.providers.grok.grok_provider.OpenAI")
    def test_cleaning_strategy_hybrid(self, mock_openai):
        """Test that hybrid cleaning strategy is selected correctly (default)."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            grok_api_key="test-api-key-123",
            transcript_cleaning_strategy="hybrid",
            speaker_detector_provider="grok",
        )

        mock_client = Mock()
        mock_openai.return_value = mock_client

        provider = GrokProvider(cfg)

        from podcast_scraper.cleaning import HybridCleaner

        self.assertIsInstance(provider.cleaning_processor, HybridCleaner)
