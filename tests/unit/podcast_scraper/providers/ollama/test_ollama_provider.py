#!/usr/bin/env python3
"""Standalone unit tests for unified Ollama provider.

These tests verify that OllamaProvider correctly implements two protocols
(SpeakerDetector, SummarizationProvider) using Ollama's local API via OpenAI SDK.

These are standalone provider tests - they test the provider itself,
not its integration with the app.

Note: Ollama does NOT support transcription (no audio API).
"""

import unittest
from unittest.mock import MagicMock, Mock, patch

import pytest

# Mock openai and httpx before importing modules that require them
# Unit tests run without openai/httpx packages installed
# Use patch.dict without 'with' to avoid context manager conflicts with @patch decorators
mock_openai = MagicMock()
mock_openai.OpenAI = Mock()
mock_httpx = MagicMock()
_patch_ollama = patch.dict(
    "sys.modules",
    {
        "openai": mock_openai,
        "httpx": mock_httpx,
    },
)
_patch_ollama.start()

from podcast_scraper import config
from podcast_scraper.providers.ml import speaker_detection
from podcast_scraper.providers.ollama.ollama_provider import OllamaProvider


@pytest.mark.unit
class TestOllamaProviderStandalone(unittest.TestCase):
    """Standalone tests for OllamaProvider - testing the provider itself."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="ollama",
            summary_provider="ollama",
            ollama_api_base="http://localhost:11434/v1",
            auto_speakers=False,  # Disable to avoid API calls
            generate_summaries=False,  # Disable to avoid API calls
        )

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    def test_provider_creation(self, mock_openai_class, mock_httpx):
        """Test that OllamaProvider can be created."""
        # Mock health check
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_httpx.get.return_value = mock_response

        provider = OllamaProvider(self.cfg)
        self.assertIsNotNone(provider)
        self.assertEqual(provider.__class__.__name__, "OllamaProvider")

        # Verify OpenAI client was created with Ollama base_url
        mock_openai_class.assert_called_once()
        call_kwargs = mock_openai_class.call_args[1]
        self.assertEqual(call_kwargs["base_url"], "http://localhost:11434/v1")
        self.assertEqual(call_kwargs["api_key"], "ollama")  # Dummy key

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    def test_provider_requires_httpx(self, mock_openai_class, mock_httpx_module):
        """Test that OllamaProvider requires httpx package."""
        # Simulate httpx not being available
        import sys

        original_httpx = sys.modules.get("podcast_scraper.providers.ollama.ollama_provider.httpx")
        sys.modules["podcast_scraper.providers.ollama.ollama_provider.httpx"] = None

        try:
            # Reload module to trigger ImportError
            from podcast_scraper.providers.ollama import ollama_provider

            # Temporarily set httpx to None
            ollama_provider.httpx = None

            with self.assertRaises(ImportError) as context:
                OllamaProvider(self.cfg)
            self.assertIn("httpx package required", str(context.exception))
        finally:
            # Restore
            if original_httpx:
                sys.modules["podcast_scraper.providers.ollama.ollama_provider.httpx"] = (
                    original_httpx
                )

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    def test_provider_requires_openai(self, mock_openai_class, mock_httpx):
        """Test that OllamaProvider requires openai package."""
        # Mock health check
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_httpx.get.return_value = mock_response

        # Simulate OpenAI not being available
        import sys

        original_openai = sys.modules.get("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
        sys.modules["podcast_scraper.providers.ollama.ollama_provider.OpenAI"] = None

        try:
            from podcast_scraper.providers.ollama import ollama_provider

            # Temporarily set OpenAI to None
            ollama_provider.OpenAI = None

            with self.assertRaises(ImportError) as context:
                OllamaProvider(self.cfg)
            self.assertIn("openai package required", str(context.exception))
        finally:
            # Restore
            if original_openai:
                sys.modules["podcast_scraper.providers.ollama.ollama_provider.OpenAI"] = (
                    original_openai
                )

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    def test_provider_validates_ollama_running(self, mock_openai_class, mock_httpx):
        """Test that provider validates Ollama server is running."""

        # Provider uses string-based fallback when httpx is mocked (exc_module/exc_type_name).
        # Use an exception that matches that fallback so ConnectionError is raised.
        class FakeConnectError(Exception):
            __module__ = "httpx"
            __name__ = "ConnectError"

        mock_httpx.get.side_effect = FakeConnectError("Connection refused")

        with self.assertRaises(ConnectionError) as context:
            OllamaProvider(self.cfg)

        self.assertIn("Ollama server is not running", str(context.exception))

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    def test_provider_implements_protocols(self, mock_openai_class, mock_httpx):
        """Test that OllamaProvider implements required protocols."""
        # Mock health check
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_httpx.get.return_value = mock_response

        provider = OllamaProvider(self.cfg)

        # SpeakerDetector protocol
        self.assertTrue(hasattr(provider, "detect_speakers"))
        self.assertTrue(hasattr(provider, "detect_hosts"))
        self.assertTrue(hasattr(provider, "analyze_patterns"))
        self.assertTrue(hasattr(provider, "initialize"))
        self.assertTrue(hasattr(provider, "cleanup"))
        self.assertTrue(hasattr(provider, "clear_cache"))

        # SummarizationProvider protocol
        self.assertTrue(hasattr(provider, "summarize"))
        self.assertTrue(hasattr(provider, "initialize"))
        self.assertTrue(hasattr(provider, "cleanup"))

        # Note: Transcription is NOT supported
        self.assertFalse(hasattr(provider, "transcribe"))

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    def test_provider_initialization_state(self, mock_openai_class, mock_httpx):
        """Test that provider tracks initialization state for each capability."""
        # Mock health check
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_httpx.get.return_value = mock_response

        provider = OllamaProvider(self.cfg)

        # Initially not initialized
        self.assertFalse(provider._speaker_detection_initialized)
        self.assertFalse(provider._summarization_initialized)
        self.assertFalse(provider.is_initialized)

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    def test_provider_thread_safe(self, mock_openai_class, mock_httpx):
        """Test that provider marks itself as thread-safe."""
        # Mock health check
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_httpx.get.return_value = mock_response

        provider = OllamaProvider(self.cfg)
        self.assertFalse(provider._requires_separate_instances)

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    def test_speaker_detection_initialization(self, mock_openai_class, mock_httpx):
        """Test that speaker detection can be initialized independently."""
        # Mock health check and model validation
        mock_health_response = Mock()
        mock_health_response.raise_for_status = Mock()
        mock_httpx.get.return_value = mock_health_response

        # Mock model list response
        mock_models_response = Mock()
        mock_models_response.raise_for_status = Mock()
        mock_models_response.json.return_value = {
            "models": [
                {"name": "llama3.1:8b"},  # Default summary model
                {"name": "llama3.3:latest"},
                {"name": "llama3.2:latest"},
            ]
        }
        mock_httpx.get.side_effect = [mock_health_response, mock_models_response]

        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            ollama_api_base=self.cfg.ollama_api_base,
            speaker_detector_provider="ollama",
            auto_speakers=True,
            generate_summaries=False,  # Disable to avoid initializing summarization
        )
        provider = OllamaProvider(cfg)

        provider.initialize()

        self.assertTrue(provider._speaker_detection_initialized)
        # Other capability should not be initialized
        self.assertFalse(provider._summarization_initialized)

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    def test_summarization_initialization(self, mock_openai_class, mock_httpx):
        """Test that summarization can be initialized independently."""
        # Mock health check and model validation
        mock_health_response = Mock()
        mock_health_response.raise_for_status = Mock()
        mock_httpx.get.return_value = mock_health_response

        # Mock model list response
        mock_models_response = Mock()
        mock_models_response.raise_for_status = Mock()
        mock_models_response.json.return_value = {
            "models": [
                {"name": "llama3.1:8b"},  # Default summary model
                {"name": "llama3.3:latest"},
                {"name": "llama3.2:latest"},
            ]
        }
        mock_httpx.get.side_effect = [mock_health_response, mock_models_response]

        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            ollama_api_base=self.cfg.ollama_api_base,
            summary_provider="ollama",
            generate_summaries=True,
            generate_metadata=True,  # Required when generate_summaries is True
            auto_speakers=False,  # Disable to avoid initializing speaker detection
        )
        provider = OllamaProvider(cfg)

        provider.initialize()

        self.assertTrue(provider._summarization_initialized)
        # Other capability should not be initialized
        self.assertFalse(provider._speaker_detection_initialized)

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    def test_cleanup_releases_all_resources(self, mock_openai_class, mock_httpx):
        """Test that cleanup releases all resources."""
        # Mock health check
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_httpx.get.return_value = mock_response

        provider = OllamaProvider(self.cfg)
        provider._speaker_detection_initialized = True
        provider._summarization_initialized = True

        provider.cleanup()

        self.assertFalse(provider._speaker_detection_initialized)
        self.assertFalse(provider._summarization_initialized)
        self.assertFalse(provider.is_initialized)

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    def test_get_pricing_returns_zero(self, mock_openai_class, mock_httpx):
        """Test that get_pricing returns zero cost (Ollama is free)."""
        # Mock health check
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_httpx.get.return_value = mock_response

        pricing = OllamaProvider.get_pricing("llama3.3:latest", "speaker_detection")
        self.assertEqual(pricing["input_cost_per_1m_tokens"], 0.0)
        self.assertEqual(pricing["output_cost_per_1m_tokens"], 0.0)

        pricing = OllamaProvider.get_pricing("llama3.2:latest", "summarization")
        self.assertEqual(pricing["input_cost_per_1m_tokens"], 0.0)
        self.assertEqual(pricing["output_cost_per_1m_tokens"], 0.0)


@pytest.mark.unit
class TestOllamaProviderSpeakerDetection(unittest.TestCase):
    """Tests for OllamaProvider speaker detection methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="ollama",
            ollama_api_base="http://localhost:11434/v1",
            auto_speakers=True,
        )

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    def test_detect_hosts_from_feed_authors(self, mock_openai_class, mock_httpx):
        """Test detect_hosts prefers feed_authors."""
        # Mock health check and model validation
        mock_health_response = Mock()
        mock_health_response.raise_for_status = Mock()
        mock_models_response = Mock()
        mock_models_response.raise_for_status = Mock()
        # Include both models in case both are validated
        mock_models_response.json.return_value = {
            "models": [
                {"name": "llama3.1:8b"},  # Default summary model
                {"name": "llama3.3:latest"},
                {"name": "llama3.2:latest"},
            ]
        }
        mock_httpx.get.side_effect = [mock_health_response, mock_models_response]

        # Create config with auto_speakers=True to enable speaker detection
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            ollama_api_base=self.cfg.ollama_api_base,
            speaker_detector_provider="ollama",
            auto_speakers=True,  # Enable to test speaker detection
            generate_summaries=False,  # Disable to avoid initializing summarization
        )
        provider = OllamaProvider(cfg)
        provider.initialize()

        hosts = provider.detect_hosts(
            feed_title="The Podcast",
            feed_description="A great podcast",
            feed_authors=["Alice", "Bob"],
        )

        self.assertEqual(hosts, {"Alice", "Bob"})

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    def test_detect_speakers_not_initialized(self, mock_openai_class, mock_httpx):
        """Test detect_speakers raises RuntimeError if not initialized."""
        # Mock health check
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_httpx.get.return_value = mock_response

        provider = OllamaProvider(self.cfg)
        # Don't call initialize()

        with self.assertRaises(RuntimeError) as context:
            provider.detect_speakers("Title", "Description", set())

        self.assertIn("not initialized", str(context.exception))

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_detect_speakers_success(self, mock_render_prompt, mock_openai_class, mock_httpx):
        """Test successful speaker detection."""
        # Mock health check and model validation
        mock_health_response = Mock()
        mock_health_response.raise_for_status = Mock()
        mock_models_response = Mock()
        mock_models_response.raise_for_status = Mock()
        # Include both models in case both are validated
        mock_models_response.json.return_value = {
            "models": [
                {"name": "llama3.1:8b"},  # Default summary model
                {"name": "llama3.3:latest"},
                {"name": "llama3.2:latest"},
            ]
        }
        mock_httpx.get.side_effect = [mock_health_response, mock_models_response]

        # Mock prompts
        mock_render_prompt.side_effect = ["System Prompt", "User Prompt"]

        # Mock OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = (
            '{"speakers": ["Alice", "Bob"], "hosts": ["Alice"], "guests": ["Bob"]}'
        )
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        # Create config with auto_speakers=True to enable speaker detection
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            ollama_api_base=self.cfg.ollama_api_base,
            speaker_detector_provider="ollama",
            auto_speakers=True,  # Enable to test speaker detection
            generate_summaries=False,  # Disable to avoid initializing summarization
        )
        provider = OllamaProvider(cfg)
        provider.initialize()

        speakers, hosts, success = provider.detect_speakers(
            episode_title="Alice interviews Bob",
            episode_description="A great conversation",
            known_hosts={"Alice"},
        )

        self.assertEqual(speakers, ["Alice", "Bob"])
        self.assertEqual(hosts, {"Alice"})
        self.assertTrue(success)

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    def test_analyze_patterns_returns_none(self, mock_openai_class, mock_httpx):
        """Test analyze_patterns returns None (not implemented)."""
        # Mock health check and model validation
        mock_health_response = Mock()
        mock_health_response.raise_for_status = Mock()
        mock_models_response = Mock()
        mock_models_response.raise_for_status = Mock()
        mock_models_response.json.return_value = {
            "models": [
                {"name": "llama3.1:8b"},  # Default summary model
                {"name": "llama3.3:latest"},
                {"name": "llama3.2:latest"},
            ]
        }
        mock_httpx.get.side_effect = [mock_health_response, mock_models_response]

        from podcast_scraper import models

        # Create config with auto_speakers=True to enable speaker detection
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            ollama_api_base=self.cfg.ollama_api_base,
            speaker_detector_provider="ollama",
            auto_speakers=True,  # Enable to test speaker detection
            generate_summaries=False,  # Disable to avoid initializing summarization
        )
        provider = OllamaProvider(cfg)
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

        # Ollama provider doesn't implement pattern analysis, returns None
        result = provider.analyze_patterns(episodes=episodes, known_hosts={"Alice"})

        self.assertIsNone(result)


@pytest.mark.unit
class TestOllamaProviderSummarization(unittest.TestCase):
    """Tests for OllamaProvider summarization methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="ollama",
            ollama_api_base="http://localhost:11434/v1",
            generate_summaries=True,
            generate_metadata=True,  # Required when generate_summaries=True
        )

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    def test_summarize_not_initialized(self, mock_openai_class, mock_httpx):
        """Test summarize raises RuntimeError if not initialized."""
        # Mock health check
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_httpx.get.return_value = mock_response

        provider = OllamaProvider(self.cfg)
        # Don't call initialize()

        with self.assertRaises(RuntimeError) as context:
            provider.summarize("Text")

        self.assertIn("not initialized", str(context.exception))

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    @patch("podcast_scraper.prompts.store.render_prompt")
    @patch("podcast_scraper.prompts.store.get_prompt_metadata")
    def test_summarize_success(
        self, mock_get_metadata, mock_render_prompt, mock_openai_class, mock_httpx
    ):
        """Test successful summarization."""
        # Mock health check and model validation
        mock_health_response = Mock()
        mock_health_response.raise_for_status = Mock()
        mock_models_response = Mock()
        mock_models_response.raise_for_status = Mock()
        # Include both models in case both are validated
        mock_models_response.json.return_value = {
            "models": [
                {"name": "llama3.1:8b"},  # Default summary model
                {"name": "llama3.3:latest"},
                {"name": "llama3.2:latest"},
            ]
        }
        mock_httpx.get.side_effect = [mock_health_response, mock_models_response]

        # Mock prompts
        mock_render_prompt.side_effect = ["System Prompt", "User Prompt"]
        mock_get_metadata.return_value = {"name": "ollama/summarization/system_v1"}

        # Mock OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "This is a summary."
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 1000
        mock_response.usage.completion_tokens = 200
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        # Create config with generate_summaries=True to enable summarization
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            ollama_api_base=self.cfg.ollama_api_base,
            summary_provider="ollama",
            generate_summaries=True,  # Enable to test summarization
            generate_metadata=True,  # Required when generate_summaries=True
            auto_speakers=False,  # Disable to avoid initializing speaker detection
        )
        provider = OllamaProvider(cfg)
        provider.initialize()

        result = provider.summarize("This is a long transcript text.")

        self.assertEqual(result["summary"], "This is a summary.")
        self.assertIn("metadata", result)
        self.assertEqual(result["metadata"]["model"], provider.summary_model)
        self.assertEqual(result["metadata"]["provider"], "ollama")

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_summarize_api_error(self, mock_render_prompt, mock_openai_class, mock_httpx):
        """Test summarization error handling."""
        # Mock health check and model validation
        mock_health_response = Mock()
        mock_health_response.raise_for_status = Mock()
        mock_models_response = Mock()
        mock_models_response.raise_for_status = Mock()
        # Include both models in case both are validated
        mock_models_response.json.return_value = {
            "models": [
                {"name": "llama3.1:8b"},  # Default summary model
                {"name": "llama3.3:latest"},
                {"name": "llama3.2:latest"},
            ]
        }
        mock_httpx.get.side_effect = [mock_health_response, mock_models_response]

        # Mock prompts
        mock_render_prompt.side_effect = ["System Prompt", "User Prompt"]

        # Mock OpenAI client with error
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        mock_openai_class.return_value = mock_client

        # Create config with generate_summaries=True to enable summarization
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            ollama_api_base=self.cfg.ollama_api_base,
            summary_provider="ollama",
            generate_summaries=True,  # Enable to test summarization
            generate_metadata=True,  # Required when generate_summaries=True
            auto_speakers=False,  # Disable to avoid initializing speaker detection
        )
        provider = OllamaProvider(cfg)
        provider.initialize()

        from podcast_scraper.exceptions import ProviderRuntimeError

        with self.assertRaises(ProviderRuntimeError) as context:
            provider.summarize("Text")

        self.assertIn("summarization failed", str(context.exception).lower())

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    def test_model_validation_warning(self, mock_openai_class, mock_httpx):
        """Test that model validation failure logs warning but doesn't fail."""
        # Mock health check
        mock_health_response = Mock()
        mock_health_response.raise_for_status = Mock()
        # Mock model validation failure (connection error)
        import httpx

        mock_httpx.get.side_effect = [mock_health_response, httpx.RequestError("Connection error")]

        # Should not raise - validation failure is logged as warning
        provider = OllamaProvider(self.cfg)
        self.assertIsNotNone(provider)

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_detect_speakers_invalid_json(self, mock_render_prompt, mock_openai_class, mock_httpx):
        """Test speaker detection handles invalid JSON response."""
        # Mock health check and model validation
        mock_health_response = Mock()
        mock_health_response.raise_for_status = Mock()
        mock_models_response = Mock()
        mock_models_response.raise_for_status = Mock()
        # Include both models in case both are validated
        mock_models_response.json.return_value = {
            "models": [
                {"name": "llama3.1:8b"},  # Default summary model
                {"name": "llama3.3:latest"},
                {"name": "llama3.2:latest"},
            ]
        }
        mock_httpx.get.side_effect = [mock_health_response, mock_models_response]

        # Mock prompts
        mock_render_prompt.side_effect = ["System Prompt", "User Prompt"]

        # Mock OpenAI client with invalid JSON
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Invalid JSON response"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        # Create config with auto_speakers=True to enable speaker detection
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            ollama_api_base=self.cfg.ollama_api_base,
            speaker_detector_provider="ollama",
            auto_speakers=True,  # Enable to test speaker detection
            generate_summaries=False,  # Disable to avoid initializing summarization
        )
        provider = OllamaProvider(cfg)
        provider.initialize()

        # Should return defaults on invalid JSON
        speakers, hosts, success = provider.detect_speakers(
            episode_title="Test Episode",
            episode_description="Test description",
            known_hosts=set(),
        )

        # Should return defaults (fallback parsing)
        self.assertIsInstance(speakers, list)
        self.assertIsInstance(hosts, set)
        # Success may be False if parsing fails completely

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    def test_timeout_configuration(self, mock_openai_class, mock_httpx):
        """Test that timeout is properly configured."""
        # Mock health check
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_httpx.get.return_value = mock_response

        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="ollama",
            ollama_api_base="http://localhost:11434/v1",
            ollama_timeout=300,  # 5 minutes
        )

        OllamaProvider(cfg)

        # Verify timeout was passed to OpenAI client
        call_kwargs = mock_openai_class.call_args[1]
        self.assertIn("timeout", call_kwargs)
        timeout = call_kwargs["timeout"]
        # Timeout is now an httpx.Timeout object
        # Check if it has a 'read' attribute (httpx.Timeout object)
        if hasattr(timeout, "read"):
            self.assertEqual(timeout.read, 300.0)
        else:
            # Fallback if simple timeout value
            self.assertEqual(timeout, 300)

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    def test_default_timeout(self, mock_openai_class, mock_httpx):
        """Test that default timeout is used when not specified."""
        # Mock health check
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_httpx.get.return_value = mock_response

        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="ollama",
            ollama_api_base="http://localhost:11434/v1",
            # ollama_timeout not specified - should use default
        )

        OllamaProvider(cfg)

        # Verify default timeout (120 seconds) was used
        call_kwargs = mock_openai_class.call_args[1]
        self.assertIn("timeout", call_kwargs)
        timeout = call_kwargs["timeout"]
        # Timeout is now an httpx.Timeout object
        # Check if it has a 'read' attribute (httpx.Timeout object)
        if hasattr(timeout, "read"):
            self.assertEqual(timeout.read, 120.0)
        else:
            # Fallback if simple timeout value
            self.assertEqual(timeout, 120)


@pytest.mark.unit
class TestOllamaProviderErrorHandling(unittest.TestCase):
    """Tests for error handling in OllamaProvider."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="ollama",
            summary_provider="ollama",
            auto_speakers=True,
            generate_summaries=True,
            generate_metadata=True,
        )

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_speaker_detection_auth_error(self, mock_render, mock_openai_class, mock_httpx):
        """Test that authentication errors are properly handled in speaker detection."""

        # Use a real Exception subclass so side_effect actually raises (openai is mocked)
        class AuthError(Exception):
            pass

        # Mock health check and model validation
        def mock_get_side_effect(url, **kwargs):
            mock_resp = Mock()
            if "/api/tags" in url:
                # Model validation check
                mock_resp.status_code = 200
                mock_resp.json.return_value = {"models": [{"name": "llama3.1:8b"}]}
            else:
                # Version check
                mock_resp.status_code = 200
                mock_resp.json.return_value = {"version": "1.0.0"}
            return mock_resp

        mock_httpx.get.side_effect = mock_get_side_effect

        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        create_mock = Mock(side_effect=AuthError("Invalid API key: authentication failed"))
        mock_client.chat.completions.create = create_mock
        mock_render.side_effect = lambda name, **kwargs: "test prompt"

        provider = OllamaProvider(self.cfg)
        provider.initialize()

        from podcast_scraper.exceptions import ProviderRuntimeError

        with self.assertRaises(ProviderRuntimeError) as context:
            provider.detect_speakers("Episode Title", "Description", set(["Host"]))

        self.assertIn("authentication", str(context.exception).lower())

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_speaker_detection_rate_limit_error(self, mock_render, mock_openai_class, mock_httpx):
        """Test that rate limit errors are properly handled in speaker detection."""

        # Use a real Exception subclass so side_effect actually raises (openai is mocked)
        class RateLimitError(Exception):
            pass

        # Mock health check and model validation
        def mock_get_side_effect(url, **kwargs):
            mock_resp = Mock()
            if "/api/tags" in url:
                # Model validation check
                mock_resp.status_code = 200
                mock_resp.json.return_value = {"models": [{"name": "llama3.1:8b"}]}
            else:
                # Version check
                mock_resp.status_code = 200
                mock_resp.json.return_value = {"version": "1.0.0"}
            return mock_resp

        mock_httpx.get.side_effect = mock_get_side_effect

        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        create_mock = Mock(side_effect=RateLimitError("Rate limit exceeded: quota exceeded"))
        mock_client.chat.completions.create = create_mock
        mock_render.side_effect = lambda name, **kwargs: "test prompt"

        provider = OllamaProvider(self.cfg)
        provider.initialize()

        from podcast_scraper.exceptions import ProviderRuntimeError

        with self.assertRaises(ProviderRuntimeError) as context:
            provider.detect_speakers("Episode Title", "Description", set(["Host"]))

        self.assertIn("rate limit", str(context.exception).lower())

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_speaker_detection_invalid_model_error(
        self, mock_render, mock_openai_class, mock_httpx
    ):
        """Test that invalid model errors are properly handled in speaker detection."""

        # Mock health check and model validation
        def mock_get_side_effect(url, **kwargs):
            mock_resp = Mock()
            if "/api/tags" in url:
                # Model validation check
                mock_resp.status_code = 200
                mock_resp.json.return_value = {"models": [{"name": "llama3.1:8b"}]}
            else:
                # Version check
                mock_resp.status_code = 200
                mock_resp.json.return_value = {"version": "1.0.0"}
            return mock_resp

        mock_httpx.get.side_effect = mock_get_side_effect

        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_chat = Mock()
        mock_client.chat.completions.create = mock_chat
        mock_chat.side_effect = ValueError("Invalid model name")
        mock_render.side_effect = lambda name, **kwargs: "test prompt"

        provider = OllamaProvider(self.cfg)
        provider.initialize()

        from podcast_scraper.exceptions import ProviderRuntimeError

        with self.assertRaises(ProviderRuntimeError) as context:
            provider.detect_speakers("Episode Title", "Description", set(["Host"]))

        error_msg = str(context.exception).lower()
        self.assertTrue("invalid model" in error_msg or "speaker detection failed" in error_msg)

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_speaker_detection_json_decode_error(self, mock_render, mock_openai_class, mock_httpx):
        """Test that JSON decode errors return default speakers."""

        # Mock health check and model validation
        def mock_get_side_effect(url, **kwargs):
            mock_resp = Mock()
            if "/api/tags" in url:
                # Model validation check
                mock_resp.status_code = 200
                mock_resp.json.return_value = {"models": [{"name": "llama3.1:8b"}]}
            else:
                # Version check
                mock_resp.status_code = 200
                mock_resp.json.return_value = {"version": "1.0.0"}
            return mock_resp

        mock_httpx.get.side_effect = mock_get_side_effect

        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        # Mock response with invalid JSON (must start with "{" to return default speakers)
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="{ invalid"))]
        create_mock = Mock(return_value=mock_response)
        mock_client.chat.completions.create = create_mock
        mock_render.side_effect = lambda name, **kwargs: "test prompt"

        provider = OllamaProvider(self.cfg)
        provider.initialize()

        # Should return default speakers on JSON decode error
        speakers, hosts, success = provider.detect_speakers(
            "Episode Title", "Description", set(["Host"])
        )

        self.assertFalse(success)
        self.assertEqual(speakers, speaker_detection.DEFAULT_SPEAKER_NAMES)

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_speaker_detection_empty_response(self, mock_render, mock_openai_class, mock_httpx):
        """Test that empty responses return default speakers."""

        # Mock health check and model validation
        def mock_get_side_effect(url, **kwargs):
            mock_resp = Mock()
            if "/api/tags" in url:
                # Model validation check
                mock_resp.status_code = 200
                mock_resp.json.return_value = {"models": [{"name": "llama3.1:8b"}]}
            else:
                # Version check
                mock_resp.status_code = 200
                mock_resp.json.return_value = {"version": "1.0.0"}
            return mock_resp

        mock_httpx.get.side_effect = mock_get_side_effect

        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_chat = Mock()
        mock_client.chat.completions.create = mock_chat

        # Mock response with empty content
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=""))]
        mock_chat.return_value = mock_response
        mock_render.side_effect = lambda name, **kwargs: "test prompt"

        provider = OllamaProvider(self.cfg)
        provider.initialize()

        speakers, hosts, success = provider.detect_speakers(
            "Episode Title", "Description", set(["Host"])
        )

        self.assertFalse(success)
        self.assertEqual(speakers, speaker_detection.DEFAULT_SPEAKER_NAMES)

    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_summarization_auth_error(self, mock_render, mock_openai_class, mock_httpx, mock_retry):
        """Test that authentication errors are properly handled in summarization."""
        mock_retry.side_effect = lambda func, **kwargs: func()

        # Use a real Exception subclass so side_effect actually raises (openai is mocked)
        class AuthError(Exception):
            pass

        # Mock health check and model validation
        def mock_get_side_effect(url, **kwargs):
            mock_resp = Mock()
            if "/api/tags" in url:
                # Model validation check
                mock_resp.status_code = 200
                mock_resp.json.return_value = {"models": [{"name": "llama3.1:8b"}]}
            else:
                # Version check
                mock_resp.status_code = 200
                mock_resp.json.return_value = {"version": "1.0.0"}
            return mock_resp

        mock_httpx.get.side_effect = mock_get_side_effect

        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        create_mock = Mock(side_effect=AuthError("Invalid API key: authentication failed"))
        mock_client.chat.completions.create = create_mock
        mock_render.side_effect = lambda name, **kwargs: "test prompt"

        provider = OllamaProvider(self.cfg)
        provider.initialize()

        from podcast_scraper.exceptions import ProviderRuntimeError

        with self.assertRaises(ProviderRuntimeError) as context:
            provider.summarize("Text to summarize")

        self.assertIn("authentication", str(context.exception).lower())

    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_summarization_rate_limit_error(
        self, mock_render, mock_openai_class, mock_httpx, mock_retry
    ):
        """Test that rate limit errors are properly handled in summarization."""
        mock_retry.side_effect = lambda func, **kwargs: func()

        # Use a real Exception subclass so side_effect actually raises (openai is mocked)
        class RateLimitError(Exception):
            pass

        # Mock health check and model validation
        def mock_get_side_effect(url, **kwargs):
            mock_resp = Mock()
            if "/api/tags" in url:
                # Model validation check
                mock_resp.status_code = 200
                mock_resp.json.return_value = {"models": [{"name": "llama3.1:8b"}]}
            else:
                # Version check
                mock_resp.status_code = 200
                mock_resp.json.return_value = {"version": "1.0.0"}
            return mock_resp

        mock_httpx.get.side_effect = mock_get_side_effect

        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        create_mock = Mock(side_effect=RateLimitError("Rate limit exceeded: quota exceeded"))
        mock_client.chat.completions.create = create_mock
        mock_render.side_effect = lambda name, **kwargs: "test prompt"

        provider = OllamaProvider(self.cfg)
        provider.initialize()

        from podcast_scraper.exceptions import ProviderRuntimeError

        with self.assertRaises(ProviderRuntimeError) as context:
            provider.summarize("Text to summarize")

        self.assertIn("rate limit", str(context.exception).lower())

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_summarization_invalid_model_error(self, mock_render, mock_openai_class, mock_httpx):
        """Test that invalid model errors are properly handled in summarization."""

        # Mock health check and model validation
        def mock_get_side_effect(url, **kwargs):
            mock_resp = Mock()
            if "/api/tags" in url:
                # Model validation check
                mock_resp.status_code = 200
                mock_resp.json.return_value = {"models": [{"name": "llama3.1:8b"}]}
            else:
                # Version check
                mock_resp.status_code = 200
                mock_resp.json.return_value = {"version": "1.0.0"}
            return mock_resp

        mock_httpx.get.side_effect = mock_get_side_effect

        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_chat = Mock()
        mock_client.chat = Mock()
        mock_client.chat.completions = Mock()
        mock_client.chat.completions.create = mock_chat
        mock_chat.side_effect = ValueError("Invalid model: unknown-model")
        mock_render.side_effect = lambda name, **kwargs: "test prompt"

        provider = OllamaProvider(self.cfg)
        provider.initialize()

        from podcast_scraper.exceptions import ProviderRuntimeError

        with self.assertRaises(ProviderRuntimeError) as context:
            provider.summarize("Text to summarize")

        error_msg = str(context.exception).lower()
        self.assertTrue("invalid model" in error_msg or "summarization failed" in error_msg)

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    def test_detect_hosts_fallback_on_error(self, mock_openai_class, mock_httpx):
        """Test that detect_hosts returns empty set on error."""

        # Mock health check and model validation
        def mock_get_side_effect(url, **kwargs):
            mock_resp = Mock()
            if "/api/tags" in url:
                # Model validation check
                mock_resp.status_code = 200
                mock_resp.json.return_value = {"models": [{"name": "llama3.1:8b"}]}
            else:
                # Version check
                mock_resp.status_code = 200
                mock_resp.json.return_value = {"version": "1.0.0"}
            return mock_resp

        mock_httpx.get.side_effect = mock_get_side_effect

        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_chat = Mock()
        mock_client.chat.completions.create = mock_chat
        mock_chat.side_effect = Exception("API error")

        provider = OllamaProvider(self.cfg)
        provider.initialize()

        # Should return empty set on error
        hosts = provider.detect_hosts("Feed Title", "Description", None)
        self.assertEqual(hosts, set())

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    def test_cleaning_strategy_pattern(self, mock_openai_class, mock_httpx):
        """Test that pattern cleaning strategy is selected correctly."""
        # Mock health check
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"version": "1.0.0"}
        mock_httpx.get.return_value = mock_response

        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcript_cleaning_strategy="pattern",
            speaker_detector_provider="ollama",
        )

        provider = OllamaProvider(cfg)

        from podcast_scraper.cleaning import PatternBasedCleaner

        self.assertIsInstance(provider.cleaning_processor, PatternBasedCleaner)

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    def test_cleaning_strategy_llm(self, mock_openai_class, mock_httpx):
        """Test that LLM cleaning strategy is selected correctly."""
        # Mock health check
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"version": "1.0.0"}
        mock_httpx.get.return_value = mock_response

        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcript_cleaning_strategy="llm",
            speaker_detector_provider="ollama",
        )

        provider = OllamaProvider(cfg)

        from podcast_scraper.cleaning import LLMBasedCleaner

        self.assertIsInstance(provider.cleaning_processor, LLMBasedCleaner)

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    def test_cleaning_strategy_hybrid(self, mock_openai_class, mock_httpx):
        """Test that hybrid cleaning strategy is selected correctly (default)."""
        # Mock health check
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"version": "1.0.0"}
        mock_httpx.get.return_value = mock_response

        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcript_cleaning_strategy="hybrid",
            speaker_detector_provider="ollama",
        )

        provider = OllamaProvider(cfg)

        from podcast_scraper.cleaning import HybridCleaner

        self.assertIsInstance(provider.cleaning_processor, HybridCleaner)


@pytest.mark.unit
class TestOllamaProviderPatchCoverage(unittest.TestCase):
    """Tests to improve patch coverage (Codecov) for ollama_provider.py."""

    def setUp(self):
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="ollama",
            summary_provider="ollama",
            ollama_api_base="http://localhost:11434/v1",
            auto_speakers=False,
            generate_summaries=False,
        )

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    def test_get_capabilities(self, mock_openai_class, mock_httpx):
        """Test get_capabilities returns correct capabilities."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_httpx.get.return_value = mock_response

        provider = OllamaProvider(self.cfg)
        caps = provider.get_capabilities()

        self.assertFalse(caps.supports_transcription)
        self.assertTrue(caps.supports_speaker_detection)
        self.assertTrue(caps.supports_summarization)
        self.assertTrue(caps.supports_semantic_cleaning)
        self.assertEqual(caps.provider_name, "ollama")

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    def test_normalize_model_name_empty(self, mock_openai_class, mock_httpx):
        """Test _normalize_model_name returns empty string unchanged."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_httpx.get.return_value = mock_response

        provider = OllamaProvider(self.cfg)
        self.assertEqual(provider._normalize_model_name(""), "")

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    def test_normalize_model_name_digit_prefix(self, mock_openai_class, mock_httpx):
        """Test _normalize_model_name adds 'llama' prefix for digit-starting names."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_httpx.get.return_value = mock_response

        provider = OllamaProvider(self.cfg)
        self.assertEqual(provider._normalize_model_name("3.1:7b"), "llama3.1:7b")

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    def test_normalize_model_name_latest_warning(self, mock_openai_class, mock_httpx):
        """Test _normalize_model_name with :latest returns name (logs warning)."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_httpx.get.return_value = mock_response

        provider = OllamaProvider(self.cfg)
        self.assertEqual(provider._normalize_model_name("llama3.1:latest"), "llama3.1:latest")

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    def test_normalize_model_name_70b_warning(self, mock_openai_class, mock_httpx):
        """Test _normalize_model_name with :70b returns name (logs error)."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_httpx.get.return_value = mock_response

        provider = OllamaProvider(self.cfg)
        self.assertEqual(provider._normalize_model_name("llama3.1:70b"), "llama3.1:70b")

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    def test_model_name_to_prompt_dir_empty(self, mock_openai_class, mock_httpx):
        """Test _model_name_to_prompt_dir with empty string returns empty."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_httpx.get.return_value = mock_response

        provider = OllamaProvider(self.cfg)
        self.assertEqual(provider._model_name_to_prompt_dir(""), "")

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    def test_model_name_to_prompt_dir_normal(self, mock_openai_class, mock_httpx):
        """Test _model_name_to_prompt_dir converts colon to underscore."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_httpx.get.return_value = mock_response

        provider = OllamaProvider(self.cfg)
        self.assertEqual(provider._model_name_to_prompt_dir("llama3.1:8b"), "llama3.1_8b")

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    def test_get_model_specific_prompt_path_empty_model(self, mock_openai_class, mock_httpx):
        """Test _get_model_specific_prompt_path with empty model returns fallback."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_httpx.get.return_value = mock_response

        provider = OllamaProvider(self.cfg)
        result = provider._get_model_specific_prompt_path(
            "", "ner", "system_ner_v1", "ollama/ner/system_ner_v1"
        )
        self.assertEqual(result, "ollama/ner/system_ner_v1")

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    def test_detect_hosts_no_feed_title_returns_empty(self, mock_openai_class, mock_httpx):
        """Test detect_hosts with feed_title=None and no feed_authors returns empty set."""
        mock_health = Mock()
        mock_health.raise_for_status = Mock()
        mock_models = Mock()
        mock_models.raise_for_status = Mock()
        mock_models.json.return_value = {"models": [{"name": "llama3.1:8b"}]}
        mock_httpx.get.side_effect = [mock_health, mock_models]

        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            ollama_api_base=self.cfg.ollama_api_base,
            speaker_detector_provider="ollama",
            auto_speakers=True,
            generate_summaries=False,
        )
        provider = OllamaProvider(cfg)
        provider.initialize()

        hosts = provider.detect_hosts(feed_title=None, feed_description=None)
        self.assertEqual(hosts, set())

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    def test_summarize_empty_content_uses_empty_string(self, mock_openai_class, mock_httpx):
        """Test summarize when API returns empty content uses empty string."""
        mock_health = Mock()
        mock_health.raise_for_status = Mock()
        mock_models = Mock()
        mock_models.raise_for_status = Mock()
        mock_models.json.return_value = {"models": [{"name": "llama3.1:8b"}]}
        mock_httpx.get.side_effect = [mock_health, mock_models]

        mock_client = Mock()
        mock_resp = Mock()
        mock_resp.choices = [Mock()]
        mock_resp.choices[0].message.content = ""
        mock_resp.usage = None
        mock_client.chat.completions.create.return_value = mock_resp
        mock_openai_class.return_value = mock_client

        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            ollama_api_base=self.cfg.ollama_api_base,
            summary_provider="ollama",
            generate_summaries=True,
            generate_metadata=True,
            auto_speakers=False,
        )
        provider = OllamaProvider(cfg)
        provider.initialize()

        result = provider.summarize("Some text.")
        self.assertEqual(result["summary"], "")
        self.assertIn("metadata", result)

    @patch("podcast_scraper.prompts.store.render_prompt", return_value="Cleaning prompt")
    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    def test_clean_transcript_connection_error_raises_with_suggestion(
        self, mock_retry, mock_openai_class, mock_httpx, mock_render_prompt
    ):
        """Test clean_transcript connection error raises ProviderRuntimeError with suggestion."""
        mock_health = Mock()
        mock_health.raise_for_status = Mock()
        mock_models = Mock()
        mock_models.raise_for_status = Mock()
        mock_models.json.return_value = {"models": [{"name": "llama3.1:8b"}]}
        mock_httpx.get.side_effect = [mock_health, mock_models]

        mock_retry.side_effect = Exception("Connection refused")

        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            ollama_api_base=self.cfg.ollama_api_base,
            summary_provider="ollama",
            generate_summaries=True,
            generate_metadata=True,
            auto_speakers=False,
        )
        provider = OllamaProvider(cfg)
        provider.initialize()

        from podcast_scraper.exceptions import ProviderRuntimeError

        with self.assertRaises(ProviderRuntimeError) as ctx:
            provider.clean_transcript("Some transcript text.")

        self.assertIn("connection", str(ctx.exception).lower())
        self.assertIsNotNone(ctx.exception.suggestion)

    @patch("podcast_scraper.prompts.store.render_prompt", return_value="Cleaning prompt")
    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    def test_clean_transcript_generic_error_raises_without_suggestion(
        self, mock_retry, mock_openai_class, mock_httpx, mock_render_prompt
    ):
        """Test clean_transcript generic error raises ProviderRuntimeError without suggestion."""
        mock_health = Mock()
        mock_health.raise_for_status = Mock()
        mock_models = Mock()
        mock_models.raise_for_status = Mock()
        mock_models.json.return_value = {"models": [{"name": "llama3.1:8b"}]}
        mock_httpx.get.side_effect = [mock_health, mock_models]

        mock_retry.side_effect = Exception("Invalid response format")

        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            ollama_api_base=self.cfg.ollama_api_base,
            summary_provider="ollama",
            generate_summaries=True,
            generate_metadata=True,
            auto_speakers=False,
        )
        provider = OllamaProvider(cfg)
        provider.initialize()

        from podcast_scraper.exceptions import ProviderRuntimeError

        with self.assertRaises(ProviderRuntimeError) as ctx:
            provider.clean_transcript("Some transcript text.")

        self.assertIn("cleaning failed", str(ctx.exception).lower())
        self.assertIsNone(ctx.exception.suggestion)
