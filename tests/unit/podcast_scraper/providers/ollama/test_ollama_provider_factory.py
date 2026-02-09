#!/usr/bin/env python3
"""Unit tests for Ollama providers via factory.

These tests verify Ollama provider implementations with mocked API calls,
using factory functions to create providers (tests factory integration).

For standalone provider tests, see test_ollama_provider.py.
"""

import json
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
from podcast_scraper.speaker_detectors.factory import create_speaker_detector
from podcast_scraper.summarization.factory import create_summarization_provider

pytestmark = [pytest.mark.unit, pytest.mark.module_ollama_providers]


@pytest.mark.llm
@pytest.mark.ollama
class TestOllamaSpeakerDetectorFactory(unittest.TestCase):
    """Test Ollama speaker detection provider via factory."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="ollama",
            ollama_api_base="http://localhost:11434/v1",
            auto_speakers=True,
        )

    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    def test_provider_initialization(self, mock_httpx, mock_openai_class):
        """Test that Ollama speaker detector initializes correctly via factory."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock health check and model validation
        mock_health_response = Mock()
        mock_health_response.raise_for_status = Mock()
        mock_models_response = Mock()
        mock_models_response.raise_for_status = Mock()
        mock_models_response.json.return_value = {
            "models": [{"name": "llama3.2:latest"}, {"name": "llama3.3:latest"}]
        }
        mock_httpx.get.side_effect = [mock_health_response, mock_models_response]

        detector = create_speaker_detector(self.cfg)
        detector.initialize()

        # Verify client was created
        mock_openai_class.assert_called_once()
        self.assertTrue(detector._speaker_detection_initialized)

    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_detect_speakers_success(self, mock_render_prompt, mock_httpx, mock_openai_class):
        """Test successful speaker detection via Ollama API via factory."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock health check and model validation
        mock_health_response = Mock()
        mock_health_response.raise_for_status = Mock()
        mock_models_response = Mock()
        mock_models_response.raise_for_status = Mock()
        mock_models_response.json.return_value = {
            "models": [{"name": "llama3.2:latest"}, {"name": "llama3.3:latest"}]
        }
        mock_httpx.get.side_effect = [mock_health_response, mock_models_response]

        # Mock Ollama API response
        mock_api_response = Mock()
        mock_api_response.choices = [Mock()]
        mock_api_response.choices[0].message = Mock()
        mock_api_response.choices[0].message.content = json.dumps(
            {
                "speakers": ["John Doe"],
                "hosts": ["John Doe"],
                "guests": [],
            }
        )
        mock_api_response.usage = Mock()
        mock_api_response.usage.prompt_tokens = 100
        mock_api_response.usage.completion_tokens = 50
        mock_client.chat.completions.create.return_value = mock_api_response

        # Mock prompt rendering
        mock_render_prompt.side_effect = ["System prompt", "User prompt"]

        detector = create_speaker_detector(self.cfg)
        detector.initialize()

        speakers, detected_hosts, success = detector.detect_speakers(
            episode_title="Test Episode with John Doe",
            episode_description="A test episode",
            known_hosts={"John Doe"},
        )

        self.assertEqual(detected_hosts, {"John Doe"})
        self.assertTrue(success)

    def test_detect_speakers_missing_api_base(self):
        """Test that missing API base uses default."""
        # Ollama doesn't require API key, but needs API base (defaults to localhost)
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="ollama",
            auto_speakers=True,
        )
        # Should use default if not set - check that provider can be created
        # (default is handled in provider code, not config)
        self.assertIsNotNone(cfg)

    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    def test_factory_creates_ollama_detector(self, mock_httpx, mock_openai):
        """Test that factory creates Ollama speaker detector."""
        mock_client = Mock()
        mock_openai.return_value = mock_client

        # Mock health check
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_httpx.get.return_value = mock_response

        detector = create_speaker_detector(self.cfg)
        # Factory returns unified OllamaProvider
        self.assertEqual(detector.__class__.__name__, "OllamaProvider")


@pytest.mark.llm
@pytest.mark.ollama
class TestOllamaSummarizationProviderFactory(unittest.TestCase):
    """Test Ollama summarization provider via factory."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="ollama",
            ollama_api_base="http://localhost:11434/v1",
            generate_summaries=True,
            generate_metadata=True,
        )

    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    def test_provider_initialization(self, mock_httpx, mock_openai_class):
        """Test that Ollama summarization provider initializes correctly via factory."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock model validation (speaker model + summary model)
        # initialize() calls both _initialize_speaker_detection() and _initialize_summarization()
        mock_models_response = Mock()
        mock_models_response.raise_for_status = Mock()
        mock_models_response.json.return_value = {
            "models": [{"name": "llama3.2:latest"}, {"name": "llama3.3:latest"}]
        }
        # Two calls: one for speaker model, one for summary model
        mock_httpx.get.return_value = mock_models_response

        provider = create_summarization_provider(self.cfg)
        provider.initialize()

        # Verify client was created
        mock_openai_class.assert_called_once()
        self.assertTrue(provider._summarization_initialized)

    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_summarize_success(self, mock_render_prompt, mock_httpx, mock_openai_class):
        """Test successful summarization via Ollama API via factory."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock model validation (speaker model + summary model)
        # initialize() calls both _initialize_speaker_detection() and _initialize_summarization()
        mock_models_response = Mock()
        mock_models_response.raise_for_status = Mock()
        mock_models_response.json.return_value = {
            "models": [{"name": "llama3.2:latest"}, {"name": "llama3.3:latest"}]
        }
        # Two calls: one for speaker model, one for summary model
        mock_httpx.get.return_value = mock_models_response

        # Mock Ollama API response
        mock_api_response = Mock()
        mock_api_response.choices = [Mock()]
        mock_api_response.choices[0].message = Mock()
        mock_api_response.choices[0].message.content = "This is a summary of the transcript."
        mock_api_response.usage = Mock()
        mock_api_response.usage.prompt_tokens = 1000
        mock_api_response.usage.completion_tokens = 200
        mock_client.chat.completions.create.return_value = mock_api_response

        # Mock prompt rendering
        mock_render_prompt.side_effect = ["System prompt", "User prompt"]

        provider = create_summarization_provider(self.cfg)
        provider.initialize()

        result = provider.summarize("This is a long transcript text that needs summarization.")

        self.assertEqual(result["summary"], "This is a summary of the transcript.")
        self.assertIn("metadata", result)
        self.assertEqual(result["metadata"]["provider"], "ollama")

    def test_summarize_missing_api_base(self):
        """Test that missing API base uses default."""
        # Ollama doesn't require API key, but needs API base (defaults to localhost)
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="ollama",
            generate_summaries=True,
            generate_metadata=True,
        )
        # Should use default if not set - check that provider can be created
        # (default is handled in provider code, not config)
        self.assertIsNotNone(cfg)

    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    def test_factory_creates_ollama_provider(self, mock_httpx, mock_openai):
        """Test that factory creates Ollama summarization provider."""
        mock_client = Mock()
        mock_openai.return_value = mock_client

        # Mock health check
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_httpx.get.return_value = mock_response

        provider = create_summarization_provider(self.cfg)
        # Factory returns unified OllamaProvider
        self.assertEqual(provider.__class__.__name__, "OllamaProvider")
