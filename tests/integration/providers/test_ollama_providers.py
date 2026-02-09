"""Integration tests for Ollama providers.

These tests verify Ollama provider implementations with mocked API calls.
"""

import unittest
from unittest.mock import Mock, patch

import pytest

from podcast_scraper import config
from podcast_scraper.speaker_detectors.factory import create_speaker_detector
from podcast_scraper.summarization.factory import create_summarization_provider

pytestmark = [pytest.mark.integration, pytest.mark.module_ollama_providers]


@pytest.mark.integration
@pytest.mark.llm
@pytest.mark.ollama
class TestOllamaSpeakerDetector(unittest.TestCase):
    """Test Ollama speaker detection provider."""

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
    def test_provider_initialization(self, mock_openai_class, mock_httpx):
        """Test that Ollama speaker detector initializes correctly via factory."""
        # Mock health check and model validation
        mock_health_response = Mock()
        mock_health_response.raise_for_status = Mock()
        mock_models_response = Mock()
        # Include both models in case both are validated
        mock_models_response.raise_for_status = Mock()
        mock_models_response.json.return_value = {
            "models": [
                {"name": "llama3.3:latest"},
                {"name": "llama3.2:latest"},
                {"name": "llama3.1:8b"},
            ]
        }
        mock_httpx.get.side_effect = [mock_health_response, mock_models_response]

        detector = create_speaker_detector(self.cfg)
        detector.initialize()

        # Verify client was created
        mock_openai_class.assert_called_once()
        self.assertTrue(detector._speaker_detection_initialized)

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_detect_speakers_success(self, mock_render_prompt, mock_openai_class, mock_httpx):
        """Test successful speaker detection via Ollama API via factory."""
        # Mock health check and model validation
        mock_health_response = Mock()
        mock_health_response.raise_for_status = Mock()
        mock_models_response = Mock()
        # Include both models in case both are validated
        mock_models_response.raise_for_status = Mock()
        mock_models_response.json.return_value = {
            "models": [
                {"name": "llama3.3:latest"},
                {"name": "llama3.2:latest"},
                {"name": "llama3.1:8b"},
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

        detector = create_speaker_detector(self.cfg)
        detector.client = mock_client
        detector.initialize()

        speakers, hosts, success = detector.detect_speakers(
            episode_title="Alice interviews Bob",
            episode_description="A great conversation",
            known_hosts={"Alice"},
        )

        self.assertEqual(speakers, ["Alice", "Bob"])
        self.assertEqual(hosts, {"Alice"})
        self.assertTrue(success)
        mock_client.chat.completions.create.assert_called_once()

    def test_factory_creates_ollama_provider(self):
        """Test that factory creates unified Ollama provider."""
        with patch("podcast_scraper.providers.ollama.ollama_provider.httpx") as mock_httpx:
            with patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI"):
                # Mock health check
                mock_response = Mock()
                mock_response.raise_for_status = Mock()
                mock_httpx.get.return_value = mock_response

                detector = create_speaker_detector(self.cfg)
                self.assertIsNotNone(detector)
                # Verify it's the unified Ollama provider
                self.assertEqual(detector.__class__.__name__, "OllamaProvider")
                # Verify protocol compliance
                self.assertTrue(hasattr(detector, "detect_speakers"))
                self.assertTrue(hasattr(detector, "detect_hosts"))
                self.assertTrue(hasattr(detector, "clear_cache"))


@pytest.mark.integration
@pytest.mark.llm
@pytest.mark.ollama
class TestOllamaSummarizationProvider(unittest.TestCase):
    """Test Ollama summarization provider."""

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
    def test_provider_initialization(self, mock_openai_class, mock_httpx):
        """Test that Ollama summarization provider initializes correctly via factory."""
        # Mock health check and model validation
        # Create reusable mock response
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3.3:latest"},
                {"name": "llama3.2:latest"},
                {"name": "llama3.1:8b"},
            ]
        }
        # Return same mock for all httpx.get calls (health check + model validation)
        mock_httpx.get.return_value = mock_response

        provider = create_summarization_provider(self.cfg)
        provider.initialize()

        # Verify client was created
        mock_openai_class.assert_called_once()
        self.assertTrue(provider._summarization_initialized)

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    @patch("podcast_scraper.prompts.store.render_prompt")
    @patch("podcast_scraper.prompts.store.get_prompt_metadata")
    def test_summarize_success(
        self, mock_get_metadata, mock_render_prompt, mock_openai_class, mock_httpx
    ):
        """Test successful summarization via Ollama API via factory."""
        # Mock health check and model validation
        # Create reusable mock response
        mock_httpx_response = Mock()
        mock_httpx_response.raise_for_status = Mock()
        mock_httpx_response.json.return_value = {
            "models": [
                {"name": "llama3.3:latest"},
                {"name": "llama3.2:latest"},
                {"name": "llama3.1:8b"},
            ]
        }
        # Return same mock for all httpx.get calls (health check + model validation)
        mock_httpx.get.return_value = mock_httpx_response

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

        provider = create_summarization_provider(self.cfg)
        provider.client = mock_client
        provider.initialize()

        result = provider.summarize("This is a long transcript text.")

        self.assertEqual(result["summary"], "This is a summary.")
        self.assertIn("metadata", result)
        self.assertEqual(result["metadata"]["provider"], "ollama")
        mock_client.chat.completions.create.assert_called_once()

    def test_factory_creates_ollama_provider(self):
        """Test that factory creates unified Ollama provider."""
        with patch("podcast_scraper.providers.ollama.ollama_provider.httpx") as mock_httpx:
            with patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI"):
                # Mock health check
                mock_response = Mock()
                mock_response.raise_for_status = Mock()
                mock_httpx.get.return_value = mock_response

                provider = create_summarization_provider(self.cfg)
                self.assertIsNotNone(provider)
                # Verify it's the unified Ollama provider
                self.assertEqual(provider.__class__.__name__, "OllamaProvider")
                # Verify protocol compliance
                self.assertTrue(hasattr(provider, "summarize"))
                self.assertTrue(hasattr(provider, "cleanup"))


@pytest.mark.integration
@pytest.mark.llm
@pytest.mark.ollama
class TestOllamaProviderErrorHandling(unittest.TestCase):
    """Test Ollama provider error handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="ollama",
            ollama_api_base="http://localhost:11434/v1",
            auto_speakers=True,
        )

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    def test_provider_creation_fails_when_ollama_not_running(self, mock_httpx):
        """Test that provider creation fails when Ollama server is not running."""
        import httpx

        mock_httpx.get.side_effect = httpx.ConnectError("Connection refused")

        from podcast_scraper.providers.ollama.ollama_provider import OllamaProvider

        with self.assertRaises(ConnectionError) as context:
            OllamaProvider(self.cfg)

        self.assertIn("Ollama server is not running", str(context.exception))

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    def test_model_validation_warning_on_failure(self, mock_openai_class, mock_httpx):
        """Test that model validation failure logs warning but doesn't fail."""
        # Mock health check
        mock_health_response = Mock()
        mock_health_response.raise_for_status = Mock()
        # Mock model validation failure (connection error)
        import httpx

        mock_httpx.get.side_effect = [mock_health_response, httpx.RequestError("Connection error")]

        from podcast_scraper.providers.ollama.ollama_provider import OllamaProvider

        # Should not raise - validation failure is logged as warning
        provider = OllamaProvider(self.cfg)
        self.assertIsNotNone(provider)
