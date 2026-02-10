"""Integration tests for Llama 3.1 8B speaker detection (Issue #394).

These tests verify that Llama 3.1 8B model-specific prompts are loaded correctly
and that speaker detection works with the optimized prompts.

Note: File name uses underscores (test_llama3_1_8b_speaker.py) because Python
module names cannot contain dots.
"""

import unittest
from unittest.mock import Mock, patch

import pytest

from podcast_scraper import config
from podcast_scraper.speaker_detectors.factory import create_speaker_detector

pytestmark = [pytest.mark.integration, pytest.mark.module_ollama_providers]


@pytest.mark.integration
@pytest.mark.llm
@pytest.mark.ollama
class TestLlama31_8BSpeakerDetection(unittest.TestCase):
    """Test Llama 3.1 8B speaker detection with model-specific prompts."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="ollama",
            ollama_speaker_model="llama3.1:8b",
            ollama_api_base="http://localhost:11434/v1",
            auto_speakers=True,
        )

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    def test_provider_initialization_with_llama_model(self, mock_openai_class, mock_httpx):
        """Test that Ollama speaker detector initializes correctly with Llama 3.1 8B model."""
        # Mock health check and model validation
        mock_health_response = Mock()
        mock_health_response.raise_for_status = Mock()
        mock_models_response = Mock()
        mock_models_response.raise_for_status = Mock()
        mock_models_response.json.return_value = {
            "models": [
                {"name": "llama3.1:8b"},
            ]
        }
        mock_httpx.get.side_effect = [mock_health_response, mock_models_response]

        detector = create_speaker_detector(self.cfg)
        detector.initialize()

        # Verify client was created
        mock_openai_class.assert_called_once()
        self.assertTrue(detector._speaker_detection_initialized)
        # Verify model name is normalized correctly
        self.assertEqual(detector.speaker_model, "llama3.1:8b")

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_detect_speakers_with_llama_prompts(
        self, mock_render_prompt, mock_openai_class, mock_httpx
    ):
        """Test successful speaker detection using Llama 3.1 8B model-specific prompts."""
        # Mock health check and model validation
        mock_health_response = Mock()
        mock_health_response.raise_for_status = Mock()
        mock_models_response = Mock()
        mock_models_response.raise_for_status = Mock()
        mock_models_response.json.return_value = {
            "models": [
                {"name": "llama3.1:8b"},
            ]
        }
        mock_httpx.get.side_effect = [mock_health_response, mock_models_response]

        # Mock prompts - verify model-specific prompts are used
        mock_render_prompt.side_effect = ["Llama System Prompt", "Llama User Prompt"]

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

        # Verify prompts were rendered (model-specific prompts should be used)
        assert mock_render_prompt.called
