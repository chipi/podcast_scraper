"""Integration tests for Qwen 2.5 32B speaker detection.

Verify model-specific prompts resolve under ollama/qwen2.5_32b/ (same contract as 7B).
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
class TestQwen2532BSpeakerDetection(unittest.TestCase):
    """Test Qwen 2.5 32B speaker detection with model-specific prompts."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="ollama",
            ollama_speaker_model="qwen2.5:32b",
            ollama_api_base="http://localhost:11434/v1",
            auto_speakers=True,
        )

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    def test_provider_initialization_with_qwen_model(self, mock_openai_class, mock_httpx):
        """Ollama speaker detector initializes with Qwen 2.5 32B model name."""
        mock_health_response = Mock()
        mock_health_response.raise_for_status = Mock()
        mock_models_response = Mock()
        mock_models_response.raise_for_status = Mock()
        mock_models_response.json.return_value = {
            "models": [
                {"name": "qwen2.5:32b"},
            ]
        }
        mock_httpx.get.side_effect = [mock_health_response, mock_models_response]

        detector = create_speaker_detector(self.cfg)
        detector.initialize()

        mock_openai_class.assert_called_once()
        self.assertTrue(detector._speaker_detection_initialized)
        self.assertEqual(detector.speaker_model, "qwen2.5:32b")

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    @patch("podcast_scraper.prompts.store.render_prompt")
    def test_detect_speakers_with_qwen_prompts(
        self, mock_render_prompt, mock_openai_class, mock_httpx
    ):
        """Speaker detection uses Qwen 2.5 32B model-specific prompts when present."""
        mock_health_response = Mock()
        mock_health_response.raise_for_status = Mock()
        mock_models_response = Mock()
        mock_models_response.raise_for_status = Mock()
        mock_models_response.json.return_value = {
            "models": [
                {"name": "qwen2.5:32b"},
            ]
        }
        mock_httpx.get.side_effect = [mock_health_response, mock_models_response]

        mock_render_prompt.side_effect = ["Qwen System Prompt", "Qwen User Prompt"]

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

        speakers, hosts, success, _ = detector.detect_speakers(
            episode_title="Alice interviews Bob",
            episode_description="A great conversation",
            known_hosts={"Alice"},
        )

        self.assertEqual(speakers, ["Alice", "Bob"])
        self.assertEqual(hosts, {"Alice"})
        self.assertTrue(success)
        mock_client.chat.completions.create.assert_called_once()
        assert mock_render_prompt.called
