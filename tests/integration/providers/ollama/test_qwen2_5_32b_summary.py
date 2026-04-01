"""Integration tests for Qwen 2.5 32B summarization.

Verify model-specific prompts resolve under ollama/qwen2.5_32b/ (same contract as 7B).
"""

import unittest
from unittest.mock import Mock, patch

import pytest

from podcast_scraper import config
from podcast_scraper.summarization.factory import create_summarization_provider

pytestmark = [pytest.mark.integration, pytest.mark.module_ollama_providers]


@pytest.mark.integration
@pytest.mark.llm
@pytest.mark.ollama
class TestQwen2532BSummarization(unittest.TestCase):
    """Test Qwen 2.5 32B summarization with model-specific prompts."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="ollama",
            ollama_summary_model="qwen2.5:32b",
            ollama_api_base="http://localhost:11434/v1",
            generate_summaries=True,
            generate_metadata=True,  # Required when generate_summaries=True
        )

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    def test_provider_initialization_with_qwen_model(self, mock_openai_class, mock_httpx):
        """Ollama summarization initializes with Qwen 2.5 32B model name."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "models": [
                {"name": "qwen2.5:32b"},
                {"name": "llama3.1:8b"},
            ]
        }
        mock_httpx.get.return_value = mock_response

        provider = create_summarization_provider(self.cfg)
        provider.initialize()

        mock_openai_class.assert_called_once()
        self.assertTrue(provider._summarization_initialized)
        self.assertEqual(provider.summary_model, "qwen2.5:32b")

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    @patch("podcast_scraper.prompts.store.render_prompt")
    @patch("podcast_scraper.prompts.store.get_prompt_metadata")
    def test_summarize_with_qwen_prompts(
        self, mock_get_metadata, mock_render_prompt, mock_openai_class, mock_httpx
    ):
        """Summarization uses Qwen 2.5 32B model-specific prompt paths when present."""
        mock_httpx_response = Mock()
        mock_httpx_response.raise_for_status = Mock()
        mock_httpx_response.json.return_value = {
            "models": [
                {"name": "qwen2.5:32b"},
                {"name": "llama3.1:8b"},
            ]
        }
        mock_httpx.get.return_value = mock_httpx_response

        mock_render_prompt.side_effect = ["Qwen System Prompt", "Qwen User Prompt"]
        mock_get_metadata.return_value = {"name": "ollama/qwen2.5_32b/summarization/system_v1"}

        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "This is a summary from Qwen 2.5 32B."
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 1000
        mock_response.usage.completion_tokens = 200
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        provider = create_summarization_provider(self.cfg)
        provider.client = mock_client
        provider.initialize()

        result = provider.summarize("This is a long transcript text.")

        self.assertEqual(result["summary"], "This is a summary from Qwen 2.5 32B.")
        self.assertIn("metadata", result)
        self.assertEqual(result["metadata"]["provider"], "ollama")
        mock_client.chat.completions.create.assert_called_once()
        assert mock_render_prompt.called
