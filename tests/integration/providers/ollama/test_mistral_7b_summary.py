"""Integration tests for Mistral 7B summarization (Issue #395).

These tests verify that Mistral 7B model-specific prompts are loaded correctly
and that summarization works with the optimized prompts (emphasizing speed).
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
class TestMistral7BSummarization(unittest.TestCase):
    """Test Mistral 7B summarization with model-specific prompts."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="ollama",
            ollama_summary_model="mistral:7b",
            ollama_api_base="http://localhost:11434/v1",
            generate_summaries=True,
            generate_metadata=True,  # Required when generate_summaries=True
        )

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    def test_provider_initialization_with_mistral_model(self, mock_openai_class, mock_httpx):
        """Test that Ollama summarization provider initializes correctly with Mistral 7B model."""
        # Mock health check and model validation
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "models": [
                {"name": "mistral:7b"},
                {"name": "llama3.1:8b"},  # Speaker model (default)
            ]
        }
        mock_httpx.get.return_value = mock_response

        provider = create_summarization_provider(self.cfg)
        provider.initialize()

        # Verify client was created
        mock_openai_class.assert_called_once()
        self.assertTrue(provider._summarization_initialized)
        # Verify model name is normalized correctly
        self.assertEqual(provider.summary_model, "mistral:7b")

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    @patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
    @patch("podcast_scraper.prompts.store.render_prompt")
    @patch("podcast_scraper.prompts.store.get_prompt_metadata")
    def test_summarize_with_mistral_prompts(
        self, mock_get_metadata, mock_render_prompt, mock_openai_class, mock_httpx
    ):
        """Test successful summarization using Mistral 7B model-specific prompts."""
        # Mock health check and model validation
        mock_httpx_response = Mock()
        mock_httpx_response.raise_for_status = Mock()
        mock_httpx_response.json.return_value = {
            "models": [
                {"name": "mistral:7b"},
                {"name": "llama3.1:8b"},  # Speaker model (default)
            ]
        }
        mock_httpx.get.return_value = mock_httpx_response

        # Mock prompts - verify model-specific prompts are used
        mock_render_prompt.side_effect = ["Mistral System Prompt", "Mistral User Prompt"]
        mock_get_metadata.return_value = {"name": "ollama/mistral_7b/summarization/system_v1"}

        # Mock OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "This is a summary from Mistral 7B."
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 1000
        mock_response.usage.completion_tokens = 200
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        provider = create_summarization_provider(self.cfg)
        provider.client = mock_client
        provider.initialize()

        result = provider.summarize("This is a long transcript text.")

        self.assertEqual(result["summary"], "This is a summary from Mistral 7B.")
        self.assertIn("metadata", result)
        self.assertEqual(result["metadata"]["provider"], "ollama")
        mock_client.chat.completions.create.assert_called_once()

        # Verify prompts were rendered (model-specific prompts should be used)
        assert mock_render_prompt.called
