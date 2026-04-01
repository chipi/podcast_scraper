"""Integration tests for additional Mistral-family Ollama tags (Nemo 12B, Small 3.2).

Prompt dirs: ``mistral-nemo_12b``, ``mistral-small3.2`` under ``prompts/ollama/``.
See docs/guides/OLLAMA_PROVIDER_GUIDE.md.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from podcast_scraper import config
from podcast_scraper.summarization.factory import create_summarization_provider

pytestmark = [pytest.mark.integration, pytest.mark.module_ollama_providers]

MISTRAL_OLLAMA_EXTRA_TAGS = (
    "mistral-nemo:12b",
    "mistral-small3.2:latest",
)


def _prompt_dir(model: str) -> str:
    return model.replace(":", "_")


@pytest.mark.integration
@pytest.mark.llm
@pytest.mark.ollama
@pytest.mark.parametrize("model_tag", MISTRAL_OLLAMA_EXTRA_TAGS)
@patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
@patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
def test_mistral_variant_summary_provider_initialization(mock_httpx, mock_openai_class, model_tag):
    mock_httpx_response = Mock()
    mock_httpx_response.raise_for_status = Mock()
    mock_httpx_response.json.return_value = {
        "models": [
            {"name": model_tag},
            {"name": "mistral:7b"},
            {"name": "llama3.1:8b"},
        ]
    }
    mock_httpx.get.return_value = mock_httpx_response

    cfg = config.Config(
        rss_url="https://example.com/feed.xml",
        summary_provider="ollama",
        ollama_summary_model=model_tag,
        ollama_api_base="http://localhost:11434/v1",
        generate_summaries=True,
        generate_metadata=True,
    )
    provider = create_summarization_provider(cfg)
    provider.initialize()

    mock_openai_class.assert_called_once()
    assert provider._summarization_initialized
    assert provider.summary_model == model_tag


@pytest.mark.integration
@pytest.mark.llm
@pytest.mark.ollama
@pytest.mark.parametrize("model_tag", MISTRAL_OLLAMA_EXTRA_TAGS)
@patch("podcast_scraper.prompts.store.get_prompt_metadata")
@patch("podcast_scraper.prompts.store.render_prompt")
@patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
@patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
def test_mistral_variant_summary_uses_model_prompts(
    mock_httpx, mock_openai_class, mock_render_prompt, mock_get_metadata, model_tag
):
    mock_httpx_response = Mock()
    mock_httpx_response.raise_for_status = Mock()
    mock_httpx_response.json.return_value = {
        "models": [
            {"name": model_tag},
            {"name": "mistral:7b"},
            {"name": "llama3.1:8b"},
        ]
    }
    mock_httpx.get.return_value = mock_httpx_response

    mock_render_prompt.side_effect = ["MistralVar System", "MistralVar User"]
    pdir = _prompt_dir(model_tag)
    mock_get_metadata.return_value = {"name": f"ollama/{pdir}/summarization/system_v1"}

    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = f"Summary from {model_tag}."
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 1000
    mock_response.usage.completion_tokens = 200
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_class.return_value = mock_client

    cfg = config.Config(
        rss_url="https://example.com/feed.xml",
        summary_provider="ollama",
        ollama_summary_model=model_tag,
        ollama_api_base="http://localhost:11434/v1",
        generate_summaries=True,
        generate_metadata=True,
    )
    provider = create_summarization_provider(cfg)
    provider.client = mock_client
    provider.initialize()

    result = provider.summarize("This is a long transcript text.")

    assert result["summary"] == f"Summary from {model_tag}."
    assert "metadata" in result
    assert result["metadata"]["provider"] == "ollama"
    mock_client.chat.completions.create.assert_called_once()
    assert mock_render_prompt.called
