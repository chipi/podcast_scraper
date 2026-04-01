"""Integration tests for Ollama Qwen 3.5 tier models (OLLAMA_PROVIDER_GUIDE checklist).

Verifies model-specific prompts load for qwen3.5:9b, qwen3.5:27b, qwen3.5:35b, and
qwen3.5:35b-a3b (Tier 3 MoE tag), matching the three-tier checklist in
docs/guides/OLLAMA_PROVIDER_GUIDE.md.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from podcast_scraper import config
from podcast_scraper.speaker_detectors.factory import create_speaker_detector
from podcast_scraper.summarization.factory import create_summarization_provider

pytestmark = [pytest.mark.integration, pytest.mark.module_ollama_providers]

# Ollama tags aligned with the guide's Tier 1–3 checklist (+ explicit MoE tag).
QWEN35_OLLAMA_TAGS = (
    "qwen3.5:9b",
    "qwen3.5:27b",
    "qwen3.5:35b",
    "qwen3.5:35b-a3b",
)


def _prompt_dir(model: str) -> str:
    return model.replace(":", "_")


@pytest.mark.integration
@pytest.mark.llm
@pytest.mark.ollama
@pytest.mark.parametrize("model_tag", QWEN35_OLLAMA_TAGS)
@patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
@patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
def test_qwen35_speaker_provider_initialization(mock_httpx, mock_openai_class, model_tag):
    mock_health_response = Mock()
    mock_health_response.raise_for_status = Mock()
    mock_models_response = Mock()
    mock_models_response.raise_for_status = Mock()
    mock_models_response.json.return_value = {"models": [{"name": model_tag}]}
    mock_httpx.get.side_effect = [mock_health_response, mock_models_response]

    cfg = config.Config(
        rss_url="https://example.com/feed.xml",
        speaker_detector_provider="ollama",
        ollama_speaker_model=model_tag,
        ollama_api_base="http://localhost:11434/v1",
        auto_speakers=True,
    )
    detector = create_speaker_detector(cfg)
    detector.initialize()

    mock_openai_class.assert_called_once()
    assert detector._speaker_detection_initialized
    assert detector.speaker_model == model_tag


@pytest.mark.integration
@pytest.mark.llm
@pytest.mark.ollama
@pytest.mark.parametrize("model_tag", QWEN35_OLLAMA_TAGS)
@patch("podcast_scraper.prompts.store.render_prompt")
@patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
@patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
def test_qwen35_speaker_detect_uses_model_prompts(
    mock_httpx, mock_openai_class, mock_render_prompt, model_tag
):
    mock_health_response = Mock()
    mock_health_response.raise_for_status = Mock()
    mock_models_response = Mock()
    mock_models_response.raise_for_status = Mock()
    mock_models_response.json.return_value = {"models": [{"name": model_tag}]}
    mock_httpx.get.side_effect = [mock_health_response, mock_models_response]

    mock_render_prompt.side_effect = ["Qwen35 System", "Qwen35 User"]

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

    cfg = config.Config(
        rss_url="https://example.com/feed.xml",
        speaker_detector_provider="ollama",
        ollama_speaker_model=model_tag,
        ollama_api_base="http://localhost:11434/v1",
        auto_speakers=True,
    )
    detector = create_speaker_detector(cfg)
    detector.client = mock_client
    detector.initialize()

    speakers, hosts, success, _ = detector.detect_speakers(
        episode_title="Alice interviews Bob",
        episode_description="A great conversation",
        known_hosts={"Alice"},
    )

    assert speakers == ["Alice", "Bob"]
    assert hosts == {"Alice"}
    assert success
    mock_client.chat.completions.create.assert_called_once()
    assert mock_render_prompt.called


@pytest.mark.integration
@pytest.mark.llm
@pytest.mark.ollama
@pytest.mark.parametrize("model_tag", QWEN35_OLLAMA_TAGS)
@patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
@patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
def test_qwen35_summary_provider_initialization(mock_httpx, mock_openai_class, model_tag):
    mock_httpx_response = Mock()
    mock_httpx_response.raise_for_status = Mock()
    mock_httpx_response.json.return_value = {
        "models": [
            {"name": model_tag},
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
@pytest.mark.parametrize("model_tag", QWEN35_OLLAMA_TAGS)
@patch("podcast_scraper.prompts.store.get_prompt_metadata")
@patch("podcast_scraper.prompts.store.render_prompt")
@patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI")
@patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
def test_qwen35_summary_uses_model_prompts(
    mock_httpx, mock_openai_class, mock_render_prompt, mock_get_metadata, model_tag
):
    mock_httpx_response = Mock()
    mock_httpx_response.raise_for_status = Mock()
    mock_httpx_response.json.return_value = {
        "models": [
            {"name": model_tag},
            {"name": "llama3.1:8b"},
        ]
    }
    mock_httpx.get.return_value = mock_httpx_response

    mock_render_prompt.side_effect = ["Qwen35 System", "Qwen35 User"]
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
