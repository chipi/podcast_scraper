"""A KG reply that is not valid JSON is asked for once more (2026-10-03).

Prod: Qwen3-30B dropped one ``},{`` in a complete 6 KB reply and the episode lost its whole KG.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock, patch

import pytest

from podcast_scraper import config

pytestmark = pytest.mark.unit

_GOOD = '{"topics": [{"label": "Institutions"}], "entities": [{"name": "Yuval Levin"}]}'
_BAD = '{"topics": [{"label": "Institutions"}], "entities": [{"name": "A"} {"name": "B"}]}'


def _response(content: str) -> Mock:
    resp = Mock()
    choice = Mock()
    choice.message.content = content
    choice.finish_reason = "stop"
    resp.choices = [choice]
    resp.usage = Mock(prompt_tokens=100, completion_tokens=50)
    return resp


def _provider(client: Mock) -> Any:
    from podcast_scraper.providers.openai.openai_provider import OpenAIProvider

    cfg = config.Config(
        transcription_provider="whisper",
        speaker_detector_provider="openai",
        summary_provider="openai",
        transcribe_missing=False,
        auto_speakers=True,
        generate_summaries=True,
        openai_api_key="test-api-key-123",
    )
    with patch("openai.OpenAI", return_value=client):
        provider = OpenAIProvider(cfg)
        provider.initialize()
    return provider


@patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
def test_bad_json_then_good_json_returns_the_good_graph(mock_retry: Mock) -> None:
    mock_retry.side_effect = lambda fn, **kwargs: fn()
    client = Mock()
    client.chat.completions.create.side_effect = [_response(_BAD), _response(_GOOD)]
    parsed = _provider(client).extract_kg_graph("a transcript long enough to matter.")
    assert parsed is not None
    assert parsed.get("topics")
    assert client.chat.completions.create.call_count == 2


@patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
def test_good_json_is_asked_for_once(mock_retry: Mock) -> None:
    mock_retry.side_effect = lambda fn, **kwargs: fn()
    client = Mock()
    client.chat.completions.create.return_value = _response(_GOOD)
    assert _provider(client).extract_kg_graph("a transcript long enough to matter.")
    assert client.chat.completions.create.call_count == 1


@patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
def test_bad_json_twice_gives_up_after_one_retry(mock_retry: Mock) -> None:
    mock_retry.side_effect = lambda fn, **kwargs: fn()
    client = Mock()
    client.chat.completions.create.return_value = _response(_BAD)
    assert _provider(client).extract_kg_graph("a transcript long enough to matter.") is None
    assert client.chat.completions.create.call_count == 2
