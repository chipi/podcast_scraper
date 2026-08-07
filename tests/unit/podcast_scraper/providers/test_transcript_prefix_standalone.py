"""RFC-111 Phase C: standalone OpenAI-style providers (grok/mistral/ollama) are transcript-first.

These providers do not subclass OpenAICompatibleProvider — each builds its own messages — so this
guards that their summary/GI/KG message sites relocate the transcript to a leading, cacheable system
block via the shared ``openai_style_messages`` helper, exactly like the base siblings.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from podcast_scraper import config as cfgmod
from podcast_scraper.providers.common.transcript_cache import TRANSCRIPT_BLOCK_HEADER

pytestmark = pytest.mark.unit

TRANSCRIPT = ("HOST: welcome. GUEST: reliable software for small teams. " * 20).strip()


def _mk_response() -> Mock:
    resp = Mock()
    resp.choices = [Mock()]
    resp.choices[0].message.content = "a summary"
    resp.choices[0].finish_reason = "stop"
    resp.usage = Mock(prompt_tokens=100, completion_tokens=20)
    resp.usage.prompt_tokens_details = Mock(cached_tokens=0)
    resp.model = "m"
    resp.id = "r"
    return resp


def _grok():
    from podcast_scraper.providers.grok.grok_provider import GrokProvider

    cfg = cfgmod.Config(
        rss="https://x/f.xml",
        summary_provider="grok",
        grok_api_key="xai-test-key",
        generate_summaries=True,
        cache_transcript_prefix=True,
    )
    p = GrokProvider(cfg)
    p._summarization_initialized = True
    client = Mock()
    client.chat.completions.create.return_value = _mk_response()
    p.client = client
    return p, lambda: client.chat.completions.create.call_args_list[0].kwargs["messages"]


def _mistral():
    from podcast_scraper.providers.mistral.mistral_provider import MistralProvider

    cfg = cfgmod.Config(
        rss="https://x/f.xml",
        summary_provider="mistral",
        mistral_api_key="test-key",
        generate_summaries=True,
        cache_transcript_prefix=True,
    )
    p = MistralProvider(cfg)
    p._summarization_initialized = True
    client = Mock()
    client.chat.complete.return_value = _mk_response()
    p.client = client
    return p, lambda: client.chat.complete.call_args_list[0].kwargs["messages"]


def _ollama():
    from podcast_scraper.providers.ollama.ollama_provider import OllamaProvider

    cfg = cfgmod.Config(
        rss="https://x/f.xml",
        summary_provider="ollama",
        generate_summaries=True,
        cache_transcript_prefix=True,
    )
    p = OllamaProvider(cfg)
    p._summarization_initialized = True
    client = Mock()
    client.chat.completions.create.return_value = _mk_response()
    p.client = client
    return p, lambda: client.chat.completions.create.call_args_list[0].kwargs["messages"]


@pytest.mark.parametrize("builder", [_grok, _mistral, _ollama], ids=["grok", "mistral", "ollama"])
def test_standalone_summary_is_transcript_first(builder) -> None:
    p, messages_of = builder()
    p.summarize(text=TRANSCRIPT + "\n", episode_title="Ep")
    messages = messages_of()
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith(TRANSCRIPT_BLOCK_HEADER), "not transcript-first"
    assert TRANSCRIPT in messages[0]["content"]
    assert TRANSCRIPT not in messages[1]["content"]


@pytest.mark.parametrize("builder", [_grok, _mistral, _ollama], ids=["grok", "mistral", "ollama"])
def test_standalone_off_is_legacy(builder) -> None:
    p, messages_of = builder()
    p._cache_transcript_prefix = False
    p.summarize(text=TRANSCRIPT + "\n", episode_title="Ep")
    messages = messages_of()
    assert not messages[0]["content"].startswith(TRANSCRIPT_BLOCK_HEADER)
    assert TRANSCRIPT in messages[1]["content"]
