"""RFC-111 Phase D: Anthropic relocates the transcript to a cache_control'd system block.

Anthropic prefix caching is opt-in, so (unlike the auto-cache providers) the transcript block must
be a ``system`` content block carrying an explicit ``cache_control`` breakpoint. Live-verified:
summary writes the block, GI reads it 100% (cache_read == the transcript token count).
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from podcast_scraper import config as cfgmod
from podcast_scraper.providers.anthropic.anthropic_provider import AnthropicProvider
from podcast_scraper.providers.common.transcript_cache import TRANSCRIPT_BLOCK_HEADER

pytestmark = pytest.mark.unit

TRANSCRIPT = ("HOST: welcome. GUEST: reliable software for small teams. " * 20).strip()


def _provider(cache: bool = True) -> AnthropicProvider:
    cfg = cfgmod.Config(
        rss="https://x/f.xml",
        summary_provider="anthropic",
        anthropic_api_key="sk-ant-test-key",
        anthropic_summary_model="claude-haiku-4-5",
        generate_summaries=True,
        cache_transcript_prefix=cache,
    )
    p = AnthropicProvider(cfg)
    p._summarization_initialized = True
    return p


def _run(p: AnthropicProvider) -> dict:
    client = Mock()
    resp = Mock()
    resp.content = [Mock(text="a summary")]
    resp.stop_reason = "end_turn"
    resp.usage = Mock(
        input_tokens=100,
        output_tokens=20,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    resp.model = "claude-haiku-4-5"
    resp.id = "r"
    client.messages.create.return_value = resp
    p.client = client
    p.summarize(text=TRANSCRIPT + "\n", episode_title="Ep")
    return dict(client.messages.create.call_args_list[0].kwargs)


def test_anthropic_transcript_first_uses_cache_control_block() -> None:
    kwargs = _run(_provider(cache=True))
    system = kwargs["system"]
    assert isinstance(system, list), "system must be a content-block list to carry cache_control"
    block = system[0]
    assert block["cache_control"] == {"type": "ephemeral"}
    assert block["text"].startswith(TRANSCRIPT_BLOCK_HEADER)
    assert TRANSCRIPT in block["text"]
    # transcript relocated out of the user message
    assert TRANSCRIPT not in kwargs["messages"][0]["content"]


def test_anthropic_off_is_plain_string_legacy() -> None:
    kwargs = _run(_provider(cache=False))
    assert isinstance(kwargs["system"], str), "flag off must keep the plain-string system"
    assert TRANSCRIPT in kwargs["messages"][0]["content"]
