"""#1852: every chat request must carry a PER-REQUEST transport timeout bounded by the deadline.

The bug: the shared OpenAI client read timeout is ``max(summarization, transcription)`` (one client
serves Whisper + chat), so for a CHAT call it is >= the 1200s summarization deadline. A single chat
call could therefore sit on a silent socket PAST the soft ``timeout_context`` deadline — the
"DEADLINE EXCEEDED ... STILL RUNNING" alert — with nothing bounding the socket. The fix sets a
per-request ``timeout`` on every ``_chat_create`` call tied to the summarization deadline, bounding
the socket without shortening Whisper's client window and without hard-aborting a legitimately long
AGGREGATE stage (an accepted overrun-is-success case).

Mutation guard: WITHOUT the fix ``_chat_create`` passes no ``timeout`` kwarg (relying on the
inflated client default), so 3 of the 4 assertions fail.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from podcast_scraper import config as cfgmod, config_constants
from podcast_scraper.providers.openai.openai_provider import OpenAIProvider

pytestmark = pytest.mark.unit

_DEADLINE = float(config_constants.DEFAULT_SUMMARIZATION_TIMEOUT_SECONDS)


def _provider(summarization_timeout: int | None = None) -> OpenAIProvider:
    if summarization_timeout is None:
        cfg = cfgmod.Config(
            rss="https://example.com/feed.xml",
            summary_provider="openai",
            openai_summary_model="gpt-4o-mini",
            openai_api_key="test-api-key-123",
        )
    else:
        cfg = cfgmod.Config(
            rss="https://example.com/feed.xml",
            summary_provider="openai",
            openai_summary_model="gpt-4o-mini",
            openai_api_key="test-api-key-123",
            summarization_timeout=summarization_timeout,
        )
    return OpenAIProvider(cfg)


def _read_timeout(call_timeout) -> float:
    """Read component of whatever _chat_create passed as timeout (httpx.Timeout or float)."""
    return float(getattr(call_timeout, "read", call_timeout))


def test_chat_call_is_bounded_by_the_summarization_deadline_by_default() -> None:
    p = _provider()
    p.client = MagicMock()
    p._chat_create(model="gpt-4o-mini", messages=[])

    kwargs = p.client.chat.completions.create.call_args.kwargs
    assert (
        "timeout" in kwargs
    ), "no per-request timeout — chat relies on the inflated client default"
    assert (
        _read_timeout(kwargs["timeout"]) == _DEADLINE
    ), "chat read must == the deadline, not client max"


def test_chat_timeout_tracks_a_custom_summarization_timeout() -> None:
    p = _provider(summarization_timeout=300)
    p.client = MagicMock()
    p._chat_create(model="gpt-4o-mini", messages=[])

    kwargs = p.client.chat.completions.create.call_args.kwargs
    assert _read_timeout(kwargs["timeout"]) == 300.0


def test_explicit_per_call_timeout_still_wins() -> None:
    """setdefault semantics: a caller that passes its own timeout is not overridden."""
    p = _provider()
    p.client = MagicMock()
    p._chat_create(model="gpt-4o-mini", messages=[], timeout=5.0)

    kwargs = p.client.chat.completions.create.call_args.kwargs
    assert kwargs["timeout"] == 5.0


def test_litellm_subclass_inherits_the_bound() -> None:
    """The prod summary provider is LiteLLM (cloud_balanced); it inherits the _chat_create seam."""
    litellm = pytest.importorskip("podcast_scraper.providers.litellm.litellm_provider")
    cfg = cfgmod.Config(
        rss="https://example.com/feed.xml",
        summary_provider="litellm",
        openai_summary_model="gpt-4o-mini",
        openai_api_key="test-api-key-123",
    )
    p = litellm.LiteLLMProvider(cfg)
    p.client = MagicMock()
    p._chat_create(model="gpt-4o-mini", messages=[])

    kwargs = p.client.chat.completions.create.call_args.kwargs
    assert "timeout" in kwargs
    assert _read_timeout(kwargs["timeout"]) == _DEADLINE
