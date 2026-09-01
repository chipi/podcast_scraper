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

#1894 STRENGTHENED THIS. The per-request value used to be the deadline ITSELF, which bounds the
socket but is not a useful bound: one stuck call could consume the ENTIRE budget, so the
deadline alert could only ever report a hang after it had already cost everything, and could
not distinguish it from the aggregate simply being expensive. Measured over 82 episodes,
summary+GI+KG totals p50=1130s against a 1200s deadline — 41% of HEALTHY episodes tripped it.

The timeout is now a FRACTION of the deadline (get_single_chat_call_timeout), so a genuine hang
surfaces in minutes and the RFC-106 ladder still has budget left to fail over. #1852's intent —
bound the socket, do not shorten Whisper's window, do not hard-abort a long aggregate stage — is
unchanged; only the value is stronger. These assertions therefore check the RELATION
(per-call < deadline) rather than a specific equality, which is the invariant that actually
matters and cannot silently regress to "equal".
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from podcast_scraper import config as cfgmod, config_constants
from podcast_scraper.providers.litellm.litellm_provider import LiteLLMProvider
from podcast_scraper.providers.openai.openai_provider import OpenAIProvider
from podcast_scraper.utils.timeout_config import (
    MIN_SINGLE_CALL_TIMEOUT_SEC,
    SINGLE_CALL_TIMEOUT_FRACTION,
)

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
    got = _read_timeout(kwargs["timeout"])
    assert got < _DEADLINE, (
        f"per-call timeout {got} must be STRICTLY LESS than the {_DEADLINE}s deadline (#1894): "
        "equal means one hung call consumes the whole budget before anything fires"
    )
    assert got == pytest.approx(_DEADLINE * SINGLE_CALL_TIMEOUT_FRACTION)


def test_chat_timeout_tracks_a_custom_summarization_timeout() -> None:
    p = _provider(summarization_timeout=300)
    p.client = MagicMock()
    p._chat_create(model="gpt-4o-mini", messages=[])

    kwargs = p.client.chat.completions.create.call_args.kwargs
    got = _read_timeout(kwargs["timeout"])
    # 300 * 1/3 = 100, below the 120s floor that protects a slow model on a short deadline.
    assert got == MIN_SINGLE_CALL_TIMEOUT_SEC
    assert got < 300.0, "must still track BELOW the configured deadline"


def test_explicit_per_call_timeout_still_wins() -> None:
    """setdefault semantics: a caller that passes its own timeout is not overridden."""
    p = _provider()
    p.client = MagicMock()
    p._chat_create(model="gpt-4o-mini", messages=[], timeout=5.0)

    kwargs = p.client.chat.completions.create.call_args.kwargs
    assert kwargs["timeout"] == 5.0


def test_litellm_subclass_inherits_the_bound() -> None:
    """The prod summary provider is LiteLLM (cloud_balanced); it inherits the _chat_create seam.

    Imported directly, not via ``importorskip``: ``litellm_provider`` imports the ``litellm``
    package lazily (not at module top), and this path only touches the inherited ``_chat_create``
    over a mocked client — so it runs under ``[dev]`` alone, which the 3-tier policy requires of a
    unit test (verified by blocking the ``litellm`` import: the bound still resolves to the
    deadline).
    """
    cfg = cfgmod.Config(
        rss="https://example.com/feed.xml",
        summary_provider="litellm",
        openai_summary_model="gpt-4o-mini",
        openai_api_key="test-api-key-123",
    )
    p = LiteLLMProvider(cfg)
    p.client = MagicMock()
    p._chat_create(model="gpt-4o-mini", messages=[])

    kwargs = p.client.chat.completions.create.call_args.kwargs
    assert "timeout" in kwargs
    got = _read_timeout(kwargs["timeout"])
    assert got == pytest.approx(_DEADLINE * SINGLE_CALL_TIMEOUT_FRACTION)
    assert got < _DEADLINE, "the prod summary provider must inherit the #1894 sub-deadline bound"
