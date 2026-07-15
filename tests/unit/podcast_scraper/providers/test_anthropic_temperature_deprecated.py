"""Claude 5 models dropped `temperature`; the provider must not die on that.

THE BUG (found in the bake-off): `claude-sonnet-5` rejects any request carrying `temperature` with
a 400 "`temperature` is deprecated for this model". The provider sent `temperature=0.3` on every
call, so the Anthropic arm crashed at episode 1 with zero output. `claude-opus-4-8` (the flagship)
does the same; the older 4.x models still accept temperature.

The fix routes every call through `_messages_create`, which strips temperature for known-deprecated
models and — crucially — LEARNS a new one from the API's own error and retries, so a future model
that drops temperature self-heals after one failed call instead of taking down the run.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from podcast_scraper import config as cfgmod
from podcast_scraper.providers.anthropic.anthropic_provider import AnthropicProvider

pytestmark = pytest.mark.unit


def _provider(model: str) -> AnthropicProvider:
    cfg = cfgmod.Config(
        rss="https://example.com/feed.xml",
        summary_provider="anthropic",
        anthropic_summary_model=model,
        anthropic_api_key="test-key",
    )
    p = AnthropicProvider(cfg)
    p.client = MagicMock()
    return p


def test_known_deprecated_model_never_sends_temperature() -> None:
    """claude-sonnet-5 is seeded as deprecated, so temperature is stripped BEFORE the call —
    no wasted 400."""
    p = _provider("claude-sonnet-5")
    p._messages_create(model="claude-sonnet-5", temperature=0.3, max_tokens=10)
    _, kwargs = p.client.messages.create.call_args
    assert "temperature" not in kwargs, "temperature must not reach a model that rejects it"
    assert p.client.messages.create.call_count == 1, "no retry needed when we already know"


def test_an_accepting_model_keeps_its_temperature() -> None:
    """sonnet-4.x still honours temperature; the fix must not strip it from accepters."""
    p = _provider("claude-sonnet-4-6")
    p._messages_create(model="claude-sonnet-4-6", temperature=0.0, max_tokens=10)
    _, kwargs = p.client.messages.create.call_args
    assert kwargs.get("temperature") == 0.0


def test_an_unknown_rejecting_model_self_heals_and_is_remembered() -> None:
    """A model we did NOT seed rejects temperature. The provider must retry without it AND learn,
    so the next call skips temperature outright."""
    p = _provider("claude-future-9")

    def _raise_then_succeed(**kwargs):
        if "temperature" in kwargs:
            raise RuntimeError(
                "Error code: 400 - {'message': '`temperature` is deprecated for this model.'}"
            )
        return MagicMock()

    p.client.messages.create.side_effect = _raise_then_succeed

    # First call: fails on temperature, retries without it, succeeds.
    p._messages_create(model="claude-future-9", temperature=0.5, max_tokens=10)
    assert p.client.messages.create.call_count == 2, "should retry once without temperature"
    assert "claude-future-9" in p._temp_deprecated, "the lesson must be remembered"

    # Second call: temperature stripped up front, single call, no retry.
    p.client.messages.create.reset_mock()
    p._messages_create(model="claude-future-9", temperature=0.5, max_tokens=10)
    assert p.client.messages.create.call_count == 1


def test_a_non_temperature_400_is_NOT_swallowed() -> None:
    """The retry is scoped to the temperature case. A different 400 (bad model, etc.) must still
    surface — otherwise the fix would hide unrelated failures."""
    p = _provider("claude-sonnet-4-6")
    p.client.messages.create.side_effect = RuntimeError(
        "Error code: 400 - {'message': 'max_tokens: must be >= 1'}"
    )
    with pytest.raises(RuntimeError, match="max_tokens"):
        p._messages_create(model="claude-sonnet-4-6", temperature=0.0, max_tokens=0)
    assert p.client.messages.create.call_count == 1, "no bogus retry on an unrelated error"
