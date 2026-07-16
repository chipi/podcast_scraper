"""The o1/o3/gpt-5 series renamed `max_tokens`; sending the old name is a hard 400.

THE BUG (found in the bake-off): `gpt-5.4-mini` rejects `max_tokens` — it requires
`max_completion_tokens`. The summarization path already branched on this, but the EVIDENCE calls
(quote extraction, entailment) still sent `max_tokens`, so on gpt-5 every grounding call 400'd, fell
back to staged, 400'd again, and the arm produced 0 grounded quotes on all 18 episodes while
reporting "completed". The fix routes every call through `_token_kwarg`, so the rename cannot be
honoured in one place and forgotten in another.
"""

from __future__ import annotations

import pytest

from podcast_scraper import config as cfgmod
from podcast_scraper.providers.openai.openai_provider import OpenAIProvider

pytestmark = pytest.mark.unit


def _provider(model: str) -> OpenAIProvider:
    cfg = cfgmod.Config(
        rss="https://example.com/feed.xml",
        summary_provider="openai",
        openai_summary_model=model,
        openai_api_key="test-api-key-123",
    )
    return OpenAIProvider(cfg)


@pytest.mark.parametrize("model", ["gpt-5.4-mini", "gpt-5", "gpt-5-mini", "o1", "o3-mini"])
def test_new_series_uses_max_completion_tokens(model: str) -> None:
    assert _provider(model)._token_kwarg(10) == {
        "max_completion_tokens": 10
    }, f"{model} rejects max_tokens; sending it 400s every evidence call and zeroes grounding"


@pytest.mark.parametrize("model", ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"])
def test_older_models_keep_max_tokens(model: str) -> None:
    assert _provider(model)._token_kwarg(10) == {"max_tokens": 10}


def test_cleaning_model_is_honoured_separately_from_summary_model() -> None:
    """The cleaning call passes its own model — a gpt-5 cleaning model must get the new kwarg even
    when the summary model is older, and vice versa."""
    p = _provider("gpt-4o-mini")  # summary model is old...
    assert p._token_kwarg(100, model="gpt-5-mini") == {"max_completion_tokens": 100}
    assert p._token_kwarg(100, model="gpt-4o") == {"max_tokens": 100}


def test_gpt55_only_allows_default_temperature_and_self_heals() -> None:
    """gpt-5.5 rejects temperature!=1; gpt-5.4-mini accepts 0. The seed strips it up front for
    gpt-5.5, and an unknown rejecter self-heals from the API error."""
    from unittest.mock import MagicMock

    p = _provider("gpt-5.5")
    p.client = MagicMock()
    assert "gpt-5.5" in p._temp_fixed_at_default
    p._chat_create(model="gpt-5.5", temperature=0.0, max_completion_tokens=10)
    _, kwargs = p.client.chat.completions.create.call_args
    assert "temperature" not in kwargs

    # a model we did NOT seed rejects a non-default temperature -> retry + remember
    p2 = _provider("gpt-6-future")
    p2.client = MagicMock()

    def _raise_then_ok(**kw):
        if "temperature" in kw:
            raise RuntimeError(
                "400 unsupported_value: temperature does not support 0 ... only default (1)"
            )
        return MagicMock()

    p2.client.chat.completions.create.side_effect = _raise_then_ok
    p2._chat_create(model="gpt-6-future", temperature=0.0, max_completion_tokens=10)
    assert p2.client.chat.completions.create.call_count == 2
    assert "gpt-6-future" in p2._temp_fixed_at_default
