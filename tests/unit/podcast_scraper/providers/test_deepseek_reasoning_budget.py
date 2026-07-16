"""A reasoning model needs token room for its reasoning BEFORE the answer.

THE BUG (found in the bake-off): deepseek-v4-flash emits a `reasoning_content` block before its
answer. The evidence stack called `score_entailment` with `max_tokens=10` — fine for a model that
replies "0.85", fatal for one that reasons first: the 10 tokens were spent on reasoning, `content`
came back EMPTY (finish_reason="length"), the parser saw nothing and returned 0.0. Every quote then
scored 0 entailment, nothing cleared the grounding threshold, and the GI invariant fired
"grounding produced NOTHING: N insights, 0 quotes" on every episode. A whole provider looked like it
could not ground, when really we had starved its reasoning.

There is no reasoning-off switch on these models (`reasoning_effort="none"` is ignored), so the fix
is headroom. These tests are deterministic — no API call — and guard both halves: reasoning models
get the headroom, non-reasoning models are left exactly as they were.
"""

from __future__ import annotations

import pytest

from podcast_scraper import config as cfgmod
from podcast_scraper.providers.deepseek.deepseek_provider import (
    _model_reasons,
    _REASONING_TOKEN_HEADROOM,
    DeepSeekProvider,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "model,reasons",
    [
        ("deepseek-v4-flash", True),
        ("deepseek-v4-pro", True),
        ("deepseek-v4-flash-2026-01-01", True),  # dated snapshot still matches
        ("deepseek-r1", True),
        ("deepseek-reasoner", True),
        ("deepseek-chat", False),  # the non-reasoning model must NOT get inflated budgets
        ("deepseek-coder", False),
    ],
)
def test_model_reasons_classification(model: str, reasons: bool) -> None:
    assert _model_reasons(model) is reasons


def _provider(model: str) -> DeepSeekProvider:
    cfg = cfgmod.Config(
        rss="https://example.com/feed.xml",
        summary_provider="deepseek",
        deepseek_summary_model=model,
        deepseek_api_key="test-api-key-123",
    )
    return DeepSeekProvider(cfg)


def test_reasoning_model_gets_headroom_on_a_tight_budget() -> None:
    """The 10-token entailment budget must grow, or content is always empty."""
    p = _provider("deepseek-v4-flash")
    assert p._is_reasoning_model is True
    assert p._evidence_max_tokens(10) == 10 + _REASONING_TOKEN_HEADROOM, (
        "a reasoning model handed max_tokens=10 spends it all on reasoning and returns empty "
        "content — the exact failure that disconnected the grounding stack"
    )


def test_non_reasoning_model_budget_is_UNCHANGED() -> None:
    """deepseek-chat does not reason, so inflating its budget would only waste tokens. The fix must
    touch reasoning models ONLY."""
    p = _provider("deepseek-chat")
    assert p._is_reasoning_model is False
    assert p._evidence_max_tokens(10) == 10
    assert p._evidence_max_tokens(300) == 300


def test_headroom_never_exceeds_the_api_cap() -> None:
    """DeepSeek caps chat max_tokens at 8192; the headroom must not push a large budget over it."""
    p = _provider("deepseek-v4-flash")
    assert p._evidence_max_tokens(8000) <= 8192
    assert p._evidence_max_tokens(7000) <= 8192
