"""The known-model allowlist: the fail-closed gate that would have stopped the sonnet-4-6 incident.

An eval arm (and, per the operator, any PROFILE) that names a cloud model we don't recognise must be
REJECTED before a cent is spent — with a "did you mean" suggestion — rather than run 18 episodes on
a wrong/fictional model and land it on the scoreboard. Local providers stay ungoverned (arbitrary
local weights, no cloud spend). These tests pin that contract, including the exact incident string.
"""

from __future__ import annotations

import pytest

from podcast_scraper.providers.known_models import (
    is_known_model,
    normalize_model_id,
    suggest_model,
    UnknownModelError,
    validate_model_or_raise,
    verify_served_model,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "provider,model",
    [
        ("anthropic", "claude-sonnet-4-5"),
        ("anthropic", "claude-sonnet-5"),
        ("anthropic", "claude-haiku-4-5"),
        ("anthropic", "claude-opus-4-8"),
        ("openai", "gpt-5.4-mini"),
        ("openai", "gpt-5.4-nano"),
        ("openai", "gpt-5.5"),
        ("gemini", "gemini-2.5-flash-lite"),
        ("mistral", "mistral-medium"),
        ("deepseek", "deepseek-v4-flash"),
        ("grok", "grok-4.3"),
    ],
)
def test_real_current_models_are_known(provider: str, model: str) -> None:
    assert is_known_model(provider, model), f"{provider}/{model} is real — must be on the allowlist"


def test_the_incident_string_is_rejected_with_a_suggestion() -> None:
    """THE INCIDENT. claude-sonnet-4-6 must fail closed and point at the model actually wanted."""
    assert is_known_model("anthropic", "claude-sonnet-4-6") is False
    assert suggest_model("anthropic", "claude-sonnet-4-6") == "claude-sonnet-4-5"
    with pytest.raises(UnknownModelError, match="claude-sonnet-4-6") as ei:
        validate_model_or_raise("anthropic", "claude-sonnet-4-6", context="bakeoff_arm")
    msg = str(ei.value)
    assert "claude-sonnet-4-5" in msg, "the error must suggest the intended model"
    assert "bakeoff_arm" in msg, "the error must name where the bad model came from"


def test_a_dated_or_pinned_suffix_still_matches_its_family() -> None:
    """A provider-pinned id (dated / -latest) must validate against its family entry."""
    assert normalize_model_id("claude-sonnet-4-5-20250219") == "claude-sonnet-4-5"
    assert normalize_model_id("mistral-medium-latest") == "mistral-medium"
    assert is_known_model("anthropic", "claude-sonnet-4-5-20250219")
    assert is_known_model("mistral", "mistral-medium-2505")


def test_local_providers_are_not_governed() -> None:
    """ollama / local weights are allow-all — validation must never block them."""
    assert is_known_model("ollama", "some-random-local:latest")
    assert is_known_model("tailnet_dgx_whisper", "whatever")
    validate_model_or_raise("ollama", "qwen3.5:35b")  # must not raise


def test_a_plain_typo_is_rejected() -> None:
    assert is_known_model("openai", "gpt-5.4-minii") is False
    assert is_known_model("gemini", "gemini-2.5-flash-litex") is False


def test_verify_served_model_flags_a_substitution() -> None:
    """Requested one model, provider served another → a mismatch reason (silent substitution)."""
    reason = verify_served_model("anthropic", "claude-sonnet-4-6", "claude-sonnet-5")
    assert reason is not None
    assert "claude-sonnet-5" in reason and "claude-sonnet-4-6" in reason


def test_verify_served_model_accepts_a_dated_pin_as_a_match() -> None:
    """A dated/pinned served id for the requested family is NOT a substitution."""
    assert (
        verify_served_model("anthropic", "claude-sonnet-4-5", "claude-sonnet-4-5-20250219") is None
    )
    assert verify_served_model("openai", "gpt-5.4-mini", "gpt-5.4-mini") is None
    # local provider → never flagged
    assert verify_served_model("ollama", "a", "b") is None
