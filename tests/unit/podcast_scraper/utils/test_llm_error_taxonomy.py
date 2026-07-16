"""Classify LLM API errors into retry / back-off / abandon — the heart of the resilience layer.

The old binary retryable-or-not check lumped every "quota" string into retryable, so the exact 400
that stopped our Anthropic account ("you have reached your specified API usage limits") got RETRIED
instead of hard-stopped. These tests pin the three-way split, using the real error strings the seven
providers actually return, so a misclassification (retrying an out-of-money error, or backing off
forever on a bad request) shows up here.
"""

from __future__ import annotations

import pytest

from podcast_scraper.utils.llm_error_taxonomy import (
    classify_llm_error,
    LLMErrorClass,
    terminal_message,
)

pytestmark = pytest.mark.unit


class _StatusError(Exception):
    """An SDK-style error carrying a numeric status_code, like the real provider SDKs."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


@pytest.mark.parametrize(
    "message,expected",
    [
        # TERMINAL — out of money / access. The class the old code missed.
        ("400 - You have reached your specified API usage limits", LLMErrorClass.TERMINAL),
        ("429 - insufficient_quota: You exceeded your current quota", LLMErrorClass.TERMINAL),
        ("402 Insufficient Balance", LLMErrorClass.TERMINAL),
        ("billing_hard_limit_reached", LLMErrorClass.TERMINAL),
        ("Your credit balance is too low to access the API", LLMErrorClass.TERMINAL),
        ("invalid x-api-key", LLMErrorClass.TERMINAL),
        ("401 Unauthorized", LLMErrorClass.TERMINAL),
        ("permission denied for this model", LLMErrorClass.TERMINAL),
        # RETRYABLE OVERLOAD — busy endpoint. Back off.
        ("503 Service Unavailable", LLMErrorClass.RETRYABLE_OVERLOAD),
        ("overloaded_error: Overloaded", LLMErrorClass.RETRYABLE_OVERLOAD),  # Anthropic 529
        ("500 Internal Server Error", LLMErrorClass.RETRYABLE_OVERLOAD),
        ("502 Bad Gateway", LLMErrorClass.RETRYABLE_OVERLOAD),
        ("504 Gateway Timeout", LLMErrorClass.RETRYABLE_OVERLOAD),
        ("The model is temporarily unavailable, try again", LLMErrorClass.RETRYABLE_OVERLOAD),
        # RETRYABLE RATE LIMIT — transient. Honour Retry-After.
        ("429 Too Many Requests: rate limit exceeded", LLMErrorClass.RETRYABLE_RATE_LIMIT),
        ("RESOURCE_EXHAUSTED", LLMErrorClass.RETRYABLE_RATE_LIMIT),  # Gemini transient
        ("quota exceeded for this minute", LLMErrorClass.RETRYABLE_RATE_LIMIT),
        # NON_RETRYABLE — our fault.
        ("400 Bad Request: unsupported parameter 'foo'", LLMErrorClass.NON_RETRYABLE),
        ("404 model not found", LLMErrorClass.NON_RETRYABLE),
        ("422 Unprocessable Entity", LLMErrorClass.NON_RETRYABLE),
    ],
)
def test_classification_by_message(message: str, expected: LLMErrorClass) -> None:
    assert classify_llm_error(Exception(message)) is expected


def test_terminal_wins_over_ratelimit_lookalike() -> None:
    """OpenAI's insufficient_quota arrives as a 429 — the SAME status as a transient rate limit.
    It must classify TERMINAL, not retry, or we loop on an out-of-money key."""
    err = _StatusError("You exceeded your current quota, insufficient_quota", status_code=429)
    assert classify_llm_error(err) is LLMErrorClass.TERMINAL


def test_status_code_attribute_is_read_not_just_the_string() -> None:
    """SDK errors often stringify without the digits; the numeric status must still drive it."""
    assert classify_llm_error(_StatusError("boom", 503)) is LLMErrorClass.RETRYABLE_OVERLOAD
    assert classify_llm_error(_StatusError("boom", 402)) is LLMErrorClass.TERMINAL
    assert classify_llm_error(_StatusError("boom", 403)) is LLMErrorClass.TERMINAL


def test_connection_errors_are_retryable_overload() -> None:
    for msg in ("Connection reset by peer", "Read timed out", "broken pipe"):
        assert classify_llm_error(Exception(msg)) is LLMErrorClass.RETRYABLE_OVERLOAD


def test_terminal_message_is_human_readable_and_names_the_cause() -> None:
    msg = terminal_message(
        "anthropic", Exception("You have reached your specified API usage limits")
    )
    assert "anthropic" in msg
    assert "no budget/credit" in msg
    assert "NOT retryable" in msg
    auth = terminal_message("openai", Exception("invalid api key"))
    assert "invalid or lacks access" in auth
