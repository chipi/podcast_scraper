"""A truncated insight list must cost us the last line, not the episode.

The guardrail is right that a truncated response is structurally unusable — for JSON. For a
newline-delimited list it is not: the cut lands in the final line and every earlier one is intact.
Re-raising discarded the whole episode to the stub fallback, which hit 1 of 3 eval episodes and
8 of 15 probe runs.

These tests pin both halves: the recoverable case is salvaged, and everything else still raises.
"""

from __future__ import annotations

from podcast_scraper.providers.guardrails.chat import (
    REASON_CHAT_BAD_JSON,
    REASON_CHAT_EMPTY,
    REASON_CHAT_FINISH_LENGTH,
)
from podcast_scraper.providers.guardrails.exceptions import GuardrailViolation
from podcast_scraper.providers.insight_salvage import salvage_truncated_lines

TRUNCATED = (
    "OpenAI renegotiated its Microsoft deal, removing revenue sharing.\n"
    "Amazon invested $50 billion and will sell OpenAI models via Bedrock.\n"
    "Senior figures tied to Stargate have left for Meta.\n"
    "The company is pivoting toward an ad-supported tier that wou"  # cut mid-word
)


def _violation(reason: str) -> GuardrailViolation:
    return GuardrailViolation("gemini", reason, "summary")


def test_length_truncation_keeps_the_complete_lines() -> None:
    out = salvage_truncated_lines(_violation(REASON_CHAT_FINISH_LENGTH), TRUNCATED)
    assert out is not None
    lines = out.splitlines()
    assert len(lines) == 3, "the partial final line must be dropped, the rest kept"
    assert "ad-supported tier that wou" not in out
    assert "Amazon invested $50 billion" in out


def test_other_guardrail_reasons_still_raise() -> None:
    """Only length truncation is recoverable. Bad JSON and empty content are not."""
    assert salvage_truncated_lines(_violation(REASON_CHAT_BAD_JSON), TRUNCATED) is None
    assert salvage_truncated_lines(_violation(REASON_CHAT_EMPTY), TRUNCATED) is None


def test_empty_content_is_not_salvageable() -> None:
    assert salvage_truncated_lines(_violation(REASON_CHAT_FINISH_LENGTH), "") is None
    assert salvage_truncated_lines(_violation(REASON_CHAT_FINISH_LENGTH), None) is None


def test_single_truncated_line_is_not_salvageable() -> None:
    """One line, itself cut off, tells us nothing reliable — do not invent an insight from it."""
    out = salvage_truncated_lines(_violation(REASON_CHAT_FINISH_LENGTH), "OpenAI renegotia")
    assert out is None


def test_blank_tail_does_not_produce_an_empty_result() -> None:
    out = salvage_truncated_lines(_violation(REASON_CHAT_FINISH_LENGTH), "\n\n   \n")
    assert out is None
