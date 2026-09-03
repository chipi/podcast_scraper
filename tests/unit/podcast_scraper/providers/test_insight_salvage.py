"""A truncated insight list must cost us the last line, not the episode.

The guardrail is right that a truncated response is structurally unusable — for JSON. For a
newline-delimited list it is not: the cut lands in the final line and every earlier one is intact.
Re-raising discarded the whole episode to the stub fallback, which hit 1 of 3 eval episodes and
8 of 15 probe runs.

These tests pin both halves: the recoverable case is salvaged, and everything else still raises.
"""

from __future__ import annotations

from podcast_scraper.providers import insight_salvage
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


# --- #1919: the ceiling cut must not be a positional head-slice --------------------------
# Providers used to end with ``cleaned[:max_insights]``. Models emit insights in transcript
# order (measured Pearson(rank, position_hint) = 0.904), so that kept the first third of the
# episode and discarded everything after it. Stride sampling preserves coverage end to end.


def test_take_within_ceiling_passes_through_when_under() -> None:
    items = ["a", "b", "c"]
    assert insight_salvage.take_within_ceiling(items, 5) == items
    assert insight_salvage.take_within_ceiling(items, 3) == items


def test_take_within_ceiling_spans_the_whole_list() -> None:
    """The regression: first AND last must survive, which a head-slice never does."""
    items = [str(i) for i in range(30)]
    kept = insight_salvage.take_within_ceiling(items, 10)
    assert len(kept) == 10
    assert kept[0] == "0"
    assert kept[-1] == "29"  # head-slice would have ended at "9"


def test_take_within_ceiling_is_not_a_head_slice() -> None:
    items = [str(i) for i in range(73)]
    kept = insight_salvage.take_within_ceiling(items, 25)
    assert kept != items[:25]
    # Coverage of the back half is the whole point: a head-slice yields zero of these.
    back_half = [k for k in kept if int(k) >= 36]
    assert len(back_half) >= 10


def test_take_within_ceiling_preserves_order_and_uniqueness() -> None:
    items = [str(i) for i in range(50)]
    kept = insight_salvage.take_within_ceiling(items, 17)
    assert kept == sorted(kept, key=lambda s: items.index(s))
    assert len(set(kept)) == len(kept)


def test_take_within_ceiling_degenerate_ceilings() -> None:
    items = ["a", "b", "c", "d"]
    assert insight_salvage.take_within_ceiling(items, 0) == []
    assert insight_salvage.take_within_ceiling(items, -3) == []
    assert insight_salvage.take_within_ceiling(items, 1) == ["a"]
    assert insight_salvage.take_within_ceiling([], 5) == []
