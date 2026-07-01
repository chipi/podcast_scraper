"""Unit tests for pairwise judging primitives.

These cover the transport-agnostic pieces of :mod:`podcast_scraper.evaluation.pairwise`
— types, prompt shape, parser (with anti-position-bias resolution), scoring
math, contest logic, and aggregation. Wire-level pairwise judge calls go
through the existing scalar transports (OllamaChatJudge / VllmChatJudge)
so no HTTP mocking is needed here — those transports have their own tests.
"""

from __future__ import annotations

import pytest

from podcast_scraper.evaluation.pairwise import (
    build_pairwise_user_message,
    is_contested,
    PAIRWISE_RUBRIC,
    pairwise_verdict_to_score,
    PairwiseVerdict,
    parse_pairwise_verdict,
    prepare_slots,
    summarize_pairwise_run,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Prompt shape
# ---------------------------------------------------------------------------


def test_build_pairwise_user_message_includes_both_slots_without_labeling_candidate() -> None:
    """Judge must NOT know which slot holds the candidate — that's the whole
    point of position randomization. The prompt names 'Summary A' /
    'Summary B', not 'candidate' / 'reference'."""
    msg = build_pairwise_user_message(
        rubric=PAIRWISE_RUBRIC,
        transcript="a transcript",
        slot_a_summary="alpha summary",
        slot_b_summary="beta summary",
    )
    assert "### Summary A" in msg
    assert "### Summary B" in msg
    assert "alpha summary" in msg
    assert "beta summary" in msg
    # Must NOT leak identity of either slot. ``preference`` in the JSON
    # schema legitimately contains "reference" as a substring — check for
    # the leak-shaped phrases, not the exact word.
    lower = msg.lower()
    assert "candidate" not in lower
    assert "silver" not in lower
    assert "reference summary" not in lower  # the actual leak we care about


def test_build_pairwise_user_message_truncates_long_transcript() -> None:
    long = "x" * 40_000
    msg = build_pairwise_user_message(
        rubric=PAIRWISE_RUBRIC,
        transcript=long,
        slot_a_summary="A",
        slot_b_summary="B",
        max_transcript_chars=1_000,
    )
    # 1000-char cap → the transcript block starts with the truncated slice
    # and doesn't grow past the cap plus a fixed prompt overhead.
    assert msg.count("x") == 1_000


# ---------------------------------------------------------------------------
# Parser + position resolution
# ---------------------------------------------------------------------------


def test_parse_pairwise_verdict_candidate_slot_A_pref_A_becomes_candidate() -> None:
    verdict = parse_pairwise_verdict(
        '{"preference": "A", "magnitude": 3, "rationale": "clearer synthesis"}',
        candidate_slot="A",
    )
    assert verdict.preference == "candidate"
    assert verdict.magnitude == 3
    assert verdict.rationale == "clearer synthesis"


def test_parse_pairwise_verdict_candidate_slot_A_pref_B_becomes_silver() -> None:
    verdict = parse_pairwise_verdict(
        '{"preference": "B", "magnitude": 2, "rationale": "silver had more coverage"}',
        candidate_slot="A",
    )
    assert verdict.preference == "silver"


def test_parse_pairwise_verdict_candidate_slot_B_pref_A_becomes_silver() -> None:
    """Anti-position-bias: candidate in slot B, judge preferred slot A →
    judge preferred the silver, not the candidate."""
    verdict = parse_pairwise_verdict(
        '{"preference": "A", "magnitude": 4, "rationale": "silver was crisper"}',
        candidate_slot="B",
    )
    assert verdict.preference == "silver"
    assert verdict.magnitude == 4


def test_parse_pairwise_verdict_candidate_slot_B_pref_B_becomes_candidate() -> None:
    verdict = parse_pairwise_verdict(
        '{"preference": "B", "magnitude": 5, "rationale": "candidate was decisive"}',
        candidate_slot="B",
    )
    assert verdict.preference == "candidate"
    assert verdict.magnitude == 5


def test_parse_pairwise_verdict_tie_resets_magnitude_to_zero() -> None:
    """A tie with reported magnitude=4 is a contradiction — treat as 0.
    Same discipline as the scalar parser: broken judge replies get
    normalized to a safe value only when the semantics are contradictory."""
    verdict = parse_pairwise_verdict(
        '{"preference": "tie", "magnitude": 4, "rationale": "indistinguishable"}',
        candidate_slot="A",
    )
    assert verdict.preference == "tie"
    assert verdict.magnitude == 0


def test_parse_pairwise_verdict_strips_markdown_fences() -> None:
    """Some judges wrap JSON in ```json ... ``` fences; the parser strips
    them so a common LLM habit doesn't become a JSONDecodeError."""
    fenced = '```json\n{"preference": "A", "magnitude": 3, "rationale": "yes"}\n```'
    verdict = parse_pairwise_verdict(fenced, candidate_slot="A")
    assert verdict.preference == "candidate"
    assert verdict.magnitude == 3


def test_parse_pairwise_verdict_rejects_bad_preference() -> None:
    with pytest.raises(ValueError, match="preference must be"):
        parse_pairwise_verdict(
            '{"preference": "either", "magnitude": 3, "rationale": ""}',
            candidate_slot="A",
        )


def test_parse_pairwise_verdict_rejects_out_of_range_magnitude() -> None:
    with pytest.raises(ValueError, match="magnitude for A/B preference must be 1-5"):
        parse_pairwise_verdict(
            '{"preference": "A", "magnitude": 7, "rationale": ""}',
            candidate_slot="A",
        )


def test_parse_pairwise_verdict_rejects_missing_magnitude() -> None:
    with pytest.raises(ValueError, match="missing 'magnitude'"):
        parse_pairwise_verdict(
            '{"preference": "A", "rationale": ""}',
            candidate_slot="A",
        )


# ---------------------------------------------------------------------------
# Score encoding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "verdict,expected",
    [
        (PairwiseVerdict("candidate", 1, ""), 0.6),
        (PairwiseVerdict("candidate", 5, ""), 1.0),
        (PairwiseVerdict("silver", 1, ""), 0.4),
        (PairwiseVerdict("silver", 5, ""), 0.0),
        (PairwiseVerdict("tie", 0, ""), 0.5),
    ],
)
def test_pairwise_verdict_to_score_covers_full_range(verdict, expected) -> None:
    assert pairwise_verdict_to_score(verdict) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Contest logic
# ---------------------------------------------------------------------------


def test_contested_when_judges_pick_opposite_parties() -> None:
    a = PairwiseVerdict("candidate", 3, "")
    b = PairwiseVerdict("silver", 3, "")
    assert is_contested(a, b) is True


def test_not_contested_when_judges_agree_on_party_even_at_different_magnitudes() -> None:
    """Magnitude disagreement is NOT a contest — the direction is what matters."""
    a = PairwiseVerdict("candidate", 1, "")
    b = PairwiseVerdict("candidate", 5, "")
    assert is_contested(a, b) is False


def test_not_contested_when_one_side_ties() -> None:
    """A tie isn't a directional statement — pair with a non-tie doesn't contest."""
    a = PairwiseVerdict("tie", 0, "")
    b = PairwiseVerdict("candidate", 4, "")
    assert is_contested(a, b) is False


# ---------------------------------------------------------------------------
# Slot assignment
# ---------------------------------------------------------------------------


def test_prepare_slots_returns_deterministic_slot_for_same_episode_id() -> None:
    """Same episode id + same process → same slot. Enables replay + debugging;
    per-process randomization comes from Python's hash randomization across
    runs."""
    slot1, _, _ = prepare_slots(episode_id="ep-42", candidate_summary="C", silver_summary="S")
    slot2, _, _ = prepare_slots(episode_id="ep-42", candidate_summary="C", silver_summary="S")
    assert slot1 == slot2


def test_prepare_slots_swaps_slot_b_correctly() -> None:
    """When the assigned slot is B, the candidate goes into the B slot and
    the silver goes into A — the caller receives them as (slot, slot_a_content,
    slot_b_content)."""
    slot, slot_a, slot_b = prepare_slots(
        episode_id="anything", candidate_summary="CANDIDATE", silver_summary="SILVER"
    )
    if slot == "A":
        assert slot_a == "CANDIDATE"
        assert slot_b == "SILVER"
    else:
        assert slot_a == "SILVER"
        assert slot_b == "CANDIDATE"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_summarize_pairwise_run_reports_win_rate_over_non_ties() -> None:
    verdicts = [
        PairwiseVerdict("candidate", 3, ""),
        PairwiseVerdict("candidate", 4, ""),
        PairwiseVerdict("silver", 2, ""),
        PairwiseVerdict("tie", 0, ""),
    ]
    summary = summarize_pairwise_run(verdicts)
    assert summary["n"] == 4
    # 2 candidate wins / 3 non-tie decisions
    assert summary["win_rate"] == pytest.approx(2 / 3)
    # 1 tie of 4 total
    assert summary["tie_rate"] == pytest.approx(0.25)
    # 1 magnitude>=4 of 4 total
    assert summary["decisive_rate"] == pytest.approx(0.25)


def test_summarize_pairwise_run_handles_all_ties_win_rate_none() -> None:
    verdicts = [
        PairwiseVerdict("tie", 0, ""),
        PairwiseVerdict("tie", 0, ""),
    ]
    summary = summarize_pairwise_run(verdicts)
    assert summary["win_rate"] is None
    assert summary["tie_rate"] == pytest.approx(1.0)
    assert summary["mean_score"] == pytest.approx(0.5)


def test_summarize_pairwise_run_handles_empty_list() -> None:
    summary = summarize_pairwise_run([])
    assert summary["n"] == 0
    assert summary["mean_score"] is None
    assert summary["win_rate"] is None
