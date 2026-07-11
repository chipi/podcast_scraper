"""Isolated unit tests for speaker_detectors.guests (E1, RFC-059).

Imports the submodule directly to exercise the guest-intent filter that decides
whether a detected PERSON is an actual guest or merely mentioned in passing.
"""

from __future__ import annotations

import pytest

from podcast_scraper.speaker_detectors.guests import (
    _has_interview_indicator,
    _has_mentioned_only_indicator,
    _is_likely_actual_guest,
    is_introduced_guest,
)

pytestmark = pytest.mark.unit


def test_is_introduced_guest_strict_intro_filter() -> None:
    # Named with an adjacent interview cue -> a guest.
    assert is_introduced_guest("Jane Doe", "welcome to the show. My guest is Jane Doe today.")
    assert is_introduced_guest("Nic Harrigan", "I'm joined by Nic Harrigan to talk quantum.")
    # Mononym / single-token noise -> never (drops "Ezra", "Kevin", "Trump", "RN").
    assert not is_introduced_guest("Ezra", "my guest is Ezra today")
    assert not is_introduced_guest("Trump", "my guest is Trump")
    # A First-Last name only MENTIONED (cue far away or absent) -> not introduced.
    assert not is_introduced_guest(
        "Elon Musk", "today we discuss what Elon Musk said about rockets"
    )
    filler = "and then a long stretch of unrelated talk " * 3
    assert not is_introduced_guest("Jane Doe", f"my guest is someone else, {filler} and Jane Doe")


def test_interview_indicator_detected() -> None:
    assert _has_interview_indicator("Jane Doe", "Interview with Jane Doe") is True
    assert _has_interview_indicator("Jane Doe", "featuring Jane Doe") is True
    assert _has_interview_indicator("Jane Doe", "A quiet morning") is False


def test_mentioned_only_indicator_detected() -> None:
    assert _has_mentioned_only_indicator("Jane Doe", "discussing Jane Doe") is True
    assert _has_mentioned_only_indicator("Jane Doe", "analysis of Jane Doe") is True
    assert _has_mentioned_only_indicator("Jane Doe", "Jane Doe joins us") is False


def test_guest_when_interview_indicator_present() -> None:
    assert _is_likely_actual_guest("Jane Doe", "Interview with Jane Doe", None) is True


def test_not_guest_when_only_mentioned() -> None:
    assert _is_likely_actual_guest("Jane Doe", "Episode discussing Jane Doe", None) is False


def test_not_guest_without_any_indicator() -> None:
    # Default is conservative: no interview cue → not treated as a guest.
    assert _is_likely_actual_guest("Jane Doe", "Random title", "Random description") is False


def test_interview_indicator_wins_over_mentioned_in_combined_text() -> None:
    # Title carries the interview cue; description merely mentions — interview wins.
    assert (
        _is_likely_actual_guest(
            "Jane Doe",
            "Conversation with Jane Doe",
            "Earlier we discussed Jane Doe",
        )
        is True
    )
