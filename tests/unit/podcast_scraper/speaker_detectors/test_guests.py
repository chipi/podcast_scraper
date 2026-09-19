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


class TestTheCueMayFollowTheNameAcrossARoleClause:
    """Descriptions write `NAME, <job title>, CUE` — and every pattern only looked forward.

    MEASURED ON PRODUCTION DESCRIPTIONS. Of 136 episodes that end with no named speaker, 126 (93%)
    have a person-shaped name in the description and only 28 trip any existing cue. The phrasings
    carrying the two biggest broken feeds put the NAME FIRST and the verb after a job title:

        "Sarah Laszlo, senior director of Visa's machine learning platform, joins the AI Podcast"
        "Mike Pritchard, Director of Climate Simulation Research at NVIDIA, discusses how AI is..."
        "Elena Burger is joined by a16z's Andy McCall and Joe Schmidt"
        "Jack Altman joins Speedrun to discuss product-market fit"

    `INTERVIEW_TRAILING_PATTERNS` already handled a trailing cue, but glued to the name with no
    gap, so a job title in between made it unreachable.

    WHY `discusses` IS SAFE HERE AND NOT A CONTRADICTION. The same verb is a mentioned-only marker,
    and direction is what separates them: mentioned-only matches cue-BEFORE-name ("discusses Mike
    Pritchard" — he is the topic); this matches name-BEFORE-cue ("Mike Pritchard ... discusses" —
    he is speaking). They cannot fire on the same text in the same direction.

    END-TO-END EFFECT, real function, real names, no NER needed — asking about a KNOWN guest and
    about a KNOWN non-speaker (a `mentioned` person in the same episode):

        recall on known guests           302/1819 (16.6%)  ->  467/1819 (25.7%)
        false positives on non-speakers   43/3689 (1.2%)   ->   72/3689 (2.0%)

    +165 guests for +29 false candidates, a 5.7:1 trade. And a candidate is not a binding: it must
    still survive the voice resolver's third-person guard, which is precisely what refutes someone
    the speakers only talk about.
    """

    @pytest.mark.parametrize(
        "name,text",
        [
            ("Jack Altman", "Jack Altman joins Speedrun to discuss product-market fit."),
            (
                "Sarah Laszlo",
                "Sarah Laszlo, senior director of Visa's machine learning platform, "
                "joins the AI Podcast to discuss fraud prevention.",
            ),
            (
                "Mike Pritchard",
                "Mike Pritchard, Director of Climate Simulation Research at NVIDIA, "
                "discusses how AI is enhancing climate models.",
            ),
            ("Andy McCall", "Elena Burger is joined by a16z's Andy McCall and Joe Schmidt."),
            ("Dan Hendrycks", "Deep dive with Dan Hendrycks, a leading AI safety researcher."),
        ],
    )
    def test_real_production_phrasings_are_recognised(self, name: str, text: str) -> None:
        assert _is_likely_actual_guest(name, "", text) is True

    def test_the_topic_of_a_sentence_is_still_not_a_guest(self) -> None:
        """Direction is load-bearing: cue-before-name still means the person is the SUBJECT."""
        assert (
            _is_likely_actual_guest("Elon Musk", "", "The hosts discuss Elon Musk's lawsuit.")
            is False
        )
        assert (
            _is_likely_actual_guest("Andy Warhol", "", "An episode about Andy Warhol and fame.")
            is False
        )

    def test_the_gap_is_still_bounded(self) -> None:
        """Unbounded, one cue introduces every name in the paragraph — the #876 failure."""
        far = "Elon Musk" + (" and other topics" * 6) + ", joins the show."
        assert _is_likely_actual_guest("Elon Musk", "", far) is False
