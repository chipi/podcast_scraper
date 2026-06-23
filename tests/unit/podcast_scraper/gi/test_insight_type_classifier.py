"""Unit tests for the rule-based insight_type classifier (RFC-097 v3.0 chunk 5).

Covers the schema enum ``claim | recommendation | observation | question
| unknown`` and the precedence ordering documented in the classifier
module's docstring.
"""

from __future__ import annotations

import pytest

from podcast_scraper.gi.insight_type_classifier import classify_insight_type

pytestmark = pytest.mark.unit


class TestQuestion:
    """Rule 1 — question. Most specific signal, evaluated first."""

    def test_question_mark_suffix(self) -> None:
        assert classify_insight_type("Is the trail safe?") == "question"

    def test_interrogative_first_word(self) -> None:
        assert classify_insight_type("Why are berms so important.") == "question"

    def test_interrogative_no_punctuation(self) -> None:
        assert classify_insight_type("How do you handle wet sections") == "question"

    def test_modal_interrogative_starter(self) -> None:
        assert classify_insight_type("Should you let air out before a wet ride?") == "question"

    def test_contracted_interrogative(self) -> None:
        # "Don't" lowercased + apostrophe stripped → "dont" — recognized.
        assert classify_insight_type("Don't you find the climb tedious?") == "question"

    def test_too_long_to_be_question_falls_through(self) -> None:
        """A 26-word sentence starting with ``How`` is not a question."""
        long_text = "How " + "filler " * 30
        # Falls through past the question heuristic. With no other signal
        # it lands in the conservative-default ``observation`` bucket.
        assert classify_insight_type(long_text) == "observation"

    def test_question_takes_precedence_over_recommendation(self) -> None:
        """A question containing 'should' is still a question."""
        assert classify_insight_type("Should we consider switching tires?") == "question"


class TestRecommendation:
    """Rule 2 — recommendation."""

    def test_should(self) -> None:
        assert classify_insight_type("You should brake earlier on berms.") == "recommendation"

    def test_must(self) -> None:
        assert (
            classify_insight_type("Riders must check pressure before every run.")
            == "recommendation"
        )

    def test_recommend_verb(self) -> None:
        assert (
            classify_insight_type("She recommends practicing on flat ground.") == "recommendation"
        )

    def test_advise_verb(self) -> None:
        assert classify_insight_type("Maya advises avoiding loose gravel.") == "recommendation"

    def test_consider(self) -> None:
        assert classify_insight_type("Consider running tubeless for grip.") == "recommendation"

    def test_avoid(self) -> None:
        assert classify_insight_type("Avoid braking mid-corner.") == "recommendation"

    def test_better_to(self) -> None:
        assert (
            classify_insight_type("It is better to walk technical sections than crash.")
            == "recommendation"
        )

    def test_contracted_shouldnt(self) -> None:
        assert classify_insight_type("You shouldn't ride without a helmet.") == "recommendation"


class TestClaim:
    """Rule 3 — claim. Strong assertions + numeric-evidence patterns."""

    def test_argued(self) -> None:
        assert classify_insight_type("She argued the new tires roll faster.") == "claim"

    def test_proved(self) -> None:
        assert classify_insight_type("Lab tests proved the casing failed at 20 PSI.") == "claim"

    def test_demonstrated(self) -> None:
        assert classify_insight_type("The data demonstrated improved cornering.") == "claim"

    def test_causes(self) -> None:
        assert classify_insight_type("Higher pressure causes more vibration.") == "claim"

    def test_percent_signal(self) -> None:
        assert classify_insight_type("Adoption grew 50% year over year.") == "claim"

    def test_pp_signal(self) -> None:
        assert classify_insight_type("Recall improved +13pp under NER pre-pass.") == "claim"

    def test_multiplier_signal(self) -> None:
        assert classify_insight_type("Latency dropped 2x with the new build.") == "claim"

    def test_dollar_amount(self) -> None:
        assert classify_insight_type("The acquisition closed at $1.2 billion.") == "claim"


class TestObservation:
    """Rule 4 — observation. Descriptive verbs + default for declarative content."""

    def test_noted_verb(self) -> None:
        assert (
            classify_insight_type("She noted the trail was drier than expected.") == "observation"
        )

    def test_observed_verb(self) -> None:
        assert classify_insight_type("Maya observed that the cadence held steady.") == "observation"

    def test_mentioned(self) -> None:
        assert (
            classify_insight_type("Liam mentioned the new tire compound briefly.") == "observation"
        )

    def test_discussed(self) -> None:
        assert (
            classify_insight_type("The hosts discussed the upcoming race weekend.") == "observation"
        )

    def test_default_for_plain_declarative(self) -> None:
        """No specific verb, no question, no modal — defaults to observation."""
        assert (
            classify_insight_type("Trail drainage matters for long-lasting tread.") == "observation"
        )

    def test_default_for_short_neutral_statement(self) -> None:
        assert classify_insight_type("Tire pressure has trade-offs.") == "observation"


class TestUnknown:
    """Rule 5 — unknown. Only empty / whitespace input."""

    def test_empty_string(self) -> None:
        assert classify_insight_type("") == "unknown"

    def test_whitespace_only(self) -> None:
        assert classify_insight_type("   \n  ") == "unknown"

    def test_none_safe(self) -> None:
        # Defensive: the upstream contract is ``str``, but a falsy value
        # shouldn't crash. classify_insight_type expects str; "" coverage
        # above is the contract guard. We don't claim to accept None here.
        # (Documented contract: str input only.)
        assert classify_insight_type("") == "unknown"


class TestPrecedence:
    """Cross-bucket precedence — question > recommendation > claim > observation."""

    def test_question_beats_recommendation(self) -> None:
        assert classify_insight_type("Should we recommend wider tires?") == "question"

    def test_recommendation_beats_claim(self) -> None:
        # Has both "recommend" and "shows" → recommendation wins.
        assert (
            classify_insight_type("The team recommends wider tires; data shows 50% better grip.")
            == "recommendation"
        )

    def test_claim_beats_observation(self) -> None:
        # Has "proved" + "noted" → claim wins.
        assert classify_insight_type("Tests proved the issue; she noted it earlier.") == "claim"

    def test_observation_default_beats_unknown(self) -> None:
        # Has no specific verb, but non-empty → observation, not unknown.
        assert classify_insight_type("The host wrapped up the conversation.") == "observation"
