"""The judge that decides KEEP or DROP was being handed a bare sentence.

`classify_insights(insights: List[str])` — that is all it ever saw. No quote, no speaker, no
timestamp. And it ran BEFORE grounding, so the quotes did not exist yet: the same defect as ADR-110,
one layer up — a decision taken at a point in the pipeline where its evidence has not been computed.

Its own rubric asks for things it could not possibly see:

    "a substantive position a NAMED PERSON took"      <- cannot see the speaker
    "a real disagreement BETWEEN SPEAKERS"            <- cannot see the speakers
    "an AD or sponsor read"                           <- we KNOW which spans are ads, and never said

On 18 episodes it deleted 358 of gemini's 756 insights on that basis. An insight quoted verbatim by
the host and one the episode never says looked identical to it.
"""

from __future__ import annotations

from typing import Any, List

import pytest

from podcast_scraper.gi.value_gate import (
    apply_value_gate,
    format_insight_for_judging,
    InsightEvidence,
)

pytestmark = pytest.mark.unit


class _Judge:
    """Records exactly what it was shown, and tiers on demand."""

    def __init__(self, tiers: List[int]) -> None:
        self.tiers = tiers
        self.saw: List[str] = []

    def classify_insights(self, insights: List[str]) -> List[int]:
        self.saw = list(insights)
        return self.tiers


def _cfg() -> Any:
    class C:
        gi_value_gate_enabled = True
        gi_value_gate_min_tier = 2
        gi_value_gate_provider = None

    return C()


class TestTheJudgeIsShownTheEvidence:
    def test_the_quote_and_the_speaker_reach_the_judge(self) -> None:
        judge = _Judge([3])
        apply_value_gate(
            [("Bubbles are not all bad.", "claim")],
            provider=judge,
            cfg=_cfg(),
            evidence=[
                InsightEvidence(
                    quote="I do think bubbles and bursts get a bad rap.",
                    speaker="Robert Armstrong",
                    voice_type=None,
                )
            ],
        )
        shown = judge.saw[0]
        assert "bad rap" in shown, "the judge never saw the quote that grounds the claim"
        assert "Robert Armstrong" in shown, "the rubric asks who said it; tell it who said it"

    def test_an_UNSUPPORTED_insight_is_marked_as_such(self) -> None:
        """The strongest FILLER signal there is, and it was invisible: the episode never says it."""
        judge = _Judge([3])
        apply_value_gate(
            [("The Fed will cut rates in March.", "claim")],
            provider=judge,
            cfg=_cfg(),
            evidence=[None],
        )
        assert "EVIDENCE: NONE" in judge.saw[0]

    def test_an_AD_read_is_labelled_an_ad(self) -> None:
        shown = format_insight_for_judging(
            "The games are different from other puzzle games.",
            InsightEvidence(
                quote="If you play our games, you probably know there is something different.",
                speaker=None,
                voice_type="commercial",
            ),
        )
        assert "[commercial]" in shown, (
            "ad copy is WRITTEN to be quotable — the most fluent false insight available — and the "
            "judge was left to guess"
        )

    def test_without_evidence_it_behaves_exactly_as_before(self) -> None:
        """Callers that cannot supply evidence (and the airgapped path) are unchanged."""
        judge = _Judge([3, 1])
        kept = apply_value_gate(
            [("a", "claim"), ("b", "claim")], provider=judge, cfg=_cfg(), evidence=None
        )
        assert judge.saw == ["a", "b"]
        assert kept == [("a", "claim")]


class TestTheQuotesStayWithTheirInsights:
    def test_dropping_an_insight_drops_ITS_quotes_too(self) -> None:
        """specs and quote-lists are index-aligned everywhere downstream. A quote left attached to
        the wrong insight is a fabricated attribution — worse than the filler we were removing."""
        from podcast_scraper.gi.grounding import GroundedQuote
        from podcast_scraper.gi.pipeline import _gate_on_evidence

        specs = [("keep me", "claim"), ("drop me", "claim"), ("keep me too", "claim")]
        quotes: List[List[Any]] = [
            [GroundedQuote(char_start=0, char_end=4, text="Q1", qa_score=0.9, nli_score=0.9)],
            [GroundedQuote(char_start=5, char_end=9, text="Q2", qa_score=0.9, nli_score=0.9)],
            [GroundedQuote(char_start=10, char_end=14, text="Q3", qa_score=0.9, nli_score=0.9)],
        ]
        out_specs, out_quotes = _gate_on_evidence(
            specs,
            quotes,
            cfg=_cfg(),
            provider=_Judge([3, 0, 3]),
            transcript_text="Q1 Q2 Q3",
            transcript_segments=None,
            pipeline_metrics=None,
        )
        assert [t for t, _ in out_specs] == ["keep me", "keep me too"]
        assert [q[0].text for q in out_quotes] == ["Q1", "Q3"], (
            "the surviving insights kept the wrong quotes — an insight is now supported by "
            "evidence for a different claim"
        )

    def test_duplicate_insight_text_does_not_confuse_the_pairing(self) -> None:
        """Two identical sentences with DIFFERENT evidence. Matching on content would mis-pair."""
        from podcast_scraper.gi.grounding import GroundedQuote
        from podcast_scraper.gi.pipeline import _gate_on_evidence

        specs = [("same", "claim"), ("same", "claim")]
        quotes: List[List[Any]] = [
            [GroundedQuote(char_start=0, char_end=2, text="FIRST", qa_score=0.9, nli_score=0.9)],
            [GroundedQuote(char_start=3, char_end=5, text="SECOND", qa_score=0.9, nli_score=0.9)],
        ]
        _, out_quotes = _gate_on_evidence(
            specs,
            quotes,
            cfg=_cfg(),
            provider=_Judge([0, 3]),
            transcript_text="ab cd",
            transcript_segments=None,
            pipeline_metrics=None,
        )
        assert [q[0].text for q in out_quotes] == ["SECOND"]


def test_the_gate_no_longer_runs_BEFORE_grounding() -> None:
    """It cannot grade the evidence if it runs before the evidence is computed."""
    import inspect

    from podcast_scraper.gi import pipeline as gi_pipeline

    resolve_src = inspect.getsource(gi_pipeline._resolve_insight_specs)
    assert "apply_value_gate(" not in resolve_src, (
        "the value gate is back in _resolve_insight_specs, which runs BEFORE grounding — it is "
        "grading a bare sentence while its rubric asks who said it"
    )

    build_src = inspect.getsource(gi_pipeline.build_artifact)
    assert "_gate_on_evidence(" in build_src
