"""Unit tests for ``DeBERTaNliScorer`` softmax calibration (#1106).

The cross-encoder returns raw logits in [contradiction, entailment, neutral]
order. Pre-fix the scorer returned the raw contradiction logit, so the enricher
thresholded an unbounded value at 0.5 and flagged ~23% of all cross-Person pairs
(0% precision on prod-v2). The scorer now softmaxes the three logits so
``.contradiction`` is a probability in [0, 1].

These tests stub the model so they need no ``[ml]`` extra ([[feedback_no_llm_in_ci]]).
"""

from __future__ import annotations

import asyncio
import math

from podcast_scraper.enrichment.scorers.nli import DeBERTaNliScorer


class _StubModel:
    """Stand-in for the CrossEncoder: returns fixed logits per predict call."""

    def __init__(self, logits: list[float]) -> None:
        self._logits = logits

    def predict(self, pairs: list[tuple[str, str]]) -> list[list[float]]:
        return [self._logits]


def _score(logits: list[float]):
    scorer = DeBERTaNliScorer()
    scorer._model = _StubModel(logits)  # skip lazy _load()
    return asyncio.run(scorer.score("a", "b"))


def test_scores_are_a_probability_distribution() -> None:
    s = _score([2.0, 0.0, -1.0])
    total = s.contradiction + s.entailment + s.neutral
    assert math.isclose(total, 1.0, abs_tol=1e-6)
    assert all(0.0 <= v <= 1.0 for v in (s.contradiction, s.entailment, s.neutral))


def test_softmax_matches_reference() -> None:
    logits = [2.0, 0.0, -1.0]  # [contradiction, entailment, neutral]
    exps = [math.exp(x) for x in logits]
    tot = sum(exps)
    s = _score(logits)
    assert math.isclose(s.contradiction, exps[0] / tot, abs_tol=1e-6)
    assert math.isclose(s.entailment, exps[1] / tot, abs_tol=1e-6)
    assert math.isclose(s.neutral, exps[2] / tot, abs_tol=1e-6)


def test_softmax_prevents_raw_logit_false_flag() -> None:
    # Regression for #1106: a small positive contradiction logit dwarfed by the
    # neutral logit must NOT flag. Pre-softmax the raw 0.6 exceeded the 0.5
    # threshold and produced a false positive; post-softmax neutral dominates.
    s = _score([0.6, 0.1, 4.0])
    assert s.contradiction < 0.5
    assert s.neutral > s.contradiction


def test_dominant_contradiction_logit_yields_high_probability() -> None:
    s = _score([6.0, 0.0, 0.0])
    assert s.contradiction > 0.9
