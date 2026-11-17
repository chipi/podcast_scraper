"""AppInsight.grounding_confidence (GI 4.0): additive, optional, 0 to 1; confidence unchanged."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from podcast_scraper.server.schemas import AppInsight

pytestmark = pytest.mark.unit


def test_absent_unless_the_grounding_build_fills_it() -> None:
    ins = AppInsight(id="insight:1", text="x", grounded=False)
    assert ins.grounding_confidence is None
    assert ins.confidence is None


def test_carries_the_best_entailment_score_beside_confidence() -> None:
    ins = AppInsight(
        id="insight:1", text="x", grounded=True, confidence=0.4, grounding_confidence=0.91
    )
    assert ins.grounding_confidence == 0.91
    assert ins.confidence == 0.4


@pytest.mark.parametrize("bad", [-0.01, 1.01])
def test_refuses_a_score_outside_zero_to_one(bad: float) -> None:
    with pytest.raises(ValidationError):
        AppInsight(id="insight:1", text="x", grounded=True, grounding_confidence=bad)
