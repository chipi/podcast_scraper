"""The single cleaning classifier (ADR-137): ad / cameo / commercial / real, computed once and
shared by the LLM resolution call and the roster so cleaning is never replicated."""

from __future__ import annotations

from typing import List, Tuple

import pytest

from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment
from podcast_scraper.providers.ml.diarization.roster import classify_voices

pytestmark = pytest.mark.unit


def _diar(turns: List[Tuple[str, float, float]]) -> DiarizationResult:
    segs = [DiarizationSegment(start=s, end=e, speaker=spk) for spk, s, e in turns]
    return DiarizationResult(segments=segs, num_speakers=len({spk for spk, _, _ in turns}))


def test_cameo_by_talk_time() -> None:
    # < CAMEO_MAX_TALK_S (20s) of total speech → cameo; a substantive voice → real.
    c = classify_voices(_diar([("S0", 0, 40), ("S1", 40, 45)]))
    assert c.real == frozenset({"S0"})
    assert c.cameo == frozenset({"S1"})
    assert c.commercial == frozenset()


def test_commercial_by_ad_overlap() -> None:
    # A voice speaking mostly inside an ad region is commercial (checked before the cameo floor).
    c = classify_voices(_diar([("S0", 0, 40), ("S1", 100, 140)]), ad_intervals=[(100.0, 140.0)])
    assert "S1" in c.commercial
    assert c.real == frozenset({"S0"})
    assert "S1" not in c.cameo  # commercial wins over cameo even when brief


def test_real_is_everything_minus_noise() -> None:
    c = classify_voices(
        _diar([("S0", 0, 40), ("S1", 40, 45), ("S2", 100, 140)]),
        ad_intervals=[(100.0, 140.0)],
    )
    assert c.real == frozenset({"S0"})
    assert c.cameo == frozenset({"S1"})
    assert c.commercial == frozenset({"S2"})
    # the four sets partition the voices with no overlap
    assert not (c.real & c.cameo) and not (c.real & c.commercial) and not (c.cameo & c.commercial)


def test_no_ad_intervals_means_no_commercial() -> None:
    # Without ad regions, only cameo-vs-real is decidable; nothing is commercial.
    c = classify_voices(_diar([("S0", 0, 40), ("S1", 40, 45)]))
    assert c.commercial == frozenset()
    assert c.real == frozenset({"S0"}) and c.cameo == frozenset({"S1"})
