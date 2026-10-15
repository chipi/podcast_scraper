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
    # Real proportions (2026-10-15): an ad read is a small share of an episode — 40 s of 640 here;
    # the toy used to give the ad voice half the talk, which the talk-share guard now keeps real.
    c = classify_voices(_diar([("S0", 0, 600), ("S1", 600, 640)]), ad_intervals=[(600.0, 640.0)])
    assert "S1" in c.commercial
    assert c.real == frozenset({"S0"})
    assert "S1" not in c.cameo  # commercial wins over cameo even when brief


def test_real_is_everything_minus_noise() -> None:
    c = classify_voices(
        # S1 mid-episode: a voice heard only near an edge is the edge-ad rule's, not a cameo.
        _diar(
            [("S0", 0, 300), ("S1", 300, 305), ("S0", 305, 600), ("S2", 600, 640), ("S0", 640, 700)]
        ),
        ad_intervals=[(600.0, 640.0)],
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


def test_a_host_who_reads_the_ads_is_not_commercial() -> None:
    """A voice holding >= COMMERCIAL_MAX_TALK_SHARE of the talk stays real however much of it sits
    in ad regions (2026-10-15): the fixture's 3-minute episodes put 63% of a HOST's talk inside an
    over-wide pre-roll region. Prod: no commercial voice above 6.5% of its episode."""
    # Host 0-120 s, 90 s of it inside the ad region (75%); guest 120-200 s.
    c = classify_voices(_diar([("HOST", 0, 120), ("GUEST", 120, 200)]), ad_intervals=[(0.0, 90.0)])
    assert "HOST" in c.real and "HOST" not in c.commercial


def test_the_talk_share_bound_and_the_ad_fraction_bound() -> None:
    from podcast_scraper.providers.ml.diarization.roster import (
        _mostly_ads,
        COMMERCIAL_AD_FRACTION,
        COMMERCIAL_MAX_TALK_SHARE,
    )

    # Just under a quarter of the episode, all of it in ads: commercial.
    assert _mostly_ads(24.0, 24.0, 100.0)
    # A quarter of the episode: never commercial by overlap.
    assert not _mostly_ads(25.0, 25.0, 100.0)
    assert COMMERCIAL_MAX_TALK_SHARE == 0.25
    # The fraction bound still holds for a small voice: 59% in ads is not enough, 60% is.
    assert not _mostly_ads(5.9, 10.0, 100.0)
    assert _mostly_ads(6.0, 10.0, 100.0)
    assert COMMERCIAL_AD_FRACTION == 0.6
    # Nothing heard: not commercial.
    assert not _mostly_ads(0.0, 0.0, 100.0)
