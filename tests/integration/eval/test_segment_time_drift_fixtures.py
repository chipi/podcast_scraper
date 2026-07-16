"""Regression guard for transcript segment-time drift on the v3 fixtures (#1173, AC2/AC4).

Recomputes turn-boundary drift from a committed words-cache (no whisper — CI-cheap) and enforces
that the word-timestamp refinement keeps drift bounded and beats the pre-fix segment-level times.

**Why p95, not max, is the gate.** The fixtures are ``say``-rendered and several carry the
``asr_garble`` failure mode, so on those the whisper transcript diverges from the source text by
design — turn-boundary alignment (and whisper's own word times on garbled runs) get unreliable,
producing rare multi-second outliers that are a *measurement* artifact, not pipeline drift. p95 is
robust to them; the strict absolute AC2 bound (p95 ≤ 300 ms / max ≤ 1000 ms) is validated
separately on real prod audio via the DGX subset. Here we guard against **regression**.

Regenerate the cache with::

    python -m tests.integration.eval.segment_drift_harness --regen
"""

from __future__ import annotations

import pytest

from podcast_scraper.evaluation.segment_time_drift import pool_drift
from tests.integration.eval.segment_drift_harness import load_cache, measure_from_cache

pytestmark = pytest.mark.integration

MAX_POOLED_P95_MS = 500.0
MAX_POOLED_MEAN_MS = 300.0
MIN_POOLED_ALIGN_FRAC = 0.70  # boundaries whose word anchored to the transcript


@pytest.fixture(scope="module")
def measured():
    cache = load_cache()
    if not cache:
        pytest.skip("segment_drift_cache.json missing — run segment_drift_harness --regen")
    return measure_from_cache(cache)


def test_refined_drift_p95_bounded(measured) -> None:
    refined = pool_drift([r for r, _ in measured.values()])
    assert refined["boundaries"] > 0
    assert refined["p95_ms"] <= MAX_POOLED_P95_MS, refined
    assert refined["mean_ms"] <= MAX_POOLED_MEAN_MS, refined


def test_refinement_beats_segment_level(measured) -> None:
    refined = pool_drift([r for r, _ in measured.values()])
    unrefined = pool_drift([u for _, u in measured.values()])
    # The #1173 fix must cut pooled p95 to a fraction of the pre-fix segment-level drift.
    assert refined["p95_ms"] <= unrefined["p95_ms"] * 0.4, (refined, unrefined)
    assert refined["mean_ms"] < unrefined["mean_ms"], (refined, unrefined)


def test_boundaries_mostly_align(measured) -> None:
    total = sum(r.boundaries_total for r, _ in measured.values())
    matched = sum(r.boundaries_matched for r, _ in measured.values())
    assert total > 0
    assert matched / total >= MIN_POOLED_ALIGN_FRAC, f"only {matched}/{total} boundaries aligned"


def test_clean_fixtures_meet_strict_bound(measured) -> None:
    # On the non-garble fixtures the alignment is trustworthy, so the strict AC2 max holds there.
    clean = {"p01_e04", "p08_e01"}
    for stem in clean & set(measured):
        refined, _ = measured[stem]
        assert refined.max_ms <= 1000.0, f"{stem}: max {refined.max_ms:.0f}ms"
        assert refined.p95_ms <= 400.0, f"{stem}: p95 {refined.p95_ms:.0f}ms"
