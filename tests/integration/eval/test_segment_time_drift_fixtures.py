"""Regression guard AND AC2 verification for transcript segment-time drift on the v3 fixtures
(#1173, AC2/AC4).

Recomputes turn-boundary drift from a committed words-cache (no whisper at test time — CI-cheap) and
enforces that the word-timestamp refinement holds the **strict AC2 bound on the clean fixtures** and
beats the pre-fix segment-level times.

The cache is transcribed with **large-v3** — the prod transcription model — so the word timestamps
are the ones production emits (~50 ms accurate). With ``base.en`` (a small model, ~320 ms word
granularity) the clean-fixture p95 was ~324 ms; large-v3 brings it to ~120–180 ms, which is what
makes the **strict AC2 bound (p95 ≤ 300 ms / max ≤ 1000 ms) genuinely met** — verified here, not
deferred to a separate DGX run.

**Why the clean subset is the AC2 gate.** The fixtures are ``say``-rendered and several carry the
``asr_garble`` failure mode, so on those the whisper transcript diverges from the source text by
design — turn-boundary alignment (and whisper's own word times on garbled runs) get unreliable,
producing multi-second outliers that are a *measurement* artifact, not pipeline drift. The strict
AC2 bound is asserted on the trustworthy (non-garble) fixtures; the pooled bound is a coarser
regression ceiling whose tail is dominated by those garble artifacts.

Regenerate the cache with (defaults to large-v3)::

    python -m tests.integration.eval.segment_drift_harness --regen
"""

from __future__ import annotations

import pytest

from podcast_scraper.evaluation.segment_time_drift import pool_drift
from tests.integration.eval.segment_drift_harness import load_cache, measure_from_cache

pytestmark = pytest.mark.integration

# Pooled = clean + garble. Its p95 tail is dominated by the asr-garble fixtures' alignment
# artifacts (multi-second, not pipeline drift), so this is a coarse regression ceiling, NOT the AC2
# gate — that is enforced strictly on the clean subset below. 600 clears the large-v3 garble tail
# (pooled p95 ~514 ms) with headroom; the refinement-beats-segment-level ratio is the real guard.
MAX_POOLED_P95_MS = 600.0
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


def test_clean_fixtures_meet_strict_ac2_bound(measured) -> None:
    """AC2, verified: on the trustworthy (non-garble) fixtures the large-v3 word timestamps hold
    the strict p95 ≤ 300 ms / max ≤ 1000 ms bound (measured ~116–180 ms p95)."""
    clean = {"p01_e04", "p07_e02", "p08_e01"}
    checked = clean & set(measured)
    assert checked, "clean fixtures missing from the cache — regenerate with --regen"
    for stem in checked:
        refined, _ = measured[stem]
        assert refined.max_ms <= 1000.0, f"{stem}: max {refined.max_ms:.0f}ms (AC2 max 1000)"
        assert refined.p95_ms <= 300.0, f"{stem}: p95 {refined.p95_ms:.0f}ms (AC2 p95 300)"
