"""The join timeout must reflect the CONFIGURED per-episode ASR budget, not a stale constant.

THE NOISE. ``_THREAD_JOIN_TIMEOUT_PER_EPISODE = 120`` was calibrated for API Whisper, where the
docstring's "each episode can take 60-90s+" was true. Measured over 74 episodes of the
2026-08-31 DGX batch: asr p50=496s, p90=706s, p99=1168s. So the allowance was **24% of the
median episode**, and "Transcription thread did not finish within 1800s (10 episodes)" fired on
every healthy multi-episode feed.

Nothing broke — the caller falls through to an unbounded ``join()``, so runs completed. But a
warning that fires on every healthy batch is one nobody reads, and it was appearing next to the
genuinely interesting ones in the same log.

THE SHAPE OF THE FIX, and why it is not just a bigger number: the per-episode allowance is
derived from ``transcription_timeout`` — the operator's own statement of how long ONE episode
may take. If a single episode cannot exceed that, N episodes cannot exceed N times it, and the
bound stays correct when the ASR backend changes again. Re-hardcoding a larger constant would
drift out of step with the timeout exactly as this one drifted out of step with the backend.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from podcast_scraper.workflow.orchestration import (
    _thread_join_timeout,
    _THREAD_JOIN_TIMEOUT_BASE,
    _THREAD_JOIN_TIMEOUT_PER_EPISODE,
)

#: Measured on the 2026-08-31 DGX batch (74 episodes with an asr duration).
_ASR_P50, _ASR_P99 = 496, 1168


def _cfg(transcription_timeout):
    return SimpleNamespace(transcription_timeout=transcription_timeout)


class TestItCoversRealAsrCost:
    @pytest.mark.parametrize("episodes", [1, 5, 10, 20])
    def test_the_bound_exceeds_measured_p99_asr_for_the_whole_feed(self, episodes):
        """THE REGRESSION GUARD: a healthy feed must not trip the warning."""
        got = _thread_join_timeout(episodes, _cfg(1800))
        assert got > episodes * _ASR_P99, (
            f"{episodes} episodes at the measured p99 asr ({_ASR_P99}s each) need "
            f"{episodes * _ASR_P99}s; the bound is only {got}s, so the warning fires on a "
            "healthy feed"
        )

    def test_the_old_constant_would_NOT_have_covered_it(self):
        """Pins why this changed — the old formula is short by several multiples."""
        old = _THREAD_JOIN_TIMEOUT_BASE + 10 * _THREAD_JOIN_TIMEOUT_PER_EPISODE
        assert old < 10 * _ASR_P50, (
            "the old bound should be below even the MEDIAN cost of a 10-episode feed; if this "
            "fails the premise of the fix no longer holds and it should be revisited"
        )


class TestItDerivesFromConfig:
    def test_a_larger_configured_timeout_raises_the_bound(self):
        small = _thread_join_timeout(10, _cfg(600))
        large = _thread_join_timeout(10, _cfg(3600))
        assert large > small

    def test_the_configured_value_is_the_per_episode_allowance(self):
        n, budget = 7, 900
        assert _thread_join_timeout(n, _cfg(budget)) == _THREAD_JOIN_TIMEOUT_BASE + n * budget


class TestDegradesSafely:
    def test_no_cfg_keeps_the_legacy_bound(self):
        """Callers that do not pass cfg must behave exactly as before."""
        assert _thread_join_timeout(10) == _THREAD_JOIN_TIMEOUT_BASE + 10 * (
            _THREAD_JOIN_TIMEOUT_PER_EPISODE
        )

    @pytest.mark.parametrize("value", [None, 0, -1, "1800", object()])
    def test_a_disabled_or_odd_timeout_falls_back_to_the_constant(self, value):
        """``transcription_timeout=None`` is documented as 'disable timeout'.

        That must not collapse the join bound to the base, which would REINTRODUCE the noisy
        warning for anyone who disabled the per-episode timeout.
        """
        assert _thread_join_timeout(10, _cfg(value)) == _THREAD_JOIN_TIMEOUT_BASE + 10 * (
            _THREAD_JOIN_TIMEOUT_PER_EPISODE
        )

    def test_the_constant_is_a_floor_not_a_ceiling(self):
        """A tiny configured timeout must not make the bound SMALLER than the legacy one."""
        assert _thread_join_timeout(10, _cfg(30)) >= _thread_join_timeout(10)

    @pytest.mark.parametrize("episodes", [0, -5])
    def test_degenerate_episode_counts(self, episodes):
        assert _thread_join_timeout(episodes, _cfg(1800)) == _THREAD_JOIN_TIMEOUT_BASE


def test_it_is_still_bounded_not_infinite():
    """The point is a MEANINGFUL bound, not the absence of one.

    The caller falls through to an unbounded join after warning, so this value only decides
    when we say something — but it must stay finite, or a genuinely wedged thread is never
    reported at all.
    """
    got = _thread_join_timeout(10, _cfg(1800))
    assert 0 < got < 10 * 60 * 60, got
