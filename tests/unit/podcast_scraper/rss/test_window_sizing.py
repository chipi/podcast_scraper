"""Unit tests for audio-minute window sizing (#1658).

The cost cap binds on modelled cost, which is dominated by transcription MINUTES, so sizing a
window by episode count means a long-form feed trips the cap on a number that is safe for a
short-form one. Tripping the cap is what caused both the G1 silent wedge and the G2 executor
crash (#1620), which makes this arithmetic a resilience concern rather than a tuning knob.

The medians below are the real ones measured per feed during onboarding.
"""

from __future__ import annotations

import pytest

from podcast_scraper.rss.window_sizing import (
    DEFAULT_TARGET_AUDIO_MINUTES,
    MAX_WINDOW_EPISODES,
    MIN_WINDOW_EPISODES,
    plan_window,
)

pytestmark = [pytest.mark.unit]


class TestPlanWindow:
    @pytest.mark.parametrize(
        "feed,median,expected",
        [
            ("The a16z Show", 49, 28),
            ("Dwarkesh Podcast", 85, 16),
            ("The Pragmatic Engineer", 87, 16),
            ("Lenny's Podcast", 92, 15),
            ("Ideas of India", 93, 15),
            ("Latent Space", 75, 18),
        ],
    )
    def test_matches_the_measured_per_feed_recommendations(
        self, feed: str, median: int, expected: int
    ) -> None:
        assert plan_window(median).episodes == expected, feed

    def test_projected_minutes_stay_under_the_target(self) -> None:
        """The point of the exercise — every window must fit the budget it was sized against."""
        for median in (20, 45, 49, 60, 75, 85, 92, 120):
            plan = plan_window(median)
            if plan.clamped != "below_minimum":
                assert plan.projected_audio_minutes <= DEFAULT_TARGET_AUDIO_MINUTES, median

    def test_a_very_long_feed_still_gets_a_workable_window(self) -> None:
        """A 4-hour episode would compute to 5 episodes; never zero, never one."""
        plan = plan_window(240)
        assert plan.episodes >= MIN_WINDOW_EPISODES
        assert plan.clamped in (None, "below_minimum")

    def test_a_very_short_feed_is_capped_rather_than_unbounded(self) -> None:
        """10-minute episodes compute to 140 — one enormous uninterruptible job."""
        plan = plan_window(10)
        assert plan.episodes == MAX_WINDOW_EPISODES
        assert plan.clamped == "above_maximum"

    def test_unknown_median_falls_back_small_not_large(self) -> None:
        """Wrong-small costs an extra job; wrong-large trips the cap and wedges the run."""
        for median in (0, -5):
            plan = plan_window(median)
            assert plan.episodes == MIN_WINDOW_EPISODES
            assert plan.clamped == "unknown_median_using_minimum"

    def test_target_is_overridable(self) -> None:
        assert plan_window(50, target_audio_minutes=500).episodes == 10

    def test_explain_carries_the_arithmetic(self) -> None:
        """A bare number invites 'round it up a bit', which is how the cap gets tripped."""
        text = plan_window(49).explain()
        assert "28 episodes" in text
        assert "median 49 min" in text
        assert "target 1400" in text

    def test_explain_reports_clamping(self) -> None:
        assert "clamped: above_maximum" in plan_window(10).explain()


class TestTheRuleItself:
    def test_the_default_target_leaves_headroom_under_the_ten_dollar_cap(self) -> None:
        """~1400 audio-minutes is ~$7 modelled — margin matters because the cap is a fuse."""
        assert 1000 <= DEFAULT_TARGET_AUDIO_MINUTES <= 1600

    def test_equal_audio_minutes_across_very_different_feeds(self) -> None:
        """The whole thesis: a16z and Lenny's should land on similar TOTAL minutes."""
        a16z = plan_window(49).projected_audio_minutes
        lennys = plan_window(92).projected_audio_minutes
        assert abs(a16z - lennys) < 0.25 * DEFAULT_TARGET_AUDIO_MINUTES
