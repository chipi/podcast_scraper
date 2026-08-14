"""Unit tests for the corpus quality report (#1647).

This report is what answers "I ran N episodes — how did it go?". Its failure mode is not
crashing; it is *reassuring*. Every test here pins a place where a plausible-looking
implementation would report health it has not established.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from podcast_scraper.quality.attribution import (
    build_report,
    EpisodeQuality,
    format_report,
    ratio,
    summarise_stage,
)

pytestmark = [pytest.mark.unit]


def _ep(**kwargs: Any) -> EpisodeQuality:
    """A healthy episode; override to describe a damaged one."""
    defaults: Dict[str, Any] = {
        "episode_id": "ep-1",
        "feed": "Hard Fork",
        "duration_seconds": 1800,
        "stage_ledger": {"speaker_detection": {"outcome": "ran"}},
        "insights_total": 10,
        "insights_surfaceable": 10,
        "voices_total": 2,
        "voices_named": 2,
    }
    defaults.update(kwargs)
    return EpisodeQuality(**defaults)


class TestRatio:
    def test_no_denominator_is_none_not_zero(self) -> None:
        """0.0 asserts total failure; None says 'no data'. A diff cannot recover the difference."""
        assert ratio(0, 0) is None
        assert ratio(5, None) is None

    def test_computes_and_rounds(self) -> None:
        assert ratio(1, 3) == 0.333333


class TestFullyZeroed:
    def test_requires_insights_to_have_existed(self) -> None:
        """An episode that produced no insights is a different bug — do not inflate the damage."""
        assert _ep(insights_total=0, insights_surfaceable=0).fully_zeroed is False
        assert _ep(insights_total=29, insights_surfaceable=0).fully_zeroed is True

    def test_unknown_attribution_is_not_zeroed(self) -> None:
        assert _ep(insights_total=None, insights_surfaceable=None).fully_zeroed is False


class TestSummariseStage:
    def test_counts_outcomes_and_breaks_skips_down_by_reason(self) -> None:
        episodes = [
            _ep(),
            _ep(
                stage_ledger={
                    "speaker_detection": {
                        "outcome": "skipped",
                        "reason": "media_over_size_limit_no_transcript_urls",
                    }
                }
            ),
            _ep(
                stage_ledger={
                    "speaker_detection": {
                        "outcome": "skipped",
                        "reason": "media_over_size_limit_no_transcript_urls",
                    }
                }
            ),
        ]
        summary = summarise_stage(episodes, "speaker_detection")
        assert summary["outcomes"] == {"ran": 1, "skipped": 2}
        assert summary["reasons"] == {"media_over_size_limit_no_transcript_urls": 2}
        assert summary["ran_ratio"] == round(1 / 3, 6)

    def test_missing_ledger_entry_is_counted_not_assumed_to_have_run(self) -> None:
        """The whole point: absence must not be silently scored as success."""
        summary = summarise_stage([_ep(stage_ledger={})], "speaker_detection")
        assert summary["no_ledger_entry"] == 1
        assert summary["outcomes"] == {}
        assert summary["ran_ratio"] == 0.0

    def test_degraded_counts_as_having_run(self) -> None:
        episodes = [_ep(stage_ledger={"speaker_detection": {"outcome": "degraded"}})]
        assert summarise_stage(episodes, "speaker_detection")["ran_ratio"] == 1.0

    def test_a_skip_without_a_reason_is_labelled_not_dropped(self) -> None:
        episodes = [_ep(stage_ledger={"speaker_detection": {"outcome": "skipped"}})]
        assert summarise_stage(episodes, "speaker_detection")["reasons"] == {
            "reason_not_recorded": 1
        }


class TestBuildReport:
    def test_healthy_corpus(self) -> None:
        report = build_report([_ep(), _ep(episode_id="ep-2")])
        assert report["attribution"]["attribution_ratio"] == 1.0
        assert report["attribution"]["episodes_fully_zeroed"] == 0
        assert report["not_measured"]["episodes_without_stage_ledger"] == 0

    def test_reproduces_the_1646_shape(self) -> None:
        """Skipped detection + everything unsurfaceable — the Latent Space episode."""
        damaged = _ep(
            feed="Latent Space",
            stage_ledger={
                "speaker_detection": {
                    "outcome": "skipped",
                    "reason": "media_over_size_limit_no_transcript_urls",
                    "detail": {"media_bytes": 42871040, "limit_bytes": 26214400},
                }
            },
            insights_total=29,
            insights_surfaceable=0,
            voices_total=4,
            voices_named=0,
        )
        report = build_report([damaged, _ep()])

        assert report["attribution"]["insights_unsurfaceable"] == 29
        assert report["attribution"]["episodes_fully_zeroed"] == 1
        assert report["attribution"]["voices_named_ratio"] == round(2 / 6, 6)
        stage = report["stages"][0]
        assert stage["reasons"]["media_over_size_limit_no_transcript_urls"] == 1
        assert report["per_feed"]["Latent Space"]["attribution_ratio"] == 0.0
        assert report["per_feed"]["Hard Fork"]["attribution_ratio"] == 1.0

    def test_episodes_without_data_are_excluded_from_ratios_and_reported(self) -> None:
        """An unreadable episode must not be averaged in as healthy."""
        report = build_report(
            [_ep(), _ep(insights_total=None, insights_surfaceable=None, notes=["gi_unreadable"])]
        )
        assert report["attribution"]["attribution_ratio"] == 1.0  # from the readable one only
        assert report["not_measured"]["episodes_without_attribution_data"] == 1
        assert report["not_measured"]["notes"] == ["gi_unreadable"]

    def test_pre_ledger_episodes_are_surfaced_as_unknown(self) -> None:
        report = build_report([_ep(stage_ledger={})])
        assert report["not_measured"]["episodes_without_stage_ledger"] == 1

    def test_semantic_correctness_is_always_declared_unmeasured(self) -> None:
        """Even a perfect structural run must not read as 'the corpus is correct'."""
        report = build_report([_ep()])
        assert "NOT MEASURED" in report["not_measured"]["semantic_correctness"]

    def test_empty_input_produces_no_false_ratios(self) -> None:
        report = build_report([])
        assert report["episodes"] == 0
        assert report["attribution"]["attribution_ratio"] is None
        assert report["attribution"]["episodes_fully_zeroed_ratio"] is None

    def test_scale_invariance_one_episode_and_many_share_a_shape(self) -> None:
        """'1, 10, 50, 5000' must be the same question with the same answer shape."""
        one = build_report([_ep()])
        many = build_report([_ep(episode_id=f"ep-{i}") for i in range(50)])
        assert set(one) == set(many)
        assert set(one["attribution"]) == set(many["attribution"])


class TestFormatReport:
    def test_renders_not_measured_section(self) -> None:
        text = format_report(build_report([_ep(stage_ledger={})]))
        assert "NOT MEASURED" in text
        assert "episodes without a stage ledger : 1" in text

    def test_renders_skip_reasons_with_counts(self) -> None:
        episodes = [
            _ep(
                stage_ledger={
                    "speaker_detection": {"outcome": "skipped", "reason": "media_over_size_limit"}
                }
            )
        ]
        text = format_report(build_report(episodes))
        assert "media_over_size_limit" in text
