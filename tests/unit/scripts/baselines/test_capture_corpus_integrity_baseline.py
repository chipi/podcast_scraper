"""Unit tests for scripts/baselines/capture_corpus_integrity_baseline.py.

The baseline is the artifact epic #1657 measures the corpus repair against, so its
aggregation has to be right *before* the repair runs — a miscounted baseline would make a
broken repair look successful, which is the exact failure mode #1646 taught us.

These cover the pure aggregation logic. Network paths are exercised in the integration tier.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[4]

_SPEC = importlib.util.spec_from_file_location(
    "capture_corpus_integrity_baseline_under_test",
    ROOT / "scripts" / "baselines" / "capture_corpus_integrity_baseline.py",
)
assert _SPEC and _SPEC.loader
_mod = importlib.util.module_from_spec(_SPEC)
sys.modules["capture_corpus_integrity_baseline_under_test"] = _mod
_SPEC.loader.exec_module(_mod)

pytestmark = [pytest.mark.unit]


def _row(**overrides: Any) -> Dict[str, Any]:
    """A healthy episode row; override fields to describe a damaged one."""
    row: Dict[str, Any] = {
        "feed": "Hard Fork",
        "episode_id": "ep-1",
        "metadata_relpath": "feeds/f/run_x/metadata/0001 - Title.metadata.json",
        "duration_seconds": 1800,
        "speaker_detection_ran": True,
        "extract_names_time": 1.2,
        "insights_total": 10,
        "insights_surfaceable": 10,
        "voices_total": 2,
        "voices_unidentified": 0,
        "persons": 2,
        "errors": [],
    }
    row.update(overrides)
    return row


class TestGiRelpath:
    def test_swaps_metadata_suffix_for_gi_keeping_the_directory(self) -> None:
        meta = "feeds/rss_x/run_y/metadata/0004 - Notion_s Token Town.metadata.json"
        assert _mod._gi_relpath(meta) == (
            "feeds/rss_x/run_y/metadata/0004 - Notion_s Token Town.gi.json"
        )

    def test_preserves_dots_inside_the_episode_title(self) -> None:
        """Titles legitimately contain dots ("Hot I.P.O Summer") — only the suffix may change."""
        meta = "feeds/f/run_z/metadata/0009 - Hot I.P.O Summer .metadata.json"
        assert _mod._gi_relpath(meta).endswith("0009 - Hot I.P.O Summer .gi.json")


class TestRatio:
    def test_computes_ratio(self) -> None:
        assert _mod._ratio(1, 4) == 0.25

    def test_zero_denominator_is_none_not_zero(self) -> None:
        """None means "no data"; 0.0 would claim perfect failure and pollute a diff."""
        assert _mod._ratio(0, 0) is None


class TestSummarise:
    def test_healthy_corpus_reports_full_attribution_and_no_skips(self) -> None:
        summary = _mod.summarise([_row(), _row(episode_id="ep-2")])
        assert summary["episodes_usable"] == 2
        assert summary["speaker_detection_skipped"] == 0
        assert summary["attribution_ratio"] == 1.0
        assert summary["episodes_fully_zeroed"] == 0

    def test_counts_skipped_stage_from_a_null_timing(self) -> None:
        """extract_names_time=None means the stage returned before recording — not zero time."""
        rows = [_row(), _row(episode_id="e2", speaker_detection_ran=False, extract_names_time=None)]
        summary = _mod.summarise(rows)
        assert summary["speaker_detection_skipped"] == 1
        assert summary["speaker_detection_skipped_ratio"] == 0.5

    def test_fully_zeroed_requires_insights_to_exist(self) -> None:
        """An episode with no insights at all is not "attribution destroyed" — do not inflate."""
        rows = [
            _row(episode_id="damaged", insights_total=29, insights_surfaceable=0),
            _row(episode_id="empty", insights_total=0, insights_surfaceable=0),
        ]
        summary = _mod.summarise(rows)
        assert summary["episodes_fully_zeroed"] == 1

    def test_attribution_ratio_reflects_partial_loss(self) -> None:
        rows = [_row(insights_total=100, insights_surfaceable=77)]
        assert _mod.summarise(rows)["attribution_ratio"] == 0.77

    def test_rows_with_errors_are_excluded_from_denominators(self) -> None:
        """An unreadable episode must not be counted as healthy — it is counted as unknown."""
        rows = [_row(), _row(episode_id="bad", errors=["gi_unreadable"], insights_total=None)]
        summary = _mod.summarise(rows)
        assert summary["episodes_probed"] == 2
        assert summary["episodes_usable"] == 1
        assert summary["episodes_with_errors"] == 1
        assert summary["attribution_ratio"] == 1.0

    def test_audio_minutes_split_tracks_the_size_gate_hypothesis(self) -> None:
        """Skipped episodes are the long ones; the split is what scales the repair cost."""
        rows = [
            _row(duration_seconds=1200),
            _row(episode_id="long", speaker_detection_ran=False, duration_seconds=4800),
        ]
        summary = _mod.summarise(rows)
        assert summary["audio_minutes_total"] == 100.0
        assert summary["audio_minutes_skipped"] == 80.0

    def test_per_feed_breakdown_separates_feeds(self) -> None:
        rows: List[Dict[str, Any]] = [
            _row(
                feed="Latent Space",
                insights_total=29,
                insights_surfaceable=0,
                speaker_detection_ran=False,
            ),
            _row(feed="Unhedged", insights_total=10, insights_surfaceable=10),
        ]
        per_feed = _mod.summarise(rows)["per_feed"]
        assert per_feed["Latent Space"]["attribution_ratio"] == 0.0
        assert per_feed["Latent Space"]["zeroed"] == 1
        assert per_feed["Unhedged"]["attribution_ratio"] == 1.0
        assert per_feed["Unhedged"]["skipped"] == 0

    def test_empty_input_does_not_divide_by_zero(self) -> None:
        summary = _mod.summarise([])
        assert summary["episodes_usable"] == 0
        assert summary["attribution_ratio"] is None
        assert summary["speaker_detection_skipped_ratio"] is None
