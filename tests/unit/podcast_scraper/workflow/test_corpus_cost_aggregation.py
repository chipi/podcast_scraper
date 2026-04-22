"""Unit tests for per-corpus cost aggregation (#650 Finding 20)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.workflow.corpus_cost_aggregation import aggregate_corpus_costs

pytestmark = [pytest.mark.unit]


def _write_metrics(root: Path, feed_slug: str, run_name: str, doc: dict) -> Path:
    run_dir = root / "feeds" / feed_slug / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "metrics.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


class TestAggregateCorpusCosts:
    def test_empty_corpus_returns_zeros(self, tmp_path: Path) -> None:
        out = aggregate_corpus_costs(tmp_path)
        assert out["total_cost_usd"] == 0.0
        assert out["total_transcription_cost_usd"] == 0.0
        assert out["total_llm_cost_usd"] == 0.0
        assert out["run_count"] == 0
        assert out["metrics_files_missing_cost_fields"] == 0
        assert all(v == 0.0 for v in out["by_stage"].values())

    def test_sums_across_multiple_runs_and_feeds(self, tmp_path: Path) -> None:
        _write_metrics(
            tmp_path,
            "feed_a",
            "run_001",
            {
                "llm_transcription_cost_usd": 0.08,
                "llm_summarization_cost_usd": 0.0013,
                "llm_gi_cost_usd": 0.0004,
                "llm_kg_cost_usd": 0.0002,
                "llm_speaker_detection_cost_usd": 0.0,
                "llm_cleaning_cost_usd": 0.0,
            },
        )
        _write_metrics(
            tmp_path,
            "feed_a",
            "run_002",
            {
                "llm_transcription_cost_usd": 0.12,
                "llm_summarization_cost_usd": 0.002,
            },
        )
        _write_metrics(
            tmp_path,
            "feed_b",
            "run_001",
            {
                "llm_transcription_cost_usd": 0.05,
                "llm_gi_cost_usd": 0.0001,
            },
        )

        out = aggregate_corpus_costs(tmp_path)

        assert out["run_count"] == 3
        assert out["metrics_files_missing_cost_fields"] == 0
        assert out["by_stage"]["llm_transcription_cost_usd"] == pytest.approx(0.25)
        assert out["by_stage"]["llm_summarization_cost_usd"] == pytest.approx(0.0033)
        assert out["by_stage"]["llm_gi_cost_usd"] == pytest.approx(0.0005)
        assert out["by_stage"]["llm_kg_cost_usd"] == pytest.approx(0.0002)
        assert out["total_transcription_cost_usd"] == pytest.approx(0.25)
        assert out["total_llm_cost_usd"] == pytest.approx(0.0033 + 0.0005 + 0.0002)
        assert out["total_cost_usd"] == pytest.approx(0.25 + 0.0033 + 0.0005 + 0.0002)

    def test_skips_missing_or_malformed_metrics_without_failing(self, tmp_path: Path) -> None:
        _write_metrics(
            tmp_path,
            "feed_a",
            "run_001",
            {"llm_transcription_cost_usd": 0.10},
        )
        # Malformed JSON should be skipped (logged).
        bad = tmp_path / "feeds" / "feed_a" / "run_002"
        bad.mkdir(parents=True)
        (bad / "metrics.json").write_text("{ not json }", encoding="utf-8")
        # Non-dict JSON should also be skipped.
        worse = tmp_path / "feeds" / "feed_a" / "run_003"
        worse.mkdir(parents=True)
        (worse / "metrics.json").write_text("[1, 2, 3]", encoding="utf-8")

        out = aggregate_corpus_costs(tmp_path)
        assert out["run_count"] == 1
        assert out["total_transcription_cost_usd"] == pytest.approx(0.10)

    def test_counts_pre_fix_runs_missing_cost_fields(self, tmp_path: Path) -> None:
        # A metrics.json with tokens but no llm_*_cost_usd fields (pre-#650).
        _write_metrics(
            tmp_path,
            "feed_a",
            "run_001",
            {"llm_transcription_calls": 2, "llm_transcription_audio_minutes": 10.0},
        )
        # A post-#650 run with zero cost (stage ran but was local / free).
        _write_metrics(
            tmp_path,
            "feed_a",
            "run_002",
            {"llm_transcription_cost_usd": 0.0, "llm_gi_cost_usd": 0.0},
        )
        out = aggregate_corpus_costs(tmp_path)
        assert out["run_count"] == 2
        assert out["metrics_files_missing_cost_fields"] == 1
        assert out["total_cost_usd"] == 0.0

    def test_non_numeric_cost_field_ignored(self, tmp_path: Path) -> None:
        _write_metrics(
            tmp_path,
            "feed_a",
            "run_001",
            {
                "llm_transcription_cost_usd": "not a number",
                "llm_summarization_cost_usd": 0.01,
            },
        )
        out = aggregate_corpus_costs(tmp_path)
        assert out["by_stage"]["llm_transcription_cost_usd"] == 0.0
        assert out["by_stage"]["llm_summarization_cost_usd"] == pytest.approx(0.01)

    def test_only_walks_feeds_subtree(self, tmp_path: Path) -> None:
        # A metrics.json in a sibling path should NOT be counted —
        # corpus_parent layout is authoritative.
        outside = tmp_path / "other" / "run_001"
        outside.mkdir(parents=True)
        (outside / "metrics.json").write_text(
            json.dumps({"llm_transcription_cost_usd": 99.0}), encoding="utf-8"
        )
        # Proper feeds-subtree file:
        _write_metrics(tmp_path, "feed_a", "run_001", {"llm_transcription_cost_usd": 0.05})

        out = aggregate_corpus_costs(tmp_path)
        assert out["run_count"] == 1
        assert out["total_transcription_cost_usd"] == pytest.approx(0.05)
