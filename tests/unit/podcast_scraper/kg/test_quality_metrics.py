"""Tests for KG quality metrics."""

from pathlib import Path

import pytest

from podcast_scraper.kg.quality_metrics import (
    compute_kg_quality_metrics,
    enforce_prd019_thresholds,
    KgQualityMetrics,
)


@pytest.mark.unit
class TestKgQualityMetricsDataclass:
    """Empty aggregates return 0.0 (PR-476)."""

    def test_avg_and_coverage_zero_when_empty(self) -> None:
        m = KgQualityMetrics()
        assert m.avg_nodes_per_artifact() == 0.0
        assert m.avg_edges_per_artifact() == 0.0
        assert m.extraction_coverage() == 0.0
        d = m.to_dict()
        assert d["errors"] == []
        assert d["artifact_paths"] == 0


@pytest.mark.unit
class TestKgQualityMetrics:
    """compute_kg_quality_metrics and enforce_prd019_thresholds."""

    def test_fixture_dir_scores(self):
        """Committed CI fixture yields one artifact and passes relaxed enforce."""
        root = Path(__file__).resolve().parents[3] / "fixtures" / "gil_kg_ci_enforce"
        m = compute_kg_quality_metrics([root], strict_schema=True)
        assert m.artifact_paths >= 1
        assert m.total_nodes >= 1
        d = m.to_dict()
        assert d["extraction_coverage"] == 1.0
        ok, failures = enforce_prd019_thresholds(
            m,
            min_artifacts=1,
            min_avg_nodes=1.0,
            min_avg_edges=0.0,
            min_extraction_coverage=1.0,
        )
        assert ok is True
        assert failures == []

    def test_enforce_fails_empty(self):
        m = compute_kg_quality_metrics([], strict_schema=False)
        ok, failures = enforce_prd019_thresholds(m, min_artifacts=1)
        assert ok is False
        assert failures
