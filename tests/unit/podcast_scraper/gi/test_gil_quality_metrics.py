"""Tests for GIL quality metrics (file aggregation)."""

import pytest

from podcast_scraper.gi import build_artifact, write_artifact
from podcast_scraper.gi.quality_metrics import (
    compute_gil_quality_metrics,
    enforce_prd017_thresholds,
    GilQualityMetrics,
)


@pytest.mark.unit
class TestGilQualityMetricsDataclass:
    """Empty aggregates and stable ``to_dict`` keys (PR-476 / CI scripts)."""

    def test_avg_rates_zero_when_no_data(self) -> None:
        m = GilQualityMetrics()
        assert m.artifact_paths == 0
        assert m.extraction_coverage() == 0.0
        assert m.grounded_insight_rate() == 0.0
        assert m.quote_validity_rate() == 0.0
        assert m.avg_insights_per_artifact() == 0.0
        assert m.avg_quotes_per_artifact() == 0.0
        d = m.to_dict()
        assert d["errors"] == []
        assert "quote_validity_rate" in d


@pytest.mark.unit
class TestGilQualityMetrics:
    """compute_gil_quality_metrics and enforce_prd017_thresholds."""

    def test_compute_metrics_stub_artifact(self, tmp_path):
        """Single stub artifact yields non-zero counts."""
        (tmp_path / "metadata").mkdir()
        p = tmp_path / "metadata" / "ep1.gi.json"
        art = build_artifact("ep:1", "Hello transcript body here.", prompt_version="v1")
        write_artifact(p, art, validate=True)
        m = compute_gil_quality_metrics([tmp_path], strict_schema=False)
        assert m.artifact_paths == 1
        assert m.total_insights >= 1
        assert m.total_quotes >= 1
        assert m.extraction_coverage() == 1.0
        assert m.grounded_insight_rate() >= 0.0
        d = m.to_dict()
        assert "quote_validity_rate" in d
        assert d["errors"] == []

    def test_enforce_fails_default_thresholds_on_sparse_stub(self, tmp_path):
        """PRD default min avg insights/quotes fails on single-insight stub."""
        (tmp_path / "metadata").mkdir()
        p = tmp_path / "metadata" / "ep1.gi.json"
        art = build_artifact("ep:1", "Hello transcript body here.", prompt_version="v1")
        write_artifact(p, art, validate=True)
        m = compute_gil_quality_metrics([tmp_path])
        ok, failures = enforce_prd017_thresholds(m)
        assert ok is False
        assert any("avg_insights" in f for f in failures)

    def test_enforce_passes_with_relaxed_thresholds(self, tmp_path):
        """Lowering density thresholds passes on stub."""
        (tmp_path / "metadata").mkdir()
        p = tmp_path / "metadata" / "ep1.gi.json"
        art = build_artifact("ep:1", "Hello transcript body here.", prompt_version="v1")
        write_artifact(p, art, validate=True)
        m = compute_gil_quality_metrics([tmp_path])
        ok, failures = enforce_prd017_thresholds(
            m,
            min_avg_insights=0.5,
            min_avg_quotes=0.5,
            min_extraction_coverage=0.5,
            min_grounded_insight_rate=0.5,
            min_quote_validity_rate=0.5,
        )
        assert ok is True
        assert failures == []
