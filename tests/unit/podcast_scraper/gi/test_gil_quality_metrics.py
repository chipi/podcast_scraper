"""Tests for GIL quality metrics (file aggregation)."""

import pytest
from conftest import artifact_with_grounded_insights

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

    def test_compute_metrics_single_insight_artifact(self, tmp_path):
        """A one-insight artifact yields non-zero counts.

        This used to build a placeholder artifact, which came with a manufactured Quote — so the
        test asserted quote counts that only a fabrication produced (#1657). It now states its
        own insight and grounds it through the evidence stack.
        """
        (tmp_path / "metadata").mkdir()
        p = tmp_path / "metadata" / "ep1.gi.json"
        art = artifact_with_grounded_insights("ep:1", "Hello transcript body with evidence here.")
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

    def test_enforce_fails_default_thresholds_on_a_sparse_artifact(self, tmp_path):
        """PRD default min avg insights/quotes fails on a one-insight artifact."""
        (tmp_path / "metadata").mkdir()
        p = tmp_path / "metadata" / "ep1.gi.json"
        art = artifact_with_grounded_insights("ep:1", "Hello transcript body with evidence here.")
        write_artifact(p, art, validate=True)
        m = compute_gil_quality_metrics([tmp_path])
        ok, failures = enforce_prd017_thresholds(m)
        assert ok is False
        assert any("avg_insights" in f for f in failures)

    def test_enforce_passes_with_relaxed_thresholds(self, tmp_path):
        """Lowering density thresholds passes on stub.

        ``min_grounded_insight_rate`` is 0.0 rather than 0.5 because the stub insight is now
        honestly ungrounded (#1657 item 9): it claimed ``grounded=True`` while its evidence was
        a transcript slice chosen by offset, for a placeholder that makes no claim. A stub
        artifact therefore has a grounded rate of exactly 0, and any positive threshold on it
        is a threshold on a lie. The sibling test above still pins that a stub FAILS the PRD
        defaults, which is the property that matters.
        """
        (tmp_path / "metadata").mkdir()
        p = tmp_path / "metadata" / "ep1.gi.json"
        art = artifact_with_grounded_insights("ep:1", "Hello transcript body with evidence here.")
        write_artifact(p, art, validate=True)
        m = compute_gil_quality_metrics([tmp_path])
        ok, failures = enforce_prd017_thresholds(
            m,
            min_avg_insights=0.5,
            min_avg_quotes=0.5,
            min_extraction_coverage=0.5,
            min_grounded_insight_rate=0.0,
            min_quote_validity_rate=0.5,
        )
        assert ok is True
        assert failures == []

    def test_an_empty_artifact_scores_zero_grounded(self, tmp_path):
        """State it directly, so the threshold change above is not mistaken for a loosening.

        An artifact with no insights has a grounded rate of zero — trivially, and honestly.
        """
        (tmp_path / "metadata").mkdir()
        p = tmp_path / "metadata" / "ep1.gi.json"
        write_artifact(
            p,
            build_artifact(
                "ep:1",
                "Hello transcript body here.",
                prompt_version="v1",
                insight_texts=["A real insight extracted from the transcript."],
            ),
            validate=True,
        )
        m = compute_gil_quality_metrics([tmp_path])
        assert m.grounded_insight_rate() == 0.0
