"""Tests for #912 Path D — bundled JSON parse-failure counter + reporter render.

Covers two surfaces:
1. workflow.metrics.Metrics.record_llm_bundled_parse_failure — counter math
   and per-kind breakdown
2. evaluation.reporter.generate_metrics_report — Markdown render of the
   "Bundled JSON Parse Failures" section when present, and the absence of
   that section when no failures were observed.
"""

from __future__ import annotations

import pytest

from podcast_scraper.evaluation.reporter import generate_metrics_report
from podcast_scraper.workflow.metrics import Metrics


@pytest.mark.unit
class TestRecordLlmBundledParseFailure:
    """Counter math on the Metrics dataclass."""

    def test_zero_failures_when_unused(self) -> None:
        m = Metrics()
        assert m.llm_bundled_parse_failures_total == 0
        assert m.llm_bundled_parse_failures_by_kind == {}

    def test_single_failure_bumps_total_and_kind(self) -> None:
        m = Metrics()
        m.record_llm_bundled_parse_failure("not_valid_json")
        assert m.llm_bundled_parse_failures_total == 1
        assert m.llm_bundled_parse_failures_by_kind == {"not_valid_json": 1}

    def test_multiple_failures_accumulate_per_kind(self) -> None:
        m = Metrics()
        for kind in ("not_valid_json", "not_valid_json", "missing_summary"):
            m.record_llm_bundled_parse_failure(kind)
        assert m.llm_bundled_parse_failures_total == 3
        assert m.llm_bundled_parse_failures_by_kind == {
            "not_valid_json": 2,
            "missing_summary": 1,
        }

    def test_all_documented_kinds_supported(self) -> None:
        """The 5 documented kinds round-trip correctly (no validation gate)."""
        m = Metrics()
        kinds = [
            "not_valid_json",
            "not_an_object",
            "missing_summary",
            "missing_bullets",
            "guardrail_violation",
        ]
        for kind in kinds:
            m.record_llm_bundled_parse_failure(kind)
        assert m.llm_bundled_parse_failures_total == 5
        assert set(m.llm_bundled_parse_failures_by_kind.keys()) == set(kinds)
        assert all(v == 1 for v in m.llm_bundled_parse_failures_by_kind.values())

    def test_finish_emits_path_d_keys(self) -> None:
        """finish() must expose Path D counters so eval_pipeline_metrics.json
        picks them up. Regression guard for the test_metrics.py expected-keys
        contract."""
        m = Metrics()
        m.record_llm_bundled_parse_failure("not_valid_json")
        m.record_llm_bundled_parse_failure("missing_summary")
        d = m.finish()
        assert d["llm_bundled_parse_failures_total"] == 2
        assert d["llm_bundled_parse_failures_by_kind"] == {
            "not_valid_json": 1,
            "missing_summary": 1,
        }


@pytest.mark.unit
class TestReporterPathDSection:
    """generate_metrics_report renders the Path D section iff failures > 0."""

    def _minimal_metrics(self) -> dict:
        return {
            "dataset_id": "curated_5feeds_dev_v1",
            "run_id": "test_run_v1",
            "episode_count": 10,
            "intrinsic": {},
        }

    def test_section_absent_when_no_failures(self) -> None:
        report = generate_metrics_report(self._minimal_metrics())
        assert "Bundled JSON Parse Failures" not in report

    def test_section_absent_when_total_is_zero(self) -> None:
        m = self._minimal_metrics()
        m["intrinsic"]["bundled_parse_failures"] = {"total": 0, "by_kind": {}}
        assert "Bundled JSON Parse Failures" not in generate_metrics_report(m)

    def test_section_rendered_with_total_and_kinds(self) -> None:
        m = self._minimal_metrics()
        m["intrinsic"]["bundled_parse_failures"] = {
            "total": 4,
            "by_kind": {"not_valid_json": 2, "missing_summary": 1, "missing_bullets": 1},
        }
        report = generate_metrics_report(m)
        assert "Bundled JSON Parse Failures" in report
        assert "Total Parse Failures:** 4" in report
        # Each kind on its own line — sorted alphabetically by the reporter
        # for stable diffs across runs.
        assert "missing_bullets: 1" in report
        assert "missing_summary: 1" in report
        assert "not_valid_json: 2" in report

    def test_section_rendered_when_only_total_present(self) -> None:
        """Defensive: by_kind missing or empty still renders the total."""
        m = self._minimal_metrics()
        m["intrinsic"]["bundled_parse_failures"] = {"total": 1}
        report = generate_metrics_report(m)
        assert "Bundled JSON Parse Failures" in report
        assert "Total Parse Failures:** 1" in report
        # by_kind block omitted when missing — no "By Kind:" line
        assert "By Kind:" not in report

    def test_path_d_section_does_not_break_existing_intrinsic_blocks(self) -> None:
        """Path D section is additive — co-exists with existing length /
        performance / cost blocks."""
        m = self._minimal_metrics()
        m["intrinsic"]["bundled_parse_failures"] = {"total": 1, "by_kind": {"not_valid_json": 1}}
        m["intrinsic"]["length"] = {"avg_tokens": 500.0, "min_tokens": 200, "max_tokens": 800}
        m["intrinsic"]["performance"] = {"avg_latency_ms": 13500.0}
        report = generate_metrics_report(m)
        assert "Length Metrics" in report
        assert "Performance Metrics" in report
        assert "Bundled JSON Parse Failures" in report
