"""Tests for evaluation schema validation."""

from __future__ import annotations

import json

import pytest

from podcast_scraper.evaluation.schema_validator import (
    validate_metrics_ner,
    validate_metrics_summarization,
    validate_schema,
    validate_summarization_reference,
)


@pytest.mark.unit
class TestValidateSchema:
    """Test validate_schema."""

    def test_missing_schema_non_strict_logs_only(self, tmp_path):
        """When strict=False, missing schema file does not raise."""
        validate_schema({}, tmp_path / "missing.json", strict=False)

    def test_missing_schema_strict_raises(self, tmp_path):
        """When strict=True, missing schema file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Schema file not found"):
            validate_schema({}, tmp_path / "missing.json", strict=True)

    def test_invalid_json_schema_strict_raises(self, tmp_path):
        """When strict=True, invalid JSON in schema raises ValueError."""
        path = tmp_path / "bad.json"
        path.write_text("not json", encoding="utf-8")
        with pytest.raises(ValueError, match="Failed to load schema"):
            validate_schema({}, path, strict=True)

    def test_valid_schema_with_required_fields_fallback(self, tmp_path):
        """Basic validation passes when obj has required fields (no jsonschema)."""
        schema_path = tmp_path / "schema.json"
        schema_path.write_text(json.dumps({"required": ["id"]}), encoding="utf-8")
        validate_schema({"id": 1}, schema_path, strict=False)


@pytest.mark.unit
class TestValidateSummarizationReference:
    """Test validate_summarization_reference."""

    def test_baseline_output_summary_final_passes_schema(self):
        """Run/baseline JSONL shape validates after normalization."""
        from pathlib import Path

        if not Path("data/eval/schemas/summarization_reference_v1.json").is_file():
            pytest.skip("summarization_reference_v1.json not in tree")
        entry = {
            "episode_id": "e1",
            "output": {"summary_final": "A real summary with enough text."},
        }
        assert validate_summarization_reference(entry, strict=False) is True

    def test_top_level_summary_passes_schema(self):
        """Legacy top-level summary still validates."""
        from pathlib import Path

        if not Path("data/eval/schemas/summarization_reference_v1.json").is_file():
            pytest.skip("summarization_reference_v1.json not in tree")
        entry = {"episode_id": "e1", "summary": "A real summary with enough text."}
        assert validate_summarization_reference(entry, strict=False) is True

    def test_missing_summary_fails_when_schema_present(self):
        """episode_id alone does not satisfy schema."""
        from pathlib import Path

        if not Path("data/eval/schemas/summarization_reference_v1.json").is_file():
            pytest.skip("summarization_reference_v1.json not in tree")
        assert validate_summarization_reference({"episode_id": "ep1"}, strict=False) is False


@pytest.mark.unit
class TestValidateMetricsSummarization:
    """Test validate_metrics_summarization."""

    def test_lenient_when_schema_missing(self):
        """Does not raise when schema file missing (default strict=False)."""
        validate_metrics_summarization({"run_id": "r1"})

    def test_v2_metrics_passes_when_schema_present(self):
        """metrics_summarization_v2-shaped dict validates when repo schema file exists."""
        metrics_v2 = {
            "schema": "metrics_summarization_v2",
            "task": "summarization",
            "dataset_id": "curated_5feeds_smoke_v1",
            "run_id": "test_run",
            "episode_count": 1,
            "intrinsic": {
                "gates": {
                    "boilerplate_leak_rate": 0.0,
                    "speaker_label_leak_rate": 0.0,
                    "truncation_rate": 0.0,
                    "failed_episodes": [],
                    "episode_gate_failures": {},
                },
                "warnings": {"speaker_name_leak_rate": 0.0},
                "length": {"avg_tokens": 10.0, "min_tokens": 10.0, "max_tokens": 10.0},
                "performance": {
                    "avg_latency_ms": 100.0,
                    "median_latency_ms": 100.0,
                    "p95_latency_ms": 100.0,
                },
            },
            "vs_reference": None,
        }
        from pathlib import Path

        schema_path = Path("data/eval/schemas/metrics_summarization_v2.json")
        if not schema_path.is_file():
            pytest.skip("metrics_summarization_v2.json not in tree")
        ok = validate_metrics_summarization(metrics_v2, strict=False)
        assert ok is True


@pytest.mark.unit
class TestValidateMetricsNer:
    """Test validate_metrics_ner."""

    def test_lenient_when_schema_missing(self):
        """Does not raise when schema file missing (default strict=False)."""
        validate_metrics_ner({"run_id": "r1"})
