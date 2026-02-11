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

    def test_calls_validate_schema(self):
        """Does not raise when schema path does not exist (lenient)."""
        validate_summarization_reference({"episode_id": "ep1"})


@pytest.mark.unit
class TestValidateMetricsSummarization:
    """Test validate_metrics_summarization."""

    def test_lenient_when_schema_missing(self):
        """Does not raise when schema file missing (default strict=False)."""
        validate_metrics_summarization({"run_id": "r1"})


@pytest.mark.unit
class TestValidateMetricsNer:
    """Test validate_metrics_ner."""

    def test_lenient_when_schema_missing(self):
        """Does not raise when schema file missing (default strict=False)."""
        validate_metrics_ner({"run_id": "r1"})
