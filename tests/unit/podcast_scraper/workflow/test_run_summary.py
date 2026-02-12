"""Tests for workflow run summary generation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from podcast_scraper.workflow.run_summary import create_run_summary, save_run_summary


@pytest.mark.unit
class TestCreateRunSummary:
    """Test create_run_summary."""

    def test_minimal_no_manifest_no_metrics(self):
        """Creates summary with schema and run_id when no manifest/metrics."""
        out = create_run_summary(None, None, "/tmp/out")
        assert out["schema_version"] == "1.0.0"
        assert "run_id" in out
        assert "created_at" in out
        assert out["index_file"] == "index.json"
        assert out["run_manifest_file"] == "run_manifest.json"

    def test_with_manifest_to_dict(self):
        """Includes manifest when it has to_dict."""
        manifest = MagicMock()
        manifest.to_dict.return_value = {"episodes": 5}
        out = create_run_summary(manifest, None, "/tmp/out")
        assert out["manifest"] == {"episodes": 5}

    def test_with_manifest_plain_dict(self):
        """Includes manifest when it is a dict."""
        manifest = {"episodes": 3}
        out = create_run_summary(manifest, None, "/tmp/out")
        assert out["manifest"] == {"episodes": 3}

    def test_with_metrics_finish(self):
        """Includes metrics from pipeline_metrics.finish()."""
        metrics = MagicMock()
        metrics.finish.return_value = {"run_duration_seconds": 10.0}
        out = create_run_summary(None, metrics, "/tmp/out")
        assert out["metrics"]["run_duration_seconds"] == 10.0

    def test_with_metrics_attributes(self):
        """Includes metrics from pipeline_metrics attributes when no finish()."""
        metrics = MagicMock(spec=["run_duration_seconds", "episodes_scraped_total"])
        metrics.run_duration_seconds = 5.0
        metrics.episodes_scraped_total = 10
        del metrics.finish
        out = create_run_summary(None, metrics, "/tmp/out")
        assert out["metrics"]["run_duration_seconds"] == 5.0
        assert out["metrics"]["episodes_scraped_total"] == 10

    def test_custom_run_id(self):
        """Uses provided run_id when given."""
        out = create_run_summary(None, None, "/tmp/out", run_id="custom-123")
        assert out["run_id"] == "custom-123"


@pytest.mark.unit
class TestSaveRunSummary:
    """Test save_run_summary."""

    def test_saves_json(self, tmp_path):
        """Saves run summary as JSON file."""
        summary = {"run_id": "r1", "schema_version": "1.0.0"}
        path = save_run_summary(summary, str(tmp_path), "run.json")
        assert path == str(tmp_path / "run.json")
        assert (tmp_path / "run.json").exists()
        loaded = json.loads((tmp_path / "run.json").read_text())
        assert loaded["run_id"] == "r1"

    def test_creates_parent_dirs(self, tmp_path):
        """Creates parent directories when needed."""
        out_dir = tmp_path / "a" / "b" / "c"
        path = save_run_summary({"run_id": "r1"}, str(out_dir))
        assert Path(path).exists()
