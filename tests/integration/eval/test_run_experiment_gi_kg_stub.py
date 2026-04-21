"""Integration: subprocess smoke for ``scripts/eval/experiment/run_experiment.py`` (GI/KG eval_stub)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _materialized_smoke_ready() -> bool:
    root = _repo_root()
    d = root / "data/eval/materialized/curated_5feeds_smoke_v1"
    return d.is_dir() and any(d.glob("*.txt"))


def test_run_experiment_gil_stub_dry_run_writes_predictions(tmp_path: Path) -> None:
    if not _materialized_smoke_ready():
        pytest.skip("Materialized curated_5feeds_smoke_v1 transcripts not found")
    repo = _repo_root()
    run_id = f"it_gil_stub_{uuid.uuid4().hex[:12]}"
    cfg_path = tmp_path / "gil_stub_one_ep.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                f"id: {run_id}",
                "task: grounded_insights",
                "backend:",
                "  type: eval_stub",
                "data:",
                "  dataset_id: curated_5feeds_smoke_v1",
                "  max_episodes: 1",
                "preprocessing_profile: cleaning_v3",
                "params:",
                "  gi_insight_source: stub",
                "  gi_require_grounding: false",
                "  gi_max_insights: 5",
                "",
            ]
        ),
        encoding="utf-8",
    )
    runs_dir = repo / "data/eval/runs" / run_id
    try:
        proc = subprocess.run(
            [
                sys.executable,
                str(repo / "scripts/eval/experiment/run_experiment.py"),
                str(cfg_path),
                "--dry-run",
                "--force",
            ],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert proc.returncode == 0, (proc.stdout, proc.stderr)
        pred = runs_dir / "predictions.jsonl"
        assert pred.is_file(), proc.stderr
        first = pred.read_text(encoding="utf-8").strip().splitlines()[0]
        rec = json.loads(first)
        assert "gil" in rec.get("output", {})
    finally:
        if runs_dir.exists():
            shutil.rmtree(runs_dir, ignore_errors=True)


def test_run_experiment_kg_stub_dry_run_writes_predictions(tmp_path: Path) -> None:
    if not _materialized_smoke_ready():
        pytest.skip("Materialized curated_5feeds_smoke_v1 transcripts not found")
    repo = _repo_root()
    run_id = f"it_kg_stub_{uuid.uuid4().hex[:12]}"
    cfg_path = tmp_path / "kg_stub_one_ep.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                f"id: {run_id}",
                "task: knowledge_graph",
                "backend:",
                "  type: eval_stub",
                "data:",
                "  dataset_id: curated_5feeds_smoke_v1",
                "  max_episodes: 1",
                "preprocessing_profile: cleaning_v3",
                "params:",
                "  kg_extraction_source: stub",
                "",
            ]
        ),
        encoding="utf-8",
    )
    runs_dir = repo / "data/eval/runs" / run_id
    try:
        proc = subprocess.run(
            [
                sys.executable,
                str(repo / "scripts/eval/experiment/run_experiment.py"),
                str(cfg_path),
                "--dry-run",
                "--force",
            ],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert proc.returncode == 0, (proc.stdout, proc.stderr)
        pred = runs_dir / "predictions.jsonl"
        assert pred.is_file()
        first = pred.read_text(encoding="utf-8").strip().splitlines()[0]
        rec = json.loads(first)
        assert "kg" in rec.get("output", {})
    finally:
        if runs_dir.exists():
            shutil.rmtree(runs_dir, ignore_errors=True)
