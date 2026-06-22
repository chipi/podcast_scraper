"""Integration: subprocess smoke for ``scripts/eval/experiment/run_experiment.py`` (GI/KG eval_stub)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from podcast_scraper.evaluation.experiment_config import PODCAST_EVAL_MATERIALIZED_ROOT_ENV

pytestmark = pytest.mark.integration


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _eval_subprocess_env(repo: Path) -> dict[str, str]:
    env = dict(os.environ)
    env[PODCAST_EVAL_MATERIALIZED_ROOT_ENV] = str(repo / "tests/fixtures/eval/materialized")
    # ``run_experiment.py`` imports ``scripts.*``; subprocess needs repo root on ``PYTHONPATH``.
    root = str(repo)
    prev = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = root if not prev else f"{root}{os.pathsep}{prev}"
    return env


def test_run_experiment_gil_stub_dry_run_writes_predictions(tmp_path: Path) -> None:
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
                "  dataset_id: integration_gi_kg_stub_v1",
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
            env=_eval_subprocess_env(repo),
        )
        assert proc.returncode == 0, (proc.stdout, proc.stderr)
        pred = runs_dir / "predictions.jsonl"
        assert pred.is_file(), proc.stderr
        first = pred.read_text(encoding="utf-8").strip().splitlines()[0]
        rec = json.loads(first)
        assert "gil" in rec.get("output", {})
        # RFC-097 v3.0 chunk-5 regression: the eval harness now threads the
        # dataset's per-episode ``duration_minutes`` to ``gi.build_artifact``
        # as ``episode_duration_ms``. End-to-end check: Episode.duration_ms
        # must be present in the artifact and match the fixture's
        # ``duration_minutes: 0.1`` (6 seconds → 6000 ms).
        gil = rec["output"]["gil"]
        episode_nodes = [n for n in gil.get("nodes", []) if n.get("type") == "Episode"]
        assert episode_nodes, "no Episode node in stub GI artifact"
        ep_props = episode_nodes[0].get("properties") or {}
        assert ep_props.get("duration_ms") == 6000, (
            f"Episode.duration_ms regression: expected 6000 (from dataset "
            f"duration_minutes=0.1), got {ep_props.get('duration_ms')!r}. "
            f"Indicates the eval harness lost the duration-threading wired "
            f"in RFC-097 chunk-5."
        )
    finally:
        if runs_dir.exists():
            shutil.rmtree(runs_dir, ignore_errors=True)


def test_run_experiment_kg_stub_dry_run_writes_predictions(tmp_path: Path) -> None:
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
                "  dataset_id: integration_gi_kg_stub_v1",
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
            env=_eval_subprocess_env(repo),
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
