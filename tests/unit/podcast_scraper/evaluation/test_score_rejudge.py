"""Behavior tests for the ``--rejudge`` flag on the autoresearch score CLI.

Full end-to-end coverage lives in the local sweep smoke; here we verify:
- ``--rejudge`` is recognized on the CLI
- ``--rejudge`` + ``--dry-run`` are mutually exclusive
- ``--rejudge`` without existing predictions surfaces a clear error

These are subprocess tests because ``score.py`` is a script (has a
side-effectful module-level import block); running it via ``python -m``
avoids polluting the pytest process with the DEFAULT_BASE_CONFIG resolution.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
SCORE_PY = REPO_ROOT / "autoresearch/initial_prompt_tuning/prompt_tuning/eval/score.py"


def _run_score(*extra_args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    # score.py eagerly imports the autoresearch eval stack (~12 s cold — the transitive eval
    # dependency graph, not the flags being tested). Even --help pays it. Under the full parallel
    # suite (xdist, many workers each loading ML models) that cold import contends for CPU and blows
    # a tight timeout non-deterministically — the flake this generous ceiling removes. A genuinely
    # hung CLI still fails, just later. (A faster fix would defer score.py's module-level imports;
    # tracked separately as an eval-tooling cleanup.)
    return subprocess.run(
        [sys.executable, str(SCORE_PY), *extra_args],
        cwd=str(cwd or REPO_ROOT),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        timeout=180,
    )


def test_rejudge_flag_recognized() -> None:
    result = _run_score("--help")
    assert result.returncode == 0
    assert "--rejudge" in result.stdout
    # The help text should mention what --rejudge does at a high level.
    assert "existing predictions" in result.stdout


def test_rejudge_and_dry_run_mutually_exclusive() -> None:
    """Both flags together must fail cleanly with a specific error, not
    silently pick one. Prevents an operator from accidentally combining
    them and getting non-obvious behavior."""
    result = _run_score(
        "--rejudge",
        "--dry-run",
        "--config",
        "data/eval/configs/summarization/autoresearch_prompt_ollama_smoke_paragraph_sweep_v1.yaml",
    )
    assert result.returncode == 1
    assert "mutually exclusive" in result.stderr


def test_rejudge_without_predictions_fails_clearly(tmp_path: Path) -> None:
    """--rejudge with a config whose run_id has no predictions.jsonl yet
    must fail with a clear error naming both the mode and the missing
    file. Silent regeneration would defeat the whole purpose of the flag."""
    # Point at a config that resolves to a run_id that definitely won't
    # have predictions on disk. The sweep uses ``id: autoresearch_prompt_ollama_<...>_sweep``
    # naming; a random config id won't collide with any real run dir.
    fake_cfg = tmp_path / "fake_config.yaml"
    fake_cfg.write_text(
        "id: autoresearch_pytest_nonexistent_run_that_never_exists\n"
        "task: summarization\n"
        "backend:\n"
        "  type: ollama\n"
        "  model: llama3.1:8b\n"
        "prompts:\n"
        "  system: ollama/llama3.1_8b/summarization/system_v1\n"
        "  user: ollama/llama3.1_8b/summarization/long_v1\n"
        "data:\n"
        "  dataset_id: curated_5feeds_smoke_v2\n"
        "params:\n"
        "  max_length: 800\n"
        "  min_length: 200\n"
        "  temperature: 0.0\n",
        encoding="utf-8",
    )

    result = _run_score("--rejudge", "--config", str(fake_cfg))
    assert result.returncode == 1
    # Error must name --rejudge (not --dry-run) so the operator knows
    # exactly which mode's precondition failed.
    assert "--rejudge" in result.stderr
    assert "predictions.jsonl" in result.stderr
