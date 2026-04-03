"""E2E subprocess tests for search/index CLIs (#484 Step 4)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def _run_cli(args: list[str], cwd: Path, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", "podcast_scraper.cli"] + args
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(cwd),
        timeout=timeout,
    )


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


@pytest.mark.e2e
@pytest.mark.critical_path
def test_search_subprocess_requires_output_dir(project_root: Path) -> None:
    proc = _run_cli(["search", "hello"], cwd=project_root)
    assert proc.returncode == 2


@pytest.mark.e2e
@pytest.mark.critical_path
def test_search_subprocess_missing_index(project_root: Path, tmp_path: Path) -> None:
    proc = _run_cli(
        ["search", "query", "--output-dir", str(tmp_path)],
        cwd=project_root,
    )
    assert proc.returncode == 3


@pytest.mark.e2e
@pytest.mark.critical_path
def test_index_subprocess_stats_missing_index(project_root: Path, tmp_path: Path) -> None:
    proc = _run_cli(
        ["index", "--output-dir", str(tmp_path), "--stats"],
        cwd=project_root,
    )
    assert proc.returncode == 3


@pytest.mark.e2e
@pytest.mark.critical_path
def test_index_subprocess_accepts_vector_flags(project_root: Path, tmp_path: Path) -> None:
    """CLI wiring: vector index mode / type filters parse and run (empty corpus OK)."""
    proc = _run_cli(
        [
            "index",
            "--output-dir",
            str(tmp_path),
            "--vector-faiss-index-mode",
            "flat",
            "--vector-index-types",
            "insight,summary",
        ],
        cwd=project_root,
    )
    assert proc.returncode == 0, proc.stderr
