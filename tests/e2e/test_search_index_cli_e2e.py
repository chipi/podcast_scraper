"""Pytest E2E: ``index`` and ``search`` CLI entrypoints (semantic corpus search, #484)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

pytestmark = [pytest.mark.e2e, pytest.mark.critical_path]


def _run_cli(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", "podcast_scraper.cli", *args]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(cwd),
        timeout=120,
        env=os.environ.copy(),
    )


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


class TestSearchIndexCliSubprocessE2E:
    def test_index_stats_exits_no_index(self, project_root: Path, tmp_path: Path) -> None:
        """``index --stats`` with empty corpus returns exit code 3 (no index)."""
        result = _run_cli(
            ["index", "--output-dir", str(tmp_path), "--stats"],
            cwd=project_root,
        )
        assert result.returncode == 3, (
            f"expected EXIT_NO_ARTIFACTS (3), got {result.returncode}: " f"stderr={result.stderr!r}"
        )

    def test_search_exits_no_index(self, project_root: Path, tmp_path: Path) -> None:
        """``search`` with no FAISS index returns exit code 3."""
        result = _run_cli(
            ["search", "climate", "--output-dir", str(tmp_path)],
            cwd=project_root,
        )
        assert (
            result.returncode == 3
        ), f"expected no index exit 3, got {result.returncode}: stderr={result.stderr!r}"
