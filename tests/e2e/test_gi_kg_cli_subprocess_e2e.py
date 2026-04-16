#!/usr/bin/env python3
"""Subprocess E2E smoke for GI and KG CLI entrypoints.

Validates real ``python -m podcast_scraper.cli gi …`` and ``kg …`` invocations using
repo fixtures. Complements unit tests in ``tests/unit/podcast_scraper/test_cli.py`` and
``tests/unit/podcast_scraper/kg/test_kg_cli.py``.

These subprocess calls contribute to **pytest E2E** coverage of ``gi`` and ``kg`` like any other
E2E subprocess invoking ``python -m podcast_scraper.cli``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)


def run_cli_subprocess(
    args: list[str],
    cwd: Path,
    timeout: int = 120,
) -> subprocess.CompletedProcess:
    """Run ``python -m podcast_scraper.cli`` with the given args.

    Same pattern as ``test_cli_subprocess_e2e``.
    """
    cmd = [sys.executable, "-m", "podcast_scraper.cli", *args]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(cwd),
        timeout=timeout,
        env=os.environ.copy(),
    )


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


@pytest.mark.e2e
@pytest.mark.critical_path
class TestGiKgCliSubprocessE2E:
    """Fast subprocess checks for GI/KG CLI."""

    def test_gi_validate_strict_via_subprocess(self, project_root: Path, tmp_path: Path) -> None:
        """``gi validate --strict`` succeeds on a corpus layout with one .gi.json artifact."""
        src = (
            project_root
            / "tests"
            / "fixtures"
            / "gil_kg_ci_enforce"
            / "metadata"
            / "ci_sample.gi.json"
        )
        assert src.is_file(), f"Missing fixture: {src}"
        meta = tmp_path / "metadata"
        meta.mkdir(parents=True)
        shutil.copy2(src, meta / "ep_ci_sample.gi.json")

        result = run_cli_subprocess(
            ["gi", "validate", "--strict", str(tmp_path)],
            cwd=project_root,
        )
        assert (
            result.returncode == 0
        ), f"gi validate failed: stdout={result.stdout!r} stderr={result.stderr!r}"

    def test_kg_validate_strict_via_subprocess(self, project_root: Path) -> None:
        """``kg validate --strict`` succeeds on the shared minimal KG fixture directory."""
        kg_dir = project_root / "tests" / "fixtures" / "kg"
        assert kg_dir.is_dir(), f"Missing fixture dir: {kg_dir}"

        result = run_cli_subprocess(
            ["kg", "validate", str(kg_dir.relative_to(project_root)), "--strict"],
            cwd=project_root,
        )
        assert (
            result.returncode == 0
        ), f"kg validate failed: stdout={result.stdout!r} stderr={result.stderr!r}"

    def test_kg_inspect_episode_path_via_subprocess(self, project_root: Path) -> None:
        """``kg inspect --episode-path`` prints JSON for a single artifact (CLI dispatch)."""
        kg_path = project_root / "tests" / "fixtures" / "kg" / "minimal.kg.json"
        assert kg_path.is_file(), f"Missing fixture: {kg_path}"
        rel = kg_path.relative_to(project_root)

        result = run_cli_subprocess(
            [
                "kg",
                "inspect",
                "--episode-path",
                str(rel),
                "--format",
                "json",
            ],
            cwd=project_root,
        )
        assert (
            result.returncode == 0
        ), f"kg inspect failed: stdout={result.stdout!r} stderr={result.stderr!r}"
        assert "fixture:minimal-kg" in result.stdout
        assert "episode_id" in result.stdout

    def test_gi_inspect_episode_path_json_via_subprocess(
        self,
        project_root: Path,
    ) -> None:
        """``gi inspect --episode-path … --format json`` exercises GI CLI dispatch."""
        gi_path = (
            project_root
            / "tests"
            / "fixtures"
            / "gil_kg_ci_enforce"
            / "metadata"
            / "ci_sample.gi.json"
        )
        assert gi_path.is_file(), f"Missing fixture: {gi_path}"
        rel = gi_path.relative_to(project_root)

        result = run_cli_subprocess(
            [
                "gi",
                "inspect",
                "--episode-path",
                str(rel),
                "--format",
                "json",
            ],
            cwd=project_root,
        )
        assert (
            result.returncode == 0
        ), f"gi inspect failed: stdout={result.stdout!r} stderr={result.stderr!r}"
        assert "ci-fixture" in result.stdout or "episode_id" in result.stdout
