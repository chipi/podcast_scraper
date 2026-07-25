"""Unit tests for scripts/pre_release_check.py (ADR-031)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]

_SPEC = importlib.util.spec_from_file_location(
    "pre_release_check_under_test",
    ROOT / "scripts" / "pre_release_check.py",
)
assert _SPEC and _SPEC.loader
_prc = importlib.util.module_from_spec(_SPEC)
sys.modules["pre_release_check_under_test"] = _prc
_SPEC.loader.exec_module(_prc)

run_checks = _prc.run_checks
check_perf_report = _prc.check_perf_report

pytestmark = [pytest.mark.unit]


def _write_perf_report(base: Path, version: str) -> None:
    d = base / "docs" / "guides" / "perf-traces" / "reports"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"2026-01-01-{version}.md").write_text(f"# perf {version}\n", encoding="utf-8")


def test_check_perf_report_found_by_filename(tmp_path: Path) -> None:
    _write_perf_report(tmp_path, "1.0.0")
    assert check_perf_report(tmp_path, "1.0.0") is True


def test_check_perf_report_missing_is_non_gating(tmp_path: Path, capsys) -> None:
    # No reports dir at all — must warn (stderr) but NEVER raise / exit.
    assert check_perf_report(tmp_path, "1.0.0") is False
    assert "NO PERF REPORT" in capsys.readouterr().err


def test_check_perf_report_content_mention_does_not_falsely_match(tmp_path: Path) -> None:
    # A report FOR 1.0.1 that merely mentions 1.0.0 in prose must not satisfy 1.0.0.
    d = tmp_path / "docs" / "guides" / "perf-traces" / "reports"
    d.mkdir(parents=True)
    (d / "2026-01-01-1.0.1.md").write_text("# perf 1.0.1 (prev was 1.0.0)\n", encoding="utf-8")
    assert check_perf_report(tmp_path, "1.0.0") is False


def test_run_checks_perf_report_absence_does_not_fail(tmp_path: Path) -> None:
    # A final release with all release docs but NO perf report still passes
    # run_checks (perf report is forced-but-non-gating).
    _write_minimal_release_tree(tmp_path, version="1.0.0")
    assert run_checks(tmp_path) == "1.0.0"


def _write_minimal_release_tree(
    base: Path, *, version: str, init_version: str | None = None
) -> None:
    init_version = init_version if init_version is not None else version
    (base / "pyproject.toml").write_text(
        f'[project]\nname = "t"\nversion = "{version}"\n',
        encoding="utf-8",
    )
    init_dir = base / "src" / "podcast_scraper"
    init_dir.mkdir(parents=True)
    (init_dir / "__init__.py").write_text(f'__version__ = "{init_version}"\n', encoding="utf-8")
    rel = base / "docs" / "releases"
    rel.mkdir(parents=True)
    (rel / f"RELEASE_v{version}.md").write_text(
        f"# Release v{version}\n\nShips {version}.\n",
        encoding="utf-8",
    )
    (rel / "index.md").write_text(
        f"# History\n\n- [{version}](RELEASE_v{version}.md)\n",
        encoding="utf-8",
    )
    (base / "docs" / "COMPATIBILITY.md").write_text(
        f"# Matrix\n\n| Code | Notes |\n| --- | --- |\n| {version} | ok |\n",
        encoding="utf-8",
    )


def test_run_checks_passes_when_consistent(tmp_path: Path) -> None:
    _write_minimal_release_tree(tmp_path, version="1.0.0")
    assert run_checks(tmp_path) == "1.0.0"


def test_run_checks_fails_on_version_mismatch(tmp_path: Path) -> None:
    _write_minimal_release_tree(tmp_path, version="1.0.0", init_version="0.9.0")
    with pytest.raises(SystemExit) as exc:
        run_checks(tmp_path)
    assert exc.value.code == 1


def test_run_checks_fails_when_release_file_missing(tmp_path: Path) -> None:
    _write_minimal_release_tree(tmp_path, version="1.0.0")
    (tmp_path / "docs" / "releases" / "RELEASE_v1.0.0.md").unlink()
    with pytest.raises(SystemExit) as exc:
        run_checks(tmp_path)
    assert exc.value.code == 1


def test_run_checks_fails_when_index_missing_version(tmp_path: Path) -> None:
    _write_minimal_release_tree(tmp_path, version="1.0.0")
    (tmp_path / "docs" / "releases" / "index.md").write_text("# No links\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        run_checks(tmp_path)
    assert exc.value.code == 1


def test_run_checks_fails_when_compatibility_matrix_missing_version(tmp_path: Path) -> None:
    _write_minimal_release_tree(tmp_path, version="1.0.0")
    (tmp_path / "docs" / "COMPATIBILITY.md").write_text(
        "# Matrix\n\n| 9.9.9 | old |\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc:
        run_checks(tmp_path)
    assert exc.value.code == 1
