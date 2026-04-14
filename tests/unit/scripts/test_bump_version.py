"""Unit tests for scripts/tools/bump_version.py."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]

_SPEC = importlib.util.spec_from_file_location(
    "bump_version_under_test",
    ROOT / "scripts" / "tools" / "bump_version.py",
)
assert _SPEC and _SPEC.loader
_bv = importlib.util.module_from_spec(_SPEC)
sys.modules["bump_version_under_test"] = _bv
_SPEC.loader.exec_module(_bv)

bump = _bv.bump

pytestmark = [pytest.mark.unit]


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _init_git_repo(repo: Path) -> None:
    _git(repo, "init")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "Test")


def _write_version_files(repo: Path, version: str) -> None:
    (repo / "pyproject.toml").write_text(
        f'[project]\nname = "t"\nversion = "{version}"\n',
        encoding="utf-8",
    )
    init_dir = repo / "src" / "podcast_scraper"
    init_dir.mkdir(parents=True)
    (init_dir / "__init__.py").write_text(f'__version__ = "{version}"\n', encoding="utf-8")


def test_bump_updates_both_files(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    _write_version_files(tmp_path, "1.0.0")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-m", "init")
    bump(tmp_path, "2.0.0")
    assert 'version = "2.0.0"' in (tmp_path / "pyproject.toml").read_text(encoding="utf-8")
    assert '__version__ = "2.0.0"' in (
        tmp_path / "src" / "podcast_scraper" / "__init__.py"
    ).read_text(
        encoding="utf-8",
    )


def test_bump_rejects_dirty_tree_without_flag(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    _write_version_files(tmp_path, "1.0.0")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-m", "init")
    (tmp_path / "extra.txt").write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        bump(tmp_path, "2.0.0")
    assert exc.value.code == 1


def test_bump_allows_dirty_with_flag(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    _write_version_files(tmp_path, "1.0.0")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-m", "init")
    (tmp_path / "extra.txt").write_text("x", encoding="utf-8")
    bump(tmp_path, "2.0.0", allow_dirty=True)


def test_bump_rejects_existing_tag(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    _write_version_files(tmp_path, "1.0.0")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-m", "init")
    _git(tmp_path, "tag", "v2.0.0")
    with pytest.raises(SystemExit) as exc:
        bump(tmp_path, "2.0.0")
    assert exc.value.code == 1


def test_bump_allows_existing_tag_with_force(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    _write_version_files(tmp_path, "1.0.0")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-m", "init")
    _git(tmp_path, "tag", "v2.0.0")
    bump(tmp_path, "2.0.0", force_tag=True)
