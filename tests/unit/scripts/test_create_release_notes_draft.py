"""Unit tests for scripts/tools/create_release_notes_draft.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]

_SPEC = importlib.util.spec_from_file_location(
    "create_release_notes_draft_under_test",
    ROOT / "scripts" / "tools" / "create_release_notes_draft.py",
)
assert _SPEC and _SPEC.loader
_crd = importlib.util.module_from_spec(_SPEC)
sys.modules["create_release_notes_draft_under_test"] = _crd
_SPEC.loader.exec_module(_crd)

pytestmark = [pytest.mark.unit]


def test_check_compatibility_matrix_fails_when_version_missing(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir(parents=True)
    (tmp_path / "docs" / "COMPATIBILITY.md").write_text(
        "# Matrix\n\n| 2.6.0 | prior |\n",
        encoding="utf-8",
    )
    assert _crd._check_compatibility_matrix(tmp_path, "2.7.0") is False


def test_check_compatibility_matrix_passes_when_version_present(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir(parents=True)
    (tmp_path / "docs" / "COMPATIBILITY.md").write_text(
        "# Matrix\n\n| 2.7.0 | current |\n",
        encoding="utf-8",
    )
    assert _crd._check_compatibility_matrix(tmp_path, "2.7.0") is True
