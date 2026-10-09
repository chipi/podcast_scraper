"""check_doc_structure skips private repos mounted at the top level, and nothing else.

ADR-162 clones the private Common and Player repos into ``apps/`` (as the eval research repo
goes into ``eval-data/``). Their docs link by their own layout, so walking them turned the gate
red for anyone with a mount. The skip is top-level only: a public directory that happens to be
named ``apps`` deeper in the tree is still checked.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "tools" / "check_doc_structure.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("check_doc_structure_mounts", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Path]:
    mod = _load()
    for rel in ("docs/a.md", "apps/player/README.md", "eval-data/README.md", "docs/apps/b.md"):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# t\n")
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    return mod, tmp_path


def test_top_level_mounts_are_skipped(tree: tuple[Any, Path]) -> None:
    mod, root = tree
    found = {p.relative_to(root).as_posix() for p in mod.markdown_files()}
    assert "apps/player/README.md" not in found
    assert "eval-data/README.md" not in found


def test_same_name_deeper_is_still_checked(tree: tuple[Any, Path]) -> None:
    mod, root = tree
    found = {p.relative_to(root).as_posix() for p in mod.markdown_files()}
    assert {"docs/a.md", "docs/apps/b.md"} <= found
