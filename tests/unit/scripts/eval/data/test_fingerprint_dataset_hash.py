"""RFC-097 fingerprint gap closure (FINGERPRINT_GAPS_ANALYSIS_2026-06-22.md §6):
dataset_content_hash makes the dataset's CONTENT a fingerprint dimension.
Operator case #4 (2026-06-22): "running different versions of datasets" used
to look identical because only the string dataset_id was captured.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.eval.data.materialize_baseline import (  # noqa: E402
    _compute_dataset_content_hash,
)

pytestmark = pytest.mark.unit


def test_returns_none_for_nonexistent_dataset(tmp_path: Path, monkeypatch) -> None:
    """No materialized directory → None, fingerprint must not raise."""
    monkeypatch.chdir(tmp_path)
    assert _compute_dataset_content_hash("does_not_exist_v1") is None


def test_returns_none_for_empty_directory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "eval" / "materialized" / "empty_v1").mkdir(parents=True)
    assert _compute_dataset_content_hash("empty_v1") is None


def test_returns_stable_hash_for_unchanged_content(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data" / "eval" / "materialized" / "stable_v1"
    root.mkdir(parents=True)
    (root / "a.txt").write_text("alpha")
    (root / "b.txt").write_text("beta")
    h1 = _compute_dataset_content_hash("stable_v1")
    h2 = _compute_dataset_content_hash("stable_v1")
    assert h1 == h2
    assert h1 is not None
    assert len(h1) == 64  # sha256 hex


def test_hash_changes_when_file_content_changes(tmp_path: Path, monkeypatch) -> None:
    """The headline regression this prevents: edit a transcript in place,
    keep dataset_id the same → fingerprint must NOT collide."""
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data" / "eval" / "materialized" / "edit_v1"
    root.mkdir(parents=True)
    (root / "ep.txt").write_text("original")
    h_before = _compute_dataset_content_hash("edit_v1")
    (root / "ep.txt").write_text("edited")
    h_after = _compute_dataset_content_hash("edit_v1")
    assert h_before != h_after


def test_hash_changes_when_file_added(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data" / "eval" / "materialized" / "addfile_v1"
    root.mkdir(parents=True)
    (root / "a.txt").write_text("alpha")
    h_one_file = _compute_dataset_content_hash("addfile_v1")
    (root / "b.txt").write_text("beta")
    h_two_files = _compute_dataset_content_hash("addfile_v1")
    assert h_one_file != h_two_files


def test_hash_independent_of_file_listing_order(tmp_path: Path, monkeypatch) -> None:
    """Hash must be deterministic regardless of filesystem traversal order.
    Sorting is required for cross-platform / cross-machine stability."""
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data" / "eval" / "materialized" / "order_v1"
    root.mkdir(parents=True)
    files = ["z_last.txt", "a_first.txt", "m_middle.txt"]
    for name in files:
        (root / name).write_text(name)
    h1 = _compute_dataset_content_hash("order_v1")
    # Touch every file (changes mtime but not content); hash must be identical
    for name in reversed(files):
        os.utime(root / name, None)
    h2 = _compute_dataset_content_hash("order_v1")
    assert h1 == h2


def test_hash_traverses_subdirectories(tmp_path: Path, monkeypatch) -> None:
    """Materialized datasets have meta.json at root + per-episode files; some
    layouts have subdirs (feed_id grouping). Hash must walk recursively."""
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data" / "eval" / "materialized" / "nested_v1"
    (root / "feeds" / "p01").mkdir(parents=True)
    (root / "meta.json").write_text('{"x": 1}')
    (root / "feeds" / "p01" / "ep.txt").write_text("transcript")
    h = _compute_dataset_content_hash("nested_v1")
    assert h is not None
    # Edit the nested file; hash must change
    (root / "feeds" / "p01" / "ep.txt").write_text("transcript-v2")
    assert _compute_dataset_content_hash("nested_v1") != h
