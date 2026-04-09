"""Unit tests for corpus discovery helpers (GitHub #505 follow-up)."""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.search.corpus_scope import discover_metadata_files


@pytest.mark.unit
def test_discover_metadata_files_merges_parent_metadata_when_feeds_present(tmp_path: Path) -> None:
    """Hybrid layout: ``feeds/…/metadata`` plus top-level ``metadata`` are both indexed."""
    root = tmp_path / "corpus"
    (root / "feeds" / "rss_a_1" / "metadata").mkdir(parents=True)
    (root / "feeds" / "rss_a_1" / "metadata" / "ep1.metadata.json").write_text(
        "{}", encoding="utf-8"
    )
    (root / "metadata").mkdir(parents=True)
    (root / "metadata" / "legacy.metadata.json").write_text("{}", encoding="utf-8")

    found = discover_metadata_files(root)
    names = sorted(p.name for p in found)
    assert names == ["ep1.metadata.json", "legacy.metadata.json"]


@pytest.mark.unit
def test_discover_metadata_files_rejects_traversal_in_root(tmp_path: Path) -> None:
    """User roots with ``..`` segments yield no metadata (path hardening)."""
    assert discover_metadata_files(tmp_path / "x" / ".." / "y") == []
