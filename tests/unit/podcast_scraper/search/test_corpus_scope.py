"""Unit tests for corpus discovery helpers (GitHub #505 follow-up)."""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.search.corpus_scope import (
    discover_metadata_files,
    episode_root_from_metadata_path,
    gi_map_lookup_key_from_vector_meta,
    index_fingerprint_scope_key,
    normalize_feed_id,
    vector_doc_scope_tag,
)


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


@pytest.mark.unit
def test_normalize_feed_id_strips_and_rejects_empty() -> None:
    assert normalize_feed_id("  abc  ") == "abc"
    assert normalize_feed_id("") is None
    assert normalize_feed_id("   ") is None
    assert normalize_feed_id(42) is None


@pytest.mark.unit
def test_index_fingerprint_scope_key_and_vector_tag() -> None:
    assert index_fingerprint_scope_key(None, "e1") == "e1"
    assert index_fingerprint_scope_key("f1", "e1") == "f1\x1fe1"
    assert vector_doc_scope_tag("", "e1") == "e1"
    assert vector_doc_scope_tag("a/b@c", "e1") == "a_b_c__e1"


@pytest.mark.unit
def test_gi_map_lookup_key_from_vector_meta() -> None:
    assert gi_map_lookup_key_from_vector_meta({}) == ""
    assert gi_map_lookup_key_from_vector_meta({"episode_id": ""}) == ""
    assert gi_map_lookup_key_from_vector_meta({"episode_id": "x", "feed_id": "f"}) == "f\x1fx"


@pytest.mark.unit
def test_episode_root_from_metadata_path(tmp_path: Path) -> None:
    meta = tmp_path / "run" / "metadata" / "ep.metadata.json"
    meta.parent.mkdir(parents=True)
    assert episode_root_from_metadata_path(meta) == (tmp_path / "run").resolve()


@pytest.mark.unit
def test_discover_metadata_flat_layout_without_feeds(tmp_path: Path) -> None:
    """Single top-level ``metadata/`` (no ``feeds/``) still discovers files."""
    m = tmp_path / "metadata"
    m.mkdir(parents=True)
    (m / "a.metadata.yml").write_text("{}", encoding="utf-8")
    names = sorted(p.name for p in discover_metadata_files(tmp_path))
    assert names == ["a.metadata.yml"]
