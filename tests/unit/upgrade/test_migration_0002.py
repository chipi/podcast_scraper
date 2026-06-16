"""Unit tests for the native two-tier reindex migration (0002, #862 / B).

Stubs ``build_two_tier_index`` so the migration's own gating logic — no-op when an
index exists, build when absent, dry-run — is tested without LanceDB.
"""

from __future__ import annotations

import pytest

from podcast_scraper.search import two_tier_indexer
from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations.m0002_two_tier_native_reindex import (
    TwoTierNativeReindexMigration,
)

pytestmark = pytest.mark.unit


def _ctx(tmp_path, dry_run=False):
    return MigrationContext(corpus_root=tmp_path, dry_run=dry_run)


def test_noop_when_index_and_sidecar_already_exist(tmp_path, monkeypatch):
    (tmp_path / "search" / "lance_index").mkdir(parents=True)  # 0001 already built it
    (tmp_path / "search" / "metadata.json").write_text("{}")  # + the #1010 offset sidecar

    def _boom(*a, **k):
        raise AssertionError("must not rebuild when an index + sidecar already exist")

    monkeypatch.setattr(two_tier_indexer, "build_two_tier_index", _boom)
    result = TwoTierNativeReindexMigration().apply(_ctx(tmp_path))
    assert result.applied and "no-op" in result.message


def test_reindexes_when_metadata_sidecar_missing(tmp_path, monkeypatch):
    """A healthy pre-#1010 index lacks search/metadata.json — reindex to emit the offset sidecar."""
    (tmp_path / "search" / "lance_index").mkdir(parents=True)  # present + non-stale, but no sidecar

    calls = {}

    def _fake(corpus, lance, **k):
        calls["built"] = True
        return two_tier_indexer.TwoTierIndexStats(episodes=1, segments=2, insights=1)

    monkeypatch.setattr(two_tier_indexer, "build_two_tier_index", _fake)
    result = TwoTierNativeReindexMigration().apply(_ctx(tmp_path))
    assert calls.get("built") is True  # sidecar missing → reindex, not no-op
    assert result.applied


def test_builds_natively_when_absent(tmp_path, monkeypatch):
    calls = {}

    def _fake(corpus, lance, **k):
        calls["corpus"] = corpus
        return two_tier_indexer.TwoTierIndexStats(episodes=3, segments=10, insights=7)

    monkeypatch.setattr(two_tier_indexer, "build_two_tier_index", _fake)
    result = TwoTierNativeReindexMigration().apply(_ctx(tmp_path))
    assert result.applied and result.details["segments"] == 10
    assert calls["corpus"] == tmp_path  # built from the corpus root


def test_rebuilds_when_index_is_schema_stale(tmp_path, monkeypatch):
    """A present-but-schema-stale index is rebuilt (not skipped) so it self-heals."""
    import json

    lance = tmp_path / "search" / "lance_index"
    lance.mkdir(parents=True)
    # Pre-schema-bump index: meta present but without the current schema_version.
    (lance / "index_meta.json").write_text(json.dumps({"embedding_model": "m", "embed_dim": 384}))

    calls = {}

    def _fake(corpus, lance_path, **k):
        calls["built"] = True
        return two_tier_indexer.TwoTierIndexStats(episodes=1, segments=2, insights=3)

    monkeypatch.setattr(two_tier_indexer, "build_two_tier_index", _fake)
    result = TwoTierNativeReindexMigration().apply(_ctx(tmp_path))
    assert calls.get("built") is True  # stale → rebuilt, not no-op
    assert result.applied and result.details["segments"] == 2


def test_dry_run_does_not_build(tmp_path, monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("dry-run must not build")

    monkeypatch.setattr(two_tier_indexer, "build_two_tier_index", _boom)
    result = TwoTierNativeReindexMigration().apply(_ctx(tmp_path, dry_run=True))
    assert result.applied is False and result.dry_run is True


def test_plan_strings(tmp_path):
    m = TwoTierNativeReindexMigration()
    assert "natively" in m.plan(_ctx(tmp_path)).lower()  # no index → native build planned
    (tmp_path / "search" / "lance_index").mkdir(parents=True)
    # index present but no offset sidecar (pre-#1010) → reindex to emit it
    assert "sidecar" in m.plan(_ctx(tmp_path)).lower()
    (tmp_path / "search" / "metadata.json").write_text("{}")
    assert "no-op" in m.plan(_ctx(tmp_path)).lower()  # index + sidecar present → no-op

    import json

    (tmp_path / "search" / "lance_index" / "index_meta.json").write_text(
        json.dumps({"embedding_model": "m", "embed_dim": 384})  # pre-versioning → stale
    )
    assert "rebuild" in m.plan(_ctx(tmp_path)).lower()  # schema-stale → rebuild planned
