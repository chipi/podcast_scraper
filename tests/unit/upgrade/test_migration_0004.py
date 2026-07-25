"""Unit tests for the insight_type reindex migration (0004, Search v3 §S8).

Stubs ``build_two_tier_index`` so the migration's gating — no-op when the index is
already at the current schema, rebuild when schema-stale (pre-insight_type), no-op
when absent, and dry-run — is tested without touching LanceDB.
"""

from __future__ import annotations

import json

import pytest

from podcast_scraper.search import two_tier_indexer
from podcast_scraper.search.backends.lancedb_backend import LANCE_SCHEMA_VERSION
from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations.m0004_insight_type_reindex import (
    InsightTypeReindexMigration,
)

pytestmark = pytest.mark.unit


def _ctx(tmp_path, dry_run=False):
    return MigrationContext(corpus_root=tmp_path, dry_run=dry_run)


def _write_index(tmp_path, schema_version):
    lance = tmp_path / "search" / "lance_index"
    lance.mkdir(parents=True)
    (lance / "index_meta.json").write_text(
        json.dumps({"embedding_model": "m", "embed_dim": 384, "schema_version": schema_version})
    )
    return lance


def test_noop_when_no_index(tmp_path, monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("must not build when no index exists")

    monkeypatch.setattr(two_tier_indexer, "build_two_tier_index", _boom)
    result = InsightTypeReindexMigration().apply(_ctx(tmp_path))
    assert result.applied and "no-op" in result.message


def test_noop_when_schema_current(tmp_path, monkeypatch):
    _write_index(tmp_path, LANCE_SCHEMA_VERSION)  # already v3 — has insight_type

    def _boom(*a, **k):
        raise AssertionError("must not rebuild a current-schema index")

    monkeypatch.setattr(two_tier_indexer, "build_two_tier_index", _boom)
    result = InsightTypeReindexMigration().apply(_ctx(tmp_path))
    assert result.applied and "no-op" in result.message


def test_rebuilds_when_schema_stale(tmp_path, monkeypatch):
    """A v2 index (no insight_type column) is rebuilt so the §S8 filter works."""
    _write_index(tmp_path, LANCE_SCHEMA_VERSION - 1)

    calls = {}

    def _fake(corpus, lance_path, **k):
        calls["built"] = True
        return two_tier_indexer.TwoTierIndexStats(episodes=2, segments=5, insights=4)

    monkeypatch.setattr(two_tier_indexer, "build_two_tier_index", _fake)
    result = InsightTypeReindexMigration().apply(_ctx(tmp_path))
    assert calls.get("built") is True
    assert result.applied and result.details["insights"] == 4


def test_dry_run_does_not_build(tmp_path, monkeypatch):
    _write_index(tmp_path, LANCE_SCHEMA_VERSION - 1)

    def _boom(*a, **k):
        raise AssertionError("dry-run must not build")

    monkeypatch.setattr(two_tier_indexer, "build_two_tier_index", _boom)
    result = InsightTypeReindexMigration().apply(_ctx(tmp_path, dry_run=True))
    assert result.applied is False and result.dry_run is True


def test_plan_strings(tmp_path):
    m = InsightTypeReindexMigration()
    assert "nothing to reindex" in m.plan(_ctx(tmp_path)).lower()  # no index
    _write_index(tmp_path, LANCE_SCHEMA_VERSION - 1)
    assert "stale" in m.plan(_ctx(tmp_path)).lower()  # v2 → rebuild planned
    # Bump the stored version to current → no-op.
    lance = tmp_path / "search" / "lance_index"
    (lance / "index_meta.json").write_text(
        json.dumps(
            {"embedding_model": "m", "embed_dim": 384, "schema_version": LANCE_SCHEMA_VERSION}
        )
    )
    assert "no-op" in m.plan(_ctx(tmp_path)).lower()
