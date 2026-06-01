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


def test_noop_when_index_already_exists(tmp_path, monkeypatch):
    (tmp_path / "search" / "lance_index").mkdir(parents=True)  # 0001 already built it

    def _boom(*a, **k):
        raise AssertionError("must not rebuild when an index already exists")

    monkeypatch.setattr(two_tier_indexer, "build_two_tier_index", _boom)
    result = TwoTierNativeReindexMigration().apply(_ctx(tmp_path))
    assert result.applied and "no-op" in result.message


def test_builds_natively_when_absent(tmp_path, monkeypatch):
    calls = {}

    def _fake(corpus, lance, **k):
        calls["corpus"] = corpus
        return two_tier_indexer.TwoTierIndexStats(episodes=3, segments=10, insights=7)

    monkeypatch.setattr(two_tier_indexer, "build_two_tier_index", _fake)
    result = TwoTierNativeReindexMigration().apply(_ctx(tmp_path))
    assert result.applied and result.details["segments"] == 10
    assert calls["corpus"] == tmp_path  # built from the corpus root


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
    assert "no-op" in m.plan(_ctx(tmp_path)).lower()  # index present → no-op
