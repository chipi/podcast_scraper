"""Verify-branch coverage for migrations 0001/0002 (empty-index path; #862)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

pytest.importorskip("lancedb")

from podcast_scraper.search.faiss_store import VECTORS_FILE  # noqa: E402
from podcast_scraper.upgrade.migration import MigrationContext  # noqa: E402
from podcast_scraper.upgrade.migrations.m0001_faiss_to_lance import (  # noqa: E402
    FaissToLanceMigration,
)
from podcast_scraper.upgrade.migrations.m0002_two_tier_native_reindex import (  # noqa: E402
    TwoTierNativeReindexMigration,
)


def _ctx(tmp_path):
    return MigrationContext(corpus_root=tmp_path)


def test_m0001_verify_fails_on_present_but_empty_index(tmp_path):
    (tmp_path / "search").mkdir()
    (tmp_path / "search" / VECTORS_FILE).write_text("", encoding="utf-8")  # FAISS present
    (tmp_path / "search" / "lance_index").mkdir()  # lance dir exists but has no tables
    ok, msg = FaissToLanceMigration().verify(_ctx(tmp_path))
    assert not ok and "no rows" in msg


def test_m0002_verify_fails_on_present_but_empty_index(tmp_path):
    (tmp_path / "search" / "lance_index").mkdir(parents=True)  # exists but empty
    ok, msg = TwoTierNativeReindexMigration().verify(_ctx(tmp_path))
    assert not ok and "empty" in msg
