"""Verify-branch coverage for migrations 0001 (no-op since #995) / 0002 (#862) / 0003."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

pytest.importorskip("lancedb")

from podcast_scraper.upgrade.migration import MigrationContext  # noqa: E402
from podcast_scraper.upgrade.migrations.m0001_faiss_to_lance import (  # noqa: E402
    FaissToLanceMigration,
)
from podcast_scraper.upgrade.migrations.m0002_two_tier_native_reindex import (  # noqa: E402
    TwoTierNativeReindexMigration,
)


def _ctx(tmp_path):
    return MigrationContext(corpus_root=tmp_path)


# A legacy v2 envelope with a MENTIONS→person edge: migrate_gi_document_v3 bumps
# schema_version 2.0→3.0 and retypes the edge, so before != after (pending).
_LEGACY_GI = {
    "schema_version": "2.0",
    "nodes": [
        {"id": "insight:1", "type": "Insight", "properties": {"text": "x"}},
        {"id": "person:p", "type": "Person", "properties": {"name": "P"}},
    ],
    "edges": [{"type": "MENTIONS", "from": "insight:1", "to": "person:p"}],
}


def test_m0001_verify_is_noop(tmp_path):
    # FAISS retired (#995): 0001 is a recorded no-op and always verifies vacuously.
    ok, msg = FaissToLanceMigration().verify(_ctx(tmp_path))
    assert ok and "no-op" in msg


def test_m0002_verify_fails_on_present_but_empty_index(tmp_path):
    (tmp_path / "search" / "lance_index").mkdir(parents=True)  # exists but empty
    ok, msg = TwoTierNativeReindexMigration().verify(_ctx(tmp_path))
    assert not ok and "empty" in msg
