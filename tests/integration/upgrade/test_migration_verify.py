"""Verify-branch coverage for migrations 0001 (no-op since #995) / 0002 (#862) / 0003."""

from __future__ import annotations

import json

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
from podcast_scraper.upgrade.migrations.m0003_gi_v3_typed_mentions import (  # noqa: E402
    GiV3TypedMentionsMigration,
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


@pytest.mark.critical_path
def test_m0003_verify_vacuous_on_empty_corpus(tmp_path):
    ok, msg = GiV3TypedMentionsMigration().verify(_ctx(tmp_path))
    assert ok and "0 .gi.json" in msg


@pytest.mark.critical_path
def test_m0003_verify_fails_on_unparsable(tmp_path):
    (tmp_path / "metadata").mkdir()
    (tmp_path / "metadata" / "ep.gi.json").write_text("{ not json", encoding="utf-8")
    ok, msg = GiV3TypedMentionsMigration().verify(_ctx(tmp_path))
    assert not ok and "unparsable" in msg


@pytest.mark.critical_path
def test_m0003_verify_fails_on_pending_then_apply_makes_it_pass(tmp_path):
    gi = tmp_path / "metadata" / "ep.gi.json"
    gi.parent.mkdir()
    gi.write_text(json.dumps(_LEGACY_GI), encoding="utf-8")

    # Legacy doc still carries pending v3 changes → verify() must fail.
    ok, msg = GiV3TypedMentionsMigration().verify(_ctx(tmp_path))
    assert not ok and "pending v3 changes" in msg

    # apply() atomically rewrites it (tmp + os.replace), leaving a valid v3 doc.
    result = GiV3TypedMentionsMigration().apply(_ctx(tmp_path))
    assert result.applied and result.details["files_changed"] == 1
    assert json.loads(gi.read_text(encoding="utf-8"))["schema_version"] == "3.0"

    # Now idempotent → verify() passes.
    ok, msg = GiV3TypedMentionsMigration().verify(_ctx(tmp_path))
    assert ok and "fully migrated" in msg


def test_m0001_verify_is_noop(tmp_path):
    # FAISS retired (#995): 0001 is a recorded no-op and always verifies vacuously.
    ok, msg = FaissToLanceMigration().verify(_ctx(tmp_path))
    assert ok and "no-op" in msg


def test_m0002_verify_fails_on_present_but_empty_index(tmp_path):
    (tmp_path / "search" / "lance_index").mkdir(parents=True)  # exists but empty
    ok, msg = TwoTierNativeReindexMigration().verify(_ctx(tmp_path))
    assert not ok and "empty" in msg
