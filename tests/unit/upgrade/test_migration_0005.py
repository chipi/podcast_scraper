"""Unit tests for the GI 3.1 route-and-tag stamp migration (0005, ADR-135/#1191).

The migration walks ``*.gi.json`` and stamps ``schema_version`` 3.0 → 3.1 via
``migrate_gi_document_v3_1`` (additive; the new Insight fields are populated by a reprocess, not
synthesised here). Idempotent, atomic write, tolerant of unparsable files.
"""

from __future__ import annotations

import json

import pytest

from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations.m0005_gi_v3_1_route_and_tag import (
    GiV31RouteAndTagMigration,
)

pytestmark = pytest.mark.unit


def _ctx(tmp_path, dry_run=False):
    return MigrationContext(corpus_root=tmp_path, dry_run=dry_run)


def _gi_doc(schema_version="3.0"):
    return {
        "schema_version": schema_version,
        "model_version": "m",
        "prompt_version": "p",
        "episode_id": "ep1",
        "nodes": [{"id": "person:alice", "type": "Person", "properties": {}}],
        "edges": [],
    }


def _write_gi(tmp_path, name="0001 - ep.gi.json", **kw):
    p = tmp_path / name
    p.write_text(json.dumps(_gi_doc(**kw)) + "\n", encoding="utf-8")
    return p


def test_noop_when_no_gi_files(tmp_path):
    result = GiV31RouteAndTagMigration().apply(_ctx(tmp_path))
    assert result.applied
    assert result.details["files_scanned"] == 0


def test_stamps_3_0_to_3_1(tmp_path):
    p = _write_gi(tmp_path, schema_version="3.0")
    result = GiV31RouteAndTagMigration().apply(_ctx(tmp_path))
    assert result.applied
    assert result.details["changed"] == 1
    assert json.loads(p.read_text())["schema_version"] == "3.1"


def test_idempotent_on_3_1(tmp_path):
    p = _write_gi(tmp_path, schema_version="3.1")
    before = p.read_text()
    result = GiV31RouteAndTagMigration().apply(_ctx(tmp_path))
    assert result.applied
    assert result.details["changed"] == 0
    assert result.details["unchanged"] == 1
    assert p.read_text() == before  # no rewrite


def test_dry_run_does_not_write(tmp_path):
    p = _write_gi(tmp_path, schema_version="3.0")
    result = GiV31RouteAndTagMigration().apply(_ctx(tmp_path, dry_run=True))
    assert result.applied and result.dry_run
    assert result.details["changed"] == 1
    assert json.loads(p.read_text())["schema_version"] == "3.0"  # untouched


def test_unparsable_file_is_recorded_not_fatal(tmp_path):
    (tmp_path / "junk.gi.json").write_text("{not json", encoding="utf-8")
    _write_gi(tmp_path, name="0001 - ep.gi.json", schema_version="3.0")
    result = GiV31RouteAndTagMigration().apply(_ctx(tmp_path))
    assert result.applied
    assert len(result.details["unparsable"]) == 1
    assert result.details["changed"] == 1
