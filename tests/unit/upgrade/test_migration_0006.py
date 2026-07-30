"""Unit tests for the KG v2.0 typed-entities migration (0006, RFC-097).

The migration walks ``*.kg.json`` and rewrites legacy ``Entity(kind=...)`` nodes to typed
``Person`` / ``Organization`` (+ id/kind normalization) via ``migrate_kg_document_v2``, stamping
``schema_version`` 2.0. Idempotent, atomic write, tolerant of unparsable files.
"""

from __future__ import annotations

import json

import pytest

from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations.m0006_kg_v2_typed_entities import (
    KgV2TypedEntitiesMigration,
)

pytestmark = pytest.mark.unit


def _ctx(tmp_path, dry_run=False):
    return MigrationContext(corpus_root=tmp_path, dry_run=dry_run)


def _legacy_kg_doc():
    """A pre-v2.0 KG doc: an Entity node carrying a ``kind`` discriminator."""
    return {
        "schema_version": "1.2",
        "episode_id": "ep1",
        "nodes": [{"id": "person:alice", "type": "Entity", "properties": {"kind": "person"}}],
        "edges": [],
    }


def _v2_kg_doc():
    return {
        "schema_version": "2.0",
        "episode_id": "ep1",
        "nodes": [{"id": "person:alice", "type": "Person", "properties": {}}],
        "edges": [],
    }


def _write_kg(tmp_path, doc, name="0001 - ep.kg.json"):
    p = tmp_path / name
    p.write_text(json.dumps(doc) + "\n", encoding="utf-8")
    return p


def test_noop_when_no_kg_files(tmp_path):
    result = KgV2TypedEntitiesMigration().apply(_ctx(tmp_path))
    assert result.applied
    assert result.details["files_scanned"] == 0


def test_rewrites_entity_to_typed_and_stamps_v2(tmp_path):
    p = _write_kg(tmp_path, _legacy_kg_doc())
    result = KgV2TypedEntitiesMigration().apply(_ctx(tmp_path))
    assert result.applied
    assert result.details["changed"] == 1
    doc = json.loads(p.read_text())
    assert doc["schema_version"] == "2.0"
    assert doc["nodes"][0]["type"] == "Person"
    assert "kind" not in doc["nodes"][0]["properties"]


def test_idempotent_on_v2(tmp_path):
    p = _write_kg(tmp_path, _v2_kg_doc())
    before = p.read_text()
    result = KgV2TypedEntitiesMigration().apply(_ctx(tmp_path))
    assert result.applied
    assert result.details["changed"] == 0
    assert result.details["unchanged"] == 1
    assert p.read_text() == before  # no rewrite


def test_dry_run_does_not_write(tmp_path):
    p = _write_kg(tmp_path, _legacy_kg_doc())
    result = KgV2TypedEntitiesMigration().apply(_ctx(tmp_path, dry_run=True))
    assert result.applied and result.dry_run
    assert result.details["changed"] == 1
    # untouched on disk
    assert json.loads(p.read_text())["schema_version"] == "1.2"


def test_unparsable_file_is_recorded_not_fatal(tmp_path):
    (tmp_path / "junk.kg.json").write_text("{not json", encoding="utf-8")
    _write_kg(tmp_path, _legacy_kg_doc(), name="0001 - ep.kg.json")
    result = KgV2TypedEntitiesMigration().apply(_ctx(tmp_path))
    assert result.applied
    assert len(result.details["unparsable"]) == 1
    assert result.details["changed"] == 1
