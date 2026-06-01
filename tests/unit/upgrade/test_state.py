"""Tests for the filesystem upgrade state store (#862)."""

from __future__ import annotations

import json

import pytest

from podcast_scraper.upgrade.state import FilesystemStateStore

pytestmark = pytest.mark.unit


def _write_manifest(root, code_version):
    (root / "corpus_manifest.json").write_text(
        json.dumps({"produced_by": {"code_version": code_version}}), encoding="utf-8"
    )


def test_current_version_from_manifest_when_no_ledger(tmp_path):
    _write_manifest(tmp_path, "2.6.0")
    store = FilesystemStateStore(tmp_path)
    assert store.current_version() == "2.6.0"
    assert store.applied_migration_ids() == set()


def test_record_advances_version_and_ledger(tmp_path):
    _write_manifest(tmp_path, "2.6.0")
    store = FilesystemStateStore(tmp_path)
    store.record_applied("0001_x", to_version="2.7.0", at="2026-06-01T00:00:00Z")

    # New store instance reads the persisted ledger.
    store2 = FilesystemStateStore(tmp_path)
    assert store2.applied_migration_ids() == {"0001_x"}
    assert store2.current_version() == "2.7.0"  # ledger version wins over manifest
    rec = store2.applied_records()[0]
    assert rec["id"] == "0001_x" and rec["to_version"] == "2.7.0"


def test_record_is_idempotent_on_id(tmp_path):
    store = FilesystemStateStore(tmp_path)
    store.record_applied("0001_x", to_version="2.7.0", at="t1")
    store.record_applied("0001_x", to_version="2.7.0", at="t2")  # re-record same id
    assert len(store.applied_records()) == 1  # not duplicated


def test_unstamped_corpus_returns_none(tmp_path):
    assert FilesystemStateStore(tmp_path).current_version() is None


def test_corrupt_ledger_degrades_gracefully(tmp_path):
    (tmp_path / "upgrade_ledger.json").write_text("{ not json", encoding="utf-8")
    store = FilesystemStateStore(tmp_path)
    assert store.applied_migration_ids() == set()
    assert store.current_version() is None
