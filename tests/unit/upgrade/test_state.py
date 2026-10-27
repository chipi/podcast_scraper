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


def _ledger(root):
    return json.loads((root / "upgrade_ledger.json").read_text())


def test_reverting_the_last_migration_rewinds_the_version_to_its_predecessor(tmp_path):
    # 2026-10-27: an undo of 0035 left the version at 2.7.29 over 2.7.28 data.
    st = FilesystemStateStore(tmp_path)
    st.record_applied("0034_x", to_version="2.7.28", at="t1")
    st.record_applied("0035_x", to_version="2.7.29", at="t2")
    assert st.record_reverted("0035_x") is True
    led = _ledger(tmp_path)
    assert led["version"] == "2.7.28"
    assert [e["id"] for e in led["applied"]] == ["0034_x"]


def test_reverting_an_earlier_migration_leaves_the_version(tmp_path):
    # Out of order: the version is the later migration's claim; nothing to rewind to safely.
    st = FilesystemStateStore(tmp_path)
    st.record_applied("0034_x", to_version="2.7.28", at="t1")
    st.record_applied("0035_x", to_version="2.7.29", at="t2")
    assert st.record_reverted("0034_x") is True
    assert _ledger(tmp_path)["version"] == "2.7.29"


def test_reverting_the_only_migration_leaves_the_version(tmp_path):
    # No recorded predecessor: guessing one would be worse than leaving it.
    st = FilesystemStateStore(tmp_path)
    st.record_applied("0034_x", to_version="2.7.28", at="t1")
    assert st.record_reverted("0034_x") is True
    assert _ledger(tmp_path)["version"] == "2.7.28"


def test_every_ledger_write_hands_the_ledger_and_its_lock_to_the_corpus_owner(
    tmp_path, monkeypatch
):
    # Prod runs `upgrade run` as root: the ledger was left root-owned after 0034 and 0035.
    from podcast_scraper.upgrade import state as state_mod

    seen = []
    monkeypatch.setattr(
        state_mod, "match_corpus_owner", lambda root, paths: seen.append(list(paths))
    )
    st = FilesystemStateStore(tmp_path)
    st.record_applied("0034_x", to_version="2.7.28", at="t1")
    st.record_reverted("0034_x")
    names = [[p.name for p in call] for call in seen]
    assert names == [["upgrade_ledger.json", "upgrade_ledger.json.lock"]] * 2
