"""Tests for the upgrade runner orchestration (#862).

Uses in-memory fakes for the state store + migrations so the runner contract
(ordering, idempotency, stop-on-failure, version gating, verify) is exercised
without touching FAISS/LanceDB.
"""

from __future__ import annotations

from typing import List, Set

import pytest

from podcast_scraper.upgrade.migration import Migration, MigrationContext, MigrationResult
from podcast_scraper.upgrade.runner import UpgradeRunner

pytestmark = pytest.mark.unit


class _FakeStore:
    def __init__(self, version=None):
        self._version = version
        self._applied: List[dict] = []

    def current_version(self):
        return self._version

    def applied_migration_ids(self) -> Set[str]:
        return {e["id"] for e in self._applied}

    def record_applied(self, migration_id, *, to_version, at):
        self._applied.append({"id": migration_id, "to_version": to_version, "at": at})
        self._version = to_version


class _FakeMigration(Migration):
    def __init__(self, mid, to_version, *, raises=False, verify_ok=True):
        self.id = mid
        self.to_version = to_version
        self.description = f"fake {mid}"
        self._raises = raises
        self._verify_ok = verify_ok
        self.apply_calls = 0

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        self.apply_calls += 1
        if self._raises:
            raise RuntimeError(f"{self.id} boom")
        return MigrationResult(self.id, applied=True, dry_run=ctx.dry_run, message="ok")

    def verify(self, ctx):
        return self._verify_ok, "verified" if self._verify_ok else "broken"


def _ctx(tmp_path, dry_run=False):
    return MigrationContext(corpus_root=tmp_path, dry_run=dry_run)


def test_status_lists_pending_in_order(tmp_path):
    store = _FakeStore("2.6.0")
    runner = UpgradeRunner(
        store, [_FakeMigration("0002_b", "2.7.0"), _FakeMigration("0001_a", "2.7.0")]
    )
    status = runner.status()
    assert [m.id for m in status.pending] == ["0001_a", "0002_b"]  # sorted by id
    assert status.target_version == "2.7.0"
    assert status.up_to_date is False


def test_run_applies_and_records(tmp_path):
    store = _FakeStore("2.6.0")
    m = _FakeMigration("0001_a", "2.7.0")
    runner = UpgradeRunner(store, [m])
    results = runner.run(_ctx(tmp_path), now="t")
    assert results[0].applied and m.apply_calls == 1
    assert store.applied_migration_ids() == {"0001_a"}
    assert store.current_version() == "2.7.0"
    assert runner.status().up_to_date is True


def test_idempotent_second_run_is_noop(tmp_path):
    store = _FakeStore("2.6.0")
    m = _FakeMigration("0001_a", "2.7.0")
    runner = UpgradeRunner(store, [m])
    runner.run(_ctx(tmp_path), now="t")
    results = runner.run(_ctx(tmp_path), now="t")  # nothing pending
    assert results == [] and m.apply_calls == 1


def test_dry_run_does_not_record(tmp_path):
    store = _FakeStore("2.6.0")
    runner = UpgradeRunner(store, [_FakeMigration("0001_a", "2.7.0")])
    runner.run(_ctx(tmp_path, dry_run=True), now="t")
    assert store.applied_migration_ids() == set()  # dry-run writes nothing


def test_stop_on_failure_preserves_prior(tmp_path):
    store = _FakeStore("2.6.0")
    good = _FakeMigration("0001_a", "2.7.0")
    bad = _FakeMigration("0002_b", "2.7.0", raises=True)
    never = _FakeMigration("0003_c", "2.7.0")
    runner = UpgradeRunner(store, [good, bad, never])
    with pytest.raises(RuntimeError):
        runner.run(_ctx(tmp_path), now="t")
    assert store.applied_migration_ids() == {"0001_a"}  # good recorded, bad/never not
    assert never.apply_calls == 0  # never reached


def test_to_version_gates_later_migrations(tmp_path):
    store = _FakeStore("2.6.0")
    runner = UpgradeRunner(
        store, [_FakeMigration("0001_a", "2.7.0"), _FakeMigration("0002_b", "2.8.0")]
    )
    runner.run(_ctx(tmp_path), to_version="2.7.0", now="t")
    assert store.applied_migration_ids() == {"0001_a"}  # 2.8.0 step skipped


def test_verify_only_applied(tmp_path):
    store = _FakeStore("2.6.0")
    m1 = _FakeMigration("0001_a", "2.7.0", verify_ok=True)
    m2 = _FakeMigration("0002_b", "2.8.0", verify_ok=False)
    runner = UpgradeRunner(store, [m1, m2])
    runner.run(_ctx(tmp_path), to_version="2.7.0", now="t")  # only m1 applied
    verify = runner.verify(_ctx(tmp_path))
    assert verify == [("0001_a", True, "verified")]  # m2 not applied → not verified
