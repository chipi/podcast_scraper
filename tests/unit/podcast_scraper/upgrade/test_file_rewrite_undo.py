"""The shared undo of every receipts-based migration: restore what it wrote, refuse what changed."""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.upgrade.file_rewrite import (
    append_receipts,
    undo_from_receipts,
    write_with_backup,
)

pytestmark = [pytest.mark.unit]

RECEIPTS, TAG, MID = "r.jsonl", "9999", "9999_x"


def _run(root: Path, payload: dict) -> None:
    receipts = [write_with_backup(root, TAG, root / "feeds/a.json", payload)]
    append_receipts(root, RECEIPTS, {"migration": MID}, receipts)


def _corpus(tmp_path: Path) -> Path:
    (tmp_path / "feeds").mkdir()
    (tmp_path / "feeds/a.json").write_text('{"v": 0}', encoding="utf-8")
    return tmp_path


def test_undo_restores_what_the_run_wrote(tmp_path: Path) -> None:
    root = _corpus(tmp_path)
    before = (root / "feeds/a.json").read_bytes()
    _run(root, {"v": 1})
    assert undo_from_receipts(root, RECEIPTS, TAG, MID) == (1, [])
    assert (root / "feeds/a.json").read_bytes() == before


def test_two_runs_in_one_receipts_file_restore_each_file_once(tmp_path: Path) -> None:
    """Apply, undo, apply again: the first run's receipt must not report the restored file as
    changed (245 false refusals on a 0031 clone, 2026-10-18)."""
    root = _corpus(tmp_path)
    before = (root / "feeds/a.json").read_bytes()
    _run(root, {"v": 1})
    undo_from_receipts(root, RECEIPTS, TAG, MID)
    _run(root, {"v": 1})
    assert undo_from_receipts(root, RECEIPTS, TAG, MID) == (1, [])
    assert (root / "feeds/a.json").read_bytes() == before


def test_a_file_changed_after_the_run_is_refused_not_overwritten(tmp_path: Path) -> None:
    root = _corpus(tmp_path)
    _run(root, {"v": 1})
    (root / "feeds/a.json").write_text('{"v": 2}', encoding="utf-8")  # neither before nor after
    restored, refused = undo_from_receipts(root, RECEIPTS, TAG, MID)
    assert restored == 0 and len(refused) == 1 and "changed since" in refused[0]
    assert (root / "feeds/a.json").read_text(encoding="utf-8") == '{"v": 2}'


def test_a_missing_backup_is_refused(tmp_path: Path) -> None:
    root = _corpus(tmp_path)
    _run(root, {"v": 1})
    backup = next((root / ".podcast_scraper").rglob("a.json"))
    backup.unlink()
    restored, refused = undo_from_receipts(root, RECEIPTS, TAG, MID)
    assert restored == 0 and "no backup" in refused[0]


def test_a_second_run_over_a_file_changed_in_between_undoes_to_that_change(
    tmp_path: Path,
) -> None:
    """Run 1 writes v1 (backup: v0). Something else rewrites the file to v2. Run 2 writes v3, and
    its receipt says "before v2". Undo must restore v2, not run 1's v0: keeping only the first
    backup silently threw away the change made in between."""
    root = _corpus(tmp_path)
    _run(root, {"v": 1})
    (root / "feeds/a.json").write_text('{"v": 2}', encoding="utf-8")
    between = (root / "feeds/a.json").read_bytes()
    _run(root, {"v": 3})
    assert undo_from_receipts(root, RECEIPTS, TAG, MID) == (1, [])
    assert (root / "feeds/a.json").read_bytes() == between


def test_a_backup_of_another_state_is_refused_not_restored(tmp_path: Path) -> None:
    """A backup whose bytes are not the receipt's before-state is never copied over the file."""
    root = _corpus(tmp_path)
    _run(root, {"v": 1})
    backup = root / ".podcast_scraper" / "upgrade-backups" / TAG / "feeds/a.json"
    backup.write_text('{"v": "unrelated"}', encoding="utf-8")
    restored, refused = undo_from_receipts(root, RECEIPTS, TAG, MID)
    assert restored == 0 and len(refused) == 1 and "no backup" in refused[0]
    assert (root / "feeds/a.json").read_text(encoding="utf-8") != '{"v": "unrelated"}'
