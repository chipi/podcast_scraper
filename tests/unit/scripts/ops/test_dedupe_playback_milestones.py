"""Unit tests for scripts/ops/dedupe_playback_milestones.py — the one-off clean-up of re-reported
playback milestones (the server keeps one per user, episode and milestone since 6c6dcd104)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]

_SPEC = importlib.util.spec_from_file_location(
    "dedupe_playback_milestones_under_test",
    ROOT / "scripts" / "ops" / "dedupe_playback_milestones.py",
)
assert _SPEC and _SPEC.loader
_mod = importlib.util.module_from_spec(_SPEC)
sys.modules["dedupe_playback_milestones_under_test"] = _mod
_SPEC.loader.exec_module(_mod)

pytestmark = [pytest.mark.unit]


def _ev(slug: str, milestone: int, ts: int) -> str:
    return json.dumps({"slug": slug, "milestone": milestone, "ts": ts, "feed_id": "f"}) + "\n"


def _appdata(tmp_path: Path, lines: str, user: str = "u1") -> Path:
    log = tmp_path / "users" / user / "playback_events.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text(lines, encoding="utf-8")
    return tmp_path


LINES = _ev("ep1", 25, 1) + _ev("ep1", 50, 2) + _ev("ep1", 25, 3) + _ev("ep2", 25, 4)


def test_keeps_the_first_of_each_milestone_and_every_other_line_as_written() -> None:
    out, dropped = _mod.dedupe_lines(LINES + "not json\n" + _ev("ep1", 25, 9))
    assert dropped == 2
    assert out == _ev("ep1", 25, 1) + _ev("ep1", 50, 2) + _ev("ep2", 25, 4) + "not json\n"


def test_dry_run_counts_and_writes_nothing(tmp_path: Path) -> None:
    appdata = _appdata(tmp_path, LINES)
    stats = _mod.run(appdata, apply=False)
    assert (stats["files"], stats["dropped"]) == (1, 1)
    assert (appdata / "users/u1/playback_events.jsonl").read_text() == LINES
    assert not (appdata / _mod.RECEIPTS_FILE).exists()


def test_apply_then_undo_keeps_what_the_app_appended_since(tmp_path: Path) -> None:
    appdata = _appdata(tmp_path, LINES)
    log = appdata / "users/u1/playback_events.jsonl"
    assert _mod.main(["--appdata", str(appdata), "--apply"]) == 0
    assert log.read_text() == _ev("ep1", 25, 1) + _ev("ep1", 50, 2) + _ev("ep2", 25, 4)
    assert _mod.run(appdata, apply=False)["dropped"] == 0  # idempotent
    with log.open("a") as fh:
        fh.write(_ev("ep3", 25, 5))
    assert _mod.main(["--appdata", str(appdata), "--undo"]) == 0
    assert log.read_text() == LINES + _ev("ep3", 25, 5)


def test_a_log_with_no_repeats_is_not_touched(tmp_path: Path) -> None:
    appdata = _appdata(tmp_path, _ev("ep1", 25, 1) + _ev("ep1", 50, 2))
    assert _mod.main(["--appdata", str(appdata), "--apply"]) == 0
    assert not (appdata / "users/u1/playback_events.jsonl.bak-dedupe-milestones").exists()
