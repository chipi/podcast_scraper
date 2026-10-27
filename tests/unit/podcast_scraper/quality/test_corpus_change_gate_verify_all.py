"""The change gate runs ``upgrade verify`` over every applied migration, not just its own."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Tuple

import pytest

pytestmark = [pytest.mark.unit]

_spec = importlib.util.spec_from_file_location(
    "corpus_change_gate",
    Path(__file__).resolve().parents[4] / "scripts" / "measure" / "corpus_change_gate.py",
)
assert _spec and _spec.loader
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


class _Fake:
    """A migration whose verify fails on the roots named in ``fails_on``."""

    def __init__(self, mid: str, *fails_on: Path) -> None:
        self.id = mid
        self.fails_on = set(fails_on)
        self.calls: list = []

    def verify(self, ctx) -> Tuple[bool, str]:
        self.calls.append(ctx.corpus_root)
        return (ctx.corpus_root not in self.fails_on, f"{self.id} on {ctx.corpus_root.name}")


def test_a_failure_this_migration_caused_is_a_regression_and_an_old_one_is_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clone, work = tmp_path / "clone", tmp_path / "work"
    caused = _Fake("0030_x", work)  # ok on the clone, failing after (the 2026-10-27 0030 case)
    old = _Fake("0025_x", clone, work)  # already failing on the snapshot
    fine = _Fake("0029_x")
    unapplied = _Fake("0031_x", work)  # not in the clone's ledger: never run
    mine = _Fake("0035_x", work)  # the migration under test: its own failure is not a regression
    monkeypatch.setattr(gate, "get_migrations", lambda: [old, fine, caused, unapplied, mine])

    rows, regressions = gate._verify_all("0035_x", {"0025_x", "0029_x", "0030_x"}, clone, work)

    assert regressions == ["0030_x"]
    assert [r["id"] for r in rows] == ["0025_x", "0029_x", "0030_x", "0035_x"]
    assert unapplied.calls == []
    assert fine.calls == [work]  # a passing check is not re-run on the clone
    assert old.calls == [work, clone] and caused.calls == [work, clone]
    assert mine.calls == [work]
    by_id = {r["id"]: r for r in rows}
    assert by_id["0025_x"]["on_clone"]["ok"] is False
    assert by_id["0030_x"]["on_clone"]["ok"] is True
    assert "on_clone" not in by_id["0029_x"] and "on_clone" not in by_id["0035_x"]
