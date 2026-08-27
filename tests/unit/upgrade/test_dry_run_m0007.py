"""The m0007 dry run must actually distinguish the two heal policies, and must never write.

It exists to answer "what would the #1685 backfill change?" against the REAL corpus before an
irreversible migration. Two ways it could be useless while looking fine:

  * it prints the same summary for both policies because the `heal` override never lands — on the
    committed fixture that is indistinguishable from a truthful answer, since that corpus happens
    to have zero resolvable bare names and both policies genuinely agree there.
  * it writes something. A "dry run" that mutates the corpus it is asked about is the worst
    possible version of this tool.

Both are pinned here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.upgrade.dry_run_m0007 import main, plan_for

pytestmark = [pytest.mark.unit]


def _episode(root: Path, name: str, persons: list) -> None:
    """One episode as a .gi.json / .kg.json pair — the shape m0007 walks."""
    meta = root / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "3.1",
        "nodes": [{"id": f"episode:{name}", "kind": "episode", "name": name}]
        + [{"id": p, "kind": "person", "name": p.split(":")[-1]} for p in persons],
        "edges": [],
    }
    (meta / f"{name}.gi.json").write_text(json.dumps(payload), encoding="utf-8")
    (meta / f"{name}.kg.json").write_text(
        json.dumps({**payload, "schema_version": "2.0"}), encoding="utf-8"
    )


def _snapshot(root: Path) -> dict:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


class TestThePolicyOverrideActuallyLands:
    def test_a_resolvable_name_heals_only_when_heal_is_true(self, tmp_path: Path) -> None:
        """`dario` + `dario-amodei` in one episode: heal=True mints the full id, heal=False scopes.

        Without a resolvable name present both policies agree, so a corpus that has none — like the
        committed fixture — cannot tell a working override from a dead one.
        """
        _episode(tmp_path, "ep1", ["person:dario", "person:dario-amodei"])

        healed = plan_for(tmp_path, heal=True)
        scoped = plan_for(tmp_path, heal=False)

        assert "1 healed" in healed, healed
        assert "0 healed" in scoped, scoped
        assert healed != scoped

    def test_an_orphan_scopes_under_both(self, tmp_path: Path) -> None:
        """No candidate anywhere in the episode -> nothing to heal, so the policies must agree."""
        _episode(tmp_path, "ep1", ["person:jensen"])
        assert "0 healed" in plan_for(tmp_path, heal=True)
        assert "0 healed" in plan_for(tmp_path, heal=False)


class TestItNeverWrites:
    def test_the_corpus_is_byte_identical_afterwards(self, tmp_path: Path) -> None:
        _episode(tmp_path, "ep1", ["person:dario", "person:dario-amodei"])
        _episode(tmp_path, "ep2", ["person:jensen"])
        before = _snapshot(tmp_path)

        assert main(["--corpus-root", str(tmp_path)]) == 0

        assert _snapshot(tmp_path) == before, "the dry run modified the corpus"


class TestItFailsLoudlyOnABadCorpus:
    def test_a_missing_root_is_a_nonzero_exit(self, tmp_path: Path, capsys) -> None:
        """Pointed at the wrong path it must not print a cheerful zero-change plan."""
        rc = main(["--corpus-root", str(tmp_path / "nope")])
        assert rc == 1
        assert "does not exist" in capsys.readouterr().out


class TestItReportsBothPolicies:
    def test_the_output_names_the_intended_one(self, tmp_path: Path, capsys) -> None:
        """The reader has to be able to tell which number is the decision and which is context."""
        _episode(tmp_path, "ep1", ["person:jensen"])
        assert main(["--corpus-root", str(tmp_path)]) == 0
        out = capsys.readouterr().out
        assert "heal=False" in out and "agreed backfill policy" in out
        assert "heal=True" in out and "NOT what we intend to run" in out
