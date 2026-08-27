"""The prod entry point for the #1685 backfill: plan must not write, apply must, twice must be safe.

This is the module `scope-bare-names-prod.yml` invokes against the live corpus, where there is no
reverse migration. Four properties matter enough to pin:

  * `--mode plan` leaves the corpus byte-identical. A "plan" that writes is the worst bug here.
  * `--mode apply` actually rewrites BOTH layers. A migration that silently no-ops would report a
    tidy summary and leave the defect in place — the shape of failure this whole workstream keeps
    running into.
  * running it twice is safe. The second pass must plan an empty map and change nothing, because
    a partial run WILL be re-run.
  * a wrong `--corpus-root` exits non-zero. `0 episodes, nothing to do` and exit 0 reads exactly
    like success, and against production that is how you conclude the job is done when it never
    started.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.upgrade.apply_m0007 import main

pytestmark = [pytest.mark.unit]


def _episode(root: Path, name: str, persons: list) -> None:
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


def _person_ids(root: Path, name: str, layer: str) -> set:
    doc = json.loads((root / "metadata" / f"{name}.{layer}.json").read_text(encoding="utf-8"))
    return {n["id"] for n in doc["nodes"] if str(n.get("id", "")).startswith("person:")}


def _snapshot(root: Path) -> dict:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


class TestPlanNeverWrites:
    def test_the_corpus_is_byte_identical(self, tmp_path: Path) -> None:
        _episode(tmp_path, "ep1", ["person:jensen"])
        before = _snapshot(tmp_path)
        assert main(["--corpus-root", str(tmp_path), "--mode", "plan"]) == 0
        assert _snapshot(tmp_path) == before


class TestApplyActuallyRewritesBothLayers:
    def test_a_bare_id_is_scoped_in_gi_and_kg(self, tmp_path: Path) -> None:
        """Both layers, or the two graphs disagree about who the person is — which is #1862."""
        _episode(tmp_path, "ep1", ["person:jensen"])
        assert main(["--corpus-root", str(tmp_path), "--mode", "apply", "--heal", "false"]) == 0
        for layer in ("gi", "kg"):
            ids = _person_ids(tmp_path, "ep1", layer)
            assert ids == {"person:unresolved-jensen-ep1"}, (layer, ids)

    def test_heal_false_does_not_mint_a_real_person(self, tmp_path: Path) -> None:
        """The whole reason heal=false is the shipped default."""
        _episode(tmp_path, "ep1", ["person:dario", "person:dario-amodei"])
        assert main(["--corpus-root", str(tmp_path), "--mode", "apply", "--heal", "false"]) == 0
        ids = _person_ids(tmp_path, "ep1", "kg")
        assert "person:unresolved-dario-ep1" in ids
        assert ids == {"person:unresolved-dario-ep1", "person:dario-amodei"}

    def test_heal_true_does_mint_it(self, tmp_path: Path) -> None:
        """The other branch must genuinely differ, or the flag is decoration."""
        _episode(tmp_path, "ep1", ["person:dario", "person:dario-amodei"])
        assert main(["--corpus-root", str(tmp_path), "--mode", "apply", "--heal", "true"]) == 0
        assert _person_ids(tmp_path, "ep1", "kg") == {"person:dario-amodei"}


class TestItIsSafeToRunTwice:
    def test_a_second_apply_changes_nothing(self, tmp_path: Path) -> None:
        """A partial run will be re-run. Idempotence is not optional here."""
        _episode(tmp_path, "ep1", ["person:jensen"])
        assert main(["--corpus-root", str(tmp_path), "--mode", "apply"]) == 0
        after_first = _snapshot(tmp_path)
        assert main(["--corpus-root", str(tmp_path), "--mode", "apply"]) == 0
        assert _snapshot(tmp_path) == after_first


class TestAWrongPathIsLoud:
    def test_missing_corpus_exits_nonzero(self, tmp_path: Path, capsys) -> None:
        rc = main(["--corpus-root", str(tmp_path / "nope"), "--mode", "apply"])
        assert rc == 1
        assert "does not exist" in capsys.readouterr().err
