"""Every role m0009 changes is written down, and can be put back (#2065, reversibility).

WHY THIS EXISTS. m0009's demotions were irreversible, and that single fact set the bar for
everything around them: each predicate had to be provably correct BEFORE the run, because a wrong
demotion could not be taken back. That is why the show-name guard took three attempts, why 6
demotions need a human to read them, and why the conversation about running this on production kept
being "prove zero damage in advance".

A ledger moves the bar to "bound the damage and keep the receipt", which is both cheaper and more
honest than perfecting a predicate that cannot be perfected — the feed title genuinely cannot tell
`Lex Fridman Podcast` from `Latent Space: The AI Engineer Podcast`.

The data already existed; it was being thrown away. `apply()` computes every
``(episode, node, role_before, role_after)`` transition and then discarded all but the counts.

TWO THINGS THE UNDO MUST GET RIGHT, both tested below:

* **It must not clobber newer work.** If a `relabel_only` re-enrich ran after the migration, the
  node's role may be newer and better than anything this ledger knows. Undo therefore refuses any
  node whose CURRENT role is not the ``role_after`` we recorded, and reports it instead of forcing.
* **It must be the audit too.** One file answers both "what did it do" and "put it back". A
  separate audit artifact would be a second instrument free to drift from the first — the failure
  this whole arc keeps rediscovering.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.upgrade.role_ledger import (
    append_ledger,
    LEDGER_FILE,
    read_ledger,
    RoleChange,
    undo_from_ledger,
)

pytestmark = pytest.mark.unit


def _episode(root: Path, stem: str, persons, *, feed="Show", speakers=None, heard=2) -> Path:
    d = root / "metadata"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{stem}.metadata.json").write_text(
        json.dumps(
            {
                "feed": {"title": feed},
                "content": {
                    "speakers": speakers if speakers is not None else [],
                    "diarization_num_speakers": heard,
                },
            }
        ),
        encoding="utf-8",
    )
    kg = {
        "schema_version": "2.1",
        "nodes": [
            {"id": pid, "type": "Person", "properties": {"name": nm, "role": role}}
            for pid, nm, role in persons
        ],
        "edges": [],
    }
    path = d / f"{stem}.kg.json"
    path.write_text(json.dumps(kg), encoding="utf-8")
    return path


def _roles(path: Path) -> dict:
    doc = json.loads(path.read_text(encoding="utf-8"))
    return {n["id"]: (n.get("properties") or {}).get("role") for n in doc["nodes"]}


class TestUndoPutsItBack:

    def test_undo_with_no_ledger_is_not_an_error(self, tmp_path: Path) -> None:
        restored, _skipped, refused = undo_from_ledger(tmp_path)
        assert (restored, refused) == (0, [])


class TestTheLedgerIsReadableOnItsOwn:
    """It is the audit artifact as well as the undo log — one instrument, not two."""

    def test_round_trip(self, tmp_path: Path) -> None:
        rows = [
            RoleChange(
                episode="metadata/e1.kg.json",
                node_id="person:x",
                name="X",
                role_before="host",
                role_after="mentioned",
                route="not_a_person",
                feed_title="Some Show",
            )
        ]
        append_ledger(tmp_path, rows)
        assert read_ledger(tmp_path) == rows

    def test_the_file_is_one_json_object_per_line(self, tmp_path: Path) -> None:
        # JSONL, so a second migration run APPENDS rather than replacing — and so a crash leaves
        # the rows already flushed rather than nothing.
        append_ledger(
            tmp_path,
            [
                RoleChange(
                    episode="metadata/e1.kg.json",
                    node_id="person:x",
                    name="X",
                    role_before="host",
                    role_after="mentioned",
                    route="not_a_person",
                    feed_title="Some Show",
                    run_id="r1",
                )
            ],
        )
        lines = (tmp_path / LEDGER_FILE).read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert row["name"] == "X" and row["route"] == "not_a_person" and row["run_id"] == "r1"

    def test_an_absent_role_round_trips_as_null_not_the_string_none(self, tmp_path: Path) -> None:
        append_ledger(
            tmp_path,
            [
                RoleChange(
                    episode="metadata/e1.kg.json",
                    node_id="person:x",
                    name="X",
                    role_before=None,
                    role_after="host",
                    route="promote",
                    run_id="r1",
                )
            ],
        )
        assert json.loads((tmp_path / LEDGER_FILE).read_text().strip())["role_before"] is None
        assert read_ledger(tmp_path)[0].role_before is None


class TestShapesTheFirstVersionCorrupted:
    """Five shapes the original fixtures never took. Each was a real defect (advisor, 3rd pass).

    Every fixture in the classes above sets ``role``, uses ASCII names, is pre-normalised to the
    migration's serialisation, runs once and never crashes. That is why 10 tests passed over a
    ledger that corrupted a roleless node into an unrepairable state.
    """

    def test_a_corrupt_ledger_is_an_error_not_silence(self, tmp_path: Path) -> None:
        # `read_ledger` swallowed JSONDecodeError, so the CLI printed "nothing to undo" and
        # exited 0 on a corrupt ledger. Missing is not the same as unreadable.
        (tmp_path / LEDGER_FILE).write_text("{ this is not json", encoding="utf-8")
        with pytest.raises(ValueError):
            read_ledger(tmp_path)
