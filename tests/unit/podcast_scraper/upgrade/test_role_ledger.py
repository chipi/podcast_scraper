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

from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations.m0009_backfill_speaker_roles import (
    BackfillSpeakerRolesMigration,
)
from podcast_scraper.upgrade.role_ledger import (
    LEDGER_FILE,
    read_ledger,
    RoleChange,
    undo_from_ledger,
    write_ledger,
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


class TestTheLedgerRecordsWhatChanged:
    def test_a_real_run_writes_every_transition(self, tmp_path: Path) -> None:
        _episode(
            tmp_path,
            "e1",
            [
                ("person:africa-tech-summit", "Africa Tech Summit", "host"),
                ("person:nixon-kanali", "Nixon Kanali", "mentioned"),
            ],
            feed="Africa Tech Summit Podcast",
            speakers=[{"name": "Nixon Kanali", "role": "host"}],
            heard=1,
        )
        BackfillSpeakerRolesMigration().apply(MigrationContext(corpus_root=tmp_path, dry_run=False))
        entries = read_ledger(tmp_path)
        by_id = {e.node_id: e for e in entries}
        assert by_id["person:africa-tech-summit"].role_before == "host"
        assert by_id["person:africa-tech-summit"].role_after == "mentioned"
        assert by_id["person:nixon-kanali"].role_before == "mentioned"
        assert by_id["person:nixon-kanali"].role_after == "host"

    def test_a_dry_run_writes_nothing(self, tmp_path: Path) -> None:
        _episode(
            tmp_path,
            "e1",
            [("person:africa-tech-summit", "Africa Tech Summit", "host")],
            feed="Africa Tech Summit Podcast",
        )
        BackfillSpeakerRolesMigration().apply(MigrationContext(corpus_root=tmp_path, dry_run=True))
        assert not (tmp_path / LEDGER_FILE).exists()
        assert read_ledger(tmp_path) == []

    def test_the_ledger_names_the_route_that_made_the_change(self, tmp_path: Path) -> None:
        # "Which rule did this?" is the first question when a demotion looks wrong.
        _episode(
            tmp_path,
            "e1",
            [("person:africa-tech-summit", "Africa Tech Summit", "host")],
            feed="Africa Tech Summit Podcast",
        )
        BackfillSpeakerRolesMigration().apply(MigrationContext(corpus_root=tmp_path, dry_run=False))
        entry = read_ledger(tmp_path)[0]
        assert entry.route in {"not_a_person", "roster_denies", "promote"}
        assert entry.route == "not_a_person"


class TestUndoPutsItBack:
    def test_undo_restores_every_role(self, tmp_path: Path) -> None:
        kg = _episode(
            tmp_path,
            "e1",
            [
                ("person:africa-tech-summit", "Africa Tech Summit", "host"),
                ("person:nixon-kanali", "Nixon Kanali", "mentioned"),
            ],
            feed="Africa Tech Summit Podcast",
            speakers=[{"name": "Nixon Kanali", "role": "host"}],
            heard=1,
        )
        before = _roles(kg)
        BackfillSpeakerRolesMigration().apply(MigrationContext(corpus_root=tmp_path, dry_run=False))
        assert _roles(kg) != before, "the migration must actually have changed something"
        restored, refused = undo_from_ledger(tmp_path)
        assert refused == []
        assert restored == 2
        assert _roles(kg) == before

    def test_undo_refuses_a_node_something_else_has_touched(self, tmp_path: Path) -> None:
        # A `relabel_only` re-enrich after the migration produces a role that is NEWER and better
        # than anything this ledger knows. Replaying `role_before` over it would be a regression
        # dressed as a rollback.
        kg = _episode(
            tmp_path,
            "e1",
            [("person:africa-tech-summit", "Africa Tech Summit", "host")],
            feed="Africa Tech Summit Podcast",
        )
        BackfillSpeakerRolesMigration().apply(MigrationContext(corpus_root=tmp_path, dry_run=False))
        doc = json.loads(kg.read_text(encoding="utf-8"))
        doc["nodes"][0]["properties"]["role"] = "guest"  # someone else moved it
        kg.write_text(json.dumps(doc), encoding="utf-8")

        restored, refused = undo_from_ledger(tmp_path)
        assert restored == 0
        assert len(refused) == 1 and "africa-tech-summit" in refused[0]
        assert _roles(kg)["person:africa-tech-summit"] == "guest", "newer work is preserved"

    def test_undo_is_idempotent(self, tmp_path: Path) -> None:
        _episode(
            tmp_path,
            "e1",
            [("person:africa-tech-summit", "Africa Tech Summit", "host")],
            feed="Africa Tech Summit Podcast",
        )
        BackfillSpeakerRolesMigration().apply(MigrationContext(corpus_root=tmp_path, dry_run=False))
        first, _ = undo_from_ledger(tmp_path)
        assert first == 1
        second, refused = undo_from_ledger(tmp_path)
        assert second == 0, "a second undo restores nothing"
        assert len(refused) == 1, "and says why, rather than silently doing nothing"

    def test_undo_with_no_ledger_is_not_an_error(self, tmp_path: Path) -> None:
        restored, refused = undo_from_ledger(tmp_path)
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
        write_ledger(tmp_path, rows)
        assert read_ledger(tmp_path) == rows

    def test_the_file_is_plain_json_a_human_can_read(self, tmp_path: Path) -> None:
        write_ledger(
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
                )
            ],
        )
        doc = json.loads((tmp_path / LEDGER_FILE).read_text(encoding="utf-8"))
        assert doc["migration"] == "0009_backfill_speaker_roles"
        assert doc["changes"][0]["name"] == "X"
        assert "written_at" in doc


class TestUndoIsAByteLevelRollback:
    """The corpus must come back byte-identical, not merely role-identical.

    A first version restored every role correctly and still left a different corpus hash, because
    it serialised compactly while the migration writes ``indent=2`` plus a trailing newline. Every
    touched file then shows as modified in git or rsync, and "did the undo work?" stops being
    answerable by comparing hashes — which is the cheapest check an operator has.
    """

    def test_the_file_comes_back_byte_identical(self, tmp_path: Path) -> None:
        kg = _episode(
            tmp_path,
            "e1",
            [
                ("person:africa-tech-summit", "Africa Tech Summit", "host"),
                ("person:nixon-kanali", "Nixon Kanali", "mentioned"),
            ],
            feed="Africa Tech Summit Podcast",
            speakers=[{"name": "Nixon Kanali", "role": "host"}],
            heard=1,
        )
        # Normalise to the migration's own serialisation first: the fixture writes compact JSON,
        # and this test is about the UNDO matching the MIGRATION, not the fixture.
        doc = json.loads(kg.read_text(encoding="utf-8"))
        kg.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        original = kg.read_bytes()

        BackfillSpeakerRolesMigration().apply(MigrationContext(corpus_root=tmp_path, dry_run=False))
        assert kg.read_bytes() != original
        undo_from_ledger(tmp_path)
        assert kg.read_bytes() == original
