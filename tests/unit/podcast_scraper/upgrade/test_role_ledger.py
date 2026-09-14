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
        restored, _skipped, refused = undo_from_ledger(tmp_path)
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

        restored, _skipped, refused = undo_from_ledger(tmp_path)
        assert restored == 0
        # Refused at EPISODE level now: the file's sha no longer matches what the migration wrote,
        # and a file we did not write last is one whose contents we cannot reason about.
        assert len(refused) == 1 and "e1.kg.json" in refused[0]
        assert _roles(kg)["person:africa-tech-summit"] == "guest", "newer work is preserved"

    def test_undo_is_idempotent(self, tmp_path: Path) -> None:
        _episode(
            tmp_path,
            "e1",
            [("person:africa-tech-summit", "Africa Tech Summit", "host")],
            feed="Africa Tech Summit Podcast",
        )
        BackfillSpeakerRolesMigration().apply(MigrationContext(corpus_root=tmp_path, dry_run=False))
        first, _s, _r = undo_from_ledger(tmp_path)
        assert first == 1
        second, skipped, refused = undo_from_ledger(tmp_path)
        assert second == 0, "a second undo restores nothing"
        assert (
            skipped and not refused
        ), "its own previous work is SKIPPED, not reported as a foreign writer"

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


class TestShapesTheFirstVersionCorrupted:
    """Five shapes the original fixtures never took. Each was a real defect (advisor, 3rd pass).

    Every fixture in the classes above sets ``role``, uses ASCII names, is pre-normalised to the
    migration's serialisation, runs once and never crashes. That is why 10 tests passed over a
    ledger that corrupted a roleless node into an unrepairable state.
    """

    def test_a_node_with_no_role_comes_back_with_no_role(self, tmp_path: Path) -> None:
        # `role_before` was the literal string "none", written back verbatim. "none" is not in
        # `_PROMOTABLE`, so a re-run could never repair it: the undo made the node WORSE than
        # not undoing. Roleless Person nodes are expected — see
        # test_a_person_with_no_role_at_all_is_promoted in the m0009 suite.
        d = tmp_path / "metadata"
        d.mkdir(parents=True)
        (d / "e1.metadata.json").write_text(
            json.dumps(
                {
                    "feed": {"title": "Show"},
                    "content": {
                        "speakers": [{"name": "Nixon Kanali", "role": "host"}],
                        "diarization_num_speakers": 1,
                    },
                }
            ),
            encoding="utf-8",
        )
        kg = d / "e1.kg.json"
        kg.write_text(
            json.dumps(
                {
                    "schema_version": "2.1",
                    "nodes": [
                        {
                            "id": "person:nixon-kanali",
                            "type": "Person",
                            "properties": {"name": "Nixon Kanali"},
                        }
                    ],
                    "edges": [],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        BackfillSpeakerRolesMigration().apply(MigrationContext(corpus_root=tmp_path, dry_run=False))
        assert json.loads(kg.read_text())["nodes"][0]["properties"]["role"] == "host"
        undo_from_ledger(tmp_path)
        props = json.loads(kg.read_text())["nodes"][0]["properties"]
        assert "role" not in props, f"the key must be GONE, not set to a sentinel: {props}"

    def test_a_node_someone_else_rewrote_is_refused_even_when_the_role_agrees(
        self, tmp_path: Path
    ) -> None:
        # THE REFUSAL PREDICATE. Role-equality detects "someone moved this node to a DIFFERENT
        # role" and is blind to "someone rewrote this file and happened to agree".
        #
        # That is the likely production sequence, not a corner case: runbook step 3 is a re-enrich
        # (`rederive_only` / `rediarize_only` cascade to GI/KG), and the rebuilt graph now reads
        # the roster — so it writes `host` for most of the same nodes m0009 promoted. An undo
        # afterwards would demote all of them with ZERO refused, reporting a clean rollback while
        # destroying the re-enrich's independently-derived answer.
        #
        # So the ledger records the sha256 of each file AS IT LEFT THE MIGRATION, and a changed
        # file refuses the whole episode.
        kg = _episode(
            tmp_path,
            "e1",
            [("person:nixon-kanali", "Nixon Kanali", "mentioned")],
            speakers=[{"name": "Nixon Kanali", "role": "host"}],
            heard=1,
        )
        BackfillSpeakerRolesMigration().apply(MigrationContext(corpus_root=tmp_path, dry_run=False))
        assert _roles(kg)["person:nixon-kanali"] == "host"

        doc = json.loads(kg.read_text(encoding="utf-8"))
        doc["nodes"].append(
            {"id": "topic:new", "type": "Topic", "properties": {"label": "added by a re-enrich"}}
        )
        kg.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

        restored, _skipped, refused = undo_from_ledger(tmp_path)
        assert restored == 0, "the role still says 'host', but the FILE changed — hands off"
        assert refused and "e1.kg.json" in refused[0]
        assert _roles(kg)["person:nixon-kanali"] == "host", "the newer answer survives"

    def test_a_second_undo_says_already_restored_not_someone_else_wrote_this(
        self, tmp_path: Path
    ) -> None:
        # The first version reported the undo's own work as "something else wrote this node",
        # which sends an operator looking for a concurrent writer that does not exist.
        _episode(
            tmp_path,
            "e1",
            [("person:africa-tech-summit", "Africa Tech Summit", "host")],
            feed="Africa Tech Summit Podcast",
        )
        BackfillSpeakerRolesMigration().apply(MigrationContext(corpus_root=tmp_path, dry_run=False))
        undo_from_ledger(tmp_path)
        restored, skipped, refused = undo_from_ledger(tmp_path)
        assert restored == 0
        assert skipped and "already restored" in skipped[0].lower(), (skipped, refused)
        assert not refused, "the undo's own previous work is not a foreign writer"

    def test_a_second_migration_run_does_not_erase_the_first_runs_ledger(
        self, tmp_path: Path
    ) -> None:
        # `write_ledger` did `os.replace`, so run 2 discarded run 1's rows — and the module
        # docstring promised the next migration could append.
        _episode(
            tmp_path,
            "e1",
            [("person:africa-tech-summit", "Africa Tech Summit", "host")],
            feed="Africa Tech Summit Podcast",
        )
        BackfillSpeakerRolesMigration().apply(MigrationContext(corpus_root=tmp_path, dry_run=False))
        first = read_ledger(tmp_path)
        assert first
        _episode(
            tmp_path,
            "e2",
            [("person:machine-learning-street", "Machine Learning Street", "host")],
            feed="Machine Learning Street Talk",
        )
        BackfillSpeakerRolesMigration().apply(MigrationContext(corpus_root=tmp_path, dry_run=False))
        all_rows = read_ledger(tmp_path)
        ids = {r.node_id for r in all_rows}
        assert "person:africa-tech-summit" in ids, "run 1's rows must survive run 2"
        assert "person:machine-learning-street" in ids

    def test_a_corrupt_ledger_is_an_error_not_silence(self, tmp_path: Path) -> None:
        # `read_ledger` swallowed JSONDecodeError, so the CLI printed "nothing to undo" and
        # exited 0 on a corrupt ledger. Missing is not the same as unreadable.
        (tmp_path / LEDGER_FILE).write_text("{ this is not json", encoding="utf-8")
        with pytest.raises(ValueError):
            read_ledger(tmp_path)
