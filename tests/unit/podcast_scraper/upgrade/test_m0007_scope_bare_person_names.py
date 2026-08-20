"""The backfill half of #1685 — the mint-time fix alone leaves production unchanged.

Without this migration, the 1931 person entities already in production keep their pooled tokens
indefinitely, and `derived_interest_tokens` keeps re-minting `person:sam` into user profiles from
listening behaviour with no click involved. The pipeline change only affects episodes processed
after it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.identity.bare_name_scope import is_bare_person_id, person_ids_in
from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations.m0007_scope_bare_person_names import (
    ScopeBarePersonNamesMigration,
)

pytestmark = [pytest.mark.unit]


def _episode(root: Path, stem: str, ep_id: str, gi_persons, kg_persons, edges=None):
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{stem}.gi.json").write_text(
        json.dumps(
            {
                "nodes": [{"id": f"episode:{ep_id}", "type": "Episode"}]
                + [{"id": p, "type": "Person"} for p in gi_persons],
                "edges": edges or [],
            }
        ),
        encoding="utf-8",
    )
    (root / f"{stem}.kg.json").write_text(
        json.dumps(
            {
                "nodes": [{"id": f"episode:{ep_id}", "type": "Episode"}]
                + [{"id": p, "type": "Person"} for p in kg_persons],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )


def _bare_ids(root: Path) -> set:
    out = set()
    for f in list(root.rglob("*.gi.json")) + list(root.rglob("*.kg.json")):
        try:
            doc = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue  # the deliberately-corrupt artifact in one of the tests below
        for pid in person_ids_in(doc):
            if is_bare_person_id(pid):
                out.add(pid)
    return out


class TestItScopesAnExistingCorpus:
    def test_bare_ids_are_gone_afterwards(self, tmp_path) -> None:
        _episode(tmp_path, "e1", "ep-1", ["person:jensen"], ["person:jensen"])
        assert _bare_ids(tmp_path) == {"person:jensen"}
        ScopeBarePersonNamesMigration().apply(MigrationContext(corpus_root=tmp_path, dry_run=False))
        assert _bare_ids(tmp_path) == set()

    def test_a_dry_run_writes_nothing(self, tmp_path) -> None:
        _episode(tmp_path, "e1", "ep-1", ["person:jensen"], ["person:jensen"])
        before = (tmp_path / "e1.gi.json").read_text()
        r = ScopeBarePersonNamesMigration().apply(
            MigrationContext(corpus_root=tmp_path, dry_run=True)
        )
        assert r.dry_run is True
        assert (tmp_path / "e1.gi.json").read_text() == before

    def test_it_is_idempotent(self, tmp_path) -> None:
        """The runner may replay it; a second pass must change nothing."""
        _episode(tmp_path, "e1", "ep-1", ["person:jensen"], ["person:jensen"])
        m = ScopeBarePersonNamesMigration()
        m.apply(MigrationContext(corpus_root=tmp_path, dry_run=False))
        after_first = (tmp_path / "e1.gi.json").read_text()
        m.apply(MigrationContext(corpus_root=tmp_path, dry_run=False))
        assert (tmp_path / "e1.gi.json").read_text() == after_first

    def test_real_people_are_untouched(self, tmp_path) -> None:
        _episode(tmp_path, "e1", "ep-1", ["person:elon-musk"], ["person:elon-musk"])
        before = (tmp_path / "e1.gi.json").read_text()
        ScopeBarePersonNamesMigration().apply(MigrationContext(corpus_root=tmp_path, dry_run=False))
        assert (tmp_path / "e1.gi.json").read_text() == before


class TestBothLayersDecideTogether:
    def test_a_full_name_in_GI_heals_a_bare_name_in_KG(self, tmp_path) -> None:
        """The `person:alex` case: the KG alone reports an orphan, the pair resolves it.

        This is why the migration pairs the artifacts instead of walking files independently
        the way 0006 does.
        """
        _episode(tmp_path, "e1", "ep-1", ["person:alex", "person:alex-mayassi"], ["person:alex"])
        ScopeBarePersonNamesMigration().apply(MigrationContext(corpus_root=tmp_path, dry_run=False))
        kg = json.loads((tmp_path / "e1.kg.json").read_text())
        ids = {n["id"] for n in kg["nodes"]}
        assert "person:alex-mayassi" in ids, "the bare id should have healed to the full name"
        assert "person:alex" not in ids

    def test_a_full_name_in_KG_heals_a_bare_name_in_GI(self, tmp_path) -> None:
        """The mirror of the case above, and the one that proves the union is real.

        My first version of this class only tested GI-carrying-the-full-name, so dropping the KG
        from the roster entirely failed ZERO tests — the sabotage caught a test that looked like
        coverage and was not. Here the full name exists ONLY in the KG, so a GI-only roster
        cannot heal it.
        """
        _episode(tmp_path, "e1", "ep-1", ["person:alex"], ["person:alex-mayassi"])
        ScopeBarePersonNamesMigration().apply(MigrationContext(corpus_root=tmp_path, dry_run=False))
        gi = json.loads((tmp_path / "e1.gi.json").read_text())
        ids = {n["id"] for n in gi["nodes"]}
        assert "person:alex-mayassi" in ids, "the KG's full name did not reach the roster"
        assert "person:alex" not in ids

    def test_the_episode_id_comes_from_the_artifact_not_the_filename(self, tmp_path) -> None:
        """Scoped ids embed the episode. If the migration derived it differently from the
        pipeline, the two would mint different ids for the same episode — the drift the shared
        rule exists to prevent."""
        _episode(tmp_path, "unrelated-filename", "ep-real-id", ["person:jensen"], [])
        ScopeBarePersonNamesMigration().apply(MigrationContext(corpus_root=tmp_path, dry_run=False))
        gi = json.loads((tmp_path / "unrelated-filename.gi.json").read_text())
        scoped = [n["id"] for n in gi["nodes"] if "unresolved" in n["id"]]
        assert scoped == ["person:unresolved-jensen-ep-real-id"]


class TestTheHealMergeTrap:
    def test_healing_onto_an_existing_node_does_not_duplicate_it(self, tmp_path) -> None:
        """`person:sam` healed into an episode that ALREADY has `person:sam-altman` must MERGE.

        A naive relabel emits two nodes with one id — a corrupt graph.
        """
        _episode(tmp_path, "e1", "ep-1", ["person:sam", "person:sam-altman"], ["person:sam-altman"])
        ScopeBarePersonNamesMigration().apply(MigrationContext(corpus_root=tmp_path, dry_run=False))
        gi = json.loads((tmp_path / "e1.gi.json").read_text())
        ids = [n["id"] for n in gi["nodes"]]
        assert ids.count("person:sam-altman") == 1, f"duplicate node ids: {ids}"

    def test_edges_follow_the_rewrite(self, tmp_path) -> None:
        _episode(
            tmp_path,
            "e1",
            "ep-1",
            ["person:jensen"],
            [],
            edges=[{"source": "insight:1", "target": "person:jensen", "type": "MENTIONS"}],
        )
        ScopeBarePersonNamesMigration().apply(MigrationContext(corpus_root=tmp_path, dry_run=False))
        gi = json.loads((tmp_path / "e1.gi.json").read_text())
        node_ids = {n["id"] for n in gi["nodes"]}
        assert gi["edges"][0]["target"] in node_ids, "edge points at a node that no longer exists"


class TestItSurvivesABrokenCorpus:
    def test_an_unparsable_artifact_is_skipped_not_fatal(self, tmp_path) -> None:
        _episode(tmp_path, "good", "ep-1", ["person:jensen"], [])
        (tmp_path / "bad.gi.json").write_text("{ not json", encoding="utf-8")
        r = ScopeBarePersonNamesMigration().apply(
            MigrationContext(corpus_root=tmp_path, dry_run=False)
        )
        assert r.applied is True
        assert r.details["unparsable"], "the unparsable file should be reported, not silently lost"
        assert _bare_ids(tmp_path) == set(), "the good episode must still have been migrated"

    def test_a_missing_kg_sibling_is_fine(self, tmp_path) -> None:
        (tmp_path / "solo.gi.json").write_text(
            json.dumps(
                {
                    "nodes": [
                        {"id": "episode:ep-1", "type": "Episode"},
                        {"id": "person:jensen", "type": "Person"},
                    ],
                    "edges": [],
                }
            ),
            encoding="utf-8",
        )
        ScopeBarePersonNamesMigration().apply(MigrationContext(corpus_root=tmp_path, dry_run=False))
        assert _bare_ids(tmp_path) == set()

    def test_an_empty_corpus_is_not_an_error(self, tmp_path) -> None:
        r = ScopeBarePersonNamesMigration().apply(
            MigrationContext(corpus_root=tmp_path, dry_run=False)
        )
        assert r.applied is True
        assert r.details["episodes_scanned"] == 0


class TestItIsRegistered:
    def test_the_runner_knows_about_it(self) -> None:
        """An unregistered migration is a file nobody runs."""
        from podcast_scraper.upgrade import registry

        ids = [m.id for m in registry.get_migrations()]
        assert "0007_scope_bare_person_names" in ids
