"""The backfill half of the name-canonicalisation work (#2130 step 1).

The mint-time fix (``identity.slugify.person_id`` / ``graph_id_utils.entity_node_id``) only
reaches episodes processed AFTER it. Production's artifacts were written before, so without this
migration the corpus holds ``person:peter-attia`` and ``person:peter-attia-md`` side by side —
two followable people for one man — and publishes ``Sophia Dew)`` on the person rail.

THE FIXTURES BELOW ARE THE MEASURED PRODUCTION CASES, not invented ones. Measured 2026-09-21 on
400 sampled production episodes:

* one id pair changes — ``person:peter-attia-md`` -> ``person:peter-attia``, in 4 episodes, and in
  all four the target already exists in the same episode, so it is a MERGE;
* eight names change with NO id change (``slugify`` already drops the punctuation), 39
  occurrences: the ``Name)`` close-paren family and ``Robert F. Kennedy Jr.``.

The second group is why an id-only migration would have been wrong and still looked finished.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations.m0010_canonical_person_names import (
    CanonicalPersonNamesMigration,
)

pytestmark = [pytest.mark.unit]


def _person(pid: str, name: str, role: str = "mentioned") -> Dict[str, Any]:
    return {"id": pid, "type": "Person", "properties": {"name": name, "role": role}}


def _episode(
    root: Path,
    stem: str,
    persons: List[Dict[str, Any]],
    *,
    edges: Optional[List[Dict[str, Any]]] = None,
    speakers: Optional[List[Dict[str, Any]]] = None,
    quotes: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """One episode's three artifacts, in the shape the pipeline writes them."""
    root.mkdir(parents=True, exist_ok=True)
    nodes: List[Dict[str, Any]] = [{"id": f"episode:{stem}", "type": "Episode"}]
    nodes += persons + (quotes or [])
    payload = {"nodes": nodes, "edges": edges or [], "episode_id": f"episode:{stem}"}
    (root / f"{stem}.gi.json").write_text(json.dumps(payload), encoding="utf-8")
    (root / f"{stem}.kg.json").write_text(json.dumps(payload), encoding="utf-8")
    if speakers is not None:
        (root / f"{stem}.metadata.json").write_text(
            json.dumps({"content": {"speakers": speakers}}), encoding="utf-8"
        )


def _run(root: Path, dry_run: bool = False):
    return CanonicalPersonNamesMigration().apply(
        MigrationContext(corpus_root=root, dry_run=dry_run)
    )


def _load(root: Path, stem: str, suffix: str) -> dict:
    return json.loads((root / f"{stem}{suffix}").read_text(encoding="utf-8"))


def _person_ids(payload: dict) -> List[str]:
    return [n["id"] for n in payload["nodes"] if n.get("type") == "Person"]


def _names(payload: dict) -> List[str]:
    return [
        (n.get("properties") or {}).get("name")
        for n in payload["nodes"]
        if n.get("type") == "Person"
    ]


# ------------------------------------------------------------------------------------------
# The id change: a MERGE, which is the case a naive rewrite corrupts.
# ------------------------------------------------------------------------------------------


class TestTheMeasuredIdMerge:
    def _attia(self, root: Path) -> None:
        """The production shape: both ids in ONE episode, the credentialed one a speaker."""
        _episode(
            root,
            "attia",
            [
                _person("person:peter-attia-md", "Peter Attia, MD", role="host"),
                _person("person:peter-attia", "Peter Attia", role="mentioned"),
            ],
            edges=[
                {"from": "episode:attia", "to": "person:peter-attia-md", "type": "FEATURES"},
                {"from": "episode:attia", "to": "person:peter-attia", "type": "MENTIONS"},
            ],
            speakers=[{"id": "speaker_1", "name": "Peter Attia, MD", "role": "host"}],
        )

    def test_the_two_ids_become_one_node(self, tmp_path: Path) -> None:
        self._attia(tmp_path)
        _run(tmp_path)
        ids = _person_ids(_load(tmp_path, "attia", ".gi.json"))
        assert ids == ["person:peter-attia"], (
            f"expected one merged person node, got {ids} — two nodes sharing one id is a corrupt "
            "graph, and emitting both ids is the duplicate this migration exists to remove"
        )

    def test_every_edge_endpoint_follows(self, tmp_path: Path) -> None:
        """A remapped node with a dangling edge is worse than no migration: the person is gone
        from the graph AND the episode still points at them."""
        self._attia(tmp_path)
        _run(tmp_path)
        targets = {e["to"] for e in _load(tmp_path, "attia", ".gi.json")["edges"]}
        assert targets == {"person:peter-attia"}, targets

    def test_the_speaking_role_survives_the_merge(self, tmp_path: Path) -> None:
        """The credentialed node was the HOST; the survivor carries ``mentioned``. Folding them
        must not demote a real speaker — the precedence `rewrite_ids` already implements."""
        self._attia(tmp_path)
        _run(tmp_path)
        nodes = [
            n for n in _load(tmp_path, "attia", ".kg.json")["nodes"] if n.get("type") == "Person"
        ]
        assert nodes[0]["properties"]["role"] == "host", nodes

    def test_the_surviving_node_carries_the_canonical_name(self, tmp_path: Path) -> None:
        self._attia(tmp_path)
        _run(tmp_path)
        assert _names(_load(tmp_path, "attia", ".gi.json")) == ["Peter Attia"]

    def test_the_published_speaker_name_is_fixed_too(self, tmp_path: Path) -> None:
        """`content.speakers` is what the player and the API read. A corpus whose graphs were
        fixed and whose metadata was not still SHOWS the old spelling to every listener."""
        self._attia(tmp_path)
        _run(tmp_path)
        speakers = _load(tmp_path, "attia", ".metadata.json")["content"]["speakers"]
        assert [s["name"] for s in speakers] == ["Peter Attia"]

    def test_a_quotes_speaker_id_follows(self, tmp_path: Path) -> None:
        """One level down: a quote attributed to an id that no longer exists loses its speaker."""
        _episode(
            tmp_path,
            "q",
            [_person("person:peter-attia-md", "Peter Attia, MD", role="host")],
            quotes=[
                {
                    "id": "quote:q0",
                    "type": "Quote",
                    "properties": {"text": "hello", "speaker_id": "person:peter-attia-md"},
                }
            ],
        )
        _run(tmp_path)
        quote = next(n for n in _load(tmp_path, "q", ".gi.json")["nodes"] if n["type"] == "Quote")
        assert quote["properties"]["speaker_id"] == "person:peter-attia"


# ------------------------------------------------------------------------------------------
# The name-only changes: 39 of the 43 measured occurrences, all invisible to an id-only fix.
# ------------------------------------------------------------------------------------------


class TestTheNameOnlyChanges:
    @pytest.mark.parametrize(
        "raw,canonical",
        [
            ("Sophia Dew)", "Sophia Dew"),
            ("Jen Kha)", "Jen Kha"),
            ("Ben Mildenhall)", "Ben Mildenhall"),
            ("Raghu Raghuram)", "Raghu Raghuram"),
            ("Andrew Chen)", "Andrew Chen"),
            ("Rayan Krishnan)", "Rayan Krishnan"),
            ("Lukasz Kaiser)", "Lukasz Kaiser"),
            ("Robert F. Kennedy Jr.", "Robert F. Kennedy Jr"),
        ],
    )
    def test_the_published_name_is_canonicalised(
        self, tmp_path: Path, raw: str, canonical: str
    ) -> None:
        """THE regression for the half an id-only migration misses. Every one of these mints the
        SAME id it always did — `slugify` drops the punctuation — so nothing about the id tells
        you the name is wrong."""
        from podcast_scraper.identity.slugify import person_id

        pid = person_id(raw)
        _episode(tmp_path, "n", [_person(pid, raw)], speakers=[{"id": "s1", "name": raw}])
        _run(tmp_path)

        assert _names(_load(tmp_path, "n", ".gi.json")) == [canonical]
        assert _names(_load(tmp_path, "n", ".kg.json")) == [canonical]
        assert _load(tmp_path, "n", ".metadata.json")["content"]["speakers"][0]["name"] == canonical

    def test_the_id_is_untouched_when_only_the_name_changes(self, tmp_path: Path) -> None:
        """Proves the two halves are independent: no id churn is generated for a punctuation fix."""
        _episode(tmp_path, "n", [_person("person:sophia-dew", "Sophia Dew)")])
        _run(tmp_path)
        assert _person_ids(_load(tmp_path, "n", ".gi.json")) == ["person:sophia-dew"]


# ------------------------------------------------------------------------------------------
# Safety: what it must NOT do.
# ------------------------------------------------------------------------------------------


class TestWhatItLeavesAlone:
    def test_an_already_canonical_corpus_is_untouched(self, tmp_path: Path) -> None:
        _episode(tmp_path, "ok", [_person("person:maya-okonkwo", "Maya Okonkwo")])
        before = (tmp_path / "ok.gi.json").read_bytes()
        result = _run(tmp_path)
        assert (tmp_path / "ok.gi.json").read_bytes() == before
        assert result.details["changed"] == 0

    def test_it_is_idempotent(self, tmp_path: Path) -> None:
        """The second run must plan nothing — a migration that keeps finding work is a migration
        that does not converge, and the ledger would re-run it forever."""
        _episode(
            tmp_path,
            "attia",
            [
                _person("person:peter-attia-md", "Peter Attia, MD", role="host"),
                _person("person:peter-attia", "Peter Attia"),
            ],
            speakers=[{"id": "s1", "name": "Peter Attia, MD"}],
        )
        _run(tmp_path)
        after_first = (tmp_path / "attia.gi.json").read_bytes()
        second = _run(tmp_path)
        assert (tmp_path / "attia.gi.json").read_bytes() == after_first
        assert second.details["changed"] == 0, second.message

    def test_a_scoped_id_is_never_re_minted(self, tmp_path: Path) -> None:
        """`person:unresolved-alex-ep42` is 0007's deliberate episode-scoping of a bare first
        name. Re-minting from its ``name`` would resolve it back to `person:alex` and silently
        undo that migration — a bare token becoming a global followable person again."""
        _episode(tmp_path, "s", [_person("person:unresolved-alex-ep42", "Alex")])
        _run(tmp_path)
        assert _person_ids(_load(tmp_path, "s", ".gi.json")) == ["person:unresolved-alex-ep42"]

    def test_a_scoped_nodes_NAME_is_still_canonicalised(self, tmp_path: Path) -> None:
        """The SCOPE is 0007's business; the SPELLING is this migration's. A scoped node showing
        ``Alex)`` on the rail is the same defect as an unscoped one."""
        _episode(tmp_path, "s", [_person("person:unresolved-alex-ep42", "Alex)")])
        _run(tmp_path)
        payload = _load(tmp_path, "s", ".gi.json")
        assert _person_ids(payload) == ["person:unresolved-alex-ep42"]
        assert _names(payload) == ["Alex"]

    def test_a_dry_run_writes_nothing(self, tmp_path: Path) -> None:
        _episode(
            tmp_path,
            "attia",
            [_person("person:peter-attia-md", "Peter Attia, MD")],
            speakers=[{"id": "s1", "name": "Peter Attia, MD"}],
        )
        before = (tmp_path / "attia.gi.json").read_bytes()
        meta_before = (tmp_path / "attia.metadata.json").read_bytes()
        result = _run(tmp_path, dry_run=True)
        assert (tmp_path / "attia.gi.json").read_bytes() == before
        assert (tmp_path / "attia.metadata.json").read_bytes() == meta_before
        assert result.details["changed"] == 1, "a dry run must still REPORT the work"

    def test_an_unparsable_artifact_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        """One corrupt file must not abandon the rest of a 678-episode corpus mid-run."""
        _episode(tmp_path, "good", [_person("person:peter-attia-md", "Peter Attia, MD")])
        (tmp_path / "bad.gi.json").write_text("{not json", encoding="utf-8")
        result = _run(tmp_path)
        assert result.applied is True
        assert result.details["unparsable"], result.details
        assert _person_ids(_load(tmp_path, "good", ".gi.json")) == ["person:peter-attia"]

    def test_an_episode_with_no_metadata_sibling_still_migrates(self, tmp_path: Path) -> None:
        """Not every artifact trio is complete on disk; a missing third file is not an error."""
        _episode(tmp_path, "attia", [_person("person:peter-attia-md", "Peter Attia, MD")])
        assert not (tmp_path / "attia.metadata.json").exists()
        _run(tmp_path)
        assert _person_ids(_load(tmp_path, "attia", ".gi.json")) == ["person:peter-attia"]


class TestItIsRegistered:
    def test_the_runner_knows_about_it(self) -> None:
        """A migration absent from the registry never runs, and nothing else would notice."""
        from podcast_scraper.upgrade.registry import get_migrations

        ids = [m.id for m in get_migrations()]
        assert "0010_canonical_person_names" in ids, ids

    def test_it_sorts_after_0009(self) -> None:
        """The registry applies in id order, and 0007's scoping must already have happened —
        this migration's scoped-id guard assumes those ids exist in their final form."""
        from podcast_scraper.upgrade.registry import get_migrations

        ids = [m.id for m in get_migrations()]
        assert ids.index("0010_canonical_person_names") > ids.index("0007_scope_bare_person_names")
        assert ids.index("0010_canonical_person_names") > ids.index("0009_backfill_speaker_roles")


class TestItNeverDemotesAResolvedPerson:
    """THE destructive bug this migration nearly shipped, caught by dry-running it over the real
    2,257-episode production snapshot rather than the 400-episode sample.

    The first draft's rule was "re-mint the id from the node's name". On that snapshot it planned
    83 remaps, of which only the 39 ``peter-attia-md`` episodes were real. The other 44 were nodes
    whose id is a FULL NAME while ``properties.name`` holds something narrower — a bare first name,
    a surname, a numbered speaker — so re-minting demoted a fully-resolved person to a global bare
    token, which is exactly what 0007 exists to prevent.

    The rule is now "the OLD rule applied to this name produces the id this node already has",
    which is the only evidence that the id came from the name at all.
    """

    @pytest.mark.parametrize(
        "pid,name",
        [
            ("person:joe-weisenthal", "Joe"),
            ("person:casey-newton", "Casey"),
            ("person:kashmir-hill", "Kashmir"),
            ("person:elad-gil", "Gil"),
            ("person:leo-strauss", "Strauss"),
            ("person:esther-miriam-wagner", "Miriam"),
            ("person:donald-trump-jr", "Trump"),
            ("person:speaker-substackpost213603862-06", "Speaker 06"),
            ("person:speaker-urnbbcpodcastp0p8cg68-01", "Speaker 01"),
        ],
    )
    def test_a_full_name_id_is_not_re_minted_from_a_narrower_name(
        self, tmp_path: Path, pid: str, name: str
    ) -> None:
        _episode(tmp_path, "d", [_person(pid, name)])
        _run(tmp_path)
        assert _person_ids(_load(tmp_path, "d", ".gi.json")) == [pid], (
            f"{pid} was re-minted from the name {name!r} — that demotes a resolved person to a "
            "global bare token, the defect 0007 exists to prevent"
        )

    def test_the_genuine_credential_case_still_migrates(self, tmp_path: Path) -> None:
        """The narrowing must not disarm the migration: `person:peter-attia-md` IS what the old
        rule mints from `Peter Attia, MD`, so it is still remapped."""
        _episode(tmp_path, "a", [_person("person:peter-attia-md", "Peter Attia, MD")])
        _run(tmp_path)
        assert _person_ids(_load(tmp_path, "a", ".gi.json")) == ["person:peter-attia"]


class TestItCanBeVerifiedAfterwards:
    """A migration that runs unattended after a deploy and cannot be checked afterwards is a
    migration you have to trust.

    The default `verify` returns "no verification defined", and this repo has already paid for
    that: `upgrade verify` said exactly that while the ledger claimed a version the data did not
    have, and nothing in the system could tell (recorded in `upgrade/state.py`).
    """

    def _dirty(self, root: Path) -> None:
        _episode(
            root,
            "attia",
            [
                _person("person:peter-attia-md", "Peter Attia, MD", role="host"),
                _person("person:peter-attia", "Peter Attia"),
            ],
            speakers=[{"id": "s1", "name": "Peter Attia, MD"}],
        )

    def test_it_fails_before_the_migration_runs(self, tmp_path: Path) -> None:
        self._dirty(tmp_path)
        ok, msg = CanonicalPersonNamesMigration().verify(
            MigrationContext(corpus_root=tmp_path, dry_run=True)
        )
        assert ok is False, msg
        assert "pre-canonical" in msg

    def test_it_passes_after_the_migration_runs(self, tmp_path: Path) -> None:
        self._dirty(tmp_path)
        _run(tmp_path)
        ok, msg = CanonicalPersonNamesMigration().verify(
            MigrationContext(corpus_root=tmp_path, dry_run=True)
        )
        assert ok is True, msg

    def test_it_catches_a_name_left_behind_even_when_every_id_is_right(
        self, tmp_path: Path
    ) -> None:
        """The half an id-only check would miss — 195 of the 234 measured production occurrences.
        `person:sophia-dew` is the correct id for `Sophia Dew)`, so nothing about the ids is
        wrong; the published name still is."""
        _episode(tmp_path, "n", [_person("person:sophia-dew", "Sophia Dew)")])
        ok, msg = CanonicalPersonNamesMigration().verify(
            MigrationContext(corpus_root=tmp_path, dry_run=True)
        )
        assert ok is False and "Sophia Dew)" in msg, msg

    def test_it_catches_a_speakers_name_in_metadata_only(self, tmp_path: Path) -> None:
        """The third surface. Graphs clean, metadata stale — what the player would still show."""
        _episode(
            tmp_path,
            "m",
            [_person("person:peter-attia", "Peter Attia")],
            speakers=[{"id": "s1", "name": "Peter Attia, MD"}],
        )
        ok, msg = CanonicalPersonNamesMigration().verify(
            MigrationContext(corpus_root=tmp_path, dry_run=True)
        )
        assert ok is False and "speakers" in msg, msg

    def test_it_does_not_demand_what_apply_deliberately_skips(self, tmp_path: Path) -> None:
        """verify must use the SAME narrow rule as apply. A node whose id is a full name and whose
        name is a bare first name is deliberately left alone — verify calling that a failure would
        make a correct corpus permanently unverifiable."""
        _episode(tmp_path, "s", [_person("person:joe-weisenthal", "Joe")])
        ok, msg = CanonicalPersonNamesMigration().verify(
            MigrationContext(corpus_root=tmp_path, dry_run=True)
        )
        assert ok is True, msg


class TestTwoNodesOneIdAreFolded:
    """MEASURED IN PRODUCTION: `The State of AI: Macro, Apps and...`, kg layer, carries
    `person:jen-kha` TWICE — one node named `Jen Kha` with role `mentioned`, one named `Jen Kha)`
    with role `host`. Two different name strings that `slugify` reduces to one id, so both were
    emitted under it. Two nodes sharing one id is a corrupt graph.

    Canonicalising the NAME alone does not fix it: the ids already agree, so the remap never runs
    and the artifact ends with two identical-looking nodes instead of two different-looking ones.
    """

    def _jen(self, root: Path) -> None:
        _episode(
            root,
            "jen",
            [
                _person("person:jen-kha", "Jen Kha", role="mentioned"),
                _person("person:jen-kha", "Jen Kha)", role="host"),
            ],
            edges=[{"from": "episode:jen", "to": "person:jen-kha", "type": "FEATURES"}],
        )

    def test_the_duplicate_is_folded_into_one_node(self, tmp_path: Path) -> None:
        self._jen(tmp_path)
        _run(tmp_path)
        nodes = [
            n for n in _load(tmp_path, "jen", ".kg.json")["nodes"] if n.get("type") == "Person"
        ]
        assert len(nodes) == 1, f"two nodes still share one id: {nodes}"
        assert nodes[0]["properties"]["name"] == "Jen Kha"

    def test_the_stated_role_wins_the_fold(self, tmp_path: Path) -> None:
        """One node says `host`, the other `mentioned`. Folding must not demote the real speaker —
        the precedence `rewrite_ids` already implements."""
        self._jen(tmp_path)
        _run(tmp_path)
        nodes = [
            n for n in _load(tmp_path, "jen", ".kg.json")["nodes"] if n.get("type") == "Person"
        ]
        assert nodes[0]["properties"]["role"] == "host", nodes

    def test_verify_fails_while_a_duplicate_remains(self, tmp_path: Path) -> None:
        self._jen(tmp_path)
        ok, msg = CanonicalPersonNamesMigration().verify(
            MigrationContext(corpus_root=tmp_path, dry_run=True)
        )
        assert ok is False and "two nodes at once" in msg, msg

    def test_verify_passes_once_it_is_folded(self, tmp_path: Path) -> None:
        self._jen(tmp_path)
        _run(tmp_path)
        ok, msg = CanonicalPersonNamesMigration().verify(
            MigrationContext(corpus_root=tmp_path, dry_run=True)
        )
        assert ok is True, msg

    def test_a_clean_episode_is_not_rewritten_by_the_dedupe_pass(self, tmp_path: Path) -> None:
        """The pass must stay inert where there is nothing to fold."""
        _episode(tmp_path, "ok", [_person("person:maya-okonkwo", "Maya Okonkwo")])
        before = (tmp_path / "ok.kg.json").read_bytes()
        _run(tmp_path)
        assert (tmp_path / "ok.kg.json").read_bytes() == before
