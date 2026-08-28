"""The roster must reach everywhere the rewriter writes — and the heal pool must not (#1868).

Two rules, opposite directions, both load-bearing:

  ROSTER (what gets scoped) must be WIDE — exactly as wide as `rewrite_ids` writes. Narrower and
  an id it cannot see survives the pass while the same person's other id is scoped, so the
  artifact holds both forms and contradicts itself.

  CANDIDATES (what a bare name may be healed INTO) must be NARROW — node-backed only. An id with
  no node is a dangling reference, the least-validated string in the artifact, and healing is the
  one branch that writes a REAL person's id onto someone else's content with no cheap undo.
  `rewrite_bridges_m0007._graph_person_ids` already applies this rule to a REVERSIBLE bridge
  substitution; the irreversible branch must not accept weaker evidence.

The shapes below are the ACTUAL production cases, not invented ones. Audit run 33123775074
(2026-08-27) found all 23 coexistence episodes at `gi:node_speaker` — quote nodes'
`properties.speaker_id` — a location the first attempted fix did not touch at all:

    host      at [gi:node_speaker]  alongside unresolved-host-c74cda10-...
    brandon   at [gi:node_speaker]  alongside unresolved-brandon-substack-post-209779219
    sam       at [gi:node_speaker]  alongside unresolved-sam-c8fefeb8-...
    gabrielle at [gi:node_speaker]  alongside unresolved-gabrielle-f29b02f2-...
                                   (heal candidate: gabrielle-steinhauser)
    erica     at [gi:node_speaker]  alongside unresolved-erica-9a23f11d-...
"""

from __future__ import annotations

import pytest

from podcast_scraper.identity.bare_name_scope import (
    _EDGE_ENDPOINT_KEYS,
    person_ids_in,
    person_node_ids_in,
    plan_bare_name_ids,
    rewrite_ids,
)

pytestmark = [pytest.mark.unit]

EP = "substack-post-209779219"


def _quote(speaker_id: str, text: str = "a quote"):
    """A quote node as the GI schema defines it (`$defs/quote_node`, gi.schema.json:472)."""
    return {
        "id": f"quote:{abs(hash(text)) % 10**8}",
        "type": "Quote",
        "properties": {"text": text, "speaker_id": speaker_id, "transcript_ref": "t.txt"},
    }


class TestTheRosterReachesTheProductionLocation:
    """All 23 real cases were quote-node `properties.speaker_id`. Nothing else mattered."""

    def test_a_quote_speaker_id_is_in_the_roster(self) -> None:
        art = {"nodes": [_quote("person:brandon")], "edges": []}
        assert "person:brandon" in person_ids_in(art)

    def test_it_was_NOT_before_which_is_why_the_23_survived(self) -> None:
        """Nodes-only is what `person_node_ids_in` still is, and it must NOT see this."""
        art = {"nodes": [_quote("person:brandon")], "edges": []}
        assert person_node_ids_in(art) == set()

    def test_the_scoping_rewrites_the_quote_too(self) -> None:
        """The whole point: the quote must not stay attributed to an id that no longer exists."""
        art = {
            "nodes": [
                {"id": "person:brandon", "kind": "person"},
                _quote("person:brandon"),
            ],
            "edges": [],
        }
        id_map = plan_bare_name_ids(
            person_ids_in(art), EP, heal=False, candidate_ids=person_node_ids_in(art)
        )
        out, changes = rewrite_ids(art, id_map)
        scoped = f"person:unresolved-brandon-{EP}"
        assert {n["id"] for n in out["nodes"] if n.get("kind") == "person"} == {scoped}
        quote = next(n for n in out["nodes"] if n.get("type") == "Quote")
        assert quote["properties"]["speaker_id"] == scoped, quote
        assert changes >= 2

    def test_the_exact_production_shape_converges(self) -> None:
        """Placeholder node already present (run 1), bare id still on the quote. Re-running must
        land BOTH on the same id — not mint a second placeholder."""
        scoped = f"person:unresolved-brandon-{EP}"
        art = {"nodes": [{"id": scoped, "kind": "person"}, _quote("person:brandon")], "edges": []}
        id_map = plan_bare_name_ids(
            person_ids_in(art), EP, heal=False, candidate_ids=person_node_ids_in(art)
        )
        out, _ = rewrite_ids(art, id_map)
        node_ids = {n["id"] for n in out["nodes"] if n.get("kind") == "person"}
        quote = next(n for n in out["nodes"] if n.get("type") == "Quote")
        assert quote["properties"]["speaker_id"] in node_ids == {scoped}


class TestTheRosterAlsoReachesEdges:
    """Not the cause of the 23, but `rewrite_ids` writes here, so the roster must read here."""

    @pytest.mark.parametrize("end", _EDGE_ENDPOINT_KEYS)
    def test_every_endpoint_key(self, end: str) -> None:
        art = {"nodes": [], "edges": [{end: "person:jensen", "type": "MENTIONS"}]}
        assert "person:jensen" in person_ids_in(art), end

    def test_edge_speaker_id(self) -> None:
        art = {"nodes": [], "edges": [{"properties": {"speaker_id": "person:sam"}}]}
        assert "person:sam" in person_ids_in(art)


class TestTheHealPoolStaysNarrow:
    """The irreversible branch must not accept an id with no node."""

    def test_a_quote_only_full_name_is_NOT_a_heal_target(self) -> None:
        """Widening the roster must not widen what a bare name can become."""
        art = {
            "nodes": [{"id": "person:dario", "kind": "person"}, _quote("person:dario-amodei")],
            "edges": [],
        }
        id_map = plan_bare_name_ids(
            person_ids_in(art), EP, heal=True, candidate_ids=person_node_ids_in(art)
        )
        assert id_map["person:dario"] == f"person:unresolved-dario-{EP}", id_map

    def test_an_edge_only_full_name_is_NOT_a_heal_target(self) -> None:
        art = {
            "nodes": [{"id": "person:dario", "kind": "person"}],
            "edges": [{"source": "person:dario-amodei", "target": "episode:e1"}],
        }
        id_map = plan_bare_name_ids(
            person_ids_in(art), EP, heal=True, candidate_ids=person_node_ids_in(art)
        )
        assert id_map["person:dario"] == f"person:unresolved-dario-{EP}", id_map

    def test_a_node_backed_full_name_IS_still_a_heal_target(self) -> None:
        """The narrowing must not break real healing — `gabrielle` in production has exactly
        this shape (heal candidate `gabrielle-steinhauser`)."""
        art = {
            "nodes": [
                {"id": "person:gabrielle", "kind": "person"},
                {"id": "person:gabrielle-steinhauser", "kind": "person"},
            ],
            "edges": [],
        }
        id_map = plan_bare_name_ids(
            person_ids_in(art), EP, heal=True, candidate_ids=person_node_ids_in(art)
        )
        assert id_map["person:gabrielle"] == "person:gabrielle-steinhauser"

    def test_two_node_backed_candidates_still_refuse(self) -> None:
        art = {
            "nodes": [
                {"id": "person:trump"},
                {"id": "person:donald-trump"},
                {"id": "person:eric-trump"},
            ],
            "edges": [],
        }
        id_map = plan_bare_name_ids(
            person_ids_in(art), EP, heal=True, candidate_ids=person_node_ids_in(art)
        )
        assert id_map["person:trump"] == f"person:unresolved-trump-{EP}"

    def test_placeholders_are_still_excluded_as_candidates(self) -> None:
        """The earlier #1685 fix must survive both changes."""
        art = {
            "nodes": [
                {"id": "person:dario"},
                {"id": "person:dario-amodei"},
                {"id": f"person:unresolved-dario-{EP}"},
            ],
            "edges": [],
        }
        id_map = plan_bare_name_ids(
            person_ids_in(art), EP, heal=True, candidate_ids=person_node_ids_in(art)
        )
        assert id_map["person:dario"] == "person:dario-amodei"


class TestItIsIdempotent:
    def test_a_second_pass_changes_nothing(self) -> None:
        """The backfill WILL be re-run over already-fixed data."""
        art = {
            "nodes": [{"id": "person:brandon", "kind": "person"}, _quote("person:brandon")],
            "edges": [],
        }
        once, _ = rewrite_ids(
            art,
            plan_bare_name_ids(
                person_ids_in(art), EP, heal=False, candidate_ids=person_node_ids_in(art)
            ),
        )
        twice, changes = rewrite_ids(
            once,
            plan_bare_name_ids(
                person_ids_in(once), EP, heal=False, candidate_ids=person_node_ids_in(once)
            ),
        )
        assert changes == 0
        assert twice == once


class TestTheMergePathRewritesToo:
    """The advisor's demonstrated bug: a duplicate node took `continue` before the speaker_id
    rewrite, so the bare id folded into the survivor UNTOUCHED.

    It broke two things at once — single-pass completeness (the roster saw the id, the plan
    contained it, the artifact still held it) and idempotence (a second pass reported changes).
    Both are the exact asymmetry this change exists to remove, reintroduced through the merge
    branch, which is why it gets its own class.
    """

    def test_a_merged_duplicate_still_gets_its_speaker_id_rewritten(self) -> None:
        scoped = f"person:unresolved-brandon-{EP}"
        art = {
            "nodes": [
                {"id": "person:brandon", "kind": "person"},
                # Merges onto the same id AND carries a speaker_id that must be rewritten.
                {"id": "person:brandon", "properties": {"speaker_id": "person:brandon"}},
            ],
            "edges": [],
        }
        out, _ = rewrite_ids(
            art,
            plan_bare_name_ids(
                person_ids_in(art), EP, heal=False, candidate_ids=person_node_ids_in(art)
            ),
        )
        assert len(out["nodes"]) == 1, "must never emit two nodes sharing one id"
        survivor = out["nodes"][0]
        assert survivor["id"] == scoped
        assert survivor["properties"]["speaker_id"] == scoped, survivor

    def test_and_that_shape_is_idempotent(self) -> None:
        """A surviving bare id makes pass 2 report changes — the tell that pass 1 was incomplete."""
        art = {
            "nodes": [
                {"id": "person:brandon", "kind": "person"},
                {"id": "person:brandon", "properties": {"speaker_id": "person:brandon"}},
            ],
            "edges": [],
        }
        once, _ = rewrite_ids(
            art,
            plan_bare_name_ids(
                person_ids_in(art), EP, heal=False, candidate_ids=person_node_ids_in(art)
            ),
        )
        _, changes = rewrite_ids(
            once,
            plan_bare_name_ids(
                person_ids_in(once), EP, heal=False, candidate_ids=person_node_ids_in(once)
            ),
        )
        assert changes == 0


class TestTheCandidatePoolIsNotOptional:
    def test_omitting_it_is_a_TypeError(self) -> None:
        """No default, on purpose. A default would BE the behaviour `candidate_ids` exists to
        remove, and a caller that forgot it would be silently unprotected — which is how this
        module got three attempts at one asymmetry."""
        with pytest.raises(TypeError):
            plan_bare_name_ids({"person:dario"}, EP, heal=True)  # type: ignore[call-arg]


class TestTheSurfaceNameConsultsBothLayers:
    """A KG-only person must get their NAME, not their own scoped slug.

    `surface_name_of` falls back to the slug and is therefore never falsy, so the old
    `surface_name_of(gi) or surface_name_of(kg)` always short-circuited on the first call — the
    kg layer was never reached. A person present only in KG got
    "unresolved-brandon-substack-post-209779219" as their display name.

    That name is precisely what a future enricher (#1801) resolves FROM: the slug is lossy (case,
    punctuation, diacritics) and the surface name is not. Handing it a slug degrades the one
    artefact episode-scoping exists to preserve.
    """

    def test_a_kg_only_person_gets_their_name(self) -> None:
        from podcast_scraper.identity.bare_name_scope import unresolved_persons_in_episode

        scoped = f"person:unresolved-brandon-{EP}"
        gi: dict = {"nodes": [], "edges": []}
        kg = {
            "nodes": [{"id": scoped, "kind": "person", "properties": {"name": "Brandon"}}],
            "edges": [],
        }
        entry = next(e for e in unresolved_persons_in_episode(gi, kg) if e["id"] == scoped)
        assert entry["surface_name"] == "Brandon", entry

    def test_gi_still_wins_when_both_carry_it(self) -> None:
        """Order must be preserved — this fix widens reach, it does not change precedence."""
        from podcast_scraper.identity.bare_name_scope import unresolved_persons_in_episode

        scoped = f"person:unresolved-brandon-{EP}"
        gi = {"nodes": [{"id": scoped, "properties": {"name": "Brandon (GI)"}}], "edges": []}
        kg = {"nodes": [{"id": scoped, "properties": {"name": "Brandon (KG)"}}], "edges": []}
        entry = next(e for e in unresolved_persons_in_episode(gi, kg) if e["id"] == scoped)
        assert entry["surface_name"] == "Brandon (GI)"

    def test_neither_layer_falls_back_to_the_slug(self) -> None:
        """The fallback must survive — a nameless node still needs SOMETHING greppable."""
        from podcast_scraper.identity.bare_name_scope import unresolved_persons_in_episode

        scoped = f"person:unresolved-brandon-{EP}"
        gi = {"nodes": [{"id": scoped}], "edges": []}
        entry = next(
            e for e in unresolved_persons_in_episode(gi, {"nodes": []}) if e["id"] == scoped
        )
        assert entry["surface_name"] == f"unresolved-brandon-{EP}"
