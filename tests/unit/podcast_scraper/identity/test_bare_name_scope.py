"""A bare first name must never become a global, followable person (#1685).

Marko, 2026-08-20: "they make no sense, they're not about real people ... your option was, well,
let's not make them clickable. But then why do they exist in the first place?"

Production, 678 episodes: 208 occurrences of 172 single-word person ids — 12 resolvable within
their own episode, 0 ambiguous, 196 orphan. So this is overwhelmingly PREVENTION (stop minting a
global id nobody can disambiguate) with a small repair attached, and the tests are weighted that
way on purpose.
"""

from __future__ import annotations

import pytest

from podcast_scraper.identity.bare_name_scope import (
    is_bare_person_id,
    person_ids_in,
    plan_bare_name_ids,
    resolve_candidates,
    rewrite_ids,
    scoped_person_id,
    SCOPED_PREFIX,
)

pytestmark = [pytest.mark.unit]


def _plan(ids, ep, *, heal=True):
    """`plan_bare_name_ids` with the candidate pool equal to the roster.

    These tests hand an explicit list of ids rather than an artifact, so there is no
    node-backed/dangling distinction to draw — the roster IS the node-backed set. That
    distinction, and the reason `candidate_ids` is required rather than defaulted, is covered in
    `test_roster_matches_rewriter.py` (#1868).
    """
    return plan_bare_name_ids(ids, ep, heal=heal, candidate_ids=ids)


EP = "ep-123"


class TestWhatCountsAsBare:
    @pytest.mark.parametrize("pid", ["person:jensen", "person:sam", "person:alex"])
    def test_a_single_token_person_is_bare(self, pid: str) -> None:
        assert is_bare_person_id(pid)

    @pytest.mark.parametrize("pid", ["person:elon-musk", "person:alex-mayassi"])
    def test_a_full_name_is_not(self, pid: str) -> None:
        assert not is_bare_person_id(pid)

    def test_a_diarization_placeholder_is_left_alone(self) -> None:
        """`person:speaker-{ep}-03` is already handled by #1b — do not re-scope it."""
        assert not is_bare_person_id("person:speaker-ep-123-03")

    def test_an_org_is_untouched(self) -> None:
        """Single-token ORG names are normal (`org:apple`). This rule is persons only."""
        assert not is_bare_person_id("org:apple")

    def test_an_already_scoped_id_is_not_bare_again(self) -> None:
        """Idempotence: the migration and the pipeline may both touch the same artifact."""
        assert not is_bare_person_id(scoped_person_id("person:jensen", EP))


class TestTheResolutionRule:
    def test_one_candidate_heals(self) -> None:
        got = _plan(["person:alex", "person:alex-mayassi"], EP)
        assert got == {"person:alex": "person:alex-mayassi"}

    def test_two_candidates_refuse_and_scope(self) -> None:
        """The Donald/Eric shape. Emitting either would be arbitrary."""
        got = _plan(["person:trump", "person:donald-trump", "person:eric-trump"], EP)
        assert got["person:trump"].startswith(f"person:{SCOPED_PREFIX}trump-")

    def test_no_candidate_scopes(self) -> None:
        """Production's dominant case — 196 of 208."""
        got = _plan(["person:jensen", "person:kevin-roose"], EP)
        assert got["person:jensen"].startswith(f"person:{SCOPED_PREFIX}jensen-")

    def test_a_surname_only_reference_resolves(self) -> None:
        assert resolve_candidates("person:musk", ["person:elon-musk"]) == ["person:elon-musk"]

    def test_a_coincidental_substring_is_not_a_candidate(self) -> None:
        """Tokens, not characters — `al` must not resolve to `alex-mayassi`."""
        assert resolve_candidates("person:al", ["person:alex-mayassi"]) == []

    def test_heal_false_scopes_even_the_resolvable_one(self) -> None:
        """The strictly-safer setting: a wrong heal is unrecoverable, a wrong scope is not."""
        got = _plan(["person:alex", "person:alex-mayassi"], EP, heal=False)
        assert got["person:alex"].startswith(f"person:{SCOPED_PREFIX}alex-")

    def test_full_names_are_never_remapped(self) -> None:
        got = _plan(["person:alex", "person:alex-mayassi"], EP)
        assert "person:alex-mayassi" not in got


class TestScopedIdsAreSafeToPutInAGraphAndAFilename:
    def test_the_episode_is_part_of_the_id(self) -> None:
        a = scoped_person_id("person:jensen", "ep-a")
        b = scoped_person_id("person:jensen", "ep-b")
        assert a != b, "the same bare name in two episodes must not collapse into one node"

    def test_the_name_is_still_readable(self) -> None:
        """An opaque handle would make a graph dump unreadable to an operator."""
        assert "jensen" in scoped_person_id("person:jensen", EP)

    def test_it_is_filename_safe(self) -> None:
        """The PKM export derives file stems from ids (`person:jane` -> `person_jane`) and does
        NOT filter placeholders — verified. So the id must not carry path-hostile characters."""
        sid = scoped_person_id("person:jensen", "Weird / Ep: 01?")
        assert not set(sid) & set('/\\:?*"<>|') - {":"}
        assert sid.count(":") == 1

    def test_it_is_stable_for_the_same_inputs(self) -> None:
        assert scoped_person_id("person:jensen", EP) == scoped_person_id("person:jensen", EP)


class TestRewritingAnArtifact:
    def test_nodes_and_edges_both_move(self) -> None:
        payload = {
            "nodes": [{"id": "person:alex", "type": "Person"}],
            "edges": [{"source": "insight:1", "target": "person:alex", "type": "MENTIONS"}],
        }
        out, changes = rewrite_ids(payload, {"person:alex": "person:alex-mayassi"})
        assert out["nodes"][0]["id"] == "person:alex-mayassi"
        assert out["edges"][0]["target"] == "person:alex-mayassi"
        assert changes == 2

    def test_healing_onto_an_existing_node_MERGES_rather_than_duplicating(self) -> None:
        """The trap that makes a naive rewrite corrupt the graph.

        Healing `person:sam` in an episode that ALREADY has `person:sam-altman` would otherwise
        emit two nodes sharing one id.
        """
        payload = {
            "nodes": [
                {"id": "person:sam-altman", "type": "Person", "properties": {"name": "Sam Altman"}},
                {"id": "person:sam", "type": "Person", "properties": {"role": "guest"}},
            ],
            "edges": [],
        }
        out, _ = rewrite_ids(payload, {"person:sam": "person:sam-altman"})
        ids = [n["id"] for n in out["nodes"]]
        assert ids == ["person:sam-altman"], f"duplicate node ids: {ids}"
        props = out["nodes"][0]["properties"]
        assert props["name"] == "Sam Altman", "the survivor's own properties must win"
        assert props["role"] == "guest", "the duplicate's extra properties must be folded in"

    def test_speaker_id_properties_are_rewritten(self) -> None:
        payload = {
            "nodes": [],
            "edges": [{"source": "q:1", "target": "x", "properties": {"speaker_id": "person:sam"}}],
        }
        out, changes = rewrite_ids(payload, {"person:sam": "person:sam-altman"})
        assert out["edges"][0]["properties"]["speaker_id"] == "person:sam-altman"
        assert changes == 1

    def test_an_empty_map_changes_nothing(self) -> None:
        payload = {"nodes": [{"id": "person:x"}], "edges": []}
        out, changes = rewrite_ids(payload, {})
        assert changes == 0
        assert out["nodes"][0]["id"] == "person:x"

    def test_the_input_payload_is_not_mutated(self) -> None:
        payload = {"nodes": [{"id": "person:alex"}], "edges": []}
        rewrite_ids(payload, {"person:alex": "person:alex-mayassi"})
        assert payload["nodes"][0]["id"] == "person:alex", "rewrote the caller's artifact in place"

    def test_running_it_twice_is_a_no_op(self) -> None:
        """Idempotence end to end — the migration may re-run over an already-migrated corpus."""
        payload = {"nodes": [{"id": "person:jensen", "type": "Person"}], "edges": []}
        plan = _plan(person_ids_in(payload), EP)
        once, _ = rewrite_ids(payload, plan)
        plan2 = _plan(person_ids_in(once), EP)
        assert plan2 == {}, f"second pass wanted to remap again: {plan2}"


class TestReadingPersonIdsOutOfAnArtifact:
    def test_it_finds_person_nodes(self) -> None:
        payload = {"nodes": [{"id": "person:a"}, {"id": "topic:b"}, {"id": "person:c"}]}
        assert person_ids_in(payload) == {"person:a", "person:c"}

    def test_a_malformed_artifact_yields_nothing_rather_than_raising(self) -> None:
        assert person_ids_in({"nodes": "not a list"}) == set()
        assert person_ids_in({}) == set()


class TestScopedIdsDisappearFromEverySurface:
    """The line that actually implements the decision (#1685).

    Marko: "let's not make them clickable. But then why do they exist in the first place?" —
    scoping alone does not answer that. `is_unresolved_speaker_placeholder` is consulted in
    twelve modules INCLUDING `app_kg_view.entities_from_kg`, which is the single source for
    entity cards, discover ranking rows AND derived interests. Extending its pattern is what
    makes a scoped id stop being a followable person everywhere at once.

    Without this the change would ship scoped ids that are still followable episode-local
    people — technically different, practically the same bug.
    """

    def test_a_scoped_bare_name_is_a_placeholder(self) -> None:
        from podcast_scraper.enrichment.enrichers._loaders import (
            is_unresolved_speaker_placeholder,
        )

        assert is_unresolved_speaker_placeholder(scoped_person_id("person:jensen", "ep-1"))

    @pytest.mark.parametrize("pid", ["person:speaker-ep-1-03", "SPEAKER_03", "person:speaker-07"])
    def test_the_existing_diarization_shapes_still_match(self, pid: str) -> None:
        """#1b must keep working — this widened the pattern, it did not replace it."""
        from podcast_scraper.enrichment.enrichers._loaders import (
            is_unresolved_speaker_placeholder,
        )

        assert is_unresolved_speaker_placeholder(pid)

    @pytest.mark.parametrize(
        "pid", ["person:jensen-huang", "person:elon-musk", "person:alex-mayassi"]
    )
    def test_real_people_are_never_dropped(self, pid: str) -> None:
        """The catastrophic failure mode: a pattern so wide it hides real entities."""
        from podcast_scraper.enrichment.enrichers._loaders import (
            is_unresolved_speaker_placeholder,
        )

        assert not is_unresolved_speaker_placeholder(pid)

    def test_an_unscoped_bare_name_still_surfaces(self) -> None:
        """Deliberate: existing corpora keep their `person:jensen` until a backfill runs.

        The filter must not silently hide them, because that would make the corpus LOOK repaired
        while the pooled nodes are still there — hiding the evidence instead of fixing it.
        """
        from podcast_scraper.enrichment.enrichers._loaders import (
            is_unresolved_speaker_placeholder,
        )

        assert not is_unresolved_speaker_placeholder("person:jensen")


class TestThePipelineRunsTheScopingPassBeforeTypedMentions:
    """Ordering is the wiring's load-bearing property (#1685).

    Typed mentions attach insight spans to entity ids. If the scoping pass ran AFTER them, the
    mentions would be bound to ids that are then rewritten underneath, and the edges would point
    at nodes that no longer exist. Running scoping first means mentions attach to final ids.

    This is a SOURCE-ORDER assertion, not a behavioural one, and that is a deliberate trade: a
    behavioural test would need the whole metadata-generation harness (providers, artifacts,
    manifests) to prove one line's position. The behaviour of the pass itself is covered above;
    what this pins is that it is invoked, and invoked in the right place. Stated plainly so
    nobody mistakes it for end-to-end coverage.
    """

    @staticmethod
    def _source() -> str:
        from pathlib import Path

        import podcast_scraper.workflow.metadata_generation as mg

        return Path(mg.__file__).read_text(encoding="utf-8")

    def test_the_pass_is_actually_called(self) -> None:
        src = self._source()
        assert "plan_bare_name_ids" in src, "the pipeline never invokes the scoping pass"
        assert "rewrite_ids" in src

    def test_it_runs_before_typed_mentions(self) -> None:
        src = self._source()
        scoping = src.index("plan_bare_name_ids")
        mentions = src.index("apply_typed_mentions_and_rewrite_gi")
        assert scoping < mentions, (
            "scoping must run BEFORE typed mentions, or mentions bind to ids that are "
            "rewritten underneath them"
        )

    def test_both_layers_feed_the_roster(self) -> None:
        """The measurement proved the KG alone gives the wrong answer — `person:alex` was
        reported orphan while `person:alex-mayassi` sat in the episode's GI layer."""
        src = self._source()
        assert "person_ids_in(bridge_gi_payload) | person_ids_in(bridge_kg_payload)" in src

    def test_the_heal_branch_is_configurable(self) -> None:
        """Marko's standing preference: these decisions live in config so a sweep can vary them."""
        from podcast_scraper import config

        assert "bare_name_heal" in config.Config.model_fields
        assert config.Config.model_fields["bare_name_heal"].default is True


class TestTheWorkListAFutureEnricherWouldStartFrom:
    """Scoping stops the harm; it does not say WHO the person was (#1685).

    Marko, 2026-08-20: "I might develop an enricher who can, based on the context of the episode,
    really unfold Alex into the real Alex ... short term we wanna put a good foundation that when
    we create such an enricher, we know what we have to go and process."

    This is that foundation: a queryable list rather than id-string parsing. Everything in it is
    DERIVED from the artifacts, because the schemas pin `additionalProperties: False` on person
    nodes (allowing only aliases/description/label/name/role), so storing a resolution_status
    would mean extending two schemas and bumping both schema_versions — a contract change that
    should wait until it has a reader.
    """

    @staticmethod
    def _scoped(bare, others, ep="ep-1"):
        from podcast_scraper.identity.bare_name_scope import rewrite_ids

        payload = {
            "nodes": [{"id": bare, "type": "Person", "properties": {"name": "Alex"}}]
            + [{"id": o, "type": "Person", "properties": {"name": o}} for o in others],
            "edges": [],
        }
        plan = _plan([bare] + list(others), ep, heal=False)
        out, _ = rewrite_ids(payload, plan)
        return out

    def test_a_scoped_person_appears_on_the_list(self) -> None:
        from podcast_scraper.identity.bare_name_scope import unresolved_persons_in_episode

        gi = self._scoped("person:jensen", [])
        rows = unresolved_persons_in_episode(gi, {"nodes": []})
        assert len(rows) == 1
        assert rows[0]["id"].startswith("person:unresolved-jensen-")

    def test_a_resolved_person_does_not(self) -> None:
        from podcast_scraper.identity.bare_name_scope import unresolved_persons_in_episode

        payload = {"nodes": [{"id": "person:elon-musk", "properties": {"name": "Elon Musk"}}]}
        assert unresolved_persons_in_episode(payload, {"nodes": []}) == []

    def test_the_surface_name_survives_scoping(self) -> None:
        """The slug is lossy (case, punctuation); the spoken name is what a resolver works from."""
        from podcast_scraper.identity.bare_name_scope import unresolved_persons_in_episode

        gi = self._scoped("person:alex", [])
        assert unresolved_persons_in_episode(gi, {"nodes": []})[0]["surface_name"] == "Alex"

    def test_an_ambiguous_case_carries_its_candidates(self) -> None:
        """The easier of the two jobs: pick between named options, do not search the world."""
        from podcast_scraper.identity.bare_name_scope import unresolved_persons_in_episode

        gi = self._scoped("person:trump", ["person:donald-trump", "person:eric-trump"])
        row = unresolved_persons_in_episode(gi, {"nodes": []})[0]
        assert row["reason"] == "ambiguous"
        assert row["candidates"] == ["person:donald-trump", "person:eric-trump"]

    def test_an_orphan_says_so_and_offers_nothing(self) -> None:
        """The harder job: the answer is not in the graph, so the enricher must look outward."""
        from podcast_scraper.identity.bare_name_scope import unresolved_persons_in_episode

        gi = self._scoped("person:jensen", [])
        row = unresolved_persons_in_episode(gi, {"nodes": []})[0]
        assert row["reason"] == "no_candidate"
        assert row["candidates"] == []

    def test_both_layers_contribute_candidates(self) -> None:
        """Same union as everywhere else — the KG alone gave the wrong answer in production."""
        from podcast_scraper.identity.bare_name_scope import unresolved_persons_in_episode

        gi = self._scoped("person:alex", [])
        kg = {"nodes": [{"id": "person:alex-karp"}, {"id": "person:alex-rampell"}]}
        row = unresolved_persons_in_episode(gi, kg)[0]
        assert row["reason"] == "ambiguous"
        assert row["candidates"] == ["person:alex-karp", "person:alex-rampell"]


class TestTheWorkListLabelsAreHonest:
    """`ambiguous` must mean what its docstring says: MORE THAN ONE candidate (#1685).

    The first version labelled every scoped id with candidates as `ambiguous`, including the
    single-candidate case — which is reachable whenever `heal` is off, or when the migration and
    the pipeline saw different rosters. That would send the enricher (#1801) hunting for a choice
    that does not exist: the same class of error as the self-matching-candidate bug this function
    already had once.
    """

    def test_exactly_one_candidate_is_resolvable_not_ambiguous(self) -> None:
        from podcast_scraper.identity.bare_name_scope import (
            rewrite_ids,
            unresolved_persons_in_episode,
        )

        payload = {
            "nodes": [
                {"id": "person:alex", "type": "Person", "properties": {"name": "Alex"}},
                {"id": "person:alex-mayassi", "type": "Person", "properties": {"name": "Alex M"}},
            ],
            "edges": [],
        }
        # heal=False is what produces a scoped id that still HAS one candidate.
        plan = _plan(["person:alex", "person:alex-mayassi"], "ep-1", heal=False)
        scoped, _ = rewrite_ids(payload, plan)
        row = unresolved_persons_in_episode(scoped, {"nodes": []})[0]
        assert row["reason"] == "resolvable"
        assert row["candidates"] == ["person:alex-mayassi"]

    def test_two_candidates_are_still_ambiguous(self) -> None:
        from podcast_scraper.identity.bare_name_scope import (
            rewrite_ids,
            unresolved_persons_in_episode,
        )

        ids = ["person:trump", "person:donald-trump", "person:eric-trump"]
        payload = {"nodes": [{"id": i, "type": "Person"} for i in ids], "edges": []}
        scoped, _ = rewrite_ids(payload, _plan(ids, "ep-1"))
        row = unresolved_persons_in_episode(scoped, {"nodes": []})[0]
        assert row["reason"] == "ambiguous"
        assert len(row["candidates"]) == 2
