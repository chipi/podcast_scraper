"""One human must not get two ids inside ONE episode (#2056).

MEASURED ON PRODUCTION, episode ``substack:post:189936942`` ("Every Agent Needs a Box — Aaron
Levie, Box", Latent Space) — the episode #2056 was reported against::

    person:aaron-levy    "Aaron Levy"    <- 65 x SPOKEN_BY        (diarization roster)
    person:aaron-levie   "Aaron Levie"   <-  2 x MENTIONS_PERSON  (entity extraction)

Two ids, one human, one episode, two minting paths. The roster heard the spoken name and ASR wrote
"Levy"; the entity extractor read "Levie" off the text. Neither is wrong about what it saw.

WHY THE EXISTING RESOLVER NEVER FIRES ON THIS. ``kg.entity_clusters._are_xep_variants`` is the
CROSS-EPISODE test — ``xep``. It runs over ``collect_entity_candidates``, which aggregates the
corpus, and is gated by ``same_show_required``. It is never asked about two ids *inside* one
episode, which is why #2056 could report "the resolver already matches every reported pair" and
still see duplicates in the product. The matcher was never the problem; nothing invoked it here.

WHAT THIS PASS DELIBERATELY WILL NOT DO — the asymmetry that drives every rule below: a false
SPLIT is cosmetic (two nodes, one person, clutter). A false MERGE is corrupt data — one person's
statements attributed to another, propagating into quotes, roles and search, with no cheap undo.
So every rule here fails toward leaving two nodes.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from podcast_scraper.identity.intra_episode_merge import (
    plan_display_names,
    plan_intra_episode_merges,
)

pytestmark = pytest.mark.unit

EP = "substack:post:189936942"


def _gi(persons, edges=()) -> Dict[str, Any]:
    """A GI payload: ``persons`` is ``[(id, name)]``, ``edges`` is ``[(type, from, to)]``."""
    return {
        "episode_id": EP,
        "nodes": [
            {"id": pid, "type": "Person", "properties": {"name": name}} for pid, name in persons
        ],
        "edges": [{"type": t, "from": f, "to": to} for t, f, to in edges],
    }


def _spoken(pid: str, n: int = 1):
    return [("SPOKEN_BY", f"quote:q{i}{pid}", pid) for i in range(n)]


def _mentioned(pid: str, n: int = 1):
    return [("MENTIONS_PERSON", f"insight:i{i}{pid}", pid) for i in range(n)]


class TestTheProductionCase:
    """The exact shape measured on substack:post:189936942."""

    def test_the_speaker_and_the_mentioned_entity_are_one_person(self) -> None:
        gi = _gi(
            [("person:aaron-levy", "Aaron Levy"), ("person:aaron-levie", "Aaron Levie")],
            [*_spoken("person:aaron-levy", 65), *_mentioned("person:aaron-levie", 2)],
        )
        plan = plan_intra_episode_merges(gi, {})
        assert plan == {"person:aaron-levie": "person:aaron-levy"}

    def test_the_roster_wins_regardless_of_edge_counts(self) -> None:
        # The roster HEARD this person speak. That is direct evidence of presence; a mention is
        # someone talking ABOUT them. Volume must not overturn that — invert the counts and the
        # winner is unchanged.
        gi = _gi(
            [("person:aaron-levy", "Aaron Levy"), ("person:aaron-levie", "Aaron Levie")],
            [*_spoken("person:aaron-levy", 1), *_mentioned("person:aaron-levie", 99)],
        )
        assert plan_intra_episode_merges(gi, {}) == {"person:aaron-levie": "person:aaron-levy"}

    def test_another_production_case(self) -> None:
        # Found by scanning 287 production artifacts with the repo's own matcher.
        gi = _gi(
            [
                ("person:bernard-leung", "Bernard Leung"),
                ("person:bernard-leong", "Bernard Leong"),
            ],
            [*_spoken("person:bernard-leung", 30), *_mentioned("person:bernard-leong")],
        )
        assert plan_intra_episode_merges(gi, {}) == {"person:bernard-leong": "person:bernard-leung"}

    def test_a_name_that_drifted_in_BOTH_tokens_is_not_merged(self) -> None:
        # THE COST OF THE PRECISION RULE, recorded rather than hidden.
        # 'Arvind Surivas'/'Aravind Srinivas' is one real human on one real production episode,
        # and this pass used to merge them. `_are_xep_variants` now refuses because BOTH tokens
        # drifted — the same rule that stops 'Jensen Huang' becoming 'Jesse Zhang'. Three of the
        # sample's 24 intra-episode merges are lost this way.
        #
        # Accepted deliberately, and with ONE matcher rather than two: this pass could afford a
        # looser name test because it ALSO has speaker evidence, but a second opinion on "is this
        # the same person" is exactly how two layers drift into disagreeing. One answer, one place.
        gi = _gi(
            [
                ("person:arvind-surivas", "Arvind Surivas"),
                ("person:aravind-srinivas", "Aravind Srinivas"),
            ],
            [*_spoken("person:arvind-surivas"), *_mentioned("person:aravind-srinivas")],
        )
        assert plan_intra_episode_merges(gi, {}) == {}


class TestTwoVoicesAreNeverOnePerson:
    """The catastrophic merge, refused explicitly.

    If BOTH ids carry ``SPOKEN_BY`` the roster heard two DISTINCT voices and named them. Merging
    them collapses two speakers into one and reassigns real quotes to the wrong human — the worst
    outcome this pass can produce, and strictly worse than the duplicate it is trying to remove.
    """

    def test_two_speakers_with_similar_names_are_left_alone(self) -> None:
        gi = _gi(
            [("person:jon-smith", "Jon Smith"), ("person:john-smith", "John Smith")],
            [*_spoken("person:jon-smith", 12), *_spoken("person:john-smith", 9)],
        )
        assert plan_intra_episode_merges(gi, {}) == {}

    def test_a_co_host_pair_is_not_collapsed(self) -> None:
        gi = _gi(
            [("person:sarah-guo", "Sarah Guo"), ("person:sarah-kuo", "Sarah Kuo")],
            [*_spoken("person:sarah-guo", 30), *_spoken("person:sarah-kuo", 25)],
        )
        assert plan_intra_episode_merges(gi, {}) == {}


class TestDistinctPeopleAreNotMerged:
    """Names that merely look alike. The matcher's precision is measured and imperfect."""

    @pytest.mark.parametrize(
        "a_name,b_name",
        [
            ("Albert Einstein", "Robert Jensen"),
            ("Alex Bregman", "Lex Friedman"),
            ("Charles I", "Charles II"),
            ("Kaiser Guo", "Kaiser Wilhelm II"),
            ("Albert Einstein", "Bert Vogelstein"),
        ],
    )
    def test_two_different_humans_stay_two_nodes(self, a_name: str, b_name: str) -> None:
        # Every pair here was produced by the CROSS-EPISODE matcher over 287 production
        # artifacts; several of them it accepts. Inside one episode we have better evidence
        # available (who spoke) and must not inherit that imprecision.
        gi = _gi(
            [("person:a-id", a_name), ("person:b-id", b_name)],
            [*_mentioned("person:a-id"), *_mentioned("person:b-id")],
        )
        plan = plan_intra_episode_merges(gi, {})
        assert plan == {}, f"{a_name!r} and {b_name!r} are different people"


class TestEpisodeScopedPlaceholdersAreLeftToTheirOwnPass:
    """``person:unresolved-<name>-<episode>`` belongs to ``identity.bare_name_scope`` (#2062).

    That pass has its own healing rule with its own evidence (``resolve_candidates``). Two passes
    rewriting the same ids on different rules is how layers drift into disagreeing about who a
    person is — the defect m0007's docstring calls worse than not migrating at all.
    """

    def test_a_scoped_placeholder_is_not_merged_here(self) -> None:
        gi = _gi(
            [
                (f"person:unresolved-ben-{EP}", "Ben"),
                ("person:ben-thompson", "Ben Thompson"),
            ],
            [*_spoken(f"person:unresolved-ben-{EP}"), *_mentioned("person:ben-thompson")],
        )
        assert plan_intra_episode_merges(gi, {}) == {}


class TestBothLayersAreConsidered:
    """A duplicate can straddle the GI and KG payloads, which are written together."""

    def test_a_person_in_kg_merges_with_the_speaker_in_gi(self) -> None:
        gi = _gi([("person:aaron-levy", "Aaron Levy")], _spoken("person:aaron-levy", 5))
        kg = {
            "episode_id": EP,
            "nodes": [
                {
                    "id": "person:aaron-levie",
                    "type": "Person",
                    "properties": {"name": "Aaron Levie"},
                }
            ],
            "edges": [],
        }
        assert plan_intra_episode_merges(gi, kg) == {"person:aaron-levie": "person:aaron-levy"}


class TestDeterminism:
    """The same episode must plan the same merge every run — ids are identity, not a ranking."""

    def test_node_order_does_not_change_the_winner(self) -> None:
        # The speaker must win from either side of the comparison, so the id that survives an
        # episode cannot depend on the order the extractor happened to emit its nodes in.
        speaker = ("person:zz-spoke", "Dario Amodei")
        mention = ("person:aa-mentioned", "Dario Amadei")
        edges = [*_spoken("person:zz-spoke", 4), *_mentioned("person:aa-mentioned", 2)]
        forward = plan_intra_episode_merges(_gi([speaker, mention], edges), {})
        reverse = plan_intra_episode_merges(_gi([mention, speaker], edges), {})
        assert forward == reverse == {"person:aa-mentioned": "person:zz-spoke"}

    def test_two_mentioned_ids_are_not_merged_at_all(self) -> None:
        # No episode-local evidence — see `plan_intra_episode_merges`. A deliberate false split:
        # visible and recoverable, where a wrong merge would not be.
        gi = _gi(
            [("person:dario-amodei", "Dario Amodei"), ("person:dario-amadei", "Dario Amadei")],
            [*_mentioned("person:dario-amodei"), *_mentioned("person:dario-amadei")],
        )
        assert plan_intra_episode_merges(gi, {}) == {}

    def test_the_plan_never_chains(self) -> None:
        # A -> B and B -> C in one plan would leave a dangling id after one rewrite pass.
        gi = _gi(
            [
                ("person:jon-smith-a", "Jon Smith"),
                ("person:jon-smyth-b", "Jon Smyth"),
                ("person:jon-smithe-c", "Jon Smithe"),
            ],
            [
                *_spoken("person:jon-smith-a", 3),
                *_mentioned("person:jon-smyth-b"),
                *_mentioned("person:jon-smithe-c"),
            ],
        )
        plan = plan_intra_episode_merges(gi, {})
        assert not (set(plan) & set(plan.values())), f"plan chains: {plan}"


class TestNothingToDo:
    def test_no_duplicates_is_an_empty_plan(self) -> None:
        gi = _gi(
            [("person:aaron-levie", "Aaron Levie"), ("person:brian-chesky", "Brian Chesky")],
            [*_spoken("person:aaron-levie"), *_mentioned("person:brian-chesky")],
        )
        assert plan_intra_episode_merges(gi, {}) == {}

    def test_empty_payloads_are_safe(self) -> None:
        assert plan_intra_episode_merges({}, {}) == {}


class TestTheFeedDecidesTheSpelling:
    """Merging the right ids can still display the wrong NAME (#2056 candidate 4).

    ``rewrite_ids`` keeps the first node's properties, so the survivor keeps the WINNER's name.
    The winner is the speaker — and my first attempt at this assumed that side was systematically
    worse, reasoning that its name arrived through ASR, generalised from two examples.

    MEASURING IT OVER 287 PRODUCTION ARTIFACTS SHOWED THAT IS WRONG more often than right. The
    speaker side holds the CORRECT spelling here::

        Andrej Karpathy   (vs 'Andrei Karpathy')      Stewart Brand  (vs 'Stuart Brand')
        Teresa Bejan      (vs 'Theresa Bejan')        Steve Brusatte (vs 'Steve Broussatti')

    and the WRONG one here::

        'Bernard Leung'   -> Bernard Leong            'Arvind Surivas' -> Aravind Srinivas

    Provenance yields no rule. The episode's own title and description do: they are human-written,
    not transcribed and not generated, which makes them authoritative for spelling in a way
    neither ASR nor an LLM is. Whichever spelling appears there wins; if both or neither appear,
    nothing is renamed.
    """

    @staticmethod
    def _with_episode(title: str, persons, edges):
        payload = _gi(persons, edges)
        payload["nodes"].append(
            {"id": f"episode:{EP}", "type": "Episode", "properties": {"title": title}}
        )
        return payload

    def test_the_title_overrules_the_survivor(self) -> None:
        # Latent Space, "Every Agent Needs a Box — Aaron Levie, Box": the feed says Levie.
        gi = self._with_episode(
            "Every Agent Needs a Box — Aaron Levie, Box",
            [("person:aaron-levy", "Aaron Levy"), ("person:aaron-levie", "Aaron Levie")],
            [*_spoken("person:aaron-levy", 65), *_mentioned("person:aaron-levie", 2)],
        )
        plan = plan_intra_episode_merges(gi, {})
        assert plan_display_names(gi, {}, plan) == {"person:aaron-levy": "Aaron Levie"}

    def test_the_title_confirming_the_survivor_is_still_emitted(self) -> None:
        # Dwarkesh, "Andrej Karpathy — AGI is still a decade away". The speaker side is RIGHT
        # here; the earlier provenance rule would have renamed this to 'Andrei Karpathy'.
        #
        # The name is emitted even though it does not CHANGE the survivor's own properties. That
        # is the point: `rewrite_ids` keeps whichever node each payload lists first, and GI and KG
        # order theirs differently, so "emit nothing" means the two artifacts disagree. Emitting
        # the decided name is what makes the merge single-valued across layers.
        gi = self._with_episode(
            "Andrej Karpathy — AGI is still a decade away",
            [
                ("person:andrej-karpathy", "Andrej Karpathy"),
                ("person:andrei-karpathy", "Andrei Karpathy"),
            ],
            [*_spoken("person:andrej-karpathy", 40), *_mentioned("person:andrei-karpathy")],
        )
        plan = plan_intra_episode_merges(gi, {})
        assert plan == {"person:andrei-karpathy": "person:andrej-karpathy"}
        got = plan_display_names(gi, {}, plan)
        assert got == {"person:andrej-karpathy": "Andrej Karpathy"}

    def test_a_real_rename_from_the_sample(self) -> None:
        gi = self._with_episode(
            "Analyse Asia with Bernard Leong",
            [("person:bernard-leung", "Bernard Leung"), ("person:bernard-leong", "Bernard Leong")],
            [*_spoken("person:bernard-leung", 20), *_mentioned("person:bernard-leong")],
        )
        plan = plan_intra_episode_merges(gi, {})
        assert plan_display_names(gi, {}, plan) == {"person:bernard-leung": "Bernard Leong"}

    def test_neither_name_in_the_prose_falls_back_to_the_side_that_spoke(self) -> None:
        """No prose evidence still has to produce ONE answer, or node order decides it.

        The tie-break is not a claim that the speaker side is better — it is measurably not. It
        is a STATED, order-independent choice, which is the property that matters: the previous
        "emit nothing" left kg.json and gi.json showing different names for the same id on 8 of
        11 real merges.
        """
        gi = self._with_episode(
            "An episode about something else entirely",
            [("person:cory-combs", "Cory Combs"), ("person:corey-combs", "Corey Combs")],
            [*_spoken("person:cory-combs"), *_mentioned("person:corey-combs")],
        )
        plan = plan_intra_episode_merges(gi, {})
        assert plan_display_names(gi, {}, plan) == {"person:cory-combs": "Cory Combs"}

    def test_both_names_in_the_prose_falls_back_the_same_way(self) -> None:
        gi = self._with_episode(
            "Featuring Bernt Bornich, sometimes written Bernt Børnich",
            [("person:bernt-brnich", "Bernt Børnich"), ("person:bernt-bornich", "Bernt Bornich")],
            [*_spoken("person:bernt-brnich"), *_mentioned("person:bernt-bornich")],
        )
        plan = plan_intra_episode_merges(gi, {})
        assert plan_display_names(gi, {}, plan) == {"person:bernt-brnich": "Bernt Børnich"}

    def test_extra_episode_text_is_consulted(self) -> None:
        # The description lives on the metadata sibling, not the graph, so the caller can pass it.
        gi = _gi(
            [
                ("person:bernard-leung", "Bernard Leung"),
                ("person:bernard-leong", "Bernard Leong"),
            ],
            [*_spoken("person:bernard-leung"), *_mentioned("person:bernard-leong")],
        )
        plan = plan_intra_episode_merges(gi, {})
        got = plan_display_names(
            gi, {}, plan, episode_text="Analyse Asia, hosted by Bernard Leong."
        )
        assert got == {"person:bernard-leung": "Bernard Leong"}

    def test_no_merge_means_no_rename(self) -> None:
        gi = self._with_episode(
            "Jon Smith and John Smith",
            [("person:jon-smith", "Jon Smith"), ("person:john-smith", "John Smith")],
            [*_spoken("person:jon-smith"), *_spoken("person:john-smith")],
        )
        assert plan_display_names(gi, {}, plan_intra_episode_merges(gi, {})) == {}


class TestTheSeamThatActuallyShipsTheName:
    """`plan -> rewrite_ids -> apply_display_names` — the only place the defect was visible.

    Every test above this class asserts on `plan_display_names`' RETURN VALUE. That is what let a
    real defect through: the function returned `{}` and the assertion passed, while the name the
    reader finally sees is decided three calls later by `rewrite_ids`, which keeps whichever node
    comes FIRST in that payload's node list.

    Production orders them badly. `kg/pipeline` appends the roster's speakers AFTER the extracted
    entities, so in kg.json the merged-away node is usually first — measured, 8 of 11 real merges
    — and the survivor inherited the LOSER's spelling while gi.json kept the winner's. One id,
    two names, and `app_kg_index` reads the kg side, so the card showed the wrong one.

    So these tests run the whole seam, in BOTH node orders, and assert the two artifacts agree.
    """

    @staticmethod
    def _kg(order, title="Stewart Brand on the Long Now"):
        """kg.json with the two person nodes in the given order, plus the Episode prose node."""
        by_key = {
            "mentioned": {
                "id": "person:stuart-brand",
                "type": "Person",
                "properties": {"name": "Stuart Brand", "role": "mentioned"},
            },
            "speaker": {
                "id": "person:stewart-brand",
                "type": "Person",
                "properties": {"name": "Stewart Brand", "role": "guest"},
            },
        }
        return {
            "episode_id": EP,
            "nodes": [
                *(by_key[k] for k in order),
                {"id": f"episode:{EP}", "type": "Episode", "properties": {"title": title}},
            ],
            "edges": [],
        }

    @staticmethod
    def _gi_side():
        """gi.json holds the speaker node and the SPOKEN_BY edge — the production split."""
        return _gi(
            [("person:stewart-brand", "Stewart Brand")],
            [*_spoken("person:stewart-brand", 12)],
        )

    @staticmethod
    def _names(payload):
        return {
            n["id"]: (n.get("properties") or {}).get("name")
            for n in payload.get("nodes", [])
            if n.get("type") == "Person"
        }

    @pytest.mark.parametrize(
        "order",
        [["mentioned", "speaker"], ["speaker", "mentioned"]],
        ids=["loser-first", "winner-first"],
    )
    def test_the_displayed_name_does_not_depend_on_node_order(self, order) -> None:
        from podcast_scraper.identity.bare_name_scope import rewrite_ids
        from podcast_scraper.identity.intra_episode_merge import apply_display_names

        gi, kg = self._gi_side(), self._kg(order)
        plan = plan_intra_episode_merges(gi, kg)
        assert plan == {"person:stuart-brand": "person:stewart-brand"}

        renames = plan_display_names(gi, kg, plan)
        kg_out = apply_display_names(rewrite_ids(kg, plan)[0], renames)
        gi_out = apply_display_names(rewrite_ids(gi, plan)[0], renames)

        # The title says Stewart. Both artifacts must say Stewart, whichever node came first.
        assert self._names(kg_out) == {"person:stewart-brand": "Stewart Brand"}
        assert self._names(gi_out) == {"person:stewart-brand": "Stewart Brand"}

    def test_kg_and_gi_agree_even_when_the_prose_decides_nothing(self) -> None:
        """The tie-break case, through the seam. No prose evidence is where order used to win."""
        from podcast_scraper.identity.bare_name_scope import rewrite_ids
        from podcast_scraper.identity.intra_episode_merge import apply_display_names

        gi = self._gi_side()
        kg = self._kg(["mentioned", "speaker"], title="An episode about something else")
        plan = plan_intra_episode_merges(gi, kg)
        renames = plan_display_names(gi, kg, plan)
        kg_out = apply_display_names(rewrite_ids(kg, plan)[0], renames)
        gi_out = apply_display_names(rewrite_ids(gi, plan)[0], renames)
        assert self._names(kg_out) == self._names(gi_out)

    def test_the_merge_still_keeps_the_speaking_role(self) -> None:
        """Order-independence must not be bought by dropping the role (#2065's live defect)."""
        from podcast_scraper.identity.bare_name_scope import rewrite_ids
        from podcast_scraper.identity.intra_episode_merge import apply_display_names

        gi, kg = self._gi_side(), self._kg(["mentioned", "speaker"])
        plan = plan_intra_episode_merges(gi, kg)
        merged = apply_display_names(rewrite_ids(kg, plan)[0], plan_display_names(gi, kg, plan))
        roles = {
            n["id"]: (n.get("properties") or {}).get("role")
            for n in merged["nodes"]
            if n.get("type") == "Person"
        }
        assert roles == {"person:stewart-brand": "guest"}
