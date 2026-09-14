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

    def test_the_survivor_keeps_its_name_when_the_title_agrees_with_it(self) -> None:
        # Dwarkesh, "Andrej Karpathy — AGI is still a decade away". The speaker side is RIGHT
        # here; the earlier provenance rule would have renamed this to 'Andrei Karpathy'.
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
        assert plan_display_names(gi, {}, plan) == {}, "the feed already agrees with the survivor"

    def test_a_real_rename_from_the_sample(self) -> None:
        gi = self._with_episode(
            "Analyse Asia with Bernard Leong",
            [("person:bernard-leung", "Bernard Leung"), ("person:bernard-leong", "Bernard Leong")],
            [*_spoken("person:bernard-leung", 20), *_mentioned("person:bernard-leong")],
        )
        plan = plan_intra_episode_merges(gi, {})
        assert plan_display_names(gi, {}, plan) == {"person:bernard-leung": "Bernard Leong"}

    def test_neither_name_in_the_prose_is_not_evidence(self) -> None:
        gi = self._with_episode(
            "An episode about something else entirely",
            [("person:cory-combs", "Cory Combs"), ("person:corey-combs", "Corey Combs")],
            [*_spoken("person:cory-combs"), *_mentioned("person:corey-combs")],
        )
        plan = plan_intra_episode_merges(gi, {})
        assert plan_display_names(gi, {}, plan) == {}, "a rename on no evidence promotes a guess"

    def test_both_names_in_the_prose_is_not_evidence_either(self) -> None:
        gi = self._with_episode(
            "Featuring Bernt Bornich, sometimes written Bernt Børnich",
            [("person:bernt-brnich", "Bernt Børnich"), ("person:bernt-bornich", "Bernt Bornich")],
            [*_spoken("person:bernt-brnich"), *_mentioned("person:bernt-bornich")],
        )
        plan = plan_intra_episode_merges(gi, {})
        assert plan_display_names(gi, {}, plan) == {}

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
