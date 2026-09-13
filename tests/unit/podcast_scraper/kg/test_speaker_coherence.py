"""Every speaker-coherence guard, proved to FIRE on the defect it exists for (#2062).

A guard that has never failed is indistinguishable from a guard that cannot fail. Each rule here is
tested from both sides:

  * POSITIVE — a coherent episode produces no violations, so the rule is not simply always-on;
  * NEGATIVE — an episode carrying the REAL defect produces one, so the rule is not vacuous.

The negative cases are not invented. Each is the shape of something that actually shipped, taken
from the production measurements in #2062 or from a fresh DGX ingest, and named as such.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from podcast_scraper.kg.speaker_coherence import (
    check_episode,
    check_no_anonymous_speakers,
    check_not_collapsed_onto_one_speaker,
    check_one_quote_one_speaker,
    check_roles_are_known,
    check_roster_speakers_reach_the_graph,
    check_speakers_actually_spoke,
    check_spoken_by_targets_exist,
    same_person,
)

pytestmark = pytest.mark.unit

HOST = "Ryan Knutson"
GUEST = "Sharon Turlip"


def meta(*speakers) -> Dict[str, Any]:
    return {"content": {"speakers": [{"name": n, "role": r} for n, r in speakers]}}


def kg(*persons) -> Dict[str, Any]:
    return {
        "nodes": [
            {
                "id": f"person:{n.lower().replace(' ', '-')}",
                "type": "Person",
                "properties": {"name": n, "role": r},
            }
            for n, r in persons
        ]
    }


def gi(edges: List[tuple], persons: List[str] | None = None) -> Dict[str, Any]:
    people = persons if persons is not None else sorted({t for _q, t in edges})
    return {
        "nodes": [{"id": p, "type": "Person", "properties": {"name": p}} for p in people],
        "edges": [{"type": "SPOKEN_BY", "from": q, "to": t} for q, t in edges],
    }


GOOD_META = meta((HOST, "host"), (GUEST, "guest"))
GOOD_KG = kg((HOST, "host"), (GUEST, "guest"), ("Elon Musk", "mentioned"))
GOOD_GI = gi([("quote:1", "person:ryan-knutson"), ("quote:2", "person:sharon-turlip")])


class TestNameMatchingToleratesWhatItMust:
    """The corpus is full of benign spelling differences; a guard that flags those is noise."""

    @pytest.mark.parametrize(
        "a,b",
        [
            ("Hanna Crebo-Rediker", "hanna krebohticker"),  # ASR variant, in the fixture corpus
            ("Noah Brier", "noah bryer"),
            ("Sophie Laurent", "sophie lorenz"),
            ("Dr. Adam Rodman", "Adam Rodman"),  # honorific
            ("Bernt Børnich", "Bernt Bornich"),  # diacritic
            ("Twiggy", "Twiggy Lawson"),  # mononym vs full name
            ("patrick  o'shaughnessy", "Patrick O'Shaughnessy"),  # case + spacing + apostrophe
        ],
    )
    def test_these_are_one_person(self, a: str, b: str) -> None:
        assert same_person(a, b)

    @pytest.mark.parametrize(
        "a,b",
        [
            ("Sarah Guo", "Elad Gil"),  # the real phantom-host case
            ("Ryan Knutson", "Jessica Mendoza"),
            ("Twiggy", "Lane Florsheim"),
            ("", "Ryan Knutson"),
        ],
    )
    def test_these_are_different_people(self, a: str, b: str) -> None:
        assert not same_person(a, b)


class TestASpeakerMustHaveSpoken:
    def test_a_coherent_episode_is_clean(self) -> None:
        assert check_speakers_actually_spoke(GOOD_META, GOOD_KG) == []

    def test_a_cohost_who_sat_the_episode_out_is_caught(self) -> None:
        # REAL: "Sarah Guo" on an episode where Elad Gil interviews Glenn Fogel.
        bad = kg(("Elad Gil", "host"), ("Glenn Fogel", "guest"), ("Sarah Guo", "host"))
        v = check_speakers_actually_spoke(meta(("Elad Gil", "host"), ("Glenn Fogel", "guest")), bad)
        assert len(v) == 1 and "Sarah Guo" in v[0]

    def test_the_show_itself_as_a_person_is_caught(self) -> None:
        # REAL: "The China-Global South Project" shipped as a host Person node.
        bad = kg(("Eric Olander", "host"), ("The China-Global South Project", "host"))
        v = check_speakers_actually_spoke(meta(("Eric Olander", "host")), bad)
        assert len(v) == 1 and "China-Global South" in v[0]

    def test_a_mentioned_person_is_not_required_to_have_spoken(self) -> None:
        # The trap: Elon Musk is discussed, not present. `mentioned` is exactly right for him.
        assert check_speakers_actually_spoke(GOOD_META, GOOD_KG) == []

    def test_no_roster_reports_nothing(self) -> None:
        # Absence of evidence is not a violation — an un-diarized episode must not be flagged.
        assert check_speakers_actually_spoke({}, GOOD_KG) == []


class TestARosterSpeakerMustReachTheGraph:
    def test_a_coherent_episode_is_clean(self) -> None:
        assert check_roster_speakers_reach_the_graph(GOOD_META, GOOD_KG) == []

    def test_the_dropped_guest_is_caught(self) -> None:
        # THE #2062 HEADLINE: 93.2% of roster-named guests never reached kg.json.
        dropped = kg((HOST, "host"), (GUEST, "mentioned"))
        v = check_roster_speakers_reach_the_graph(GOOD_META, dropped)
        assert len(v) == 1 and GUEST in v[0]

    def test_an_episode_with_no_speakers_at_all_is_caught(self) -> None:
        v = check_roster_speakers_reach_the_graph(GOOD_META, kg(("Elon Musk", "mentioned")))
        assert len(v) == 2


class TestAnonymousVoices:
    def test_a_coherent_episode_is_clean(self) -> None:
        assert check_no_anonymous_speakers(GOOD_KG) == []

    def test_a_diarization_label_as_guest_is_caught(self) -> None:
        v = check_no_anonymous_speakers(kg(("SPEAKER_07", "guest")))
        assert len(v) == 1 and "SPEAKER_07" in v[0]

    def test_a_real_person_whose_name_starts_with_speaker_is_not_caught(self) -> None:
        # "Speaker John Knight" is a person; the same trap `is_scoped_placeholder_person_id` avoids.
        assert check_no_anonymous_speakers(kg(("Speaker John Knight", "host"))) == []


class TestRoleVocabulary:
    def test_a_coherent_episode_is_clean(self) -> None:
        assert check_roles_are_known(GOOD_KG) == []

    def test_an_unknown_role_is_caught(self) -> None:
        v = check_roles_are_known(kg(("Someone Else", "contributor")))
        assert len(v) == 1 and "contributor" in v[0]

    def test_a_roleless_node_is_not_a_violation(self) -> None:
        assert (
            check_roles_are_known(
                {"nodes": [{"id": "person:x", "type": "Person", "properties": {"name": "X"}}]}
            )
            == []
        )


class TestSpokenByIntegrity:
    def test_a_coherent_episode_is_clean(self) -> None:
        assert check_spoken_by_targets_exist(GOOD_GI) == []

    def test_a_dangling_edge_is_caught(self) -> None:
        bad = gi([("quote:1", "person:ghost")], persons=["person:ryan-knutson"])
        v = check_spoken_by_targets_exist(bad)
        assert len(v) == 1 and "person:ghost" in v[0]


class TestOneQuoteOneSpeaker:
    def test_a_coherent_episode_is_clean(self) -> None:
        assert check_one_quote_one_speaker(GOOD_GI) == []

    def test_the_duplicate_identity_is_caught(self) -> None:
        # REAL: a fresh DGX ingest produced 116 edges for 59 quotes — `person:twiggy` and
        # `person:unresolved-twiggy-{ep}` were the same voice.
        bad = gi([("quote:1", "person:twiggy"), ("quote:1", "person:unresolved-twiggy-ep1")])
        v = check_one_quote_one_speaker(bad)
        assert len(v) == 1 and "quote:1" in v[0]

    def test_the_same_edge_twice_is_not_two_speakers(self) -> None:
        bad = gi([("quote:1", "person:x"), ("quote:1", "person:x")])
        assert check_one_quote_one_speaker(bad) == []


class TestCollapseOntoOneSpeaker:
    def test_a_coherent_episode_is_clean(self) -> None:
        assert check_not_collapsed_onto_one_speaker(GOOD_META, GOOD_GI) == []

    def test_the_operator_reported_symptom_is_caught(self) -> None:
        # Every insight on the panel showing the same name: 36.9% of production episodes.
        bad = gi([(f"quote:{i}", "person:ryan-knutson") for i in range(8)])
        v = check_not_collapsed_onto_one_speaker(GOOD_META, bad)
        assert len(v) == 1 and "8 attributed quotes" in v[0]

    def test_a_genuine_monologue_is_exempt(self) -> None:
        # One voice on the roster: everything landing on that voice is CORRECT, not a collapse.
        solo = meta((HOST, "host"))
        bad = gi([(f"quote:{i}", "person:ryan-knutson") for i in range(8)])
        assert check_not_collapsed_onto_one_speaker(solo, bad) == []

    def test_a_short_episode_is_not_judged(self) -> None:
        # Two quotes on one speaker is a normal episode opening, not a signature.
        few = gi([("quote:1", "person:ryan-knutson"), ("quote:2", "person:ryan-knutson")])
        assert check_not_collapsed_onto_one_speaker(GOOD_META, few) == []


class TestTheCombinedCheck:
    def test_a_coherent_episode_reports_nothing(self) -> None:
        assert check_episode(GOOD_META, GOOD_KG, GOOD_GI) == []

    def test_it_reports_every_independent_defect_at_once(self) -> None:
        # A partially-fixed corpus must not look clean because one rule passes.
        bad_kg = kg((HOST, "host"), ("Sarah Guo", "host"), ("SPEAKER_07", "guest"))
        bad_gi = gi([("quote:1", "person:ghost")], persons=[])
        v = check_episode(GOOD_META, bad_kg, bad_gi, label="ep1")
        joined = " | ".join(v)
        assert "Sarah Guo" in joined
        assert "SPEAKER_07" in joined
        assert "person:ghost" in joined
        assert all(x.startswith("ep1: ") for x in v), v
