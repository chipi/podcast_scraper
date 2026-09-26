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
    check_no_show_as_speaker,
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


class TestATitleAndAMisspellingTogether:
    """Either alone was tolerated; the two at once were not, and that demoted a real guest.

    Found by running m0009's preview against the 2026-09-20 production snapshot: *Ground Truths*
    episode "Bruce Lanphear: Chronic Lead Exposure" has a graph node `Bruce Lanphear` and a roster
    that heard `Professor Bruce Lanphier`. m0009 reads "matches no roster entry" as "did not
    speak", so the episode's actual guest was demoted to `mentioned` — the one direction the
    migration's own docstring promises never to take for a spelling variant.
    """

    def test_either_alone_already_matched(self) -> None:
        assert same_person("Bruce Lanphear", "Bruce Lanphier")  # misspelling, no title
        assert same_person("Bruce Lanphear", "Professor Bruce Lanphear")  # title, no misspelling

    def test_both_at_once_now_matches(self) -> None:
        assert same_person("Bruce Lanphear", "Professor Bruce Lanphier")

    @pytest.mark.parametrize(
        "a,b",
        [
            ("Eric Topol", "Professor Bruce Lanphier"),  # the co-speaker on that same episode
            ("Sarah Guo", "Dr. Elad Gil"),
            ("Ryan Knutson", "Senator Jessica Mendoza"),
            # A TITLE OVER A MONONYM is the shape that merges strangers: strip the title and the
            # subset rule reads the bare surname as "the same human as anyone sharing it".
            ("Dr. Smith", "Jane Smith"),
            ("Senator Warren", "Elizabeth Warren"),
            ("Prof. Jones", "Indiana Jones"),
        ],
    )
    def test_dropping_a_title_does_not_merge_strangers(self, a: str, b: str) -> None:
        assert not same_person(a, b)

    @pytest.mark.parametrize(
        "a,b",
        [
            ("Justice Smith", "Will Smith"),  # two actors
            ("Major Garrett", "Garrett Smith"),  # a journalist
            ("Sister Souljah", "Souljah Boy"),
            ("Gen Kato", "Kato Hiroshi"),
            ("Lady Gaga", "Gaga Smith"),
        ],
    )
    def test_a_title_that_is_also_a_name_is_not_treated_as_a_title(self, a: str, b: str) -> None:
        """`HONORIFIC_PREFIXES` is deliberately short: a token that heads real names stays out.

        The asymmetry is the whole argument — missing one variant costs a single unmatched name,
        while admitting `Justice` or `Major` merges two different people into one identity.
        """
        assert not same_person(a, b)

    def test_a_real_mononym_still_matches(self) -> None:
        """The rule the mononym guard must not break."""
        assert same_person("Twiggy", "Twiggy Lawson")

    def test_a_name_that_is_only_a_title_keeps_its_token(self) -> None:
        """Folding it away would leave "", which every caller reads as "no name at all"."""
        assert same_person("Professor", "professor")
        assert not same_person("Professor", "Bruce Lanphear")

    def test_a_title_that_is_also_a_surname_survives_in_place(self) -> None:
        """`Major` and `Rev` are real surnames; only a LEADING title is dropped."""
        assert not same_person("John Major", "John Smith")
        assert same_person("Sir John Major", "John Major")


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

    # --- the interview-shape exemption (#2158) ------------------------------------------------
    #
    # The rule's own docstring recorded this false positive and could not act on it: an interview
    # where one voice says everything quotable. Talk Eastern Europe, "Book Talk: Betrayal" — roster
    # [Adam Reichardt host, Luke Harding guest], diarization cleanly separated them (273/118
    # segments), all 82 quotes on the guest, and 81 of 82 CORRECT on a char-offset check. 22 prod
    # episodes had this shape and were being counted as damage.
    #
    # These fixtures give Person nodes a real display name, which the shared ``gi()`` helper does
    # not (it uses the node id), because the role lookup resolves edge -> node name -> roster entry.

    @staticmethod
    def _named_gi(target_name: str, target_id: str, n: int = 8):
        return {
            "nodes": [{"id": target_id, "type": "Person", "properties": {"name": target_name}}],
            "edges": [
                {"type": "SPOKEN_BY", "from": f"quote:{i}", "to": target_id} for i in range(n)
            ],
        }

    def test_every_quote_on_the_GUEST_is_an_interview_not_a_collapse(self) -> None:
        g = self._named_gi(GUEST, "person:sharon-turlip")

        assert check_not_collapsed_onto_one_speaker(GOOD_META, g) == [], (
            "the host asks questions; questions are not claims. This is the documented shape of "
            "22 prod episodes that were being reported as damage"
        )

    def test_every_quote_on_the_HOST_is_still_reported(self) -> None:
        """The defect this rule exists for: marker-blind attribution latches onto the host."""
        g = self._named_gi(HOST, "person:ryan-knutson")

        v = check_not_collapsed_onto_one_speaker(GOOD_META, g)
        assert len(v) == 1 and "8 attributed quotes" in v[0]

    def test_an_unresolvable_target_is_reported_not_exempted(self) -> None:
        """An exemption must never be granted on a lookup failure — fail toward reporting."""
        g = self._named_gi("Somebody Not On The Roster", "person:stranger")

        assert len(check_not_collapsed_onto_one_speaker(GOOD_META, g)) == 1

    def test_guest_only_with_NO_host_on_the_roster_is_still_reported(self) -> None:
        """The exemption is the host/guest PAIR shape. Two guests and no host is not that."""
        two_guests = meta((GUEST, "guest"), ("Someone Else", "guest"))
        g = self._named_gi(GUEST, "person:sharon-turlip")

        assert len(check_not_collapsed_onto_one_speaker(two_guests, g)) == 1


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


class TestAShowIsNotAPersonWhoHostsIt:
    """#2064 as a coherence rule — the one the roster cannot supply.

    Feed host detection seeded the show's own name, so `content.speakers` SAYS
    `host='Africa Tech Summit'` and every other rule here agrees with it. It is a wrong entry rather
    than a missing one, which is exactly why `check_speakers_actually_spoke` waves it through. 19
    episodes in a 279-episode production sample carry one.
    """

    def _meta(self, title: str, *speakers):
        return {
            "feed": {"title": title},
            "content": {"speakers": [{"name": n, "role": r} for n, r in speakers]},
        }

    def test_the_show_seated_as_host_is_caught(self) -> None:
        meta = self._meta(
            "Africa Tech Summit Podcast",
            ("Africa Tech Summit", "host"),
            ("Mukami Wairaina", "guest"),
        )
        bad = kg(("Africa Tech Summit", "host"), ("Mukami Wairaina", "guest"))
        v = check_no_show_as_speaker(meta, bad)
        assert len(v) == 1 and "Africa Tech Summit" in v[0]

    def test_every_other_rule_misses_it(self) -> None:
        # The point of adding this one: the show IS on the roster, so the "did they speak?" rule
        # cannot object. Without this rule the episode looks coherent.
        meta = self._meta(
            "Africa Tech Summit Podcast",
            ("Africa Tech Summit", "host"),
            ("Mukami Wairaina", "guest"),
        )
        bad = kg(("Africa Tech Summit", "host"), ("Mukami Wairaina", "guest"))
        assert check_speakers_actually_spoke(meta, bad) == []

    def test_a_real_host_whose_name_is_in_the_title_is_not_caught(self) -> None:
        meta = self._meta(
            "Invest Like the Best with Patrick O'Shaughnessy", ("Patrick O'Shaughnessy", "host")
        )
        assert check_no_show_as_speaker(meta, kg(("Patrick O'Shaughnessy", "host"))) == []

    def test_a_mentioned_show_name_is_not_a_violation(self) -> None:
        # Only a SPEAKING role is a problem: the show may legitimately be mentioned.
        meta = self._meta("Africa Tech Summit Podcast", ("Mukami Wairaina", "guest"))
        assert check_no_show_as_speaker(meta, kg(("Africa Tech Summit", "mentioned"))) == []

    def test_no_feed_title_means_no_opinion(self) -> None:
        assert check_no_show_as_speaker({"content": {}}, kg(("Anything At All", "host"))) == []

    def test_it_is_part_of_the_combined_check(self) -> None:
        meta = self._meta("Africa Tech Summit Podcast", ("Africa Tech Summit", "host"))
        v = check_episode(meta, kg(("Africa Tech Summit", "host")), {"nodes": [], "edges": []})
        assert any("names the show" in x for x in v), v


class TestFoldingMustNotEraseANonLatinName:
    """A CJK or Cyrillic name must at minimum match ITSELF (advisor S2).

    ``_fold`` ran ``re.sub(r"[^a-z0-9]+", " ", ...)`` after accent-stripping, which deletes every
    character of an entirely non-Latin name. The fold returned ``""``, the empty guard returned
    False, and so::

        same_person('张川红', '张川红')             -> False
        same_person('Владимир Путин', ...same...)  -> False

    A speaker who cannot match themselves is "never spoke" on every coherence check and can never
    be fuzzy-matched by the m0009 migration. Production carries Round Table China, China Plus,
    ChinaTalk and The Naked Pravda.
    """

    @pytest.mark.parametrize(
        "name",
        ["张川红", "Владимир Путин", "محمد صلاح", "김정은", "Ἀριστοτέλης"],
    )
    def test_a_name_matches_itself(self, name: str) -> None:
        assert same_person(name, name) is True

    def test_latin_behaviour_is_unchanged(self) -> None:
        assert same_person("Dr. Adam Rodman", "Adam Rodman") is True
        assert same_person("Bernard Leong", "Bernard Leung") is True
        assert same_person("Sarah Guo", "Elad Gil") is False

    def test_two_different_cjk_names_are_not_the_same_person(self) -> None:
        assert same_person("张川红", "李明") is False
