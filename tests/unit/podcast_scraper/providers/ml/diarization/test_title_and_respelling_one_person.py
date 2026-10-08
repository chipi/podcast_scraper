"""One person is one entry on an episode's record, whatever the title or spelling (2026-10-08).

A census of 2,531 served episodes found 12 in 9 feeds listing one person twice: a voice under the
spelling or title it said ("Professor Hannah Frye"), and the feed's or show notes' spelling of the
same person beside it ("Hannah Fry"). Three defects, one test group each.
"""

from __future__ import annotations

import pytest

from podcast_scraper.providers.ml.diarization.roster import (
    _canonicalize_to_known_host,
    _core_name_tokens,
    _recover_stated_names,
    _same_person,
    _snap_near_identical_host,
    SpeakerRole,
)
from podcast_scraper.workflow.metadata_generation import _unplaced_speakers, SpeakerInfo

pytestmark = [pytest.mark.unit]


# --- 1. A leading title is not the given name ---------------------------------------------------


def test_a_title_does_not_hide_the_host_from_the_host_spelling() -> None:
    # Google DeepMind 0018: the host's own "I'm Professor Hannah Frye" never snapped to the feed's
    # Hannah Fry, because "Professor" was compared against "Hannah".
    assert _canonicalize_to_known_host("Professor Hannah Frye", ["Hannah Fry"]) == "Hannah Fry"


def test_a_title_alone_does_not_make_a_stated_host_a_stranger() -> None:
    # Google DeepMind 0014: "I'm Professor Hannah Fry" was read as a name NOT in the host pool, so
    # the host's own voice was barred from the host seat.
    assert _snap_near_identical_host("Professor Hannah Fry", ["Hannah Fry"]) == "Hannah Fry"
    assert _snap_near_identical_host("Dr. Peter Attia", ["Peter Attia"]) == "Peter Attia"


def test_a_title_with_only_a_surname_keeps_its_title() -> None:
    # Dropping it would leave the surname to be read as a given name.
    assert _core_name_tokens("Professor Pape") == ["Professor", "Pape"]
    assert _core_name_tokens("Professor Hannah Fry") == ["Hannah", "Fry"]
    assert _same_person("Professor Pape", "Robert Pape")


def test_a_different_person_with_a_title_is_still_a_different_person() -> None:
    assert _snap_near_identical_host("Professor Hannah Smith", ["Hannah Fry"]) == (
        "Professor Hannah Smith"
    )
    assert _snap_near_identical_host("Dr. Kevin Ross", ["Kevin Roose"]) == "Dr. Kevin Ross"


# --- 2. The record does not list a placed person again as unplaced ------------------------------


def _unplaced(placed, known_hosts=(), unbound=()):
    return [
        s.name
        for s in _unplaced_speakers(
            [
                SpeakerInfo(id=f"p{i}", name=n, role="guest", placed=True, voices=[f"V{i}"])
                for i, n in enumerate(placed)
            ],
            diagnostics={
                "tried": {"known_hosts": list(known_hosts)},
                "summary": {"unbound_names": list(unbound)},
            },
            detected_hosts=None,
            detected_guests=None,
            feed_title="Some Show",
        )
    ]


@pytest.mark.parametrize(
    "placed, stated",
    [
        ("Joe Wiesenthal", "Joe Weisenthal"),  # Odd Lots
        ("Anita Arnond", "Anita Anand"),
        ("Professor Hannah Frye", "Hannah Fry"),  # title AND spelling
    ],
)
def test_a_respelt_placed_person_is_not_listed_again(placed: str, stated: str) -> None:
    assert _unplaced([placed], known_hosts=[stated]) == []


def test_two_unplaced_spellings_of_one_person_are_listed_once() -> None:
    # The feed states "Bernard Leong" and the show notes "Bernard Leung": the first source wins.
    assert _unplaced([], known_hosts=["Bernard Leong"], unbound=["Bernard Leung"]) == [
        "Bernard Leong"
    ]


@pytest.mark.parametrize(
    "placed, stated",
    [
        ("Anna Smith", "Anna Jones"),  # a different surname is a different person
        ("Robert Pape", "Karen Pape"),  # a shared surname is not enough
        ("Sam Lee Jr", "Sam Leigh"),  # a generation on one side only, and a different spelling
    ],
)
def test_different_people_are_still_listed(placed: str, stated: str) -> None:
    assert _unplaced([placed], known_hosts=[stated]) == [stated]


def test_a_missing_generation_across_sources_is_still_one_person() -> None:
    # Across sources a feed routinely drops the "Jr."; `_same_person` keeps that a match.
    assert _unplaced(["Robert Pape Jr."], known_hosts=["Robert Pape"]) == []


def test_two_swapped_letters_are_one_slip_of_the_host_name() -> None:
    # Odd Lots: "I'm Joe Wiesenthal" stayed a guest beside the feed's host Joe Weisenthal.
    assert _snap_near_identical_host("Joe Wiesenthal", ["Joe Weisenthal"]) == "Joe Weisenthal"
    # Not a looser match: two separate edits, or a short surname, still stay apart.
    assert _snap_near_identical_host("Joe Wiesenthel", ["Joe Weisenthal"]) == "Joe Wiesenthel"
    assert _snap_near_identical_host("Kevin Rsos", ["Kevin Ross"]) == "Kevin Rsos"
    # Two letters traded across the word are two edits, not one slip.
    assert _snap_near_identical_host("Joe Waisenthel", ["Joe Weisenthal"]) == "Joe Waisenthel"


# --- 3. A stated participant in a polluted host pool is not a host -----------------------------


def _roster(name: str, role: str) -> dict:
    return {"V": SpeakerRole(name=name, role=role, named=True, source="self_intro")}


def test_a_stated_participant_in_a_polluted_host_pool_gets_its_spelling_not_the_host_role() -> None:
    # The a16z Show lists its interviewee Lukasz Kaiser among nine "hosts"; his voice said "Lucas".
    by_voice = _roster("Lucas Kaiser", "guest")
    _recover_stated_names(
        by_voice, ["Lukasz Kaiser"], ["Lukasz Kaiser"], participants=["Lukasz Kaiser"]
    )
    assert (by_voice["V"].name, by_voice["V"].role) == ("Lukasz Kaiser", "guest")


def test_without_the_episode_naming_them_a_guest_still_never_takes_a_hosts_spelling() -> None:
    by_voice = _roster("Kevin Ross", "guest")
    _recover_stated_names(by_voice, ["Kevin Roose"], ["Kevin Roose"])
    assert by_voice["V"].name == "Kevin Ross"


# --- 4. A spoken respelling of a person already on a voice is not a spare guest name ------------


def test_a_spoken_respelling_of_a_placed_guest_is_not_a_spare_guest_name() -> None:
    # The Long Run: the host's spoken "Andy Ratcliffe" stayed spare beside the claimed Andy
    # Rachleff and was forced onto the co-guest Yung Lie's voice.
    from podcast_scraper.providers.ml.diarization.roster import _spare_guest_names

    declared = ["Andy Ratcliffe", "Young Lee", "Jill Lepore"]
    claimed = ["Luke Timmerman", "Andy Rachleff"]
    assert _spare_guest_names(declared, {"luke timmerman"}, {"young lee"}, claimed) == [
        "Jill Lepore"
    ]
    # A different person with the same given name stays spare.
    assert _spare_guest_names(["Andy Jassy"], set(), set(), claimed) == ["Andy Jassy"]


# --- 5. A respelling of a stated participant publishes the stated spelling ---------------------


def test_a_respelt_participant_takes_the_metadata_spelling() -> None:
    # Made In Africa: the host said "Scholastic Gatobu"; the show notes say "Schola Gatobu".
    by_voice = _roster("Scholastic Gatobu", "guest")
    _recover_stated_names(by_voice, ["SCHOLA GATOBU"], [], participants=["SCHOLA GATOBU"])
    assert by_voice["V"].name == "SCHOLA GATOBU"


@pytest.mark.parametrize(
    "name, participants, claimed_elsewhere",
    [
        ("Scholastic Gatobu", ["SCHOLA GATOBU"], "SCHOLA GATOBU"),  # another voice has it
        ("Anna Smith", ["Anna Jones"], None),  # a different surname is a different person
        ("Anita Anant", ["Anita Anand", "Anita Anaut"], None),  # two candidates: no answer
    ],
)
def test_otherwise_the_voice_keeps_its_spelling(name, participants, claimed_elsewhere) -> None:
    by_voice = _roster(name, "guest")
    if claimed_elsewhere:
        by_voice["W"] = SpeakerRole(
            name=claimed_elsewhere, role="guest", named=True, source="llm_resolution"
        )
    _recover_stated_names(by_voice, [], [], participants=participants)
    assert by_voice["V"].name == name
