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
    _same_person,
    _snap_near_identical_host,
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
