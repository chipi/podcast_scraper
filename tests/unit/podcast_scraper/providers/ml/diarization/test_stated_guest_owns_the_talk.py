"""K8 (#2338): the one stated guest on the one voice that owns the talk, and its cross-post guard.

Measured 2026-10-14 on prod (2,632 served episodes): +12 named voices, each read right on its text;
gold dev_v1 1 better / 0 worse, val_v1 2 better / 0 worse. All fixtures are synthetic.
"""

from __future__ import annotations

import pytest

from podcast_scraper.providers.ml.diarization.roster import (
    _named_as_another_shows_host,
    _stated_guest_who_owns_the_talk,
)

pytestmark = pytest.mark.unit

TALK = {"SPEAKER_00": 200.0, "SPEAKER_01": 800.0}


def _owner(metadata_named, texts, roster_names=("Tobias Wren",), talk=TALK):
    return _stated_guest_who_owns_the_talk(
        metadata_named=list(metadata_named),
        unassigned=["SPEAKER_01"],
        talk=talk,
        roster_names=list(roster_names),
        known_hosts=["Tobias Wren"],
        host_names_lower={"tobias wren"},
        voice_texts=texts,
    )


TEXTS = {
    "SPEAKER_00": "Welcome, I'm Tobias Wren. I have been reading Maria Lindqvist's book.",
    "SPEAKER_01": "The ports moved because the river silted.",
}


def test_the_stated_guest_lands_on_the_voice_that_owns_the_talk() -> None:
    assert _owner(["Maria Lindqvist"], TEXTS) == ("Maria Lindqvist", "SPEAKER_01")


def test_a_surname_nobody_speaks_names_nobody() -> None:
    """The floor the reverted design lacked: the description names someone never mentioned."""
    texts = {**TEXTS, "SPEAKER_00": "Welcome, I'm Tobias Wren. Today: the Baltic trade."}
    assert _owner(["Maria Lindqvist"], texts) is None


def test_two_stated_people_name_nobody() -> None:
    assert _owner(["Maria Lindqvist", "Henrik Sallow"], TEXTS) is None


def test_no_voice_owning_the_talk_names_nobody() -> None:
    assert (
        _owner(["Maria Lindqvist"], TEXTS, talk={"SPEAKER_00": 600.0, "SPEAKER_01": 400.0}) is None
    )


@pytest.mark.parametrize(
    "text",
    [
        "now, tobias was recently on henrik sallow's show to talk about investing.",
        "this week a conversation from henrik sallow’s podcast.",
        "tobias was a guest on henrik sallow, lifestyle mastery.",
    ],
)
def test_another_shows_host_is_recognised(text: str) -> None:
    assert _named_as_another_shows_host("Henrik Sallow", text)


@pytest.mark.parametrize(
    "text",
    [
        "so henrik, you were masquerading as a historian of ireland.",
        "i have been reading henrik sallow's new book all week.",
        "henrik sallow showed us the archive.",
    ],
)
def test_a_guest_spoken_of_is_not_another_shows_host(text: str) -> None:
    assert not _named_as_another_shows_host("Henrik Sallow", text)


def test_a_cross_post_names_nobody() -> None:
    texts = {
        "SPEAKER_00": "Now, Tobias was recently a guest on Henrik Sallow's show.",
        "SPEAKER_01": "I started as an engineer and then moved into investing.",
    }
    assert _owner(["Henrik Sallow"], texts) is None
