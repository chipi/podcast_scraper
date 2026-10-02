"""The last gate before a name is published: what it refuses, what it repairs, what it keeps.

Measured on the published corpus 2026-10-02 (4,638 names): "Host" on 20 voices, an organisation as
host on 10, interjections ("OK", "Thank", "Right"), committees and job titles ("House Select
Committee", "Roblox CEO"), and self-introductions that kept the job or the show in front of the
person ("Your Host Luisa Leni", "Planet Money's Kenny Malone").

Each shape is its own case — and so is every real name that must survive.
"""

from __future__ import annotations

import pytest

from podcast_scraper.speaker_detectors.hosts import is_publishable_speaker_name, strip_role_prefix

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("label", ["Host", "Guest", "Narrator", "Co-Host"])
def test_a_role_word_is_not_a_person(label: str) -> None:
    assert not is_publishable_speaker_name(label)


@pytest.mark.parametrize("word", ["OK", "Okay", "Thank", "Right", "Yeah"])
def test_an_interjection_is_not_a_person(word: str) -> None:
    assert not is_publishable_speaker_name(word)


@pytest.mark.parametrize(
    "name",
    ["House Select Committee", "Roblox CEO", "Treasury Foreign Exchange", "Redwood Research"],
)
def test_an_organisation_or_a_job_is_not_a_person(name: str) -> None:
    assert not is_publishable_speaker_name(name)


@pytest.mark.parametrize("abbr", ["GE", "AI", "IDF"])
def test_an_abbreviation_is_not_a_person(abbr: str) -> None:
    assert not is_publishable_speaker_name(abbr)


def test_a_possessive_is_not_published_as_part_of_a_name() -> None:
    assert not is_publishable_speaker_name("Planet Money's Kenny Malone")


@pytest.mark.parametrize(
    "raw, person",
    [
        ("Your Host Luisa Leni", "Luisa Leni"),
        ("Deputy Editor Eilish Hart", "Eilish Hart"),
        ("Planet Money's Kenny Malone", "Kenny Malone"),
        ("Host, Tobias Wren", "Tobias Wren"),
    ],
)
def test_the_job_or_the_show_in_front_of_a_person_is_stripped(raw: str, person: str) -> None:
    assert strip_role_prefix(raw) == person
    assert is_publishable_speaker_name(person)


@pytest.mark.parametrize(
    "name", ["Karen Elliott House", "Dick Spring", "swyx", "Twiggy", "Jo", "Maria Lindqvist"]
)
def test_real_names_survive(name: str) -> None:
    # A surname that is an ordinary word, a lowercase handle, a mononym, a two-letter name.
    assert is_publishable_speaker_name(name)


def test_stripping_leaves_a_plain_name_alone() -> None:
    assert strip_role_prefix("Hostetler Brown") == "Hostetler Brown"
    assert strip_role_prefix("Maria Lindqvist") == "Maria Lindqvist"
