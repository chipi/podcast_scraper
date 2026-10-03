"""The one person check (`is_publishable_speaker_name`) refuses the junk the census found published.

Show-sidecar census and gold gate, 2026-10-02: show names, places, counts, role-word captures and
product names were published as people ("Norman Conquest" 132 times, "Americas Online" 62). Each
family of junk is one test; the real names that resemble them are pinned as accepted beside it.
"""

from __future__ import annotations

import pytest

from podcast_scraper.speaker_detectors.hosts import is_publishable_speaker_name as ok

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "name",
    [
        "Timmerman Report",
        "Americas Online",
        "Africa Tech Summit",
        "Brilliant Experience",
        "Turkey Book",
        "Fable Tech",
        "Ruse Tech",
        "Norman Conquest",
    ],
)
def test_a_show_or_brand_tail_is_not_a_person(name: str) -> None:
    assert not ok(name)


@pytest.mark.parametrize("name", ["Trivium China", "Carnegie India", "West Africa"])
def test_a_region_as_the_last_word_is_not_a_person(name: str) -> None:
    assert not ok(name)


@pytest.mark.parametrize("name", ["Two Carnegie Mellon", "One McKinsey"])
def test_a_name_that_starts_with_a_count_is_not_a_person(name: str) -> None:
    assert not ok(name)


@pytest.mark.parametrize(
    "name", ["Host Mike", "Guest Mike", "Speaker Mike", "Guest Host Tim", "Mister Rob"]
)
def test_a_captured_role_word_before_a_first_name_is_not_a_person(name: str) -> None:
    assert not ok(name)


def test_a_job_title_inside_a_long_name_is_not_a_clean_person_name() -> None:
    assert not ok("Senior User Experience Specialist Therese Fessenden")


def test_punctuation_a_name_never_carries_rejects_it() -> None:
    assert not ok("Premier Unbelievable?")


@pytest.mark.parametrize("name", ["Gemini", "Claude", "Apple", "ChatGPT"])
def test_a_product_or_company_introducing_itself_is_not_a_person(name: str) -> None:
    assert not ok(name)


@pytest.mark.parametrize(
    "name",
    [
        "Christopher Guest",  # role word as a real surname
        "Christian Schmidt",
        "Donald S. Lopez Jr.",
        "Peter Attia, MD",
        "Picabo Street",  # "street" is deliberately not a show tail
        "Sam Brazil",  # a country that is also a surname
        "Professor Hannah Fry",  # a title is stripped by identity, not a reason to reject
        "India Arie",  # a place as a GIVEN name
        "Aaron Levie)",  # a stray bracket is cleaned by the canonicaliser, not rejected
        "swyx",
        "Ok Taecyeon",
    ],
)
def test_real_names_that_resemble_the_junk_are_accepted(name: str) -> None:
    assert ok(name)


@pytest.mark.parametrize("name", ["Pulitzer Prize-winning", "Award-winning", "London-based"])
def test_a_hyphenated_descriptor_is_not_a_name(name: str) -> None:
    """Freakonomics, 2026-10-03: "Pulitzer Prize-winning" was published as a guest."""
    assert not ok(name)


@pytest.mark.parametrize("name", ["Jean-Paul Sartre", "Mary-Kate Olsen", "Hannah Fry"])
def test_hyphenated_real_names_still_pass(name: str) -> None:
    assert ok(name)
