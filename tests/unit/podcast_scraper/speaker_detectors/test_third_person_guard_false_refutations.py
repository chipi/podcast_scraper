"""The third-person guard must not discard a person's name from their OWN voice.

Measured on prod (2026-10-02, 28 discards reviewed by hand): the guard threw away correct names
when the voice introduced itself in a form the check did not recognise ("I am your host, David
Beckworth", "And I am Rob Armstrong").

Each shape is its own case, and so is each shape that must STILL refute.

All fixtures are synthetic (never-commit-real-episodes).
"""

from __future__ import annotations

import pytest

from podcast_scraper.speaker_detectors.resolution import refuted_by_third_person

pytestmark = pytest.mark.unit


def test_your_host_introduction_is_a_self_introduction() -> None:
    text = "Welcome to the show. I am your host, Tobias Wren, and today: rivers. Wren here."
    assert not refuted_by_third_person(text, "Tobias Wren")


def test_a_short_first_name_with_the_surname_is_a_self_introduction() -> None:
    text = "And I am Tob Wren, coming to you from the studio. Wren's take is that rates fall."
    assert not refuted_by_third_person(text, "Tobias Wren")


def test_a_relative_with_the_same_surname_is_not_a_self_introduction() -> None:
    # "Jac" is not a prefix of "Tobias"; this voice is somebody else in the family.
    text = "I'm Jacqueline Wren, his mother. Tobias Wren was always a difficult child."
    assert refuted_by_third_person(text, "Tobias Wren")


def test_the_full_name_in_the_third_person_still_refutes() -> None:
    text = "Today we look at what Maria Lindqvist found in the archive and why it matters."
    assert refuted_by_third_person(text, "Maria Lindqvist")


def test_a_possessive_after_this_is_is_still_not_a_self_introduction() -> None:
    text = "This is Maria Lindqvist's fourth book, and it is her best."
    assert refuted_by_third_person(text, "Maria Lindqvist")


def test_the_full_first_name_spoken_for_a_stated_short_form_is_a_self_introduction() -> None:
    # Stated "Tob Wren", heard "I am Tobias Wren": the prefix holds in the other direction too.
    text = "And I am Tobias Wren, coming to you from the studio. Wren's take is that rates fall."
    assert not refuted_by_third_person(text, "Tob Wren")


def test_an_honorific_between_the_cue_and_the_name_is_a_self_introduction() -> None:
    text = "Hello, I'm Dr. Tobias Wren and this is the show. Wren here, back after the break."
    assert not refuted_by_third_person(text, "Tobias Wren")
