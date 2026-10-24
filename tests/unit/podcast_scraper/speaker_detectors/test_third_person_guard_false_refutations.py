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


# A host reading out their own handle: "On Twitter, I'm at <name>". Measured on the DGX relabel of
# 2026-10-23 (val_v4, two episodes of one interview show): the guard refused the host's name on
# the host's own voice, and the two-voice complement / swap then put it on the guest's voice.


def test_a_host_reading_their_own_handle_is_a_self_introduction() -> None:
    text = (
        "Hello, everyone, and welcome back to Talks with Tobias. Thanks for listening. "
        "On Twitter, I'm at Tobias Wren, and the show is at Wren Talks."
    )
    assert not refuted_by_third_person(text, "Tobias Wren")


def test_i_am_at_a_name_is_a_self_introduction() -> None:
    assert not refuted_by_third_person("You can find me, I am at Tobias Wren.", "Tobias Wren")


def test_i_am_at_someones_place_is_still_a_mention() -> None:
    text = "Today I'm at Tobias Wren's studio, talking about what Wren built."
    assert refuted_by_third_person(text, "Tobias Wren")


def test_this_is_at_a_name_is_not_a_self_introduction() -> None:
    # only "I'm" / "I am" take the handle form; "this is at X" is no introduction of anyone
    text = "The reading this is at Tobias Wren, the bookshop on Main Street. Wren opens at nine."
    assert refuted_by_third_person(text, "Tobias Wren")


def test_the_host_keeps_their_name_and_the_guest_is_not_given_it() -> None:
    """The episode shape end to end: the model answers both voices right; the host's voice is not
    refused, so no complement or swap moves the host's name onto the guest's voice."""
    import json

    from podcast_scraper.speaker_detectors.resolution import resolve_voices_and_roles

    voices = {
        "SPEAKER_00": "Welcome back to Talks with Tobias. Today I'm chatting with Ana Ruiz. "
        "Thanks for listening. On Twitter, I'm at Tobias Wren, and the show is at Wren Talks.",
        "SPEAKER_01": "Thanks for having me. That feels very nanny to you, Tobias.",
    }
    answer = {
        "voices": {
            "SPEAKER_00": {"name": "Tobias Wren", "role": "host"},
            "SPEAKER_01": {"name": "Ana Ruiz", "role": "guest"},
        }
    }
    report: dict = {}
    out = resolve_voices_and_roles(
        ["Ana Ruiz", "Tobias Wren"],
        voices,
        lambda _prompt: json.dumps(answer),
        known_hosts=["Tobias Wren"],
        report=report,
    )
    assert (out["SPEAKER_00"].name, out["SPEAKER_01"].name) == ("Tobias Wren", "Ana Ruiz")
    assert not report.get("complement")
