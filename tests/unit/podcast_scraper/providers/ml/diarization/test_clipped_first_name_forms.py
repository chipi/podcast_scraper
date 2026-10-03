"""A clipped given name is never an ordinary word, and "between you and me, X" names nobody.

Synthetic, one behaviour per test (never-commit-real-episodes).
"""

from __future__ import annotations

import re

import pytest

from podcast_scraper.providers.ml.diarization.roster import (
    _first_name_forms,
    _introduces_itself_as_host,
    _says_cohost_formula,
    _vocative_count,
)

pytestmark = pytest.mark.unit


def _forms(name: str, text: str) -> set:
    return set(_first_name_forms(name, text).split("|"))


def test_a_real_clipped_name_is_still_a_form() -> None:
    assert "kev" in _forms("Kevin Okafor", "Thanks, Kev. Over to you.")


def test_just_is_not_a_form_of_justin() -> None:
    assert "just" not in _forms("Justin Marlowe", "And so I'm just looking at all of it.")


def test_and_is_not_a_form_of_andy() -> None:
    assert "and" not in _forms("Andy Pellow", "Right. And, you know, it worked.")


def test_case_is_not_a_form_of_casey() -> None:
    assert "case" not in _forms("Casey Tarrant", "In that case, we should go.")


def test_a_sentence_initial_and_is_not_a_vocative_of_andy() -> None:
    assert _vocative_count("Right. And, you know, it worked.", "Andy Pellow") == 0


def test_im_just_is_not_a_self_introduction_as_justin() -> None:
    assert not _introduces_itself_as_host("So I'm just looking at all of it.", "Justin Marlowe")


def test_between_you_and_me_is_not_the_cohost_formula() -> None:
    assert not _says_cohost_formula(
        "It's between you and me, Bethany. Nobody else.", "Bethany Ruiz"
    )


def test_between_you_and_me_is_not_a_self_introduction() -> None:
    text = "It's between you and me, Bethany. Nobody else."
    assert not _introduces_itself_as_host(text, "Bethany Ruiz")


def test_the_cohost_formula_still_names_its_speaker() -> None:
    assert _says_cohost_formula(
        "Welcome to the show with me, Priya. And me, Bethany Ruiz.", "Bethany Ruiz"
    )


def test_a_nickname_from_the_table_is_still_a_form() -> None:
    forms = _first_name_forms("Robert Hale", "Thanks, Rob.")
    assert re.fullmatch(forms, "rob", re.IGNORECASE)
