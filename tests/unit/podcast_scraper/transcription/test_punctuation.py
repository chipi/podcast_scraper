"""Detecting an unpunctuated transcript (#2284). Synthetic text shaped like the prod defect:
lowercase, no sentence ends, from the first word to the last."""

from __future__ import annotations

import pytest

from podcast_scraper.transcription.punctuation import (
    echoes_prompt,
    is_unpunctuated,
    prompt_suits_language,
    PUNCTUATION_PROMPT,
    sentence_ends_per_1000_words,
)

pytestmark = pytest.mark.unit

UNPUNCTUATED = "so the ports moved north over a decade and the trading families followed " * 30
PUNCTUATED = (
    "So the ports moved north over a decade. Why? The delta silted up, and trade moved. " * 25
)


def test_the_prod_shape_is_detected() -> None:
    assert is_unpunctuated(UNPUNCTUATED)
    assert sentence_ends_per_1000_words(UNPUNCTUATED) == 0.0


def test_a_punctuated_transcript_is_not() -> None:
    assert not is_unpunctuated(PUNCTUATED)


def test_sentences_glued_without_a_space_still_count() -> None:
    """Publisher transcripts that drop the space after a full stop ("factor.Now,") are punctuated;
    37 In Moscow's Shadows / Explaining Brazil transcripts were falsely flagged without this."""
    assert not is_unpunctuated("The ports moved north.Then the families followed.Why?Trade. " * 40)


def test_a_short_transcript_is_not_judged() -> None:
    assert not is_unpunctuated("so the ports moved north and the families followed " * 5)
    assert not is_unpunctuated("")
    assert not is_unpunctuated(None)


def test_capitalised_names_without_sentences_are_still_the_defect() -> None:
    """Whisper often keeps capitalising names and acronyms while dropping every full stop (5 of the
    121 prod cases: 3-10% capitals, 0-2 sentence ends) -- capitals must not hide the defect."""
    text = (
        "deep in Bolivia's southwest lies the Salar de Uyuni and the AI teams of Potosi went " * 25
    )
    assert is_unpunctuated(text)


def test_a_screenplay_transcript_is_judged_on_its_speech() -> None:
    """The prod transcripts are screenplays (``SPEAKER_07: ...``); the labels are not speech."""
    lines = "\n".join(
        f"SPEAKER_0{i % 3}: so the ports moved north over a decade and the families followed"
        for i in range(40)
    )
    assert is_unpunctuated(lines)


def test_a_script_without_latin_punctuation_is_not_judged() -> None:
    text = "港口向北移动了十年 贸易家族跟随着河水 " * 200
    assert not is_unpunctuated(text)


def test_a_prompt_echo_is_recognised() -> None:
    assert echoes_prompt(PUNCTUATION_PROMPT + " So the ports moved north.")
    assert not echoes_prompt(PUNCTUATED)


@pytest.mark.parametrize(
    "language, suits",
    [(None, True), ("en", True), ("en-GB", True), ("EN", True), ("de", False), ("fr-CA", False)],
)
def test_the_english_prompt_is_only_sent_for_english(language, suits) -> None:
    assert prompt_suits_language(language) is suits
