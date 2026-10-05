"""Every spelling of an English tag takes the English branch, wherever a language is compared.

The 2026-10-05 English-path audit found the same bug in sixteen places: a tag reduced by
``.split("-")[0]`` turns ``en_US`` / ``English`` / ``eng`` into a code no map has, so an English
episode carrying such a tag read as non-English. They all go through
``languages.primary_language`` now; this pins the decision points a raw tag can reach.
"""

from __future__ import annotations

import pytest

from podcast_scraper.languages_guard import is_target_language
from podcast_scraper.speaker_detectors import naming_vocabulary
from podcast_scraper.speaker_detectors.constants import interview_cue_patterns_for
from podcast_scraper.transcription.punctuation import prompt_suits_language

pytestmark = pytest.mark.unit

ENGLISH_TAGS = ["en", "EN", "en-US", "en_US", "en_GB", "English", "english", "eng"]


@pytest.mark.parametrize("tag", ENGLISH_TAGS)
def test_english_only_nlp_runs(tag: str) -> None:
    assert is_target_language(tag) is True


@pytest.mark.parametrize("tag", ENGLISH_TAGS)
def test_the_english_whisper_prompt_is_kept(tag: str) -> None:
    assert prompt_suits_language(tag) is True


@pytest.mark.parametrize("tag", ENGLISH_TAGS)
def test_the_english_interview_cues_load(tag: str) -> None:
    assert interview_cue_patterns_for(tag) == interview_cue_patterns_for("en")
    assert interview_cue_patterns_for(tag) is not None


@pytest.mark.parametrize("tag", ENGLISH_TAGS)
def test_the_english_vocabulary_row_is_read(tag: str) -> None:
    rows = naming_vocabulary.SELF_INTRO_WORDS
    assert naming_vocabulary.vocabulary_row(rows, tag) == rows["en"]


@pytest.mark.parametrize("tag", ["es", "es-ES", "Spanish", "spa"])
def test_another_language_still_takes_its_own_branch(tag: str) -> None:
    assert is_target_language(tag) is False
    assert prompt_suits_language(tag) is False
    rows = naming_vocabulary.SELF_INTRO_WORDS
    assert naming_vocabulary.vocabulary_row(rows, tag) == rows["es"]
