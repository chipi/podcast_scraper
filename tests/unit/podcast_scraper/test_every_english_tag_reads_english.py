"""Every spelling of an English tag takes the English branch, wherever a language is compared.

The 2026-10-05 English-path audit found the same bug in sixteen places: a tag reduced by
``.split("-")[0]`` turns ``en_US`` / ``English`` / ``eng`` into a code no map has, so an English
episode carrying such a tag read as non-English. They all go through
``languages.primary_language`` now; this pins the decision points a raw tag can reach.
"""

from __future__ import annotations

import pytest

from podcast_scraper.languages_guard import is_target_language
from podcast_scraper.transcription.punctuation import prompt_suits_language

pytestmark = pytest.mark.unit

ENGLISH_TAGS = ["en", "EN", "en-US", "en_US", "en_GB", "English", "english", "eng"]


@pytest.mark.parametrize("tag", ENGLISH_TAGS)
def test_english_only_nlp_runs(tag: str) -> None:
    assert is_target_language(tag) is True


@pytest.mark.parametrize("tag", ENGLISH_TAGS)
def test_the_english_whisper_prompt_is_kept(tag: str) -> None:
    assert prompt_suits_language(tag) is True
