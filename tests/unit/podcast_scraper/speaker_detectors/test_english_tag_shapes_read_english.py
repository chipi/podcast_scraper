"""Every spelling of an English feed tag reads the ENGLISH naming rows — main's behaviour.

Main ignored the feed's ``<language>`` here and always used the English patterns. The multilingual
work keys the patterns by language, and looked the tag up by splitting on ``-`` only, so a feed
tagged ``en_US``, ``en_GB``, ``English`` or ``eng`` — tags ``normalize_language_tag`` already maps
to ``en``, so the language gate accepts the feed — got NO host-statement patterns and lost its
stated hosts (found 2026-10-05 in the English-path audit: ``["Jane Smith", "John Doe"]`` on main,
``[]`` on the branch). English must behave as main; this pins it for every shape.
"""

from __future__ import annotations

import pytest

from podcast_scraper.speaker_detectors.hosts import (
    _clean_stated_name,
    _first_name_presenters,
    detect_hosts_from_feed,
)

_ENGLISH_TAGS = ["en", "en-US", "EN", "en_US", "en_GB", "English", "english", "eng", " en-gb "]
_DESC = "The weekly show hosted by Jane Smith and John Doe."


@pytest.mark.parametrize("tag", _ENGLISH_TAGS)
def test_the_feed_statement_reads_english_for_every_english_tag(tag: str) -> None:
    assert detect_hosts_from_feed("Weekly Show", _DESC, None, None, language=tag) == {
        "Jane Smith",
        "John Doe",
    }


@pytest.mark.parametrize("tag", _ENGLISH_TAGS)
def test_a_stated_name_is_cleaned_as_english_for_every_english_tag(tag: str) -> None:
    assert _clean_stated_name("Bloomberg's Joe Weisenthal", tag) == _clean_stated_name(
        "Bloomberg's Joe Weisenthal", "en"
    )


@pytest.mark.parametrize("tag", _ENGLISH_TAGS)
def test_first_name_presenters_read_english_for_every_english_tag(tag: str) -> None:
    desc = "Join Jane and John every week as they talk markets."
    assert _first_name_presenters(desc, tag) == _first_name_presenters(desc, "en")
