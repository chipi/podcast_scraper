"""A job title on its own is not a speaker's name, at any length.

"General Partner" and "Founding Partner" were published as speakers on the DGX block of
2026-10-23: the publish gate checked job titles only from three words up. All fixtures are
synthetic.
"""

from __future__ import annotations

import pytest

from podcast_scraper.speaker_detectors.hosts import is_publishable_speaker_name

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "name",
    [
        "General Partner",
        "Founding Partner",
        "Senior Fellow",
        "Managing Partner",
        "Chief Executive",
        "Senior Editor",
        "Partner",
        "Lead Analyst",
    ],
)
def test_a_title_alone_is_refused(name: str) -> None:
    assert not is_publishable_speaker_name(name)


@pytest.mark.parametrize(
    "name",
    ["Tobias Partner", "Ada Fellow", "Mira Head", "Tobias Wren", "Ada Wren", "Leo Lead"],
)
def test_a_person_whose_name_holds_such_a_word_is_kept(name: str) -> None:
    assert is_publishable_speaker_name(name)


def test_a_non_english_feed_keeps_the_english_titles_refused() -> None:
    assert not is_publishable_speaker_name("General Partner", language="es")
