"""An opener takes a SECOND host seat only when it presents (#2304 Phase C, prod 2026-10-13).

On two-host shows the opener rule seated the guest whose cluster carried the network's merged ad
tag, or a guest's cold-open soundbite; the real host openers name the show or the guest.
"""

from __future__ import annotations

import pytest

from podcast_scraper.providers.ml.diarization.roster import _opener_presents

pytestmark = pytest.mark.unit

FEED = "Market Hour"
STATED = ["Jane Example"]


def _turns(*opener_lines: str):
    out = [("SPEAKER_01", "Welcome, I'm the other host.")]
    out += [("SPEAKER_00", line) for line in opener_lines]
    return out


@pytest.mark.parametrize(
    "lines",
    [
        ("Listeners, do you love the Market Hour podcast?", "Come see it live."),
        ("Today we are lucky to have with us Jane Example.", "She runs a bakery."),
        ("Okay.", "Right.", "Anyone who listens to market hour knows this."),  # third turn
    ],
)
def test_an_opener_naming_the_show_or_the_stated_guest_presents(lines) -> None:
    assert _opener_presents("SPEAKER_00", _turns(*lines), FEED, STATED)


@pytest.mark.parametrize(
    "lines",
    [
        ("This network podcast is supported by ads.", "Yes, so the cell is old."),
        ("It's not just a force of nature.", "Capitalism is made."),
        ("Okay.", "Right.", "Sure.", "Welcome to Market Hour."),  # only after the first 3 turns
        ("Jane said that already.",),  # a first name is not the stated full name
    ],
)
def test_an_ad_tag_or_a_soundbite_does_not(lines) -> None:
    assert not _opener_presents("SPEAKER_00", _turns(*lines), FEED, STATED)


def test_a_one_word_stated_name_is_ordinary_speech() -> None:
    """Metadata can list a mononym ("Guy"); "you guy" in a soundbite is not an introduction."""
    turns = _turns("Come on, guy, which stone is real?")
    assert not _opener_presents("SPEAKER_00", turns, FEED, ["Guy"])


def test_other_voices_and_missing_inputs_say_nothing() -> None:
    turns = [("SPEAKER_01", "Welcome to Market Hour with Jane Example."), ("SPEAKER_00", "Hi.")]
    assert not _opener_presents("SPEAKER_00", turns, FEED, STATED)
    assert not _opener_presents("SPEAKER_00", None, FEED, STATED)
    assert not _opener_presents("SPEAKER_00", _turns("Hello."), None, [])


def test_the_network_ad_tag_is_skipped_before_reading_the_opening() -> None:
    """BBC's tag is merged into the host's first turn; the presenting starts after it (Curious
    Cases, 2026-10-14)."""
    turns = _turns(
        "This BBC podcast is supported by ads outside the UK.",
        "Okay.",
        "Right.",
        "Welcome to Market Hour.",  # the 4th turn, the 3rd once the tag is skipped
    )
    assert _opener_presents("SPEAKER_00", turns, FEED, STATED)


def test_a_long_turn_that_mentions_ads_is_not_a_tag() -> None:
    line = (
        "This podcast is supported by advertising and we talk about that a lot because the whole "
        "industry runs on it, as does this one."
    )
    turns = _turns(line, "Okay.", "Right.", "Welcome to Market Hour.")  # 4th turn if not skipped
    assert not _opener_presents("SPEAKER_00", turns, FEED, STATED)


def test_handing_the_floor_to_a_stated_guest_by_first_name_presents() -> None:
    turns = _turns("First of all, Jane, can you explain what a phantom pain is?")
    assert _opener_presents("SPEAKER_00", turns, FEED, STATED)


def test_a_hail_at_the_start_of_a_turn_counts_as_turns_arrive() -> None:
    """Turns arrive with a leading space ("Going Viral", "Melvyn Bragg meets Misha Glenny")."""
    turns = [
        ("SPEAKER_01", " Welcome, I'm the other host."),
        ("SPEAKER_00", " This BBC podcast is supported by ads outside the UK."),
        ("SPEAKER_00", " Jane, let's start with the basics."),
    ]
    assert _opener_presents("SPEAKER_00", turns, FEED, STATED)
