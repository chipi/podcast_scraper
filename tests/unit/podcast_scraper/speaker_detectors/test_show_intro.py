"""A voice that presents THIS show by name ("you're listening to <show>") is presenting it.

Each test names one behaviour of `show_name_pattern` (the show's name as a voice says it) or
`performs_show_intro` (the presenting formula plus that name). Transcript lines are invented.
"""

from __future__ import annotations

from typing import Optional

import pytest

from podcast_scraper.speaker_detectors.hosts import performs_show_intro, show_name_pattern

pytestmark = pytest.mark.unit


def test_subtitle_after_a_colon_is_not_part_of_the_name() -> None:
    """A title's subtitle is never said aloud, so the name stops at the colon."""
    assert performs_show_intro("Welcome to No Priors.", "No Priors: AI | Tech")


def test_a_pipe_subtitle_is_dropped() -> None:
    """A subtitle after a pipe is dropped like one after a colon."""
    assert performs_show_intro("This is Quiet Harbour.", "Quiet Harbour | Stories from the coast")


def test_a_parenthetical_tag_is_not_part_of_the_name() -> None:
    """A parenthesised acronym is not said aloud, so the full name alone matches."""
    title = "Machine Learning Street Talk (MLST)"
    assert performs_show_intro("Welcome to Machine Learning Street Talk.", title)


def test_the_first_words_of_a_long_title_do_not_stand_for_it() -> None:
    """Ordinary speech that merely starts like the title is not the show's name."""
    title = "Machine Learning Street Talk (MLST)"
    assert not performs_show_intro("this is machine learning in the wild", title)


def test_a_with_host_suffix_is_dropped() -> None:
    """A trailing "with <Host>" names the presenter, not the show."""
    assert performs_show_intro("Welcome to Lantern Hour.", "Lantern Hour with Dov Pell")


def test_a_trailing_podcast_word_is_optional() -> None:
    """The title says "Podcast"; the voice may or may not."""
    assert performs_show_intro("Welcome to Lantern Hour.", "Lantern Hour Podcast")
    assert performs_show_intro("Welcome to the Lantern Hour podcast.", "Lantern Hour Podcast")


def test_a_leading_article_is_optional() -> None:
    """A title's leading "The" is not required of the voice."""
    assert performs_show_intro("You're listening to Lantern Hour.", "The Lantern Hour")


def test_asr_joined_tokens_match_a_two_token_prefix_that_ends_the_clause() -> None:
    """The ASR writes "Roundtable" for "Round Table"; that prefix ending the clause is the show."""
    assert performs_show_intro("this is Roundtable.", "Round Table China")


def test_asr_joined_initialism_matches_the_whole_title() -> None:
    """The ASR writes "NNG" for "NN/G"."""
    assert performs_show_intro("This is the NNG UX Podcast", "NN/G UX Podcast")


def test_a_two_token_prefix_must_end_the_clause() -> None:
    """ "this is in our view" is not "In Our Time": the prefix runs on into the sentence."""
    assert not performs_show_intro("this is in our view a mistake", "In Our Time")


def test_a_short_two_token_prefix_ending_the_clause_is_not_enough() -> None:
    """ "this is in our." ends the clause, but two short words are too little to be the show."""
    assert not performs_show_intro("this is in our.", "In Our Time")


def test_a_short_one_word_title_has_no_pattern() -> None:
    """ "The Daily" is five letters: "this is daily" is ordinary speech, so no pattern at all."""
    assert show_name_pattern("The Daily") is None


def test_a_long_one_word_title_matches() -> None:
    """A distinctive one-word title with a question mark still matches."""
    assert performs_show_intro("today on Unbelievable", "Unbelievable?")


def test_no_title_has_no_pattern() -> None:
    """Without a feed title there is nothing to compare."""
    assert show_name_pattern(None) is None
    assert show_name_pattern("") is None
    assert not performs_show_intro("welcome to anything", None)


def test_credits_are_not_presenting() -> None:
    """ "This episode of <show> was produced by" is the credits, said by anyone."""
    assert not performs_show_intro(
        "This episode of Planet Money was produced by Ana Roy.", "Planet Money"
    )


def test_a_plug_for_the_show_is_not_presenting() -> None:
    """A guest's "listen to this episode of <show>" is a plug, not the show's own intro."""
    assert not performs_show_intro(
        "you should listen to this episode of MLST", "MLST Weekly Podcast"
    )


def test_a_greeting_followed_by_the_show_name_is_presenting() -> None:
    """ "Hello, <show> episode 276" opens the episode."""
    assert performs_show_intro("Hello, Turkey Book Talk episode 276", "Turkey Book Talk")


def test_a_greeting_to_the_listeners_is_presenting() -> None:
    """ "Hi, <show> listeners" greets the audience of this show."""
    assert performs_show_intro("Hi, Planet Money listeners", "Planet Money")


def test_a_guest_thanking_the_host_for_having_them_is_not_presenting() -> None:
    """ "Thanks for having me on <show>" names the show without presenting it."""
    assert not performs_show_intro("Thanks for having me on Hard Fork", "Hard Fork")


def test_another_shows_name_is_not_this_show() -> None:
    """A promo for ANOTHER show after the same formula does not count."""
    assert not performs_show_intro("You're listening to Lantern Hour.", "Quiet Harbour")


def test_empty_text_is_not_presenting() -> None:
    """No words, no presenting."""
    empty: Optional[str] = None
    assert not performs_show_intro(empty, "Lantern Hour")
    assert not performs_show_intro("", "Lantern Hour")
