"""The episode description names the interviewer; read it (#2061).

FOUND ON A FRESH DGX INGEST of WSJ's "The Journal." (2026-09-13). The episode "My Monday Morning:
Twiggy" came out of the pipeline with BOTH people filed as guests and no host:

    kg.json:  guest  Lane Florsheim      <- the interviewer
              guest  Twiggy              <- the actual guest

The role was never missing. It is stated in the episode's own description:

    "In My Monday Morning, a new Journal video series ..., LANE FLORSHEIM SITS DOWN WITH TWIGGY
     about her career and life."

``sits down with`` is already a recognised interview cue — but only to confirm that the name AFTER
it is a guest. The name BEFORE it, the person doing the interviewing, was never read, so a stand-in
interviewer who is not on the feed's regular-host list ("Hosted by Ryan Knutson and Jessica
Mendoza") could only come out as a guest.

WHY THE FEED-LEVEL LIST CANNOT COVER THIS. ``known_hosts`` is parsed from the FEED description and
is therefore the same for every episode of the show. A series that rotates interviewers, a guest
host, or a spin-off strand inside a main feed — exactly this case — has an episode host who is not
a show host. The episode description is the only place that distinguishes them, and it says so in
plain language.
"""

from __future__ import annotations

import pytest

from podcast_scraper.speaker_detectors.guests import interviewers_in_text

pytestmark = pytest.mark.unit

REAL_DESCRIPTION = (
    "Twiggy is one of the most recognizable supermodels of the 20th century. Now the documentary, "
    "“Twiggy” about her career and life is being released in the U.S. In My Monday "
    "Morning, a new Journal video series inspired by the longstanding column of the same name, "
    "Lane Florsheim sits down with Twiggy about her career, her life and her Monday mornings."
)


class TestTheRealEpisodeThatExposedThis:
    def test_the_interviewer_is_identified(self) -> None:
        assert "Lane Florsheim" in interviewers_in_text(
            REAL_DESCRIPTION, ["Lane Florsheim", "Twiggy"]
        )

    def test_the_guest_is_not_called_an_interviewer(self) -> None:
        # Twiggy is on the RIGHT of the cue — she is being interviewed, not interviewing.
        assert "Twiggy" not in interviewers_in_text(REAL_DESCRIPTION, ["Lane Florsheim", "Twiggy"])


class TestTheCueForms:
    @pytest.mark.parametrize(
        "text",
        [
            "Alice Smith sits down with Bob Jones to discuss the merger.",
            "Alice Smith sat down with Bob Jones last week.",
            "Alice Smith interviews Bob Jones about the deal.",
            "Alice Smith speaks with Bob Jones on the record.",
            "Alice Smith talks to Bob Jones about the filing.",
            "Alice Smith chats with Bob Jones.",
            "Alice Smith is joined by Bob Jones.",
            "Alice Smith welcomes Bob Jones to the show.",
            "Alice Smith in conversation with Bob Jones.",
        ],
    )
    def test_the_left_hand_name_is_the_interviewer(self, text: str) -> None:
        found = interviewers_in_text(text, ["Alice Smith", "Bob Jones"])
        assert found == ["Alice Smith"], f"{text!r} -> {found!r}"


class TestItStaysQuietWithoutEvidence:
    def test_no_cue_means_no_interviewer(self) -> None:
        # A description that merely names people must not manufacture a host.
        text = "Alice Smith and Bob Jones both appear in the documentary."
        assert interviewers_in_text(text, ["Alice Smith", "Bob Jones"]) == []

    def test_a_name_after_the_cue_only_is_not_an_interviewer(self) -> None:
        text = "This week we are joined by Bob Jones."
        assert interviewers_in_text(text, ["Bob Jones"]) == []

    def test_a_name_far_from_the_cue_is_not_an_interviewer(self) -> None:
        # Adjacency is the whole signal; a name a paragraph away is not the subject of the verb.
        text = (
            "Alice Smith wrote the book. "
            + ("Filler sentence here. " * 12)
            + "Our reporter sits down with Bob Jones."
        )
        assert "Alice Smith" not in interviewers_in_text(text, ["Alice Smith", "Bob Jones"])

    def test_empty_inputs_are_safe(self) -> None:
        assert interviewers_in_text("", ["Alice Smith"]) == []
        assert interviewers_in_text(REAL_DESCRIPTION, []) == []
        assert interviewers_in_text(None, ["Alice Smith"]) == []


class TestOrderAndHygiene:
    def test_first_appearance_order_is_preserved(self) -> None:
        text = "Alice Smith sits down with Carol White. Later Bob Jones interviews Dan Brown."
        found = interviewers_in_text(text, ["Bob Jones", "Alice Smith", "Carol White", "Dan Brown"])
        assert found == ["Alice Smith", "Bob Jones"]

    def test_matching_is_case_insensitive(self) -> None:
        assert interviewers_in_text("alice smith sits down with bob jones.", ["Alice Smith"]) == [
            "Alice Smith"
        ]
