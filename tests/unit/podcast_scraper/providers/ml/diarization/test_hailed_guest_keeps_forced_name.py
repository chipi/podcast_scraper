"""A forced guest name survives a surname mention when the other side hails that guest (#2304).

The third-person guard refuses a forced guest name on a voice that says the name -- it was written
for the host's voice wearing the guest's name. It also fired on the guest's own voice: on The Living
Philosophy "#10 Michael Montgomery" the host opens "Hi, Michael. How are you?", the next turn is the
guest's, and the guest was refused for mentioning his domain, "Dr. Montgomery".
"""

from __future__ import annotations

from podcast_scraper.providers.ml.diarization.roster import _refuted_unless_hailed

NAME = "Michael Montgomery"
GUEST = "Hi. Great to be here. But the fact that I got Dr. Montgomery as a domain was amazing."
HOST = "Hi, Michael. How are you? I believe you're the first Irishman I've had on the show."


def _turns(*pairs):
    return list(pairs)


class TestHailedGuestKeepsTheName:
    def test_hailed_by_the_host_and_answering_next(self) -> None:
        turns = _turns(("S0", HOST), ("S1", GUEST))
        assert not _refuted_unless_hailed("S1", NAME, {"S0": HOST, "S1": GUEST}, turns)

    def test_never_mentioning_the_name_is_not_refuted_either(self) -> None:
        texts = {"S0": HOST, "S1": "Hi. Great to be here."}
        assert not _refuted_unless_hailed("S1", NAME, texts, _turns(("S0", HOST), ("S1", "Hi.")))


class TestTheGuardStillHolds:
    def test_no_hail_by_anyone(self) -> None:
        texts = {"S0": "Welcome to the show.", "S1": GUEST}
        turns = _turns(("S0", "Welcome to the show."), ("S1", GUEST))
        assert _refuted_unless_hailed("S1", NAME, texts, turns)

    def test_the_voice_says_the_full_name(self) -> None:
        # The host introducing the guest: the case the guard was written for.
        intro = "Michael Montgomery is a psychotherapist. Hi, Michael."
        texts = {"S0": intro, "S1": "Hi, Michael. Good to see you."}
        turns = _turns(("S1", "Hi, Michael. Good to see you."), ("S0", intro))
        assert _refuted_unless_hailed("S0", NAME, texts, turns)

    def test_hailed_but_the_voice_says_the_full_name_in_the_third_person(self) -> None:
        # Being hailed does not outweigh speaking the WHOLE name about someone: that is not a
        # surname slip, it is talking about a person.
        said = "Thanks. As Michael Montgomery wrote in his book, fear is old."
        texts = {"S0": HOST, "S1": said}
        turns = _turns(("S0", HOST), ("S1", said))
        assert _refuted_unless_hailed("S1", NAME, texts, turns)

    def test_the_voice_hails_that_first_name_itself(self) -> None:
        own = "Well, Michael, thank you. Montgomery is a fine name."
        texts = {"S0": HOST, "S1": own}
        turns = _turns(("S0", HOST), ("S1", own))
        assert _refuted_unless_hailed("S1", NAME, texts, turns)

    def test_hailed_but_someone_else_answers(self) -> None:
        texts = {"S0": HOST, "S1": GUEST, "S2": "Yes, hello."}
        turns = _turns(("S0", HOST), ("S2", "Yes, hello."), ("S1", GUEST))
        assert _refuted_unless_hailed("S1", NAME, texts, turns)

    def test_no_turn_order_means_no_rescue(self) -> None:
        assert _refuted_unless_hailed("S1", NAME, {"S0": HOST, "S1": GUEST}, None)
