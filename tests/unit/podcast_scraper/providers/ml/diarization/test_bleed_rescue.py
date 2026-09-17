"""A host whose cluster caught the guest's reply (#2075).

Diarization puts one exchange in one cluster: "So, Charles, welcome to the podcast. Thanks, KG.
I'm very happy to be here." The guest-act veto reads the whole cluster and refuses the host —
139 seats on the production snapshot, the bulk of this branch's single-host coverage loss.

THE TRAP THESE TESTS EXIST TO PIN: that bled pair reads identically whether it landed in the
host's cluster or the guest's. A rule that accepts "the seat performs a host act" cannot tell them
apart; one shipped (53df0349) and was reverted for putting `Gergely Orosz` on Elizabeth Stone's
voice. So the rescue rests only on evidence bleed cannot fabricate, and each test below is one of
the wrong seats that version published.
"""

from __future__ import annotations

import pytest

from podcast_scraper.providers.ml.diarization.roster import _rescued_from_bleed

pytestmark = pytest.mark.unit

_WELCOME = "So, Elizabeth, welcome to the podcast."
_REPLY = "Thank you. Thank you for having me."
_FILLER = "We talked about scale and culture for a while. " * 12  # ~560 chars


def _call(text, *, seat="S0", name="Gergely Orosz", intro=None, texts=None, share=None):
    return _rescued_from_bleed(
        seat,
        text,
        name,
        voice_intro=intro or {},
        voice_texts=texts or {},
        talk_share=share or {},
    )


def test_a_host_act_clear_of_the_reply_rescues_the_seat() -> None:
    """The host presents somewhere OTHER than the contested exchange, so the welcome is his."""
    text = f"{_WELCOME} {_REPLY} {_FILLER} And my guest today is Elizabeth Stone. {_FILLER}"
    assert _call(text, share={"S0": 0.4, "S1": 0.6}) is True


def test_only_the_bled_pair_is_not_evidence() -> None:
    """The whole of the seat's host evidence is one turn wide — exactly what bleed produces."""
    text = f"{_WELCOME} {_REPLY} {_FILLER}"
    assert _call(text, share={"S0": 0.4, "S1": 0.6}) is False


def test_the_dominant_voice_of_a_two_party_interview_is_declined() -> None:
    """Elizabeth Stone at 74%: a presenter does not own most of an interview. 11 of the 14 seats
    in this shape were the guest or a merged cluster."""
    text = f"{_WELCOME} {_REPLY} {_FILLER} And my guest today is Elizabeth Stone. {_FILLER}"
    assert _call(text, share={"S0": 0.74, "S1": 0.26}) is False


def test_a_dominant_voice_on_a_crowded_episode_is_not_declined() -> None:
    """The decline is about two-party interviews; a panel's most active voice is not the same
    claim, so the rule stays out of it."""
    text = f"{_WELCOME} {_REPLY} {_FILLER} And my guest today is Elizabeth Stone. {_FILLER}"
    share = {"S0": 0.55, "S1": 0.2, "S2": 0.15, "S3": 0.1}
    assert _call(text, share=share) is True


def test_a_name_another_voice_already_claimed_is_not_spare() -> None:
    """ "One spare name, one spare seat" is false when Kevin has already said he is Kevin — the
    arithmetic cannot see the self-introduction, so it is checked here."""
    text = f"{_WELCOME} {_REPLY} {_FILLER} And my guest today is Elizabeth Stone. {_FILLER}"
    assert (
        _call(
            text,
            name="Kevin Roose",
            intro={"S1": "Kevin Rooze"},
            share={"S0": 0.4, "S1": 0.6},
        )
        is False
    )


def test_a_wider_reply_form_still_counts_as_a_reply() -> None:
    """ "It's nice to be here" is not in the narrow veto list, and that is what let the Novo
    Nordisk CEO be published as a host of The Journal."""
    text = f"Mike, welcome to the journal. It's nice to be here. {_FILLER}"
    assert _call(text, name="Ryan Knutson", share={"S0": 0.4, "S1": 0.6}) is False


def test_a_seat_with_no_host_act_is_never_rescued() -> None:
    assert _call(f"Thanks for having me. {_FILLER}", share={"S0": 0.3}) is False


def test_empty_inputs_decline() -> None:
    assert _call("", share={"S0": 0.3}) is False
    assert _call(f"{_WELCOME} {_FILLER}", name="", share={"S0": 0.3}) is False
