"""An episode whose feed-listed hosts are all ABSENT still has whoever ran the interview (#2061).

THE EPISODE. WSJ's "The Journal." states two hosts in its feed description — Ryan Knutson and
Jessica Mendoza. Its "My Monday Morning" strand is presented by a different reporter. On
"My Monday Morning: Twiggy" the roster produced:

    SPEAKER_01  Lane Florsheim  role=guest  source=self_intro     <- the interviewer
    SPEAKER_02  Twiggy          role=guest  source=self_intro     <- the actual guest
    known_hosts: [Jessica Mendoza, Ryan Knutson]                  <- neither is in the episode

Nobody hosts. Both people are guests. Downstream that is an episode with no host and two guests,
and #2062's roster-authoritative graph faithfully publishes exactly that.

WHY THE EXISTING SIGNALS ALL DECLINE. `_select_host_voices` runs five ordered steps:

  1. a voice that self-introduces as a STATED host — neither stated host is present;
  2. a voice that PERFORMS the host role (`conv_hosts`, from speech acts) — BLOCKED, see below;
  3. the opener does the intro — skipped whenever `conv_hosts` is non-empty;
  4. fill a counted-but-unmatched slot from intro voices — blocked by the same set as 2;
  5. anchor a show that states NO hosts — this show states two.

Step 2's guard is `stated_non_host_voices`: a voice that says a name absent from the feed's host
pool is treated as evidence it is not a host. That guard is right about what it was built for — it
stops a GUEST filling a vacant host seat when a stated host is merely absent (No Priors: Andy Fang
over absent Sarah Guo). But "not one of the feed's hosts" is not "not hosting this episode", and
applied to a stand-in interviewer it leaves the episode with no host at all.

THE RULE THIS PINS. When every other signal has declined and NO host has been seated, a voice that
PERFORMS host speech acts, is not heard as a guest, is not an ad, and is NAMED may host. Behaviour
outranks a feed-level roster that does not describe this episode. It is last-resort by
construction, so an episode whose stated host did turn up is untouched.
"""

from __future__ import annotations

import pytest

from podcast_scraper.providers.ml.diarization.roster import _select_host_voices

pytestmark = pytest.mark.unit


class _Diar:
    """Minimal stand-in: the opener probe only reads ``segments``."""

    def __init__(self, segments=None):
        self.segments = segments or []


def _call(**over):
    kw = dict(
        diarization=_Diar(),
        voice_intro={"SPEAKER_01": "Lane Florsheim", "SPEAKER_02": "Twiggy"},
        host_pool=[("Ryan Knutson", "feed"), ("Jessica Mendoza", "feed")],
        known_hosts=["Ryan Knutson", "Jessica Mendoza"],
        conv_hosts=["SPEAKER_01"],
        conv_guests={"SPEAKER_02"},
        voices_by_intro=["SPEAKER_01", "SPEAKER_02"],
        llm_named=set(),
        llm_voice_roles=None,
        content_start=0.0,
        intro_window_s=120.0,
        ad_intervals=None,
        ad_voices=set(),
    )
    kw.update(over)
    return _select_host_voices(**kw)


class TestTheStandInInterviewerIsSeated:
    def test_the_interviewer_becomes_a_host(self) -> None:
        assert "SPEAKER_01" in _call(), "the episode was left with no host at all"

    def test_the_guest_does_not_become_a_host(self) -> None:
        assert "SPEAKER_02" not in _call()

    def test_a_voice_heard_as_a_guest_never_seats(self) -> None:
        # "thanks for having me" is positive evidence against hosting, whatever else is true.
        got = _call(conv_hosts=["SPEAKER_02"], conv_guests={"SPEAKER_02"})
        assert "SPEAKER_02" not in got

    def test_an_unnamed_performing_voice_still_seats_via_the_existing_step(self) -> None:
        # NOT a new restriction: a voice that performs host acts and never stated a name was
        # already seatable by step 2 (it is not in `stated_non_host_voices` at all). Pinned here so
        # the new last-resort step is not mistaken for having tightened it.
        got = _call(voice_intro={"SPEAKER_02": "Twiggy"}, conv_hosts=["SPEAKER_01"])
        assert "SPEAKER_01" in got

    def test_an_ad_voice_never_seats(self) -> None:
        got = _call(ad_voices={"SPEAKER_01"})
        assert "SPEAKER_01" not in got


class TestItIsTrulyLastResort:
    def test_a_stated_host_who_turned_up_is_still_the_host(self) -> None:
        # The ordinary episode: Ryan self-introduces. Nothing about it may change.
        got = _call(
            voice_intro={"SPEAKER_00": "Ryan Knutson", "SPEAKER_02": "Twiggy"},
            conv_hosts=["SPEAKER_00"],
            voices_by_intro=["SPEAKER_00", "SPEAKER_02"],
        )
        assert got and got[0] == "SPEAKER_00"
        assert "SPEAKER_02" not in got

    def test_a_guest_does_not_take_a_vacant_seat_without_host_behaviour(self) -> None:
        # The No Priors case the original guard exists for: the stated host is absent and the
        # guest performs NO host speech acts, so nobody is seated rather than the guest.
        got = _call(conv_hosts=[], conv_guests={"SPEAKER_02"})
        assert "SPEAKER_02" not in got
