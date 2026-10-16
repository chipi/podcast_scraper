"""A sponsor read is cut whole, and a show's own plug never takes the talk before it.

Read on the prod snapshot of 2026-10-15, every stretch that changes across all 2,685 served
episodes:

* An opener and its call to action clustered apart whenever the pitch between them ran past
  ``CLUSTER_GAP``: the opener alone is a lone cue (skipped mid-episode), the call to action started
  its own cut, and the pitch stayed — "This episode is brought to you by our season's presenting
  sponsor, WorkOS…" then ~800 chars of it before "Visit workos.com". Joined, 84 more stretches are
  cut, all of them sponsor reads (WorkOS, Mercury, Plaud, Meter, Sonar…).
* A two-cue block near the top reached ``PREROLL_PITCH`` chars back for a pitch; before a show's
  OWN plug there is none, and it took a Lenny's Podcast guest introduction, Colombia Calling's
  talk. Reaching back only to the sentence naming the call to action's domain keeps 8 such
  stretches and lets 6 pitches back in (the brand misheard: "Tire" for entire.io).
"""

from __future__ import annotations

import pytest

from podcast_scraper.gi.ad_regions import (
    CLUSTER_GAP,
    detect_ad_cut_ranges,
    PREROLL_LEADIN,
    READ_SPAN,
)

pytestmark = pytest.mark.unit

TALK = "We went to the lab and looked at how the eye sees colour in the dark. " * 80
PITCH = "Acme turns your whole stack into one clean dashboard that your team will love. " * 12


def _cut(text: str) -> str:
    kept, last = [], 0
    for lo, hi in detect_ad_cut_ranges(text, threshold=1):
        kept.append(text[last:lo])
        last = hi
    return "".join(kept) + text[last:]


def test_an_opener_and_its_call_to_action_far_apart_are_one_read() -> None:
    opener = "This episode is brought to you by Acme. "
    cta = "Visit acme.com slash show to learn more. "
    assert CLUSTER_GAP < len(PITCH) < READ_SPAN
    out = _cut(TALK + opener + PITCH + cta + TALK)
    assert "Acme" not in out and "acme" not in out
    assert out == TALK + TALK


def test_a_turn_change_between_them_keeps_them_apart() -> None:
    """The next speaker's talk is not the read, whatever follows it."""
    opener = "SPEAKER_00: This episode is brought to you by Acme.\n"
    guest = "SPEAKER_01: " + PITCH + "Visit acme.com slash show to learn more. "
    out = _cut(TALK + opener + guest + TALK)
    assert PITCH[:40] in out


def test_a_bare_sponsored_by_never_brackets_a_read() -> None:
    life = "We were sponsored by a church. "
    out = _cut(TALK + life + PITCH + "Visit acme.com slash show to learn more. " + TALK)
    assert PITCH[:40] in out


def test_a_show_plug_near_the_top_keeps_the_introduction_before_it() -> None:
    intro = "My guest today ran the largest team at a famous company for many years. " * 16
    plug = "Check out showpass.com for deals, and subscribe at showpodcast.com slash best. "
    assert len(intro) > PREROLL_LEADIN
    out = _cut(intro + plug + TALK)
    assert out.startswith(intro)
    assert "showpass" not in out


def test_a_pitch_that_names_its_product_is_still_taken_with_the_call_to_action() -> None:
    intro = "My guest today ran the largest team at a famous company for many years. " * 16
    pitch = "Acmeflow ships your features behind flags and measures them for you. "
    cta = "To get started, go to acmeflow.com slash show, use code SHOW. "
    out = _cut(intro + pitch + cta + TALK)
    assert out.startswith(intro) and "Acmeflow" not in out


def test_an_opener_and_a_call_to_action_past_the_read_span_stay_apart() -> None:
    far = "We went back to the archive and read the letters again, one by one. " * 40
    assert len(far) > READ_SPAN
    out = _cut(
        TALK + "This episode is brought to you by Acme. " + far + "Visit acme.com slash show. "
    )
    assert far[:40] in out


def test_a_call_to_action_naming_no_domain_does_not_reach_back() -> None:
    intro = "My guest today ran the largest team at a famous company for many years. " * 16
    out = _cut(intro + "Use code SHOW for 20% off your first order. " + TALK)
    assert out.startswith(intro) and "code SHOW" not in out
