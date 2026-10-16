"""A single ad cue near the top cuts the read it marks, not the talk before it.

Read on the prod snapshot of 2026-10-15 (every stretch the cut changes, all 2,685 served episodes):
the first block of an episode was cut as a pre-roll from one cue alone, and how far back it reached
came from rules written for pre-roll STACKS.

* An opener ("brought to you by") within ``PREROLL_LEADIN`` cut from 0, and the cold open before it
  went too (The Rest Is Science, "YouTube economics now").
* A lone call to action past the lead-in reached ``PREROLL_PITCH`` chars back for its pitch, and
  took the guest's introduction before "check out lennysproductpass.com" (14 stretches of Lenny's
  Podcast) and a founder's account of his own product before "go to dreamer.com".

The replay of this change keeps 22 stretches of talk and lets 3 ads back in (a pitch whose call to
action is still cut, two German-language reads that were only cut because the cut ran from 0).
Talk that shares an ASR segment with a cut sentence still goes with it: the cut drops segments.
"""

from __future__ import annotations

import pytest

from podcast_scraper.gi.ad_regions import detect_ad_cut_ranges, PREROLL_LEADIN, PREROLL_PITCH

pytestmark = pytest.mark.unit

TALK = "We went to the lab and looked at how the eye sees colour in the dark. " * 60


def _cut(text: str) -> str:
    kept, last = [], 0
    for lo, hi in detect_ad_cut_ranges(text, threshold=1):
        kept.append(text[last:lo])
        last = hi
    return "".join(kept) + text[last:]


def test_a_cold_open_before_an_opener_is_kept() -> None:
    cold_open = "Welcome to the show. Today I brought an object from my childhood to test. "
    assert len(cold_open) < PREROLL_LEADIN
    ad = "This episode is brought to you by Acme. "
    out = _cut(cold_open + ad + TALK)
    assert out.startswith(cold_open)
    assert "brought to you by" not in out


def test_an_opener_on_the_first_line_is_cut_from_the_top() -> None:
    out = _cut("SPEAKER_00: This episode is brought to you by Acme.\n" + TALK)
    assert out.startswith("We went to the lab")


def test_a_lone_call_to_action_past_the_lead_in_takes_only_its_sentence() -> None:
    opening = "My guest today was an early investor in many companies and ran a large team. " * 25
    assert len(opening) > PREROLL_LEADIN + PREROLL_PITCH
    cta = "Check out acmepass.com for deals for subscribers. "
    out = _cut(opening + cta + TALK)
    assert out.startswith(opening)
    assert "acmepass" not in out


def test_a_call_to_action_at_the_top_still_takes_its_pitch() -> None:
    pitch = "Our games are the best way to start your day, with friends. "
    out = _cut(pitch + "Subscribe now at acme.com slash games for an offer. " + TALK)
    assert out.startswith("We went to the lab")


def test_a_read_with_two_kinds_of_cue_still_takes_its_pitch() -> None:
    opening = "My guest today was an early investor in many companies and ran a large team. " * 25
    pitch = "Acme makes your banking simple and your books clean every single month. "
    read = "Go to acme.com and use code SHOW for 20% off. "
    out = _cut(opening + pitch + read + TALK)
    assert "Acme makes your banking" not in out and "use code" not in out
