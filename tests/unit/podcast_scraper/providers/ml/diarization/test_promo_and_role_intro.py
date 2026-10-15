"""Two Gray Area naming errors (#2304, prod 2026-10-14).

A NEW promo campaign for another show was named as guests: the recurring-ad strategy learns a
promo only after it has run before. And a stand-in presenter who said "I'm Vox reporter and
producer Miles Bryan, sitting in for Sean today" stayed unnamed, because a role phrase stood
between "I'm" and the name.
"""

from __future__ import annotations

import pytest

from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment
from podcast_scraper.providers.ml.diarization.roster import (
    _ad_voices_for,
    _metadata_anchored_self_intro,
    _self_introduced_promo_voices,
)

pytestmark = pytest.mark.unit

PROMO = (
    "Hi, I'm Phoebe Judge, host of the podcast Criminal. This fall I'm headed out on the road, "
    "come see us in person."
)


def test_a_short_self_introduced_promo_is_an_ad_voice() -> None:
    assert _self_introduced_promo_voices({"S5": PROMO}, {"S5": 30.0}) == {"S5"}


@pytest.mark.parametrize(
    "text, talk",
    [
        ("Hi, I'm Max and I'm nine. Why do cats purr?", 12.0),  # a kid's question, no promo words
        ("Tickets are on sale now for our live tour in November.", 20.0),  # no self-introduction
        (PROMO, 600.0),  # a participant who also mentions a tour
    ],
)
def test_one_half_alone_is_not_a_promo(text, talk) -> None:
    assert _self_introduced_promo_voices({"S5": text}, {"S5": talk}) == set()


def test_the_introduced_name_must_be_capitalised() -> None:
    """ "I'm in this class" and "This is the fifth" are not introductions (MLST, NVIDIA)."""
    texts = {
        "S1": "So I'm in this class and I went to sign up for physics.",
        "S2": "This is the fifth of our episodes on the road to GTC live.",
    }
    assert _self_introduced_promo_voices(texts, {"S1": 45.0, "S2": 44.0}) == set()


def test_a_stated_person_promoting_this_show_is_not_an_ad() -> None:
    """The Journal's host: "I'm Lane Florsheim ... follow us wherever you get your podcasts"."""
    text = "I'm Lane Florsheim. Follow us on Spotify or wherever you get your podcasts."
    talk = {"S1": 86.0}
    assert _self_introduced_promo_voices({"S1": text}, talk) == {"S1"}
    assert _self_introduced_promo_voices({"S1": text}, talk, ["Lane Florsheim"]) == set()
    assert _self_introduced_promo_voices({"S1": text}, talk, ["Lane"]) == {"S1"}  # full names only


def test_saying_the_hosts_name_is_not_being_the_host() -> None:
    # A supporter shout-out names the show's host in the third person; it is not the show's own
    # person promoting the show (StarTalk, prod snapshot 2026-10-15: 3 shout-outs named as guests).
    text = (
        "I'm Dana Okafor, and I support the show on Patreon. This is Night Sky with Ada Brightwell."
    )
    assert _self_introduced_promo_voices({"S0": text}, {"S0": 8.2}, ["Ada Brightwell"]) == {"S0"}


def test_a_stated_person_introducing_themselves_in_the_asr_spelling_is_still_this_show() -> None:
    text = "I'm Lane Florshime. Follow us on Spotify or wherever you get your podcasts."
    assert _self_introduced_promo_voices({"S1": text}, {"S1": 86.0}, ["Lane Florsheim"]) == set()


def test_a_three_word_stated_name_is_read_whole() -> None:
    text = "I'm Ana Ruiz Okafor. Subscribe now wherever you get your podcasts."
    assert _self_introduced_promo_voices({"S1": text}, {"S1": 40.0}, ["Ana Ruiz Okafor"]) == set()


def test_a_cue_split_across_two_segments_is_still_a_cue() -> None:
    # Joined segments leave two spaces at the seam: "support the show on  Patreon" read as no
    # promo cue, and a 10 s supporter shout-out was named as a guest (prod snapshot 2026-10-15).
    text = " Hi, I'm Dana Okafor from Ohio.  We listen every night and support the show on  Patreon"
    assert _self_introduced_promo_voices({"S0": text}, {"S0": 9.7}) == {"S0"}


def test_the_boundary_is_strict() -> None:
    assert _self_introduced_promo_voices({"S5": PROMO}, {"S5": 90.0}) == set()


def test_the_promo_voice_joins_the_episode_ad_voices() -> None:
    """Mid-episode, so the edge rule does not catch it, and no recurring text exists yet."""
    segs = [
        DiarizationSegment(start=0.0, end=600.0, speaker="HOST"),
        DiarizationSegment(start=600.0, end=630.0, speaker="PROMO"),
        DiarizationSegment(start=630.0, end=1800.0, speaker="HOST"),
    ]
    diar = DiarizationResult(segments=segs, num_speakers=2)
    texts = {"HOST": "Welcome back to the show. Let's get into it.", "PROMO": PROMO}
    assert "PROMO" in _ad_voices_for(diar, None, texts, None, None)
    assert "HOST" not in _ad_voices_for(diar, None, texts, None, None)
    assert "PROMO" not in _ad_voices_for(diar, None, texts, None, None, ["Phoebe Judge"])


STATED = ["Miles Bryan", "Sean Illing", "Jane Doe"]


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "Welcome to the Gray Area. I'm Vox reporter and producer Miles Bryan, sitting in for "
            "Sean today.",
            "Miles Bryan",
        ),
        ("I'm your guest host Jane Doe.", "Jane Doe"),
        ("I'm joined by producer Jane Doe.", None),
        ("I'm here with reporter Jane Doe.", None),
        ("I'm your host and today's guest is novelist Jane Doe.", None),
        ("I'm a reporter and Miles Bryan is a producer.", None),
        ("I'm a big fan of Jane Doe.", None),
        ("I'm voting Jane Doe.", None),  # no role in the gap
        (
            "I'm the one person who reads every word of every page here in this office, Jane Doe.",
            None,
        ),
        ("I'm Vox producer Miles Brian.", None),  # no fuzzy surname through a gap
        ("I'm sure every producer in this business knows Jane Doe.", None),  # gap too long
        ("I'm Vox reporter Jane.", None),  # a first name alone
    ],
)
def test_a_role_phrase_before_a_stated_name(text, expected) -> None:
    assert _metadata_anchored_self_intro(text, STATED) == expected


def test_an_unstated_name_is_never_bound() -> None:
    assert _metadata_anchored_self_intro("I'm Vox producer Ana Ruiz.", STATED) is None


def test_only_english_reads_the_role_phrase() -> None:
    text = "I'm Vox reporter and producer Miles Bryan, sitting in for Sean today."
    assert _metadata_anchored_self_intro(text, STATED, language="de") is None
