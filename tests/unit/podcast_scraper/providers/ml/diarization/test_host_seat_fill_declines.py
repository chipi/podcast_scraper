"""The vacant host seat (step 4 of `_select_host_voices`) is filled from the show's intro voices —
but never by the voice that OWNS the conversation while a guest is present, nor by a voice that
barely speaks.

Measured on prod (2026-10-02): with two stated hosts and only one of them in the room, step 4
seated the guest (66-77% of the talk) or a 19-46s ad, and the one-name-one-seat rule then put the
absent host's name on it. Each shape below is its own case: what must be declined, and what must
still be seated.

All fixtures are synthetic (never-commit-real-episodes).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import pytest

from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment
from podcast_scraper.providers.ml.diarization.roster import resolve_speaker_roster

pytestmark = pytest.mark.unit

HOST_A = "Tobias Wren"
HOST_B = "Greta Holm"


def _roster(turns: List[Tuple[str, str, float]], known_hosts: Sequence[str] = (HOST_A, HOST_B)):
    segs: List[DiarizationSegment] = []
    chunks: Dict[str, List[str]] = {}
    ordered: List[Tuple[str, str]] = []
    t = 0.0
    for spk, text, dur in turns:
        segs.append(DiarizationSegment(start=t, end=t + dur, speaker=spk))
        t += dur
        chunks.setdefault(spk, []).append(" " + text)
        ordered.append((spk, " " + text))
    voice_texts = {v: " ".join(c) for v, c in chunks.items()}
    return resolve_speaker_roster(
        DiarizationResult(segments=segs, num_speakers=len(voice_texts)),
        " ".join(x for _, x in ordered),
        known_hosts=list(known_hosts),
        voice_texts=voice_texts,
        ordered_turns=ordered,
    )


def _not_host_b(roster, voice: str) -> None:
    r = roster.by_voice[voice]
    assert r.name != HOST_B
    assert r.role != "host"


def test_a_dominant_guest_does_not_take_the_absent_hosts_seat() -> None:
    # Host A alone with a guest who owns the conversation, plus a third voice (a producer's
    # sign-off), so the episode is plainly not two hosts talking to each other.
    turns = [
        ("SPEAKER_00", f"Welcome to the show, I'm {HOST_A}.", 15.0),
        ("SPEAKER_01", "The river silted, so the ports moved north within a decade.", 60.0),
        ("SPEAKER_00", "And the merchants?", 10.0),
        ("SPEAKER_01", "They followed the trade, most of them within a generation.", 900.0),
        ("SPEAKER_00", "What did the crown make of that?", 60.0),
        ("SPEAKER_01", "It taxed whatever was left behind and called it reform.", 600.0),
        ("SPEAKER_00", "Another question on the guilds then.", 200.0),
        ("SPEAKER_02", "This episode was produced by the team, with music by the band.", 150.0),
    ]
    roster = _roster(turns)
    assert roster.by_voice["SPEAKER_00"].name == HOST_A
    _not_host_b(roster, "SPEAKER_01")


def test_with_only_two_voices_the_second_is_seated_even_when_it_dominates() -> None:
    # Two voices on a two-host show cannot be told apart from host + guest by talk share alone,
    # so the stated count wins. Declining needs a third voice in the room (case above).
    turns = [
        ("SPEAKER_00", f"Welcome to the show, I'm {HOST_A}.", 15.0),
        ("SPEAKER_01", "The river silted, so the ports moved north within a decade.", 60.0),
        ("SPEAKER_00", "And the merchants?", 200.0),
        ("SPEAKER_01", "They followed the trade, most of them within a generation.", 900.0),
    ]
    roster = _roster(turns)
    assert roster.by_voice["SPEAKER_01"].name == HOST_B


def test_two_hosts_without_a_guest_are_both_seated_though_one_talks_more_than_half() -> None:
    # Dominance only counts when someone BESIDES the stated hosts takes part.
    turns = [
        ("SPEAKER_00", f"Welcome to the show, I'm {HOST_A}.", 15.0),
        ("SPEAKER_01", "Right, so where do we start this week?", 40.0),
        ("SPEAKER_00", "So the bond market had a strange week.", 300.0),
        ("SPEAKER_01", "It did, yields moved the wrong way twice.", 700.0),
        ("SPEAKER_00", "Let's take that apart.", 200.0),
    ]
    roster = _roster(turns)
    assert roster.by_voice["SPEAKER_00"].name == HOST_A
    assert roster.by_voice["SPEAKER_01"].name == HOST_B
    assert roster.by_voice["SPEAKER_01"].role == "host"


def test_a_voice_under_five_percent_of_the_talk_does_not_take_a_seat() -> None:
    # A short promo early in the episode, not caught as an ad, outranks nobody for a host seat.
    turns = [
        ("SPEAKER_00", f"Welcome to the show, I'm {HOST_A}.", 15.0),
        ("SPEAKER_02", "Tickets for the summer tour are on sale now at the usual place.", 40.0),
        ("SPEAKER_01", "The ports moved because the river silted.", 20.0),
        ("SPEAKER_00", "And the merchants after that?", 400.0),
        ("SPEAKER_01", "Most of them followed the trade north.", 600.0),
    ]
    roster = _roster(turns)
    _not_host_b(roster, "SPEAKER_02")


def test_the_second_host_behind_a_dominant_guest_is_still_seated() -> None:
    # Both hosts present with a dominant guest who comes first in the intro: the guest is skipped,
    # the search goes ON to the second host (stopping there loses real co-hosts).
    turns = [
        ("SPEAKER_00", f"Welcome to the show, I'm {HOST_A}.", 10.0),
        ("SPEAKER_02", "Thanks, the short version is that the river silted.", 50.0),
        ("SPEAKER_01", "Let's start with the merchants, then.", 25.0),
        ("SPEAKER_02", "They followed the trade north within a generation.", 1300.0),
        ("SPEAKER_00", "And what did the crown make of it?", 250.0),
        ("SPEAKER_01", "And the guilds, where were they in all this?", 250.0),
        ("SPEAKER_02", "Mostly in the way.", 300.0),
    ]
    roster = _roster(turns)
    assert roster.by_voice["SPEAKER_00"].name == HOST_A
    assert roster.by_voice["SPEAKER_01"].name == HOST_B
    _not_host_b(roster, "SPEAKER_02")


def test_a_voice_under_five_percent_does_not_take_the_seat_as_the_opener() -> None:
    # Step 3 seats whoever speaks first. Once a pre-roll ad is recognised as an ad, the first voice
    # can be a 3-second fragment; the floor that guards step 4 guards the opener too.
    turns = [
        ("SPEAKER_02", "Okay, that sounds lovely.", 3.0),
        ("SPEAKER_00", "So the ports moved north because the river silted up.", 400.0),
        ("SPEAKER_01", "And the merchants went with them?", 300.0),
        ("SPEAKER_00", "Most of them, within a generation.", 100.0),
        ("SPEAKER_02", "Okay, that sounds lovely.", 3.0),
        ("SPEAKER_01", "That is a remarkable story about the trade.", 300.0),
    ]
    roster = _roster(turns, known_hosts=(HOST_A,))
    assert roster.by_voice["SPEAKER_02"].name != HOST_A
