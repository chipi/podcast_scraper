"""F7 (The Rest Is History): the storyteller host is seated, and two seats are named by address.

Measured 2026-10-14 on prod (2,664 served episodes, replayed back to back against main): +10 named
voices, 0 episodes losing a name; gold dev_v1 2 better / 0 worse, val_v1 1 / 0. All fixtures are
synthetic.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import pytest

from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment
from podcast_scraper.providers.ml.diarization.roster import (
    _pair_by_dominant_address,
    resolve_speaker_roster,
)

pytestmark = pytest.mark.unit

ANN, BEN = "Ann Rivers", "Ben Stone"


def _talk(ann: int, ben: int) -> str:
    return " ".join(["Well, Ann, that is the story."] * ann + ["Right, Ben, go on."] * ben)


def test_two_seats_that_mostly_hail_different_hosts_are_named() -> None:
    """The storyteller carries one stray hail of the other name; the strict pair refuses it."""
    texts = {"S_BIG": _talk(ann=0, ben=0) + " " + _talk(ann=6, ben=1), "S_SMALL": _talk(0, 3)}
    assert _pair_by_dominant_address(texts, ["S_BIG", "S_SMALL"], [ANN, BEN]) == {
        "S_BIG": BEN,
        "S_SMALL": ANN,
    }


@pytest.mark.parametrize(
    "big, small",
    [
        ((6, 1), (5, 0)),  # both hail Ann: both cannot be Ben
        ((6, 1), (1, 1)),  # the small seat says nothing decisive
        ((2, 0), (0, 3)),  # below the floor of 3 hails
    ],
)
def test_undecided_or_contradicting_seats_name_nobody(big, small) -> None:
    texts = {"S_BIG": _talk(*big), "S_SMALL": _talk(*small)}
    assert _pair_by_dominant_address(texts, ["S_BIG", "S_SMALL"], [ANN, BEN]) is None


def test_only_two_seats() -> None:
    assert _pair_by_dominant_address({"A": _talk(6, 0)}, ["A"], [ANN, BEN]) is None


def _roster(turns: List[Tuple[str, str, float]], metadata_named=(), host_copresence=None):
    segs: List[DiarizationSegment] = []
    texts: Dict[str, str] = {}
    ordered: List[Tuple[str, str]] = []
    t = 0.0
    for spk, text, dur in turns:
        segs.append(DiarizationSegment(start=t, end=t + dur, speaker=spk))
        t += dur
        texts[spk] = (texts.get(spk, "") + " " + text).strip()
        ordered.append((spk, text))
    return resolve_speaker_roster(
        DiarizationResult(segments=segs, num_speakers=len(texts)),
        " ".join(x for _, x in ordered),
        known_hosts=[ANN, BEN],
        metadata_named=list(metadata_named),
        voice_texts=texts,
        ordered_turns=ordered,
        feed_title="Deep Past",
        host_copresence=host_copresence,
    )


def _storyteller_episode(extra: Sequence[Tuple[str, str, float]] = ()):
    """SPEAKER_00 opens and interjects; SPEAKER_01 tells the story, first heard at minute 6."""
    turns = [
        ("SPEAKER_00", "Hello everyone and welcome back to Deep Past.", 40.0),
        ("SPEAKER_00", "Some housekeeping first about the live shows.", 300.0),
    ]
    for i in range(12):
        turns.append(("SPEAKER_01", f"Well, Ann, so the fleet sailed north, chapter {i}.", 300.0))
        if i == 5:
            # One stray hail of the other name in a seat's cluster (diarization bleed) makes the
            # strict pair rule refuse; the dominant-address pair still reads it.
            turns.append(("SPEAKER_00", "Well, Ann, that is the story.", 20.0))
        turns.append(("SPEAKER_00", f"Right, Ben, and what happened next, question {i}?", 40.0))
    return turns + list(extra)


def test_the_storyteller_takes_the_second_seat_and_both_are_named() -> None:
    """The feed's history says its two hosts are rarely BOTH named -- the very gap this closes --
    so only this episode's own addresses can vouch for the second seat."""
    roster = _roster(_storyteller_episode(), host_copresence={1: 1.0, 2: 0.0, 3: 0.0})
    assert roster.by_voice["SPEAKER_01"].role == "host"
    assert roster.by_voice["SPEAKER_01"].name == BEN
    assert roster.by_voice["SPEAKER_00"].name == ANN


def test_with_a_guest_in_the_room_the_storyteller_rule_stays_out() -> None:
    guest = [
        ("SPEAKER_02", "Thanks for having me. The archive shows the ships were late.", 900.0),
        ("SPEAKER_02", "And the letters confirm it, which surprised me.", 900.0),
    ]
    roster = _roster(_storyteller_episode(guest), metadata_named=["Cara Hale"])
    assert roster.by_voice["SPEAKER_02"].role != "host"
