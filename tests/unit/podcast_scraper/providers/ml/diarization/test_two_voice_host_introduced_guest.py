"""The one stated guest the named host introduces by full name, in a two-voice interview.

Measured on prod: two-voice episodes whose host is named and whose single metadata-stated guest
stays a raw SPEAKER_NN, although the host's own voice says "With me today is <guest>". The rule is
deliberately narrow — every case below that is NOT that shape must leave the guest unnamed.

All fixtures are synthetic (never-commit-real-episodes).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import pytest

from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment
from podcast_scraper.providers.ml.diarization.roster import resolve_speaker_roster

pytestmark = pytest.mark.unit

HOST = "Tobias Wren"
GUEST = "Maria Lindqvist"


def _roster(turns: List[Tuple[str, str, float]], metadata_named: Sequence[str] = (GUEST,)):
    segs: List[DiarizationSegment] = []
    chunks: Dict[str, List[str]] = {}
    ordered: List[Tuple[str, str]] = []
    t = 30.0
    for spk, text, dur in turns:
        segs.append(DiarizationSegment(start=t, end=t + dur, speaker=spk))
        t += dur
        chunks.setdefault(spk, []).append(" " + text)
        ordered.append((spk, " " + text))
    voice_texts = {v: " ".join(c) for v, c in chunks.items()}
    return resolve_speaker_roster(
        DiarizationResult(segments=segs, num_speakers=len(voice_texts)),
        " ".join(x for _, x in ordered),
        known_hosts=[HOST],
        metadata_named=list(metadata_named),
        voice_texts=voice_texts,
        ordered_turns=ordered,
    )


def _interview(host_intro: str, guest_answer: str = "The ports moved because the river silted."):
    return [
        ("SPEAKER_00", f"Welcome to the show, I'm {HOST}. {host_intro}", 40.0),
        ("SPEAKER_01", guest_answer, 400.0),
        ("SPEAKER_00", "And what happened to the merchants after that?", 60.0),
        ("SPEAKER_01", "Most of them followed the trade north within a generation.", 400.0),
    ]


@pytest.mark.parametrize(
    "intro",
    [
        f"With me today is {GUEST}, a historian of medieval trade.",
        f"Today's guest is the author of four books on the Baltic, {GUEST}.",
        f"I am really delighted today to welcome Dr. {GUEST}.",
    ],
)
def test_the_guest_the_host_introduces_is_named(intro: str) -> None:
    roster = _roster(_interview(intro))
    assert roster.by_voice["SPEAKER_00"].name == HOST
    assert roster.by_voice["SPEAKER_01"].name == GUEST
    assert roster.by_voice["SPEAKER_01"].role == "guest"


def test_a_mentioned_stated_guest_who_owns_the_talk_is_named() -> None:
    """K8 (#2338): the ONE person the metadata states, spoken of in the episode, on the ONE voice
    holding most of the talk. This test used to pin "a mention names nobody"; measured 2026-10-14
    the strict reading (an introduction required) kept 2 of the 12 names this rule adds on prod,
    all 12 read right on their text, and the gold sets were 1/0 (dev) and 2/0 (val_v1)."""
    roster = _roster(_interview(f"I have been reading {GUEST}'s new book all week."))
    assert roster.by_voice["SPEAKER_01"].name == GUEST
    assert roster.by_voice["SPEAKER_00"].name == HOST


def test_two_stated_people_is_a_guess_and_names_nobody() -> None:
    roster = _roster(
        _interview(f"With me today is {GUEST}."), metadata_named=(GUEST, "Henrik Sallow")
    )
    assert roster.by_voice["SPEAKER_01"].name != GUEST


def test_a_short_third_voice_does_not_stop_the_voice_that_owns_the_talk() -> None:
    """K8 (#2338): a 90 s archive clip next to a voice holding 80% of the talk is no competitor;
    the clip itself is never named (this test used to pin "a third voice names nobody")."""
    turns = _interview(f"With me today is {GUEST}.") + [
        ("SPEAKER_02", "A listener's question, recorded at the archive.", 90.0)
    ]
    roster = _roster(turns)
    assert roster.by_voice["SPEAKER_01"].name == GUEST
    assert roster.by_voice["SPEAKER_02"].name != GUEST


def test_a_cross_post_never_names_the_feed_host_after_the_interviewer() -> None:
    """K8 guard: the feed's host is the GUEST on someone else's show, and the metadata names that
    show's host (BRAVE E720, "a guest on Rohit Malhotra's show")."""
    interviewer = "Henrik Sallow"
    turns = [
        ("SPEAKER_02", f"Now, {HOST} was recently a guest on {interviewer}'s show.", 20.0),
        ("SPEAKER_00", "You know, Tobias, you've had a very interesting journey.", 60.0),
        ("SPEAKER_01", "Thanks. I started as an engineer and then moved into investing.", 600.0),
        ("SPEAKER_00", "And what did founding teach you?", 30.0),
        ("SPEAKER_01", "That distribution matters more than the product, most of the time.", 600.0),
    ]
    roster = _roster(turns, metadata_named=(interviewer,))
    assert all(r.name != interviewer for v, r in roster.by_voice.items() if v != "SPEAKER_00")
    assert roster.by_voice["SPEAKER_01"].name != interviewer


def test_the_remaining_voice_that_talks_about_the_guest_in_the_third_person_is_not_them() -> None:
    roster = _roster(
        _interview(
            f"With me today is {GUEST}.",
            guest_answer=f"As {GUEST} argued in her book, the ports moved north.",
        )
    )
    assert roster.by_voice["SPEAKER_01"].name != GUEST


def test_a_third_voice_above_the_cameo_floor_names_nobody() -> None:
    # The name must not fall through to a short ad promo when the second substantial voice is
    # already a host seat (Unhedged: a 47s live-show promo was named as the guest).
    turns = _interview(f"With me today is {GUEST}.") + [
        ("SPEAKER_02", "Love the show? Come and see it live this summer.", 47.0)
    ]
    roster = _roster(turns)
    assert roster.by_voice["SPEAKER_02"].name != GUEST
