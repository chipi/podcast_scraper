"""A feed's known host is not forced onto a voice that speaks of that host in the third person.

Record: docs/wip/changes/known_host_third_person_veto.md (#2094 D5). The shape on record: a guest
interviews the feed's one host, opens the episode, and says the host's full name and surname
throughout; the forced host seat went to that voice because it speaks first.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import pytest

from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment
from podcast_scraper.providers.ml.diarization.roster import resolve_speaker_roster

pytestmark = pytest.mark.unit

HOST = "Tobias Wren"
GUEST = "Mara Quell"
HOST_B = "Greta Holm"

Turn = Tuple[str, str, float]


def _roster(
    turns: Sequence[Turn],
    known_hosts: Sequence[str] = (HOST,),
    *,
    metadata_named: Sequence[str] = (GUEST,),
    feed_title: Optional[str] = None,
):
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
        metadata_named=list(metadata_named),
        voice_texts=voice_texts,
        ordered_turns=ordered,
        feed_title=feed_title,
    )


def _interview_of_the_host(opener: str) -> List[Turn]:
    """SPEAKER_00 opens and asks; SPEAKER_01 answers at length. Neither introduces itself."""
    return [
        ("SPEAKER_00", opener, 40.0),
        ("SPEAKER_01", "Thanks. The ports moved north because the river silted up.", 400.0),
        ("SPEAKER_00", "And what became of the merchants after that?", 60.0),
        ("SPEAKER_01", "Most of them followed the trade within a generation.", 400.0),
        ("SPEAKER_00", "Let's leave it there for today.", 30.0),
    ]


SPEAKS_OF_HOST = (
    f"I'm thrilled to be talking to Dr. {HOST} today. Wren has written about the old ports, "
    "and I have read every page of it."
)


def test_a_voice_that_speaks_of_the_host_by_full_name_and_surname_is_not_seated_as_the_host() -> (
    None
):
    roster = _roster(_interview_of_the_host(SPEAKS_OF_HOST))
    assert roster.by_voice["SPEAKER_00"].name != HOST


def test_control_the_same_opener_without_the_host_name_is_seated_as_the_host() -> None:
    roster = _roster(_interview_of_the_host("Welcome back. Today we talk about the old ports."))
    assert roster.by_voice["SPEAKER_00"].name == HOST


def test_the_veto_names_no_other_voice_with_the_host() -> None:
    """Refusing the seat does not move the host's name onto the other voice by itself."""
    roster = _roster(_interview_of_the_host(SPEAKS_OF_HOST))
    assert [v for v, r in roster.by_voice.items() if r.name == HOST] in ([], ["SPEAKER_01"])
    assert roster.by_voice["SPEAKER_00"].name != HOST


def test_the_show_named_after_the_host_is_not_a_mention_of_the_host() -> None:
    """ "Welcome back to Wren on Ports" is read on the host's own voice: the surname inside the
    feed's title is the show, not the host spoken of."""
    opener = "Welcome back to Wren on Ports. Today we talk about the old ports."
    roster = _roster(_interview_of_the_host(opener), feed_title="Wren on Ports")
    assert roster.by_voice["SPEAKER_00"].name == HOST


def test_a_seat_that_introduces_itself_as_the_host_keeps_the_name() -> None:
    opener = f"I'm {HOST}, and this is the show. {HOST} here, as every week."
    roster = _roster(_interview_of_the_host(opener))
    assert roster.by_voice["SPEAKER_00"].name == HOST


def _two_host(seat_text: str) -> List[Turn]:
    """Host B introduces itself; SPEAKER_01 is the spare seat for host A."""
    return [
        ("SPEAKER_00", f"Hello and welcome, I'm {HOST_B}.", 30.0),
        ("SPEAKER_01", seat_text, 300.0),
        ("SPEAKER_00", "And the guilds, what became of them?", 300.0),
        ("SPEAKER_01", "Most of the merchants followed the trade within a generation.", 300.0),
        ("SPEAKER_00", "Let's leave it there for today.", 30.0),
    ]


def test_a_seat_that_keeps_addressing_the_cohost_keeps_the_host_name_despite_a_bled_mention() -> (
    None
):
    """The co-host's line about the host, bled into the host's cluster, does not unseat the host
    while the seat keeps talking TO the co-host (the given-name veto's exemption, kept here)."""
    text = "Greta, the ports moved north. Greta, yes, exactly. Wren wrote about that once."
    roster = _roster(_two_host(text), known_hosts=(HOST, HOST_B), metadata_named=())
    assert roster.by_voice["SPEAKER_01"].name == HOST


def test_control_without_the_cohost_address_the_bled_mention_unseats() -> None:
    text = "The ports moved north. Yes, exactly. Wren wrote about that once."
    roster = _roster(_two_host(text), known_hosts=(HOST, HOST_B), metadata_named=())
    assert roster.by_voice["SPEAKER_01"].name != HOST
