"""The host of a two-voice interview is seated on the host's voice, not the guest's (#2224).

Measured on prod (Conversations with Tyler, 42 episodes): the deployed roster put the stated host on
the GUEST's voice, or on no voice at all, because

1. ASR segments start with a space and turns are joined with " ", so the host's greeting arrived as
   "welcome back to  Conversations…" — two spaces, which no host speech-act pattern matches;
2. with no host act, a guest act won instead: the guest's "Thank you so much for having me", merged
   into the host's diarized cluster, or the host's own "happy to be here today with…";
3. a hypothetical — "Let's say I'm Cass Sunstein" — was read as the host introducing himself.

All fixtures are synthetic (never-commit-real-episodes); the shapes mirror those episodes.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pytest

from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment
from podcast_scraper.providers.ml.diarization.roster import resolve_speaker_roster
from podcast_scraper.speaker_detectors.hosts import (
    distinct_self_introductions,
    extract_self_introduced_host,
)

pytestmark = pytest.mark.unit

HOST = "Tobias Wren"
GUEST = "Maria Lindqvist"


def _as_production(turns: List[Tuple[str, str, float]]):
    """(diarization, voice_texts, ordered_turns) built the way the pipeline builds them.

    Each segment's text starts with a space (as ASR emits it) and a voice's turns are joined with a
    single space (``pipeline._voice_texts_from_aligned``) — which leaves two at every boundary.
    """
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
    return DiarizationResult(segments=segs, num_speakers=len(voice_texts)), voice_texts, ordered


def _roster(turns):
    dz, vt, ordered = _as_production(turns)
    return resolve_speaker_roster(
        dz,
        " ".join(t for _, t in ordered),
        known_hosts=[HOST],
        metadata_named=[GUEST],
        voice_texts=vt,
        ordered_turns=ordered,
    )


_GREETING = [
    ("SPEAKER_00", "Hello, everyone, and welcome back to", 4.0),
    ("SPEAKER_00", "Conversations with Tobias. Today I'm chatting with Maria Lindqvist.", 6.0),
    # The guest's biography, read by the host: it is what puts the host's greeting more than one
    # bled exchange away from the "thank you for having me" that diarization merged into his voice.
    (
        "SPEAKER_00",
        "She is a historian of medieval trade, the author of four books on the Baltic ports, and "
        "her new book follows a single merchant family across three centuries of shifting routes, "
        "wars, plagues and the slow silting of the river that once made their fortune.",
        20.0,
    ),
]


def test_host_is_seated_on_the_opener_when_the_guests_thanks_merged_into_his_voice() -> None:
    # Diarization put the guest's reply inside the host's cluster, so the host's text carries a
    # guest speech act. The host act must still win — it only can if it matches at all.
    roster = _roster(
        _GREETING
        + [
            ("SPEAKER_00", "Maria, welcome. Thank you so much for having me.", 5.0),
            ("SPEAKER_01", "The archive tells a different story than the textbooks do.", 400.0),
            ("SPEAKER_00", "Why did the trade routes shift north?", 60.0),
            ("SPEAKER_01", "Because the river silted up and the ports moved with it.", 400.0),
        ]
    )
    assert roster.by_voice["SPEAKER_00"].name == HOST
    assert roster.by_voice["SPEAKER_00"].role == "host"
    assert roster.by_voice["SPEAKER_01"].name != HOST


def test_host_who_says_happy_to_be_here_keeps_the_seat_and_the_guest_never_gets_his_name() -> None:
    # "I'm very happy to be here today with Maria" is the host's own sentence; it matches a guest
    # act, and without a matching host act the seat went to the guest's voice. The seat is now the
    # host's. The NAME stays off it: the guest act sits inside one bled exchange of his greeting, so
    # `_rescued_from_bleed` declines — the honest SPEAKER_NN state, never the guest wearing it.
    roster = _roster(
        [
            ("SPEAKER_00", "Hello, everyone, and welcome back to", 4.0),
            (
                "SPEAKER_00",
                "Conversations with Tobias. I'm very happy to be here today with Maria.",
                6.0,
            ),
            _GREETING[2],
            ("SPEAKER_00", "Maria, welcome.", 2.0),
            ("SPEAKER_01", "The archive tells a different story than the textbooks do.", 400.0),
            ("SPEAKER_00", "Why did the trade routes shift north?", 60.0),
            ("SPEAKER_01", "Cheers, Tobias. Because the river silted up.", 400.0),
        ]
    )
    assert roster.by_voice["SPEAKER_00"].role == "host"
    assert roster.by_voice["SPEAKER_00"].name in (HOST, "SPEAKER_00")
    assert roster.by_voice["SPEAKER_01"].name != HOST
    assert roster.by_voice["SPEAKER_01"].role != "host"


def test_hypothetical_i_am_is_not_a_self_introduction() -> None:
    text = "Just an on-the-spot test. Let's say I'm Cass Ellery. I know Cass, he would say no."
    assert extract_self_introduced_host(text) is None
    assert distinct_self_introductions(text) == []
    for lead in ("Suppose I'm", "Imagine I'm", "Pretend I'm", "If I'm", "Let us say, I'm"):
        assert extract_self_introduced_host(f"OK. {lead} Cass Ellery, what then?") is None


def test_a_real_introduction_is_still_read_and_still_found_after_a_hypothetical() -> None:
    assert extract_self_introduced_host("Hi, I'm Tobias Wren.") == HOST
    text = "Let's say I'm Cass Ellery for a second. Anyway, I'm Tobias Wren and this is the show."
    assert extract_self_introduced_host(text) == HOST
    assert distinct_self_introductions(text) == [HOST]


def test_host_posing_a_hypothetical_is_not_named_after_it() -> None:
    roster = _roster(
        _GREETING
        + [
            ("SPEAKER_01", "Thank you so much for having me, Tobias. Excited to be here.", 30.0),
            ("SPEAKER_00", "Let's say I'm Cass Ellery. I know Cass. How would you test me?", 60.0),
            ("SPEAKER_01", "We would hand you a case you have never seen before.", 400.0),
        ]
    )
    assert roster.by_voice["SPEAKER_00"].name == HOST
    assert all(r.name != "Cass Ellery" for r in roster.by_voice.values())
