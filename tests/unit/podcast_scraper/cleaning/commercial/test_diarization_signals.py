"""Unit tests for diarization-enhanced commercial signals."""

from __future__ import annotations

import pytest

from podcast_scraper.cleaning.commercial.diarization_signals import diarization_sponsor_signals

pytestmark = pytest.mark.unit


def test_guest_speaker_disqualifies_candidate() -> None:
    text = "Intro\nSponsored by Acme\nOutro"
    segments = [
        {"start": 0.0, "end": 5.0, "text": "Intro", "speaker": "SPEAKER_00"},
        {"start": 5.0, "end": 35.0, "text": "Sponsored by Acme", "speaker": "SPEAKER_01"},
    ]
    sponsor_start = text.index("Sponsored")

    signals = diarization_sponsor_signals(
        sponsor_start,
        len(text),
        text,
        segments,
        host_speaker_id="SPEAKER_00",
    )

    assert signals.disqualify is True


def test_host_monologue_boosts_confidence() -> None:
    text = "A" * 100 + "Sponsored by Acme" + "B" * 100
    segments = [
        {"start": 0.0, "end": 5.0, "text": "A" * 100, "speaker": "SPEAKER_00"},
        {
            "start": 40.0,
            "end": 75.0,
            "text": "Sponsored by Acme",
            "speaker": "SPEAKER_00",
        },
    ]
    sponsor_start = text.index("Sponsored")

    signals = diarization_sponsor_signals(
        sponsor_start,
        sponsor_start + len("Sponsored by Acme"),
        text,
        segments,
        host_speaker_id="SPEAKER_00",
    )

    assert signals.disqualify is False
    assert signals.confidence_delta > 0.0


def test_signals_map_char_offsets_to_time_proportionally() -> None:
    """Speaker attribution follows the candidate's position in the timeline, not
    char offsets into the (differently-sized) concatenated segment text. Regression
    for B1: the old index assumed segment texts reproduced the cleaned transcript
    char-for-char, so attribution was effectively random after cleaning."""
    # 200-char cleaned transcript; the segment texts are a totally different length
    # (10 chars each) -> any char-offset-based index would mis-map.
    text = "X" * 200
    segments = [
        {"start": 0.0, "end": 50.0, "text": "g" * 10, "speaker": "GUEST"},
        {"start": 50.0, "end": 100.0, "text": "h" * 10, "speaker": "HOST"},
    ]

    # Candidate at chars 160-180 -> ~80-90% of the timeline -> HOST half -> kept.
    host_side = diarization_sponsor_signals(160, 180, text, segments, "HOST")
    assert host_side.disqualify is False

    # Candidate at chars 20-40 -> ~10-20% of the timeline -> GUEST half -> disqualified.
    guest_side = diarization_sponsor_signals(20, 40, text, segments, "HOST")
    assert guest_side.disqualify is True
