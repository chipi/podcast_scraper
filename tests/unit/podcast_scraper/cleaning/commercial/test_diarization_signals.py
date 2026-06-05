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
