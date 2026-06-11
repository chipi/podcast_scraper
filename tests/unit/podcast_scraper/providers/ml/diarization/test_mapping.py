"""Unit tests for diarization name mapping."""

from __future__ import annotations

import pytest

from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment
from podcast_scraper.providers.ml.diarization.mapping import map_speakers_to_names

pytestmark = pytest.mark.unit


def test_map_speakers_to_names_keeps_host_raw_and_names_guest() -> None:
    # SPEAKER_00 dominates the intro -> treated as host (kept raw). SPEAKER_01 talks most
    # overall -> the (only) guest. ``detected_names`` is guest-only (#876).
    diarization = DiarizationResult(
        segments=[
            DiarizationSegment(start=0.0, end=60.0, speaker="SPEAKER_00"),
            DiarizationSegment(start=60.0, end=120.0, speaker="SPEAKER_01"),
            DiarizationSegment(start=120.0, end=180.0, speaker="SPEAKER_01"),
        ],
        num_speakers=2,
        model_name="test",
    )

    mapping = map_speakers_to_names(diarization, ["Guest Name"])

    assert mapping["SPEAKER_00"] == "SPEAKER_00", "host must keep its raw label"
    assert mapping["SPEAKER_01"] == "Guest Name", "guest must be named"


def test_map_speakers_to_names_never_paints_guest_name_on_host() -> None:
    # #876 regression: the ILtB shape — 3 speakers, a single guest name. The host (intro
    # speaker) must NOT receive the guest's name; the guest (most total talk-time) gets it;
    # the third speaker (e.g. a disclaimer voice) stays raw.
    diarization = DiarizationResult(
        segments=[
            DiarizationSegment(start=0.0, end=80.0, speaker="HOST"),  # owns the intro
            DiarizationSegment(start=80.0, end=400.0, speaker="GUEST"),  # most total time
            DiarizationSegment(start=400.0, end=405.0, speaker="EXTRA"),
        ],
        num_speakers=3,
        model_name="test",
    )

    mapping = map_speakers_to_names(diarization, ["Brian Chesky"])

    assert mapping["HOST"] == "HOST", "guest name must never land on the host slot"
    assert mapping["GUEST"] == "Brian Chesky"
    assert mapping["EXTRA"] == "EXTRA"
