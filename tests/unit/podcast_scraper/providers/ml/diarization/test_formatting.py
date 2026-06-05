"""Unit tests for diarized screenplay formatting."""

from __future__ import annotations

import pytest

from podcast_scraper.providers.ml.diarization.formatting import (
    format_diarized_screenplay_from_segments,
)

pytestmark = pytest.mark.unit


def test_format_diarized_screenplay_merges_same_speaker_lines() -> None:
    segments = [
        {"start": 0.0, "end": 1.0, "text": "Hello", "speaker_label": "Host"},
        {"start": 1.0, "end": 2.0, "text": "there", "speaker_label": "Host"},
        {"start": 2.0, "end": 3.0, "text": "Hi", "speaker_label": "Guest"},
    ]

    formatted = format_diarized_screenplay_from_segments(segments)

    assert formatted == "Host: Hello there\nGuest: Hi\n"
