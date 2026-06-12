"""Unit tests for diarized screenplay formatting."""

from __future__ import annotations

import pytest

from podcast_scraper.providers.ml.diarization.formatting import (
    format_diarized_screenplay_from_segments,
    format_diarized_screenplay_with_offsets,
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


def test_offsets_match_text_only_formatter_byte_for_byte() -> None:
    # The offset-emitting formatter must produce identical text to the legacy one.
    segments = [
        {"start": 0.0, "end": 1.0, "text": "  Hello ", "speaker_label": "Host"},
        {"start": 1.0, "end": 2.0, "text": "there", "speaker_label": "Host"},
        {"start": 2.0, "end": 3.0, "text": "Hi", "speaker_label": "Guest"},
        {"start": 3.0, "end": 4.0, "text": "", "speaker_label": "Guest"},  # dropped
        {"start": 4.0, "end": 5.0, "text": "Back", "speaker_label": "Host"},
    ]
    text, offset_segs = format_diarized_screenplay_with_offsets(segments)
    assert text == format_diarized_screenplay_from_segments(segments)
    assert text == "Host: Hello there\nGuest: Hi\nHost: Back\n"


def test_each_segment_range_slices_back_to_its_text() -> None:
    segments = [
        {"start": 0.0, "end": 1.0, "text": "Hello", "speaker_label": "Patrick"},
        {"start": 1.0, "end": 2.0, "text": "there friend", "speaker_label": "Patrick"},
        {"start": 2.0, "end": 3.0, "text": "Great to be here", "speaker_label": "Brian"},
        {"start": 3.0, "end": 4.0, "text": "So", "speaker_label": "Patrick"},
    ]
    text, offset_segs = format_diarized_screenplay_with_offsets(segments)
    assert len(offset_segs) == 4
    for seg in offset_segs:
        cs, ce = seg["char_start"], seg["char_end"]
        assert text[cs:ce] == seg["text"]
    # speaker label carried through for the speaker-mapping consumer
    assert offset_segs[2]["speaker_label"] == "Brian"
    # the "Brian" turn's text starts after its "Brian: " marker, not at a naive concat pos
    brian = offset_segs[2]
    assert text[brian["char_start"] - len("Brian: ") : brian["char_start"]] == "Brian: "


def test_empty_segments_return_empty() -> None:
    assert format_diarized_screenplay_with_offsets([]) == ("", [])
    assert format_diarized_screenplay_from_segments([]) == ""


def test_falls_back_to_speaker_then_default_label() -> None:
    segments = [
        {"start": 0.0, "end": 1.0, "text": "Raw", "speaker": "SPEAKER_01"},
        {"start": 1.0, "end": 2.0, "text": "None"},
    ]
    text, offset_segs = format_diarized_screenplay_with_offsets(segments)
    assert text == "SPEAKER_01: Raw\nSPEAKER: None\n"
    assert offset_segs[0]["speaker_label"] == "SPEAKER_01"
    assert offset_segs[1]["speaker_label"] == "SPEAKER"
