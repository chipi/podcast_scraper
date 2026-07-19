"""Unit tests for the plain (un-diarized) segment-per-line transcript formatter (#1212)."""

from __future__ import annotations

import pytest

from podcast_scraper.transcript_formats.plain import format_plain_transcript_with_offsets

pytestmark = pytest.mark.unit


def _segs():
    return [
        {"id": 0, "start": 0.0, "end": 6.0, "text": " Welcome back to the show."},
        {"id": 1, "start": 6.0, "end": 12.0, "text": "Today we discuss trail building. "},
        {"id": 2, "start": 12.0, "end": 18.0, "text": "It is harder than it looks."},
    ]


def test_one_line_per_segment_no_labels():
    text, out = format_plain_transcript_with_offsets(_segs())
    assert text == (
        "Welcome back to the show.\n"
        "Today we discuss trail building.\n"
        "It is harder than it looks."
    )
    assert len(out) == 3
    assert "SPEAKER" not in text and ":" not in text.split("\n")[0]


def test_offsets_map_back_verbatim():
    text, out = format_plain_transcript_with_offsets(_segs())
    # The core invariant the char->timestamp mapping relies on.
    for seg in out:
        assert text[seg["char_start"] : seg["char_end"]] == seg["text"]


def test_preserves_original_fields_and_strips_text():
    _, out = format_plain_transcript_with_offsets(_segs())
    assert out[0]["start"] == 0.0 and out[0]["end"] == 6.0 and out[0]["id"] == 0
    assert out[0]["text"] == "Welcome back to the show."  # leading space stripped


def test_blank_and_nondict_segments_dropped():
    segs = [
        {"start": 0.0, "end": 1.0, "text": "   "},  # blank -> dropped
        "not a dict",  # skipped
        {"start": 1.0, "end": 2.0, "text": "Only real line."},
    ]
    text, out = format_plain_transcript_with_offsets(segs)
    assert text == "Only real line."
    assert len(out) == 1 and out[0]["char_start"] == 0


def test_alignment_within_delta_would_fail_but_offsets_win():
    # Many segments => many newlines; the raw cumulative len(text) sum would diverge from the
    # joined text by (n-1) newlines, well over SEGMENT_TRANSCRIPT_ALIGNMENT_MAX_DELTA=50. The
    # explicit offsets are what keep the mapping exact (#1212).
    segs = [
        {"start": float(i), "end": float(i + 1), "text": f"line number {i}."} for i in range(200)
    ]
    text, out = format_plain_transcript_with_offsets(segs)
    assert text.count("\n") == 199
    for seg in out:
        assert text[seg["char_start"] : seg["char_end"]] == seg["text"]


def test_empty_input():
    assert format_plain_transcript_with_offsets([]) == ("", [])
