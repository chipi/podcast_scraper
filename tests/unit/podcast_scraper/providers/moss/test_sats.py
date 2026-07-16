"""Unit tests for the MOSS SATS parser (#1177).

The model is days old and its output format is the least settled thing about it, so these lean
hard on the malformed cases: a decoder-only ASR *will* emit a dangling span or a stray tag, and
losing one segment must never cost us a 90-minute episode.
"""

from __future__ import annotations

import pytest

from podcast_scraper.providers.moss.sats import parse_sats, speakers, transcript_text

pytestmark = pytest.mark.unit

CANONICAL = "[0.48][S01]Welcome everyone[1.66][12.26][S02]The new pipeline is ready[13.81]"


def test_parses_the_canonical_stream() -> None:
    segments = parse_sats(CANONICAL)
    assert segments == [
        {"start": 0.48, "end": 1.66, "text": "Welcome everyone", "speaker": "S01"},
        {"start": 12.26, "end": 13.81, "text": "The new pipeline is ready", "speaker": "S02"},
    ]


def test_transcript_and_speakers_derive_from_the_same_parse() -> None:
    segments = parse_sats(CANONICAL)
    assert transcript_text(segments) == "Welcome everyone The new pipeline is ready"
    # Anonymous + relative, in first-appearance order — the roster resolves them downstream.
    assert speakers(segments) == ["S01", "S02"]


def test_acoustic_event_tags_are_stripped_from_text() -> None:
    """MOSS interleaves non-speech annotations; they are not transcript and not speakers."""
    segments = parse_sats("[0.0][S01]That is [laughter] genuinely funny[3.2]")
    assert segments[0]["text"] == "That is genuinely funny"
    assert segments[0]["speaker"] == "S01"


def test_malformed_spans_are_skipped_not_raised() -> None:
    """A dangling span must cost one segment, not the whole episode."""
    raw = (
        "[0.0][S01]good one[1.0]"
        "[2.0][S02]"  # no end timestamp, no text — dangling
        "[3.0][S01]another good one[4.0]"
    )
    segments = parse_sats(raw)
    assert [s["text"] for s in segments] == ["good one", "another good one"]


def test_backwards_timestamps_are_dropped() -> None:
    """end < start is incoherent; keeping it would poison every downstream time calculation."""
    raw = "[10.0][S01]impossible[2.0][20.0][S01]fine[21.0]"
    segments = parse_sats(raw)
    assert [s["text"] for s in segments] == ["fine"]


def test_empty_and_garbage_input_return_no_segments() -> None:
    assert parse_sats("") == []
    assert parse_sats("   ") == []
    assert parse_sats("the model said something with no spans at all") == []


def test_integer_timestamps_are_accepted() -> None:
    """Nothing guarantees the model always emits a decimal point."""
    segments = parse_sats("[1][S01]terse[2]")
    assert segments == [{"start": 1.0, "end": 2.0, "text": "terse", "speaker": "S01"}]


def test_multiline_text_is_flattened() -> None:
    segments = parse_sats("[0.0][S01]line one\nline two[5.0]")
    assert segments[0]["text"] == "line one line two"
