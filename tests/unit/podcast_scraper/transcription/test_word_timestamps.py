"""Unit tests for word-level segment-time refinement (#1173)."""

from __future__ import annotations

from podcast_scraper.transcription.word_timestamps import apply_word_timestamps, word_dicts


def test_resets_segment_times_from_words() -> None:
    """Each segment's start/end come from the accurate words, not the coarse segment times."""
    segments = [
        {"start": 0.0, "end": 30.0, "text": "Hello there world"},  # coarse/drifted
        {"start": 30.0, "end": 60.0, "text": "second segment here"},
    ]
    words = [
        {"word": "Hello", "start": 1.0, "end": 1.4},
        {"word": " there", "start": 1.4, "end": 1.9},
        {"word": " world.", "start": 1.9, "end": 2.5},
        {"word": " Second", "start": 5.0, "end": 5.4},
        {"word": " segment", "start": 5.4, "end": 5.9},
        {"word": " here.", "start": 5.9, "end": 6.3},
    ]
    out = apply_word_timestamps(segments, words)
    assert (out[0]["start"], out[0]["end"]) == (1.0, 2.5)
    assert (out[1]["start"], out[1]["end"]) == (5.0, 6.3)
    assert out[0]["text"] == "Hello there world"  # text/grouping preserved


def test_noop_without_words() -> None:
    """No word timestamps → segments returned unchanged (old segment-level behaviour)."""
    segs = [{"start": 0.0, "end": 5.0, "text": "unchanged"}]
    out = apply_word_timestamps(segs, [])
    assert out == segs and out is not segs  # copied, not mutated


def test_punctuation_and_spacing_tolerant() -> None:
    """Segment text vs word tokens differ in punctuation/case/spacing — still aligns."""
    segments = [{"start": 0.0, "end": 9.0, "text": "It's a test!"}]
    words = [
        {"word": "It's", "start": 2.0, "end": 2.3},
        {"word": " a", "start": 2.3, "end": 2.4},
        {"word": " test!", "start": 2.4, "end": 2.9},
    ]
    out = apply_word_timestamps(segments, words)
    assert (out[0]["start"], out[0]["end"]) == (2.0, 2.9)


def test_word_dicts_normalizes_objects_and_drops_untimed() -> None:
    class _W:
        def __init__(self, word, start, end):
            self.word, self.start, self.end = word, start, end

    raw = [_W("a", 0.0, 0.5), {"word": "b", "start": 0.5, "end": 1.0}, _W("c", None, None)]
    out = word_dicts(raw)
    assert [w["word"] for w in out] == ["a", "b"]  # untimed "c" dropped
