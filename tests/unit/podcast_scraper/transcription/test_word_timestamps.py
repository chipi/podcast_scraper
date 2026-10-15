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


def test_word_dicts_ignores_non_list_words() -> None:
    """A provider whose ``words`` attribute isn't a list must degrade, not crash."""
    from unittest.mock import Mock

    assert word_dicts(Mock()) == []
    assert word_dicts(None) == []


def test_word_dicts_normalizes_objects_and_drops_untimed() -> None:
    class _W:
        def __init__(self, word, start, end):
            self.word, self.start, self.end = word, start, end

    raw = [_W("a", 0.0, 0.5), {"word": "b", "start": 0.5, "end": 1.0}, _W("c", None, None)]
    out = word_dicts(raw)
    assert [w["word"] for w in out] == ["a", "b"]  # untimed "c" dropped


def _w(word, start):
    return {"word": word, "start": start, "end": start + 0.4}


def test_a_word_split_across_two_segments_does_not_shift_the_rest() -> None:
    """faster-whisper-server can cut one word across two segments ("and" | "ocracy") while its
    word list holds it whole ("andocracy"). Consuming words per segment by character count then
    hands every later segment the NEXT segment's word, and on short segments the error grows into
    minutes (#2370: 470 overlapping segments, 775 moved by >5 s on a 37-minute episode)."""
    segments = [
        {"start": 0.0, "end": 0.5, "text": " and"},
        {"start": 0.5, "end": 1.0, "text": "ocracy"},
        {"start": 1.0, "end": 1.5, "text": " coming"},
        {"start": 1.5, "end": 2.0, "text": " from"},
        {"start": 2.0, "end": 2.5, "text": " the"},
        {"start": 2.5, "end": 3.0, "text": " north"},
    ]
    words = [_w(" andocracy", 0.0), _w(" coming", 1.0), _w(" from", 1.5), _w(" the", 2.0)]
    words.append(_w(" north", 2.5))
    out = apply_word_timestamps(segments, words)
    assert [s["start"] for s in out[2:]] == [1.0, 1.5, 2.0, 2.5]
    assert out[1]["start"] == 0.0  # the second half of the split word takes that word's time


def test_many_splits_never_accumulate() -> None:
    texts, words, t = [], [], 0.0
    for i in range(200):
        texts += [" pre", f"fix{i}", f" word{i}"]
        words += [_w(f" prefix{i}", t), _w(f" word{i}", t + 1.0)]
        t += 2.0
    segments = [{"start": 0.0, "end": 0.0, "text": x} for x in texts]
    out = apply_word_timestamps(segments, words)
    for i in range(200):
        assert out[3 * i + 2]["start"] == 2.0 * i + 1.0, f"segment {3 * i + 2} drifted"


def test_segment_times_never_go_backwards_on_a_split() -> None:
    segments = [{"start": 0, "end": 0, "text": x} for x in (" and", "ocracy", " coming")]
    out = apply_word_timestamps(segments, [_w(" andocracy", 0.0), _w(" coming", 1.0)])
    starts = [s["start"] for s in out]
    assert starts == sorted(starts)


def test_a_punctuation_only_segment_keeps_its_own_times() -> None:
    segments = [
        {"start": 0.0, "end": 1.0, "text": " Yes"},
        {"start": 7.0, "end": 8.0, "text": " ..."},
        {"start": 1.0, "end": 2.0, "text": " no"},
    ]
    out = apply_word_timestamps(segments, [_w(" Yes", 0.2), _w(" no", 1.2)])
    assert (out[1]["start"], out[1]["end"]) == (7.0, 8.0)
    assert out[2]["start"] == 1.2
