"""Diarized speech with no transcript under it is detected and recorded (#2187).

WHY THIS EXISTS. The first real ASR run on non-English audio (V.6b, Spanish p10_e01, 2026-10-07)
lost a whole 19.1 s turn: the diarizer found a third voice at 166.7-185.8 s (the ad read), and
Whisper returned no words for it in long-form decoding — while the same clip transcribed on its
own. Nothing noticed: the transcript simply had no segment there. `untranscribed_speech` makes the
gap a recorded fact (asr.json + a manifest quality flag). DETECTION ONLY — no transcript changes.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from podcast_scraper.providers.ml.diarization.pipeline import (
    stretched_words_over_speech,
    untranscribed_speech,
    UNTRANSCRIBED_SPEECH_MIN_S,
)

pytestmark = pytest.mark.unit


def _turn(start: float, end: float, speaker: str) -> SimpleNamespace:
    return SimpleNamespace(start=start, end=end, speaker=speaker)


def _seg(start: float, end: float) -> dict:
    return {"start": start, "end": end, "text": "x"}


class TestTheV6bShape:
    """The measured Spanish case, reduced: a 19.1 s turn with nothing under it, and seams."""

    TURNS = [
        _turn(0.0, 26.9, "SPEAKER_01"),
        _turn(162.9, 166.8, "SPEAKER_01"),
        _turn(166.7, 185.8, "SPEAKER_00"),  # the ad voice
        _turn(185.7, 192.8, "SPEAKER_01"),
    ]
    SEGMENTS = [_seg(0.0, 26.3), _seg(163.0, 166.7), _seg(186.0, 192.4)]

    def test_the_skipped_turn_is_found_with_its_speaker(self) -> None:
        gaps = untranscribed_speech(self.TURNS, self.SEGMENTS)
        assert len(gaps) == 1
        assert gaps[0]["speaker"] == "SPEAKER_00"
        assert gaps[0]["start"] == pytest.approx(166.7)
        assert gaps[0]["end"] == pytest.approx(185.8)
        assert gaps[0]["duration_s"] == pytest.approx(19.1)

    def test_the_seams_do_not_count(self) -> None:
        """0.5 s edges between a turn and Whisper's segment boundaries are ordinary."""
        gaps = untranscribed_speech(self.TURNS, self.SEGMENTS, min_gap_s=0.0)
        assert all(
            g["duration_s"] < UNTRANSCRIBED_SPEECH_MIN_S
            for g in gaps
            if g["speaker"] != "SPEAKER_00"
        )


class TestTheArithmetic:
    def test_a_fully_covered_episode_has_no_gaps(self) -> None:
        turns = [_turn(0, 10, "A"), _turn(10, 20, "B")]
        assert untranscribed_speech(turns, [_seg(0, 20)]) == []

    def test_a_gap_inside_a_turn(self) -> None:
        gaps = untranscribed_speech([_turn(0, 30, "A")], [_seg(0, 10), _seg(20, 30)])
        assert [(g["start"], g["end"]) for g in gaps] == [(10.0, 20.0)]

    def test_a_turn_with_no_transcript_at_all(self) -> None:
        gaps = untranscribed_speech([_turn(5, 12, "A")], [])
        assert [(g["start"], g["end"], g["speaker"]) for g in gaps] == [(5.0, 12.0, "A")]

    def test_overlapping_and_unsorted_segments(self) -> None:
        segs = [_seg(15, 25), _seg(0, 8), _seg(6, 12)]
        gaps = untranscribed_speech([_turn(0, 30, "A")], segs)
        # 0-8 and 6-12 merge to 0-12; 12-15 is a real 3.0 s hole; 25-30 trails the last segment.
        assert [(g["start"], g["end"]) for g in gaps] == [(12.0, 15.0), (25.0, 30.0)]

    def test_the_threshold_is_inclusive(self) -> None:
        gaps = untranscribed_speech([_turn(0, 3, "A")], [], min_gap_s=3.0)
        assert len(gaps) == 1

    def test_dict_turns_and_degenerate_input(self) -> None:
        turns = [{"start": 0, "end": 5, "speaker": "A"}, {"start": 9, "end": 9}, {"bad": True}]
        gaps = untranscribed_speech(turns, [{"start": "x"}])
        assert [(g["start"], g["end"], g["speaker"]) for g in gaps] == [(0.0, 5.0, "A")]


class TestStretchedWordsAreEvidenceNotGaps:
    """A word Whisper stretched over seconds of speech (80k_06: "case," 209.5-213.98 s over a
    skipped quote). Re-transcribing 11 such spans (2026-10-08) found lost speech under 3 and only
    the word, a stutter or fillers under 6 — so they are recorded on their own, never as gaps."""

    TURNS = [_turn(189.2, 212.76, "SPEAKER_00"), _turn(213.47, 218.8, "SPEAKER_00")]

    def _segs(self, case_end: float) -> list:
        words = [
            {"start": 208.62, "end": 209.06, "word": " In"},
            {"start": 209.06, "end": 209.5, "word": " another"},
            {"start": 209.5, "end": case_end, "word": " case,"},
            {"start": case_end + 0.28, "end": case_end + 0.42, "word": " rather"},
        ]
        return [
            {"start": 189.2, "end": 208.6, "text": "x"},
            {"start": 208.62, "end": 218.8, "text": "In this case, though", "words": words},
        ]

    def test_a_stretched_word_is_not_untranscribed_speech(self) -> None:
        assert untranscribed_speech(self.TURNS, self._segs(213.98)) == []

    def test_it_is_reported_with_the_speech_under_it(self) -> None:
        assert stretched_words_over_speech(self.TURNS, self._segs(213.98)) == [
            {"start": 209.5, "end": 213.98, "duration_s": 4.48, "word": "case,", "speech_s": 3.77}
        ]

    def test_an_ordinary_long_word_is_not_reported(self) -> None:
        """The longest ordinary word on the six V.6b real feeds was 2.4 s."""
        assert stretched_words_over_speech(self.TURNS, self._segs(209.5 + 2.9)) == []

    def test_a_stretched_word_over_silence_is_not_reported(self) -> None:
        turns = [_turn(189.2, 209.5, "SPEAKER_00"), _turn(214.0, 218.8, "SPEAKER_00")]
        assert stretched_words_over_speech(turns, self._segs(213.98)) == []

    def test_word_objects_are_read_too(self) -> None:
        word = SimpleNamespace(start=2.0, end=9.0, word="so")
        seg = SimpleNamespace(start=0.0, end=10.0, words=[word])
        got = stretched_words_over_speech([_turn(0, 10, "A")], [seg])
        assert [(g["start"], g["end"], g["word"]) for g in got] == [(2.0, 9.0, "so")]
        assert untranscribed_speech([_turn(0, 10, "A")], [seg]) == []
