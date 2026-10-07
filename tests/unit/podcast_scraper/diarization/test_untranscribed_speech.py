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
    UNTRANSCRIBED_SPEECH_MIN_S,
    untranscribed_speech,
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
