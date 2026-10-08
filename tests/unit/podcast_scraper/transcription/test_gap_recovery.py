"""Speech the long-form transcript skipped is re-transcribed and spliced in (#2187, A2).

WHY. Whisper's long-form decode can drop a stretch of speech with no trace: on 80k_03 (raw audio)
its segments jump from 23.4 s to 48.8 s while one speaker talks throughout, losing 77 words, and
the V.6b Spanish fixture lost a 19 s ad read. The diarizer hears the stretch
(``untranscribed_speech``) and the same audio transcribes when sent alone. These tests pin what may
be spliced in (only words inside the gap, only above the confidence and loop filters), that
nothing already transcribed changes, and that a failed recovery never costs the transcript.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import pytest

from podcast_scraper.transcription import gap_recovery as G

pytestmark = pytest.mark.unit

AUDIO = Path(__file__).resolve().parents[3] / "fixtures" / "audio" / "v3" / "p01_e01.mp3"


def _w(word: str, start: float, end: float) -> Dict[str, Any]:
    return {"word": word, "start": start, "end": end}


@pytest.fixture
def no_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> List[tuple]:
    cuts: List[tuple] = []
    monkeypatch.setattr(G, "cut_clip", lambda path, s, e, out: cuts.append((s, e)))
    return cuts


class TestIsRepetitive:
    @pytest.mark.parametrize(
        "text",
        [
            "Thank you. Thank you. Thank you. Thank you.",
            "gracias gracias gracias gracias",
            "la la la la la la la la la",
            "and so and so and so and so on",
        ],
    )
    def test_a_loop_is_repetitive(self, text: str) -> None:
        assert G.is_repetitive(text)

    @pytest.mark.parametrize(
        "text",
        [
            "So the question is whether the model actually learned anything at all.",
            "Bueno, yo creo que eso no es así, pero vamos a verlo con calma.",
            "Thank you, thank you, thank you.",  # three is speech; four in a row is a loop
        ],
    )
    def test_speech_is_not(self, text: str) -> None:
        assert not G.is_repetitive(text)

    def test_no_fixture_sentence_is_called_a_loop(self) -> None:
        """A false loop verdict throws away real speech, so the v3 transcripts — authored prose in
        six languages — must not trip it, sentence by sentence."""
        v3 = Path(__file__).resolve().parents[3] / "fixtures" / "transcripts" / "v3"
        flagged = [
            line
            for path in sorted(v3.glob("p??_e??.txt"))
            for line in path.read_text(encoding="utf-8").splitlines()
            if G.is_repetitive(line)
        ]
        assert flagged == []


class TestSegmentsInsideGap:
    def test_only_words_inside_the_gap_are_kept_in_episode_time(self) -> None:
        # The clip starts at 99 s (1 s pad before a 100-104 s gap). Its first and last words are
        # padding — already in the transcript — and must not be duplicated.
        seg = {
            "start": 0.0,
            "end": 6.0,
            "text": "end. Lost words here. next",
            "avg_logprob": -0.2,
            "compression_ratio": 1.3,
            "words": [
                _w("end.", 0.1, 0.8),
                _w("Lost", 1.2, 1.6),
                _w("words", 1.7, 2.2),
                _w("here.", 2.3, 4.6),
                _w("next", 5.2, 5.8),
            ],
        }
        kept, rejected = G.segments_inside_gap(
            [seg], clip_start=99.0, gap_start=100.0, gap_end=104.0
        )
        assert rejected == []
        assert len(kept) == 1
        assert kept[0]["text"] == "Lost words here."
        assert kept[0]["recovered"] is True
        assert kept[0]["start"] == pytest.approx(100.2)
        assert kept[0]["end"] == pytest.approx(103.6)
        assert [w["start"] for w in kept[0]["words"]] == pytest.approx([100.2, 100.7, 101.3])

    def test_without_word_times_the_segment_midpoint_decides(self) -> None:
        segs = [
            {"start": 0.0, "end": 0.9, "text": "pad"},
            {"start": 1.0, "end": 4.0, "text": "inside"},
        ]
        kept, _ = G.segments_inside_gap(segs, clip_start=9.0, gap_start=10.0, gap_end=13.0)
        assert [(s["text"], s["start"], s["end"]) for s in kept] == [("inside", 10.0, 13.0)]

    def test_a_low_confidence_segment_is_rejected(self) -> None:
        seg = {"start": 1.0, "end": 4.0, "text": "maybe words", "avg_logprob": -1.4}
        kept, rejected = G.segments_inside_gap([seg], clip_start=0.0, gap_start=1.0, gap_end=4.0)
        assert kept == [] and rejected == ["low_confidence"]

    @pytest.mark.parametrize(
        "seg",
        [
            {"start": 1.0, "end": 4.0, "text": "words words", "compression_ratio": 2.9},
            {"start": 1.0, "end": 4.0, "text": "Thank you. Thank you. Thank you. Thank you."},
        ],
    )
    def test_a_loop_is_rejected(self, seg: Dict[str, Any]) -> None:
        kept, rejected = G.segments_inside_gap([seg], clip_start=0.0, gap_start=1.0, gap_end=4.0)
        assert kept == [] and rejected == ["repetitive"]

    def test_the_filters_sit_at_whispers_own_thresholds(self) -> None:
        at_edge = {
            "start": 1.0,
            "end": 4.0,
            "text": "kept at the edge",
            "avg_logprob": G.RECOVERY_MIN_AVG_LOGPROB,
            "compression_ratio": G.RECOVERY_MAX_COMPRESSION_RATIO,
        }
        kept, _ = G.segments_inside_gap([at_edge], clip_start=0.0, gap_start=1.0, gap_end=4.0)
        assert len(kept) == 1


class TestRecoverUntranscribedSpeech:
    RESULT = {
        "text": "Before the gap. After the gap.",
        "segments": [
            {"start": 90.0, "end": 99.5, "text": "Before the gap."},
            {"start": 105.0, "end": 110.0, "text": "After the gap."},
        ],
        "language_requested": "es",
    }
    GAP = [{"start": 99.5, "end": 105.0, "duration_s": 5.5, "speaker": "SPEAKER_01"}]

    def _clip(self, text: str = "Las palabras perdidas.") -> Dict[str, Any]:
        return {
            "segments": [
                {
                    "start": 1.2,
                    "end": 5.0,
                    "text": text,
                    "avg_logprob": -0.3,
                    "compression_ratio": 1.2,
                }
            ]
        }

    def test_no_gaps_means_no_call_and_the_same_result(self, no_ffmpeg: List[tuple]) -> None:
        calls: List[str] = []

        def transcribe(path: str) -> Dict[str, Any]:
            calls.append(path)
            return {}

        out = G.recover_untranscribed_speech(self.RESULT, [], "a.mp3", transcribe)
        assert out is self.RESULT
        assert calls == [] and no_ffmpeg == []

    def test_the_recovered_segment_is_spliced_in_time_order(self, no_ffmpeg: List[tuple]) -> None:
        out = G.recover_untranscribed_speech(self.RESULT, self.GAP, "a.mp3", lambda p: self._clip())
        assert [s["text"] for s in out["segments"]] == [
            "Before the gap.",
            "Las palabras perdidas.",
            "After the gap.",
        ]
        assert [bool(s.get("recovered")) for s in out["segments"]] == [False, True, False]
        assert out["segments"][1]["start"] == pytest.approx(99.7)
        assert out["text"] == "Before the gap. Las palabras perdidas. After the gap."
        assert out["asr_speech_recovery"] == [
            {
                "start": 99.5,
                "end": 105.0,
                "speaker": "SPEAKER_01",
                "status": "recovered",
                "words": 3,
                "rejected": [],
            }
        ]
        # The clip is the gap plus RECOVERY_PAD_S of context either side.
        assert no_ffmpeg == [(98.5, 106.0)]

    def test_the_input_is_not_mutated(self, no_ffmpeg: List[tuple]) -> None:
        before = json.dumps(self.RESULT, sort_keys=True)
        G.recover_untranscribed_speech(self.RESULT, self.GAP, "a.mp3", lambda p: self._clip())
        assert json.dumps(self.RESULT, sort_keys=True) == before

    def test_a_failed_call_keeps_the_transcript_and_says_so(self, no_ffmpeg: List[tuple]) -> None:
        def boom(path: str) -> Dict[str, Any]:
            raise ConnectionError("dgx down")

        out = G.recover_untranscribed_speech(self.RESULT, self.GAP, "a.mp3", boom)
        assert out["segments"] == self.RESULT["segments"]
        assert out["text"] == self.RESULT["text"]
        assert out["asr_speech_recovery"][0]["status"] == "failed"
        assert out["asr_speech_recovery"][0]["error"] == "ConnectionError"

    def test_a_rejected_clip_adds_nothing_and_records_why(self, no_ffmpeg: List[tuple]) -> None:
        loop = self._clip("Gracias. Gracias. Gracias. Gracias.")
        out = G.recover_untranscribed_speech(self.RESULT, self.GAP, "a.mp3", lambda p: loop)
        assert out["segments"] == self.RESULT["segments"]
        assert out["asr_speech_recovery"][0]["status"] == "rejected"
        assert out["asr_speech_recovery"][0]["rejected"] == ["repetitive"]

    def test_an_empty_clip_is_reported_as_empty(self, no_ffmpeg: List[tuple]) -> None:
        out = G.recover_untranscribed_speech(
            self.RESULT, self.GAP, "a.mp3", lambda p: {"segments": []}
        )
        assert out["segments"] == self.RESULT["segments"]
        assert out["asr_speech_recovery"][0]["status"] == "empty"

    def test_an_invented_credit_is_not_spliced_in(self, no_ffmpeg: List[tuple]) -> None:
        """V.6b French: the credit came back at avg_logprob -0.16, so only its text betrays it."""
        credit = self._clip("Sous-titrage Société Radio-Canada")
        out = G.recover_untranscribed_speech(self.RESULT, self.GAP, "a.mp3", lambda p: credit)
        assert out["segments"] == self.RESULT["segments"]
        assert out["asr_speech_recovery"][0]["status"] == "rejected"
        assert out["asr_speech_recovery"][0]["rejected"] == ["invented_line"]

    def test_a_credit_cut_by_the_gap_edge_is_still_rejected(self) -> None:
        """Only "Radio-Canada" falls inside the gap; the whole clip segment is what is judged."""
        seg = {
            "start": 0.0,
            "end": 4.0,
            "text": " Sous-titrage Société Radio-Canada",
            "avg_logprob": -0.16,
            "words": [
                _w(" Sous-titrage", 0.0, 1.5),
                _w(" Société", 1.5, 2.5),
                _w(" Radio-Canada", 2.6, 4.0),
            ],
        }
        kept, rejected = G.segments_inside_gap([seg], clip_start=0.0, gap_start=2.55, gap_end=4.0)
        assert kept == [] and rejected == ["invented_line"]


@pytest.mark.skipif(not AUDIO.is_file(), reason="fixture audio missing")
def test_cut_clip_cuts_the_requested_span(tmp_path: Path) -> None:
    out = tmp_path / "clip.wav"
    G.cut_clip(str(AUDIO), 10.0, 14.5, str(out))
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=sample_rate,channels:format=duration",
            "-of",
            "json",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(probe.stdout)
    assert float(info["format"]["duration"]) == pytest.approx(4.5, abs=0.05)
    assert info["streams"][0]["sample_rate"] == "16000"
    assert info["streams"][0]["channels"] == 1


class TestThroughTheDiarizationPipeline:
    """The splice happens BEFORE alignment, so a recovered segment gets its speaker the same way
    every other segment does, and ``asr_untranscribed_speech`` reports only what is still missing.
    """

    def _run(self, monkeypatch: pytest.MonkeyPatch, transcribe_clip: Any) -> Dict[str, Any]:
        from podcast_scraper.config import Config
        from podcast_scraper.providers.ml.diarization import pipeline as P
        from podcast_scraper.providers.ml.diarization.base import (
            DiarizationResult,
            DiarizationSegment,
        )

        monkeypatch.setattr(G, "cut_clip", lambda path, s, e, out: None)
        diar = DiarizationResult(
            segments=[
                DiarizationSegment(0, 30, "SPEAKER_00"),
                DiarizationSegment(30, 50, "SPEAKER_01"),
                DiarizationSegment(50, 80, "SPEAKER_00"),
            ],
            num_speakers=2,
        )
        result = {
            "text": "Welcome to the show. Back to you.",
            "segments": [
                {"start": 0, "end": 30, "text": "Welcome to the show."},
                {"start": 50, "end": 80, "text": "Back to you."},
            ],
        }
        cfg = Config(speaker_resolution_llm=False)
        return P.apply_diarization_to_result(
            result, "ep.mp3", cfg, [], precomputed_diarization=diar, transcribe_clip=transcribe_clip
        )

    def test_the_recovered_turn_is_attributed_to_the_voice_that_spoke_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clip = {"segments": [{"start": 1.5, "end": 19.0, "text": "Thanks, it is good to be here."}]}
        out = self._run(monkeypatch, lambda path: clip)
        recovered = [s for s in out["segments"] if s.get("recovered")]
        assert len(recovered) == 1
        assert recovered[0]["text"] == "Thanks, it is good to be here."
        others = {s.get("speaker") for s in out["segments"] if not s.get("recovered")}
        assert recovered[0].get("speaker") not in others
        assert out["asr_untranscribed_speech"] == []
        assert out["asr_speech_recovery"][0]["status"] == "recovered"

    def test_without_a_transcriber_nothing_is_recovered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = self._run(monkeypatch, None)
        assert not any(s.get("recovered") for s in out["segments"])
        assert "asr_speech_recovery" not in out
        assert [(g["start"], g["end"]) for g in out["asr_untranscribed_speech"]] == [(30.0, 50.0)]
