"""A word stretched over speech is re-transcribed alone and replaced only by more speech (D16).

Radio Ambulante 2026-09-10: "¿no?" timed 15.6 s over 11.5 s of diarized speech hid 55 reference
words; SWR "einige" (3.9 s) hid 13. Of 11 stretched words re-transcribed on 2026-10-08, 3 hid
speech and 6 held only the word, a stutter or fillers. So the span is re-transcribed on its own
and the word is REPLACED -- never spliced beside it, which would duplicate it -- and only when
the result holds the word itself plus at least ``STRETCHED_MIN_EXTRA_WORDS`` other words that are
not fillers.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

import pytest

from podcast_scraper.transcription import gap_recovery as G


@pytest.fixture
def no_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> List[tuple]:
    cuts: List[tuple] = []
    monkeypatch.setattr(G, "cut_clip", lambda path, s, e, out: cuts.append((s, e)))
    return cuts


def _w(word: str, start: float, end: float) -> Dict[str, Any]:
    return {"word": word, "start": start, "end": end}


def _result() -> Dict[str, Any]:
    words = [_w(" Eso", 10.0, 10.4), _w(" fue", 10.4, 10.7), _w(" así,", 10.7, 11.0)]
    words.append(_w(" ¿no?", 11.0, 26.6))
    words.append(_w(" Bueno.", 26.6, 27.0))
    text = "".join(w["word"] for w in words)
    return {"segments": [{"start": 10.0, "end": 27.0, "text": text, "words": words}], "text": text}


_STRETCHED = [{"start": 11.0, "end": 26.6, "word": "¿no?", "duration_s": 15.6, "speech_s": 11.5}]


def _clip(*words: tuple) -> Any:
    """A clip transcriber returning ``words`` as (text, clip-relative start, end)."""

    def transcribe(_path: str) -> Dict[str, Any]:
        ws = [_w(t, s, e) for t, s, e in words]
        return {
            "segments": [
                {
                    "start": ws[0]["start"],
                    "end": ws[-1]["end"],
                    "text": "".join(w["word"] for w in ws),
                    "words": ws,
                    "avg_logprob": -0.3,
                    "compression_ratio": 1.4,
                }
            ]
        }

    return transcribe


def test_the_word_is_replaced_by_the_speech_it_hid(no_ffmpeg: List[tuple]) -> None:
    # Clip starts at 10.0 (11.0 - pad). The hidden narration sits inside 11.0-26.6.
    clip = _clip(
        (" ¿no?", 1.0, 1.4),
        (" Y", 2.0, 2.2),
        (" entonces", 2.2, 2.8),
        (" llegó", 2.8, 3.2),
        (" la", 3.2, 3.3),
        (" policía", 3.3, 3.9),
    )
    out = G.recover_stretched_words(_result(), _STRETCHED, "a.mp3", clip)
    seg = out["segments"][0]
    assert seg["text"] == " Eso fue así, ¿no? Y entonces llegó la policía Bueno."
    assert [w["word"] for w in seg["words"]].count(" ¿no?") == 1
    assert out["asr_stretched_word_recovery"][0]["status"] == "replaced"
    assert out["text"] == seg["text"].strip()


def test_only_the_word_again_changes_nothing(no_ffmpeg: List[tuple]) -> None:
    out = G.recover_stretched_words(_result(), _STRETCHED, "a.mp3", _clip((" ¿no?", 1.0, 1.4)))
    assert out["segments"] == _result()["segments"]
    assert out["asr_stretched_word_recovery"][0]["status"] == "declined"


def test_a_stutter_with_fillers_is_not_more_speech(no_ffmpeg: List[tuple]) -> None:
    clip = _clip((" eh", 1.0, 1.2), (" ¿no?", 1.2, 1.5), (" eh,", 1.5, 1.8), (" ¿no?", 1.8, 2.1))
    out = G.recover_stretched_words(_result(), _STRETCHED, "a.mp3", clip)
    assert out["segments"] == _result()["segments"]


def test_speech_without_the_word_is_not_a_replacement(no_ffmpeg: List[tuple]) -> None:
    """The word must be heard again, or the result may be another stretch of audio."""
    clip = _clip(
        (" Y", 2.0, 2.2), (" entonces", 2.2, 2.8), (" llegó", 2.8, 3.2), (" ella", 3.2, 3.6)
    )
    out = G.recover_stretched_words(_result(), _STRETCHED, "a.mp3", clip)
    assert out["segments"] == _result()["segments"]


def test_a_low_confidence_clip_is_rejected(no_ffmpeg: List[tuple]) -> None:
    clip = _clip(
        (" ¿no?", 1.0, 1.4), (" Y", 2.0, 2.2), (" entonces", 2.2, 2.8), (" llegó", 2.8, 3.2)
    )

    def low(path: str) -> Dict[str, Any]:
        r: Dict[str, Any] = clip(path)
        r["segments"][0]["avg_logprob"] = -1.6
        return r

    out = G.recover_stretched_words(_result(), _STRETCHED, "a.mp3", low)
    assert out["segments"] == _result()["segments"]


def test_a_failed_call_keeps_the_transcript(no_ffmpeg: List[tuple]) -> None:
    def boom(_path: str) -> Dict[str, Any]:
        raise RuntimeError("endpoint down")

    out = G.recover_stretched_words(_result(), _STRETCHED, "a.mp3", boom)
    assert out["segments"] == _result()["segments"]
    assert out["asr_stretched_word_recovery"][0]["status"] == "failed"


def test_nothing_stretched_means_no_call(no_ffmpeg: List[tuple]) -> None:
    result = _result()
    assert G.recover_stretched_words(result, [], "a.mp3", _clip((" x", 0, 1))) is result
    assert no_ffmpeg == []


def test_the_input_is_not_mutated(no_ffmpeg: List[tuple]) -> None:
    result = _result()
    before = copy.deepcopy(result)
    clip = _clip(
        (" ¿no?", 1.0, 1.4), (" Y", 2.0, 2.2), (" entonces", 2.2, 2.8), (" llegó", 2.8, 3.2)
    )
    G.recover_stretched_words(result, _STRETCHED, "a.mp3", clip)
    assert result == before


def test_the_diarization_pipeline_replaces_it_before_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from podcast_scraper.config import Config
    from podcast_scraper.providers.ml.diarization import pipeline as P
    from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment

    monkeypatch.setattr(G, "cut_clip", lambda path, s, e, out: None)
    diar = DiarizationResult(segments=[DiarizationSegment(0, 40, "SPEAKER_00")], num_speakers=1)
    clip = _clip(
        (" ¿no?", 1.0, 1.4),
        (" Y", 2.0, 2.2),
        (" entonces", 2.2, 2.8),
        (" llegó", 2.8, 3.2),
        (" la", 3.2, 3.3),
        (" policía", 3.3, 3.9),
    )
    out = P.apply_diarization_to_result(
        _result(),
        "ep.mp3",
        Config(speaker_resolution_llm=False),
        [],
        precomputed_diarization=diar,
        transcribe_clip=clip,
    )
    assert out["asr_stretched_word_recovery"][0]["status"] == "replaced"
    assert "policía" in " ".join(s["text"] for s in out["segments"])
    assert out["asr_stretched_words"] == []
