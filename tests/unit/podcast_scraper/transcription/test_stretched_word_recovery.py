"""A word stretched over speech is re-transcribed alone and replaced by the speech it hid (D16).

Radio Ambulante 07-21: "¿no?" timed 15.6 s over 11.5 s of diarized speech hid a whole exchange
the clip transcribes (about 35 words). SWR's "Europäer": the clip's words inside the span are the
episode's NEXT words, timed later, so adding them would duplicate them. The clip and the episode's
own words around the stretched one are aligned; only words the episode lacks replace the stretched
word, and only when they are ``STRETCHED_MIN_EXTRA_WORDS`` or more, not fillers. Scored against
the human references on 7 stretched words (2026-10-10): adjusted errors -33, -3, -5, 0.
"""

from __future__ import annotations

import copy
from pathlib import Path
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


def test_speech_the_episode_lacks_replaces_the_word_even_unheard(no_ffmpeg: List[tuple]) -> None:
    """Radio Ambulante: the clip does not hear "¿no?" at all, only the exchange it covered."""
    clip = _clip(
        (" Y", 2.0, 2.2), (" entonces", 2.2, 2.8), (" llegó", 2.8, 3.2), (" ella", 3.2, 3.6)
    )
    out = G.recover_stretched_words(_result(), _STRETCHED, "a.mp3", clip)
    assert out["segments"][0]["text"] == " Eso fue así, Y entonces llegó ella Bueno."


def test_words_the_episode_already_has_beside_it_are_not_added(no_ffmpeg: List[tuple]) -> None:
    """SWR "Europäer": the clip re-hears the episode's next words inside the span."""
    words = [_w(" Ländern.", 9.0, 10.9), _w(" Europäer", 11.0, 16.0)]
    words += [_w(" Wenn", 16.2, 16.4), _w(" man", 16.4, 16.6), _w(" die", 16.6, 16.8)]
    words += [_w(" Kriminalität", 16.8, 17.5)]
    text = "".join(w["word"] for w in words)
    result = {"segments": [{"start": 9.0, "end": 17.5, "text": text, "words": words}]}
    stretched = [{"start": 11.0, "end": 16.0, "word": "Europäer"}]
    clip = _clip(
        (" Wenn", 4.0, 4.3), (" man", 4.3, 4.5), (" die", 4.5, 4.8), (" Kriminalität", 5.2, 6.0)
    )
    out = G.recover_stretched_words(result, stretched, "a.mp3", clip)
    assert out["segments"][0]["text"] == text
    assert out["asr_stretched_word_recovery"][0]["reason"] == "not_more_speech"


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


def test_the_recovery_is_kept_in_the_asr_record(tmp_path: Path) -> None:
    """The first real run (SWR, 2026-10-10) left no trace of what was decided: ``.asr.json`` copies
    named keys only, and this one was not among them."""
    import json

    from podcast_scraper.config import Config
    from podcast_scraper.workflow import episode_processor as epx

    (tmp_path / "transcripts").mkdir()
    rel = "transcripts/0001 - ep.txt"
    (tmp_path / rel).write_text("x")
    record = [{"start": 11.0, "end": 26.6, "word": "¿no?", "status": "declined"}]
    result = {"speech_audio_ratio": 0.9, "asr_stretched_word_recovery": record}
    epx._save_asr_provenance_file(result, Config(), rel, str(tmp_path))
    asr = json.loads((tmp_path / "transcripts" / "0001 - ep.asr.json").read_text())
    assert asr["stretched_word_recovery"] == record
