"""A window whose punctuation broke off is re-transcribed and replaced only by a real repair.

#2187. WHY THIS EXISTS. Five of the six V.6b real feeds (2026-10-08) and 121 of 2,421 prod
episodes of 20+ minutes lose their punctuation part-way and keep it lost. Re-transcribing such a
window on its own with a punctuated prompt fixed all 25 broken windows on five V.6b feeds. The
broken decode also DROPS speech (an English prod window covered 377 of 600 s), so a real repair can
carry many more words — in the time the old decode left empty. These tests pin what counts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import pytest

from podcast_scraper.transcription import punctuation_repair as R

pytestmark = pytest.mark.unit

FLAT = "so the ports moved north over a decade and the trading families followed them there"
PROMPT = "Hola, y bienvenidos. En este episodio hablamos del tema en cuestión."
WINDOW = [[600.0, 1200.0]]
SEG_TIMES = [(600.0 + 10 * i, 609.0 + 10 * i) for i in range(20)]  # 9 s of every 10 s covered


def _punctuated(text: str) -> str:
    """The same words, as a punctuated decode writes them."""
    words = text.split()
    return words[0].capitalize() + " " + " ".join(words[1:8]) + ". " + " ".join(words[8:]) + "."


@pytest.fixture
def no_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> List[tuple]:
    cuts: List[tuple] = []
    monkeypatch.setattr(R, "cut_clip", lambda audio, a, b, out: cuts.append((a, b)))
    return cuts


def _result() -> Dict[str, Any]:
    first = "So the ports moved north. Why? The delta silted up, and trade moved on. " * 40
    segs: List[Dict[str, Any]] = [{"start": 0.0, "end": 590.0, "text": first}]
    segs += [{"start": a, "end": b, "text": FLAT} for a, b in SEG_TIMES]
    return {"text": " ".join([first] + [FLAT] * 20), "segments": segs, "language": "es"}


def _clip(texts: List[str], extra: Sequence[Dict[str, Any]] = (), **seg: Any) -> Dict[str, Any]:
    """A clip answer in clip time (the window is cut from 600.0)."""
    segs: List[Dict[str, Any]] = [
        {"start": a - 600.0, "end": b - 600.0, "text": t, **seg}
        for (a, b), t in zip(SEG_TIMES, texts)
    ] + list(extra)
    return {"text": " ".join(s["text"] for s in segs), "segments": segs}


def _repair() -> Dict[str, Any]:
    return _clip([_punctuated(FLAT)] * 20)


def test_a_real_repair_replaces_the_window(no_ffmpeg: List[tuple]) -> None:
    calls: List[tuple] = []

    def transcribe(path: str, prompt: str) -> Dict[str, Any]:
        calls.append((path, prompt))
        return _repair()

    out = R.repair_unpunctuated_windows(_result(), WINDOW, "a.mp3", transcribe, PROMPT)
    assert no_ffmpeg == [(600.0, 799.0)], "cut at segment edges inside the window"
    assert [c[1] for c in calls] == [PROMPT]
    assert [s["start"] for s in out["segments"][:3]] == [0.0, 600.0, 610.0]
    assert len(out["segments"]) == 21
    assert all(s.get("punctuation_repaired") for s in out["segments"][1:])
    assert out["segments"][1]["text"].startswith("So the ports")
    [entry] = out["asr_punctuation_repair"]
    assert entry["status"] == "repaired"
    assert entry["sentence_ends_per_1000_words_before"] == 0.0
    assert entry["sentence_ends_per_1000_words_after"] > 50


def test_speech_the_broken_decode_dropped_is_kept(no_ffmpeg: List[tuple]) -> None:
    """English prod: the old decode covered 377 of 600 s and the re-decode filled the empty
    seconds at speech rate — twice the words in all, which a plain cap on words refused."""
    first = "So the ports moved north. Why? The delta silted up, and trade moved on. " * 40
    sparse = [(600.0 + 10 * i, 605.0 + 10 * i) for i in range(20)]  # 5 s of every 10 s
    old: List[Dict[str, Any]] = [{"start": 0.0, "end": 590.0, "text": first}] + [
        {"start": a, "end": b, "text": FLAT} for a, b in sparse
    ]
    result = {"text": " ".join(s["text"] for s in old), "segments": old}
    filled = "And then they talked about the river trade for years and years after."
    new_segs: List[Dict[str, Any]] = [
        {"start": a - 600.0, "end": b - 600.0, "text": _punctuated(FLAT)} for a, b in sparse
    ] + [{"start": 5.5 + 10 * i, "end": 9.5 + 10 * i, "text": filled} for i in range(20)]
    clip = {"text": " ".join(s["text"] for s in new_segs), "segments": new_segs}
    out = R.repair_unpunctuated_windows(result, WINDOW, "a.mp3", lambda p, pr: clip, PROMPT)
    [entry] = out["asr_punctuation_repair"]
    assert entry["status"] == "repaired"
    assert entry["words_after"] >= 1.8 * entry["words_before"], "far past the old 130% cap"


@pytest.mark.parametrize(
    ("clip", "reason"),
    [
        (lambda: _clip([FLAT] * 20), "still_unpunctuated"),
        (
            lambda: _clip([PROMPT + " " + _punctuated(FLAT)] + [_punctuated(FLAT)] * 19),
            "echoed_prompt",
        ),
        (lambda: _clip([_punctuated(FLAT)] * 8), "lost_words"),
        (lambda: _clip([_punctuated(FLAT + " " + FLAT)] * 20), "word_count_changed"),
        (
            lambda: _clip(
                [_punctuated(FLAT)] * 20,
                extra=[{"start": 9.1, "end": 9.9, "text": " ".join(["word"] * 100) + "."}],
            ),
            "too_many_new_words",
        ),
        (lambda: _clip([_punctuated(FLAT)] * 20, compression_ratio=3.1), "repetitive"),
    ],
)
def test_anything_else_keeps_the_original(no_ffmpeg: List[tuple], clip, reason: str) -> None:
    before = _result()
    out = R.repair_unpunctuated_windows(before, WINDOW, "a.mp3", lambda p, pr: clip(), PROMPT)
    assert out["segments"] == before["segments"]
    assert out["text"] == before["text"]
    assert out["asr_punctuation_repair"][0]["status"] == "refused"
    assert out["asr_punctuation_repair"][0]["reason"] == reason


def test_an_ordinary_filler_run_does_not_veto_a_repair(no_ffmpeg: List[tuple]) -> None:
    """V.6b es: two good windows were refused over ONE segment each of a filler said four times
    (compression ratio 1.6). Only Whisper's loop measure refuses a window."""
    texts = [_punctuated(FLAT)] * 20
    texts[3] = texts[3] + " Bla, bla, bla, bla."
    clip = _clip(texts, compression_ratio=1.6)
    out = R.repair_unpunctuated_windows(_result(), WINDOW, "a.mp3", lambda p, pr: clip, PROMPT)
    assert out["asr_punctuation_repair"][0]["status"] == "repaired"


def test_a_failed_call_keeps_the_window(no_ffmpeg: List[tuple]) -> None:
    def boom(path: str, prompt: str) -> Dict[str, Any]:
        raise ConnectionError("dgx down")

    before = _result()
    out = R.repair_unpunctuated_windows(before, WINDOW, "a.mp3", boom, PROMPT)
    assert out["segments"] == before["segments"]
    assert out["asr_punctuation_repair"][0]["status"] == "failed"
    assert out["asr_punctuation_repair"][0]["error"] == "ConnectionError"


def test_no_windows_means_no_call_and_the_same_result(no_ffmpeg: List[tuple]) -> None:
    before = _result()
    out = R.repair_unpunctuated_windows(before, [], "a.mp3", lambda p, pr: {}, PROMPT)
    assert out is before and no_ffmpeg == []


def test_the_input_is_not_mutated(no_ffmpeg: List[tuple]) -> None:
    before = _result()
    snapshot = [dict(s) for s in before["segments"]]
    R.repair_unpunctuated_windows(before, WINDOW, "a.mp3", lambda p, pr: _repair(), PROMPT)
    assert before["segments"] == snapshot
    assert "asr_punctuation_repair" not in before
