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


def test_after_a_failed_call_the_remaining_windows_are_skipped(no_ffmpeg: List[tuple]) -> None:
    """A failed call means the endpoint is not answering; each further window would wait out its
    own timeout under the DGX lock, so the rest are skipped and reported, not attempted."""
    calls: List[str] = []

    def boom(path: str, prompt: str) -> Dict[str, Any]:
        calls.append(path)
        raise TimeoutError("dgx wedged")

    before = _result()
    before["segments"] += [{"start": 1300.0 + i, "end": 1300.5 + i, "text": FLAT} for i in range(3)]
    out = R.repair_unpunctuated_windows(before, WINDOW + [[1300.0, 1400.0]], "a.mp3", boom, PROMPT)
    assert len(calls) == 1, "one failed call, then no more"
    assert [e["status"] for e in out["asr_punctuation_repair"]] == ["failed", "skipped"]
    assert out["asr_punctuation_repair"][1]["reason"] == "earlier_call_failed"
    assert out["segments"] == before["segments"]


def test_a_failed_cut_does_not_stop_the_other_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cut failure is local to its clip — it says nothing about the endpoint."""
    cuts: List[float] = []

    def cut(audio: str, a: float, b: float, out: str) -> None:
        cuts.append(a)
        if len(cuts) == 1:
            raise OSError("ffmpeg")

    monkeypatch.setattr(R, "cut_clip", cut)
    calls: List[str] = []

    def ok(path: str, prompt: str) -> Dict[str, Any]:
        calls.append(path)
        return {"text": "", "segments": []}

    before = _result()
    before["segments"] += [{"start": 1300.0 + i, "end": 1300.5 + i, "text": FLAT} for i in range(3)]
    out = R.repair_unpunctuated_windows(before, WINDOW + [[1300.0, 1400.0]], "a.mp3", ok, PROMPT)
    assert [e["status"] for e in out["asr_punctuation_repair"]][0] == "failed"
    assert len(calls) == 1, "the second window was still attempted"


def test_a_segment_without_times_is_left_alone(no_ffmpeg: List[tuple]) -> None:
    before = _result()
    before["segments"].insert(1, {"text": "no times on this one"})
    out = R.repair_unpunctuated_windows(before, WINDOW, "a.mp3", lambda p, pr: _repair(), PROMPT)
    assert {"text": "no times on this one"} in out["segments"]


# Up First 10-09 (2026-10-12): the repair of 602-768 s was refused as "repetitive" although its
# 160 s of text were good. Over the clip's near-silent end Whisper re-emitted lines the credits had
# already said, and its reported ratio for that last window (2.23 in a replay, over 2.4 in the
# run) counts text it never put in a segment: the segments' own text measures 1.47.

UPFIRST_TAIL = [
    # (seek, start, text, reported compression_ratio)
    (14028, 140.3, "As we finish today, we'll remind you the NPR Network needs you.", 1.48),
    (14028, 142.0, "I'm Steve Inskeep.", 1.48),
    (14028, 142.9, "And I'm Michelle Martin.", 1.48),
    (14028, 160.1, "And our executive producer is Jay Shaler.", 1.48),
    (14028, 162.3, "We hope you'll join us again on Monday.", 1.48),
    (16128, 163.9, "I'm Steve Inskeep.", 3.1),
    (16128, 167.5, "We hope you'll join us again on Monday.", 3.1),
    (16128, 167.8, "We hope you'll join us again on Monday.", 3.1),
    (16128, 167.9, "Bye.", 3.1),
]


def _segs(rows: Sequence[tuple]) -> List[Dict[str, Any]]:
    return [
        {"seek": k, "start": t, "end": t + 0.5, "text": x, "compression_ratio": c}
        for k, t, x, c in rows
    ]


def test_lines_an_earlier_window_holds_are_dropped_from_the_last_window() -> None:
    kept, dropped = R.drop_tail_repeats(_segs(UPFIRST_TAIL))
    assert dropped == [
        "I'm Steve Inskeep.",
        "We hope you'll join us again on Monday.",
        "We hope you'll join us again on Monday.",
    ]
    assert [s["text"] for s in kept][-2:] == ["We hope you'll join us again on Monday.", "Bye."]


def test_a_window_whose_kept_text_is_ordinary_is_not_repetitive() -> None:
    kept, _ = R.drop_tail_repeats(_segs(UPFIRST_TAIL))
    assert R.repetitive_window(kept) is None


def test_a_loop_inside_one_window_is_still_repetitive_and_reported() -> None:
    """A line looping within one window is not an earlier window's line, so it is not dropped,
    and its own text measures high."""
    loop = _segs([(0, 3.4 * i, "Grazie a tutti.", 6.0) for i in range(10)])
    kept, dropped = R.drop_tail_repeats(loop)
    assert dropped == [] and len(kept) == 10
    found = R.repetitive_window(kept)
    assert found is not None and found["ratio"] > 2.4 and found["reported_ratio"] == 6.0
    assert found["text"].startswith("Grazie a tutti. Grazie a tutti.")


def test_without_seek_the_reported_ratio_decides_as_before() -> None:
    segs = [{"start": 0.0, "end": 1.0, "text": "Fine.", "compression_ratio": 3.1}]
    assert R.repetitive_window(segs) is not None
    assert R.drop_tail_repeats(segs) == (segs, [])


def test_the_repair_drops_tail_repeats_and_records_them(no_ffmpeg: List[tuple]) -> None:
    line = _punctuated(FLAT)
    tail = [
        {"seek": 3000, "start": 198.5 + 0.2 * i, "end": 198.6 + 0.2 * i, "text": line,
         "compression_ratio": 3.1}
        for i in range(2)
    ]  # fmt: skip
    clip = _clip([line] * 20, extra=tail, seek=0, compression_ratio=1.6)
    out = R.repair_unpunctuated_windows(_result(), WINDOW, "a.mp3", lambda p, pr: clip, PROMPT)
    [entry] = out["asr_punctuation_repair"]
    assert entry["status"] == "repaired"
    assert entry["tail_repeats_dropped"] == [line, line]
    assert len(out["segments"]) == 21
