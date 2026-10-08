"""Re-transcribe the windows where Whisper's punctuation broke off, and keep only real repairs.

Long-form decoding conditions each 30-second window on the previous one's text, so once a window
comes out unpunctuated the style can hold to the end of the episode: five of the six V.6b real
feeds (2026-10-08) and 121 of 2,421 prod episodes of 20+ minutes. A prompt on the whole file
does not fix it -- it shapes only the first window (pt-PT: still 3 windows flagged), and VAD only
delays it (minute 30 instead of 10). Re-transcribing the broken window ON ITS OWN with a
punctuated prompt in the episode's language does (pt-PT: 4.2 / 1.6 / 2.7 -> 74.5 / 83.0 / 99.5
sentence ends per 1,000 words).

A window is replaced only when the new text is punctuated, does not echo the prompt, has no
segment over Whisper's loop threshold (compression ratio), keeps the old window's words
(``MIN_RETAINED``), adds no more than ``MAX_WORD_RATIO`` where the old decode already had text,
and adds new words only at speech rate (``MAX_NEW_WORDS_PER_S``) where it had none. Anything
else keeps the original window.

The broken decode also DROPS speech: on an English prod episode the old windows covered 377-417
of 600 s, with 18-24 s holes, and the re-decode filled them at speech rate (35-88 words per
hole) while keeping 98-99.6% of the old words — 35-55% more words in all. A plain cap on total
words refused those repairs; the old decode's coverage is what separates recovered speech from
invention.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
from collections import Counter
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from podcast_scraper.transcription import punctuation
from podcast_scraper.transcription.gap_recovery import cut_clip, RECOVERY_MAX_COMPRESSION_RATIO

logger = logging.getLogger(__name__)

#: Share of the old window's words (as a bag) the repair must also contain: the same speech,
#: nothing dropped. Measured on 34 repairs (25 V.6b, 9 English prod): 91.8-99.5%.
MIN_RETAINED = 0.9
#: Where the old decode already had text, the repair may carry at most this many times its
#: words: more there is not a better hearing of the same speech.
MAX_WORD_RATIO = 1.3
#: Where the old decode had NO text, new words at most this fast: fast conversational speech is
#: ~3.5 words/s; the recovered English holes ran 1.8-4.4 (35-88 words in 18-24 s).
MAX_NEW_WORDS_PER_S = 4.5

_TOKEN = re.compile(r"[^\w]+", re.UNICODE)

#: ``(clip_path, prompt) -> {"text", "segments"}`` in clip-relative time.
WindowTranscriber = Callable[[str, str], Mapping[str, Any]]


def _f(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _starts_in(seg: Mapping[str, Any], lo: float, hi: float) -> bool:
    """A segment with usable times that starts in ``[lo, hi)``; one without is left alone."""
    start, end = _f(seg.get("start")), _f(seg.get("end"))
    return start is not None and end is not None and lo <= start < hi


def _shift(seg: Mapping[str, Any], offset: float) -> Dict[str, Any]:
    out = dict(seg)
    for key in ("start", "end"):
        if _f(seg.get(key)) is not None:
            out[key] = round(float(seg[key]) + offset, 3)
    words = seg.get("words")
    if isinstance(words, list):
        out["words"] = [
            {
                **w,
                "start": round(float(w["start"]) + offset, 3),
                "end": round(float(w["end"]) + offset, 3),
            }
            for w in words
            if isinstance(w, Mapping) and _f(w.get("start")) is not None
        ]
    out["punctuation_repaired"] = True
    return out


def _bag(text: str) -> Counter:
    return Counter(t for t in (_TOKEN.sub("", w.lower()) for w in text.split()) if t)


def _merged(intervals: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _verdict(
    old: Sequence[Mapping[str, Any]],
    new_segments: Sequence[Mapping[str, Any]],
    new_text: str,
    span: Tuple[float, float],
    prompt: str,
) -> Optional[str]:
    """None when the new window is a real repair, else why it is refused.

    ``old`` and ``new_segments`` are in episode time; ``span`` is the window as cut.
    """
    old_text = " ".join(str(s.get("text", "")) for s in old)
    if punctuation.is_unpunctuated(new_text):
        return "still_unpunctuated"
    if punctuation.echoes_prompt(new_text, prompt):
        return "echoed_prompt"
    # Whisper's own loop measure only. The short-phrase rule gap recovery uses (``is_repetitive``)
    # vetoed two good V.6b Spanish windows over ONE ordinary filler run each ("bla, bla, bla,
    # bla" in 113 segments); a real loop over a window also inflates the counts below.
    for seg in new_segments:
        ratio = _f(seg.get("compression_ratio"))
        if ratio is not None and ratio > RECOVERY_MAX_COMPRESSION_RATIO:
            return "repetitive"
    old_bag, new_bag = _bag(old_text), _bag(new_text)
    old_n = sum(old_bag.values())
    if old_n and sum((old_bag & new_bag).values()) / old_n < MIN_RETAINED:
        return "lost_words"
    covered = _merged([(float(s["start"]), float(s["end"])) for s in old])
    inside_n = outside_n = 0
    for seg in new_segments:
        mid = (float(seg["start"]) + float(seg["end"])) / 2
        n = len(str(seg.get("text", "")).split())
        if any(a <= mid <= b for a, b in covered):
            inside_n += n
        else:
            outside_n += n
    if old_n and inside_n > MAX_WORD_RATIO * old_n:
        return "word_count_changed"
    uncovered_s = (span[1] - span[0]) - sum(b - a for a, b in covered)
    if outside_n and outside_n > MAX_NEW_WORDS_PER_S * max(uncovered_s, 0.0):
        return "too_many_new_words"
    return None


def repair_unpunctuated_windows(
    result: Dict[str, Any],
    windows: Sequence[Sequence[float]],
    audio_path: str,
    transcribe_window: WindowTranscriber,
    prompt: str,
) -> Dict[str, Any]:
    """Return ``result`` with each repairable window re-transcribed and spliced in.

    Each window is cut at segment edges (first segment starting in it to the end of the last
    one), transcribed with ``prompt``, judged by ``_verdict`` and either replaces the window's
    segments (tagged ``punctuation_repaired``) or is refused. The attempts are recorded as
    ``asr_punctuation_repair``. A failed call keeps the window. With no windows, the input is
    returned unchanged.
    """
    if not windows:
        return result
    segments = [s for s in (result.get("segments") or []) if isinstance(s, Mapping)]
    report: List[Dict[str, Any]] = []
    work_dir = tempfile.mkdtemp(prefix="punct_repair_")
    # As in gap recovery: after a failed CALL the endpoint is not answering, so the remaining
    # windows are skipped rather than each waiting out a timeout under the DGX lock.
    call_failed = False
    try:
        for i, (w_start, w_end) in enumerate(windows):
            inside = [s for s in segments if _starts_in(s, w_start, w_end)]
            if not inside:
                continue
            a, b = float(inside[0]["start"]), max(float(s["end"]) for s in inside)
            old_text = " ".join(str(s.get("text", "")).strip() for s in inside)
            entry: Dict[str, Any] = {
                "start": round(a, 3),
                "end": round(b, 3),
                "sentence_ends_per_1000_words_before": round(
                    punctuation.sentence_ends_per_1000_words(old_text), 1
                ),
                "words_before": len(old_text.split()),
            }
            if call_failed:
                report.append({**entry, "status": "skipped", "reason": "earlier_call_failed"})
                continue
            clip = os.path.join(work_dir, f"window_{i:03d}.wav")
            try:
                cut_clip(audio_path, a, b, clip)
            except Exception as exc:  # noqa: BLE001 - never lose the transcript to a repair call
                logger.warning("punctuation repair: %.0f-%.0fs could not be cut: %s", a, b, exc)
                report.append({**entry, "status": "failed", "error": type(exc).__name__})
                continue
            try:
                new = transcribe_window(clip, prompt)
            except Exception as exc:  # noqa: BLE001 - never lose the transcript to a repair call
                logger.warning(
                    "punctuation repair: %.0f-%.0fs failed (%s); skipping the remaining windows",
                    a,
                    b,
                    exc,
                )
                report.append({**entry, "status": "failed", "error": type(exc).__name__})
                call_failed = True
                continue
            new_text = str(new.get("text") or "")
            entry["sentence_ends_per_1000_words_after"] = round(
                punctuation.sentence_ends_per_1000_words(new_text), 1
            )
            entry["words_after"] = len(new_text.split())
            new_segs = [
                _shift(s, a)
                for s in (new.get("segments") or [])
                if isinstance(s, Mapping)
                and str(s.get("text", "")).strip()
                and _f(s.get("start")) is not None
                and _f(s.get("end")) is not None
            ]
            refused = _verdict(inside, new_segs, new_text, (a, b), prompt)
            if refused is not None:
                report.append({**entry, "status": "refused", "reason": refused})
                continue
            replaced = {id(s) for s in inside}
            keep = [s for s in segments if id(s) not in replaced]
            segments = sorted([*keep, *new_segs], key=lambda s: float(s.get("start", 0.0)))
            report.append({**entry, "status": "repaired"})
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    out = dict(result)
    out["asr_punctuation_repair"] = report
    if any(r["status"] == "repaired" for r in report):
        out["segments"] = segments
        out["text"] = " ".join(str(s.get("text", "")).strip() for s in segments if s.get("text"))
    logger.info(
        "punctuation repair: %d of %d window(s) repaired",
        sum(1 for r in report if r["status"] == "repaired"),
        len(report),
    )
    return out
