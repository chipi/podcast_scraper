"""Re-transcribe diarized speech the long-form transcript has no words for (#2187, A2).

Whisper's long-form decode can skip a stretch of speech outright: on 80k_03 (raw audio) its
segments jump from 23.4 s to 48.8 s while one speaker talks throughout, losing 77 words; the V.6b
Spanish fixture lost a 19 s ad read the same way. The diarizer hears those stretches
(``untranscribed_speech``), and the same audio transcribes when it is cut out and sent alone.

So each gap is cut from the episode audio with a little context either side, transcribed in the
episode's declared language, and what Whisper says INSIDE the gap is spliced into the transcript as
segments tagged ``recovered: True``. Nothing outside the gap is taken, so no transcribed word is
duplicated, and nothing already in the transcript is changed.

What a recovered clip must pass, because a short clip is where Whisper invents text:

- confidence: ``avg_logprob`` at or above ``RECOVERY_MIN_AVG_LOGPROB``;
- not a loop: ``compression_ratio`` at or below ``RECOVERY_MAX_COMPRESSION_RATIO`` and no run of
  the same short phrase repeated (``is_repetitive``);
- not an invented line: a subtitle credit or video sign-off (``invented_lines``), which passes
  both checks above (V.6b French feed: avg_logprob -0.1 to -0.3).

``no_speech_prob`` is NOT used: the DGX faster-whisper server returns 0.0 for every segment (all
six 80k A/B responses, 2026-10-08), so it cannot tell speech from silence there. The diarizer's
speaker turn is what says the gap is speech.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from podcast_scraper.preprocessing.audio.ffmpeg_processor import _run_text_subprocess
from podcast_scraper.transcription.invented_lines import is_invented_line

logger = logging.getLogger(__name__)

#: Context either side of a gap, so the clip does not start or end mid-word.
RECOVERY_PAD_S = 1.0
#: Below this mean token log-probability a recovered segment is a guess, not a hearing.
RECOVERY_MIN_AVG_LOGPROB = -1.0
#: Whisper's own loop threshold (``compression_ratio_threshold`` in openai-whisper).
RECOVERY_MAX_COMPRESSION_RATIO = 2.4

ClipTranscriber = Callable[[str], Mapping[str, Any]]

_WORD = re.compile(r"\w+", re.UNICODE)


def is_repetitive(text: str) -> bool:
    """True when ``text`` is one short phrase said over and over — Whisper's loop failure.

    A phrase of 1-4 words repeated 4 or more times back to back, or 8+ words of which fewer than a
    third are distinct.
    """
    words = [w.lower() for w in _WORD.findall(text)]
    if len(words) >= 8 and len(set(words)) / len(words) < 1 / 3:
        return True
    for n in range(1, 5):
        for i in range(0, len(words) - 4 * n + 1):
            phrase = words[i : i + n]
            if all(words[i + k * n : i + (k + 1) * n] == phrase for k in range(1, 4)):
                return True
    return False


def cut_clip(audio_path: str, start: float, end: float, out_path: str) -> None:
    """Cut ``[start, end)`` of ``audio_path`` to a mono 16 kHz WAV (what ASR resamples to)."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not available for gap recovery")
    _run_text_subprocess(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            f"{max(0.0, start):.3f}",
            "-t",
            f"{max(0.0, end - start):.3f}",
            "-i",
            audio_path,
            "-ac",
            "1",
            "-ar",
            "16000",
            out_path,
        ],
        timeout=120.0,
        check=True,
    )


def _f(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _inside(start: Optional[float], end: Optional[float], lo: float, hi: float) -> bool:
    if start is None or end is None:
        return False
    return lo <= (start + end) / 2 < hi


def _rejection(seg: Mapping[str, Any], text: str) -> Optional[str]:
    # The WHOLE clip segment, not the words inside the gap: a credit line sliced by the gap edges
    # no longer reads as one.
    if is_invented_line(seg.get("text")):
        return "invented_line"
    logprob = _f(seg.get("avg_logprob"))
    if logprob is not None and logprob < RECOVERY_MIN_AVG_LOGPROB:
        return "low_confidence"
    ratio = _f(seg.get("compression_ratio"))
    if (ratio is not None and ratio > RECOVERY_MAX_COMPRESSION_RATIO) or is_repetitive(text):
        return "repetitive"
    return None


def segments_inside_gap(
    clip_segments: Sequence[Mapping[str, Any]],
    *,
    clip_start: float,
    gap_start: float,
    gap_end: float,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """The clip's words that fall inside the gap, as absolute-time segments tagged ``recovered``.

    Word timestamps decide what is inside when the segment has them (a word belongs where its
    midpoint is); otherwise the whole segment does. Returns the kept segments and one rejection
    reason per segment dropped by the filters.
    """
    kept: List[Dict[str, Any]] = []
    rejected: List[str] = []
    for seg in clip_segments:
        s, e = _f(seg.get("start")), _f(seg.get("end"))
        if s is None or e is None:
            continue
        words = [w for w in (seg.get("words") or []) if isinstance(w, Mapping)]
        if words:
            inside = [
                {
                    **w,
                    "start": round(clip_start + float(w["start"]), 3),
                    "end": round(clip_start + float(w["end"]), 3),
                }
                for w in words
                if _inside(
                    _f(w.get("start")),
                    _f(w.get("end")),
                    gap_start - clip_start,
                    gap_end - clip_start,
                )
            ]
            if not inside:
                continue
            text = " ".join(str(w.get("word", "")).strip() for w in inside).strip()
            start, end = inside[0]["start"], inside[-1]["end"]
        else:
            if not _inside(s, e, gap_start - clip_start, gap_end - clip_start):
                continue
            inside = []
            text = str(seg.get("text", "")).strip()
            start, end = round(clip_start + s, 3), round(clip_start + e, 3)
        if not text:
            continue
        reason = _rejection(seg, text)
        if reason is not None:
            rejected.append(reason)
            continue
        out: Dict[str, Any] = {"start": start, "end": end, "text": text, "recovered": True}
        for key in ("avg_logprob", "compression_ratio"):
            if key in seg:
                out[key] = seg[key]
        if inside:
            out["words"] = inside
        kept.append(out)
    return kept, rejected


def recover_untranscribed_speech(
    result: Dict[str, Any],
    gaps: Sequence[Mapping[str, Any]],
    audio_path: str,
    transcribe_clip: ClipTranscriber,
) -> Dict[str, Any]:
    """Splice re-transcribed gaps into ``result``; return a new dict with ``asr_speech_recovery``.

    ``transcribe_clip(path)`` transcribes one clip file in the episode's language and returns a
    dict with ``segments`` (clip-relative times). A gap whose call fails is reported and skipped:
    the transcript the episode already has is never lost to a recovery attempt.

    With no gaps the input is returned unchanged — no key added, no call made.
    """
    if not gaps:
        return result
    segments = [s for s in (result.get("segments") or []) if isinstance(s, Mapping)]
    report: List[Dict[str, Any]] = []
    added: List[Dict[str, Any]] = []
    work_dir = tempfile.mkdtemp(prefix="gap_recovery_")
    # A failed CALL means the endpoint is not answering, and every further clip would wait out its
    # own timeout while holding the DGX lock; the rest are skipped. A failed CUT is local to its
    # clip and says nothing about the endpoint.
    call_failed = False
    try:
        for i, gap in enumerate(gaps):
            gap_start, gap_end = float(gap["start"]), float(gap["end"])
            clip_start = max(0.0, gap_start - RECOVERY_PAD_S)
            entry: Dict[str, Any] = {
                "start": gap_start,
                "end": gap_end,
                "speaker": gap.get("speaker"),
            }
            if call_failed:
                report.append({**entry, "status": "skipped", "reason": "earlier_call_failed"})
                continue
            clip = os.path.join(work_dir, f"gap_{i:03d}.wav")
            try:
                cut_clip(audio_path, clip_start, gap_end + RECOVERY_PAD_S, clip)
            except Exception as exc:  # noqa: BLE001 - never lose the transcript to a recovery call
                logger.warning(
                    "gap recovery: %.1f-%.1fs could not be cut: %s", gap_start, gap_end, exc
                )
                report.append({**entry, "status": "failed", "error": type(exc).__name__})
                continue
            try:
                clip_result = transcribe_clip(clip)
            except Exception as exc:  # noqa: BLE001 - never lose the transcript to a recovery call
                logger.warning(
                    "gap recovery: %.1f-%.1fs could not be re-transcribed (%s); skipping the "
                    "remaining gaps",
                    gap_start,
                    gap_end,
                    exc,
                )
                report.append({**entry, "status": "failed", "error": type(exc).__name__})
                call_failed = True
                continue
            kept, rejected = segments_inside_gap(
                [s for s in (clip_result.get("segments") or []) if isinstance(s, Mapping)],
                clip_start=clip_start,
                gap_start=gap_start,
                gap_end=gap_end,
            )
            words = sum(len(_WORD.findall(s["text"])) for s in kept)
            status = "recovered" if kept else ("rejected" if rejected else "empty")
            report.append({**entry, "status": status, "words": words, "rejected": rejected})
            added.extend(kept)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    out = dict(result)
    out["asr_speech_recovery"] = report
    if added:
        merged = sorted([*segments, *added], key=lambda s: float(s.get("start", 0.0)))
        out["segments"] = merged
        out["text"] = " ".join(str(s.get("text", "")).strip() for s in merged if s.get("text"))
    recovered = sum(1 for r in report if r["status"] == "recovered")
    logger.info(
        "gap recovery: %d of %d untranscribed stretch(es) recovered, %d word(s) added",
        recovered,
        len(report),
        sum(int(r.get("words", 0)) for r in report),
    )
    return out


#: A stretched word is replaced only when its span re-transcribes as the word plus at least this
#: many other words that are not fillers (D16). Of 11 stretched words re-transcribed on 2026-10-08,
#: 6 held only the word, a stutter or fillers; the 3 that hid speech held 13 to 55 words.
STRETCHED_MIN_EXTRA_WORDS = 3
#: Hesitation sounds, in the languages the pipeline transcribes. They are speech, not lost speech.
_FILLERS = frozenset(
    "uh um uhm erm er ah eh ehm hmm mm mhm äh ähm öh euh heu hum bah ahn ãh éé".split()
)


def _tokens(text: str) -> List[str]:
    return [t.lower() for t in _WORD.findall(text)]


def _replacement(word: str, kept: Sequence[Mapping[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """The recovered words to put in place of ``word``, or None when they are not more speech."""
    heard = [dict(w) for s in kept for w in (s.get("words") or [])]
    target = _tokens(word)
    said = _tokens(" ".join(str(w.get("word", "")) for w in heard))
    if not heard or not target or not all(t in said for t in target):
        return None
    extra = [t for t in said if t not in target and t not in _FILLERS]
    if len(extra) < STRETCHED_MIN_EXTRA_WORDS:
        return None
    for w in heard:
        text = str(w.get("word", ""))
        w["word"] = text if text.startswith(" ") else " " + text
        w["recovered"] = True
    return heard


def _replace_word(
    segments: List[Dict[str, Any]], start: float, end: float, new_words: List[Dict[str, Any]]
) -> Optional[str]:
    """Replace the word timed ``start``-``end`` in place; the reason it could not be, else None."""
    for seg in segments:
        words = list(seg.get("words") or [])
        for i, w in enumerate(words):
            w_start, w_end = _f(w.get("start")), _f(w.get("end"))
            # ``stretched_words_over_speech`` rounds to the millisecond.
            if w_start is None or w_end is None:
                continue
            if abs(w_start - start) > 0.001 or abs(w_end - end) > 0.001:
                continue
            # The segment text is rebuilt from its words, so it must BE its words.
            if (
                "".join(str(x.get("word", "")) for x in words).strip()
                != str(seg.get("text", "")).strip()
            ):
                return "text_differs_from_words"
            words[i : i + 1] = new_words
            seg["words"] = words
            seg["text"] = "".join(str(x.get("word", "")) for x in words)
            return None
    return "word_not_found"


def recover_stretched_words(
    result: Dict[str, Any],
    stretched: Sequence[Mapping[str, Any]],
    audio_path: str,
    transcribe_clip: ClipTranscriber,
) -> Dict[str, Any]:
    """Re-transcribe each stretched word's span alone; replace the word when it hid speech (D16).

    ``stretched`` is ``stretched_words_over_speech`` output. The span is cut with the usual pad,
    transcribed in the episode's language, and the clip words inside the span go through the same
    filters as gap recovery. The word is REPLACED by them -- so it is never duplicated -- and only
    when they hold the word itself plus ``STRETCHED_MIN_EXTRA_WORDS`` non-filler words; otherwise
    the transcript is unchanged. Reported as ``asr_stretched_word_recovery``.
    """
    if not stretched:
        return result
    out = dict(result)
    segments = [dict(s) for s in (result.get("segments") or []) if isinstance(s, Mapping)]
    report: List[Dict[str, Any]] = []
    work_dir = tempfile.mkdtemp(prefix="stretched_recovery_")
    call_failed = False
    try:
        for i, item in enumerate(stretched):
            ws, we, word = float(item["start"]), float(item["end"]), str(item.get("word", ""))
            entry: Dict[str, Any] = {"start": ws, "end": we, "word": word}
            if call_failed:
                report.append({**entry, "status": "skipped", "reason": "earlier_call_failed"})
                continue
            clip_start = max(0.0, ws - RECOVERY_PAD_S)
            clip = os.path.join(work_dir, f"stretched_{i:03d}.wav")
            try:
                cut_clip(audio_path, clip_start, we + RECOVERY_PAD_S, clip)
                clip_result = transcribe_clip(clip)
            except Exception as exc:  # noqa: BLE001 - never lose the transcript to a recovery call
                logger.warning("stretched word %.1f-%.1fs not re-transcribed: %s", ws, we, exc)
                report.append({**entry, "status": "failed", "error": type(exc).__name__})
                call_failed = True
                continue
            kept, rejected = segments_inside_gap(
                [s for s in (clip_result.get("segments") or []) if isinstance(s, Mapping)],
                clip_start=clip_start,
                gap_start=ws,
                gap_end=we,
            )
            new_words = None if rejected else _replacement(word, kept)
            reason = "rejected" if rejected else ("not_more_speech" if new_words is None else None)
            if new_words is not None:
                reason = _replace_word(segments, ws, we, new_words)
            heard = " ".join(str(s.get("text", "")) for s in kept)
            if reason is None:
                report.append({**entry, "status": "replaced", "words": len(_tokens(heard))})
            else:
                report.append({**entry, "status": "declined", "reason": reason, "heard": heard})
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    out["asr_stretched_word_recovery"] = report
    if any(r["status"] == "replaced" for r in report):
        out["segments"] = segments
        out["text"] = " ".join(str(s.get("text", "")).strip() for s in segments if s.get("text"))
    return out
