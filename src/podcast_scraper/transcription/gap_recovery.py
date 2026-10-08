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
    try:
        for i, gap in enumerate(gaps):
            gap_start, gap_end = float(gap["start"]), float(gap["end"])
            clip_start = max(0.0, gap_start - RECOVERY_PAD_S)
            entry: Dict[str, Any] = {
                "start": gap_start,
                "end": gap_end,
                "speaker": gap.get("speaker"),
            }
            try:
                clip = os.path.join(work_dir, f"gap_{i:03d}.wav")
                cut_clip(audio_path, clip_start, gap_end + RECOVERY_PAD_S, clip)
                clip_result = transcribe_clip(clip)
            except Exception as exc:  # noqa: BLE001 - never lose the transcript to a recovery call
                logger.warning(
                    "gap recovery: %.1f-%.1fs could not be re-transcribed: %s",
                    gap_start,
                    gap_end,
                    exc,
                )
                report.append({**entry, "status": "failed", "error": type(exc).__name__})
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
