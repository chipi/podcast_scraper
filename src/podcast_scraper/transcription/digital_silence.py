"""Long digital silence at either end of an episode's audio, cut off before ASR.

Whisper fills a long stretch of digital silence (true zeros) with an invented line and can carry
that line through the whole file. Camera seduta 717 (2026-10-12) opens with 103 s of zeros: the
whole 66-minute sitting came back as "Grazie a tutti." 139 times, 417 words, on every attempt;
with the silence cut off the same audio gave 8,957 words against a reference of 8,836. Its two
siblings, with 90 s of zeros, decoded, but with "Grazie a tutti." invented three times in the
lead and silent-gap recovery adding invented words in the 70 s tail.

Only digital silence, and only at the ends. ``silencedetect`` compares every sample's peak with
the threshold: MP3 coding noise inside true zeros peaks just above -90 dB, so -90 found nothing
on 717's preprocessed audio and -70 finds both ends; broadcast room tone sits near -50 to -60.
Of the 30 preprocessed files of the 2026-10-12 reference runs, only the three Camera sittings are
cut; every other lead is under 3 s. The cut keeps ``MARGIN_S`` of the silence so a word at the
edge is not clipped; the ASR result is shifted back so every later stage reads episode time.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess  # nosec B404 - ffmpeg with a fixed argv
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

DIGITAL_SILENCE_DB = -70.0
MIN_SILENCE_S = 10.0
MARGIN_S = 1.0

_START = re.compile(r"silence_start:\s*(-?[\d.]+)")
_END = re.compile(r"silence_end:\s*([\d.]+)")
_DURATION = re.compile(r"Duration:\s*(\d+):(\d+):([\d.]+)")
_DECODED = re.compile(r"time=(\d+):(\d+):([\d.]+)")


@dataclass(frozen=True)
class SilentEdges:
    """Seconds of digital silence at the start and end of a file, and its duration."""

    lead_s: float
    tail_s: float
    duration_s: float


def silent_edges_from_log(log: str) -> Optional[SilentEdges]:
    """The digital silence at each end, from ffmpeg ``silencedetect`` output; None if unreadable.

    A silence still open at the end of the stream has no ``silence_end`` on some ffmpeg versions
    and one at the stream's duration on others; both count as reaching the end.
    """
    # The stream ends where decoding ended (the last progress ``time=``), not at the header's
    # estimate: on an MP3, ffmpeg 6.1 and 7.1 estimate 47.09 s for a file that decodes to 47.00 and
    # report its final silence ending at 47, so against the header the tail read as none (CI,
    # 2026-10-13). The header is the fallback when no progress line was printed.
    decoded = _DECODED.findall(log)
    m = _DURATION.search(log)
    if decoded:
        h, mi, se = decoded[-1]
        duration = int(h) * 3600 + int(mi) * 60 + float(se)
    elif m:
        duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    else:
        return None
    spans: List[Tuple[float, float]] = []
    pending: Optional[float] = None
    for line in log.splitlines():
        s = _START.search(line)
        if s:
            pending = max(0.0, float(s.group(1)))
            continue
        e = _END.search(line)
        if e and pending is not None:
            spans.append((pending, float(e.group(1))))
            pending = None
    if pending is not None:
        spans.append((pending, duration))
    lead = spans[0][1] if spans and spans[0][0] <= 0.05 else 0.0
    tail = duration - spans[-1][0] if spans and spans[-1][1] >= duration - 0.05 else 0.0
    if spans and len(spans) == 1 and lead and tail:
        # One span covering the whole file: all silence, nothing to keep.
        tail = 0.0
    return SilentEdges(lead_s=lead, tail_s=tail, duration_s=duration)


def silent_edges(audio_path: str) -> Optional[SilentEdges]:
    """Measure the digital silence at each end of ``audio_path``; None when ffmpeg cannot."""
    if not shutil.which("ffmpeg"):
        return None
    proc = subprocess.run(  # nosec B603 B607 - fixed argv, the path is the episode's own file
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            audio_path,
            "-af",
            f"silencedetect=noise={DIGITAL_SILENCE_DB:.0f}dB:d={MIN_SILENCE_S:.0f}",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        # ffmpeg echoes the file's tags as stored, often Latin-1 (GitHub #558); only the
        # ASCII silencedetect/Duration lines are parsed.
        encoding="utf-8",
        errors="replace",
        timeout=600,
        check=False,
    )
    return silent_edges_from_log(proc.stderr or "")


@dataclass(frozen=True)
class Trim:
    """The cut copy ASR hears, and the span of the original it kept (episode time)."""

    path: str
    kept_start_s: float
    kept_end_s: float
    edges: SilentEdges

    def record(self) -> Dict[str, Any]:
        """What the ASR record keeps: the silence at each end and the span ASR heard."""
        return {
            "lead_s": round(self.edges.lead_s, 2),
            "tail_s": round(self.edges.tail_s, 2),
            "kept": [round(self.kept_start_s, 3), round(self.kept_end_s, 3)],
        }


def kept_span(edges: SilentEdges) -> Optional[Tuple[float, float]]:
    """The part of the audio ASR should hear, or None when no end has ``MIN_SILENCE_S`` of zeros."""
    start = edges.lead_s - MARGIN_S if edges.lead_s >= MIN_SILENCE_S else 0.0
    end = (
        edges.duration_s - (edges.tail_s - MARGIN_S)
        if edges.tail_s >= MIN_SILENCE_S
        else edges.duration_s
    )
    if start <= 0.0 and end >= edges.duration_s:
        return None
    if end - start < MIN_SILENCE_S:
        return None
    return (max(0.0, start), end)


def trim_for_asr(audio_path: str, out_dir: str) -> Optional[Trim]:
    """A copy of ``audio_path`` without its long silent ends, or None when there is nothing to cut.

    Never raises: a file that cannot be measured or cut is transcribed as it is.
    """
    try:
        edges = silent_edges(audio_path)
        span = kept_span(edges) if edges else None
        if edges is None or span is None:
            return None
        start, end = span
        _, ext = os.path.splitext(audio_path)
        out = os.path.join(out_dir, f"asr_trimmed{ext or '.mp3'}")
        subprocess.run(  # nosec B603 B607 - fixed argv
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-ss",
                f"{start:.3f}",
                "-to",
                f"{end:.3f}",
                "-i",
                audio_path,
                "-c",
                "copy",
                out,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
            check=True,
        )
        return Trim(path=out, kept_start_s=start, kept_end_s=end, edges=edges)
    except Exception:  # noqa: BLE001 - trimming is a protection, never a reason to lose the episode
        logger.warning(
            "digital-silence trim failed for %s; transcribing as is", audio_path, exc_info=True
        )
        return None


def _shifted(item: Mapping[str, Any], offset: float) -> Dict[str, Any]:
    out = dict(item)
    for key in ("start", "end"):
        value = out.get(key)
        if isinstance(value, (int, float)):
            out[key] = float(value) + offset
    words = out.get("words")
    if isinstance(words, list):
        out["words"] = [_shifted(w, offset) if isinstance(w, Mapping) else w for w in words]
    return out


def shift_result(result: Dict[str, Any], offset: float) -> Dict[str, Any]:
    """``result`` with every segment and word time moved by ``offset`` seconds."""
    if not offset:
        return result
    out = dict(result)
    for key in ("segments", "words"):
        items = out.get(key)
        if isinstance(items, list):
            out[key] = [_shifted(s, offset) if isinstance(s, Mapping) else s for s in items]
    return out


def transcribed_span(result: Mapping[str, Any]) -> Optional[Tuple[float, float]]:
    """The episode-time span ASR heard, when its silent ends were cut; else None."""
    rec = result.get("asr_digital_silence_trimmed")
    if not isinstance(rec, Mapping):
        return None
    kept = rec.get("kept")
    if isinstance(kept, (list, tuple)) and len(kept) == 2:
        return (float(kept[0]), float(kept[1]))
    return None
