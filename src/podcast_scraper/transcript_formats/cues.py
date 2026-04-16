"""Parse WebVTT and SubRip captions into plain text and GI-compatible segment lists.

Each segment is ``{"start": float, "end": float, "text": str}`` (seconds), matching
Whisper-style sidecars used by ``_char_range_to_ms``. Plain text is the concatenation
of segment ``text`` values with no separator (exact length alignment for issue #545).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

# WebVTT cue timing: optional hours; comma or dot for fractional seconds.
_WEBVTT_CUE_LINE = re.compile(
    r"^(\d{1,2}:\d{2}(?::\d{2})?[.,]\d{3})\s*-->\s*(\d{1,2}:\d{2}(?::\d{2})?[.,]\d{3})"
)
# SRT typical line (hours optional in some files; require full h:m:s)
_SRT_CUE_LINE = re.compile(r"^(\d{1,2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{1,2}:\d{2}:\d{2},\d{3})")
_HTML_TAG = re.compile(r"<[^>]+>")


def _timestamp_to_seconds(ts: str) -> float:
    """Parse VTT/SRT timestamp fragment to seconds."""
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(ts)


def _normalize_cue_text(raw: str) -> str:
    """Strip simple HTML-like tags; newlines to space; collapse horizontal runs.

    Leading/trailing spaces within a cue are preserved so adjacent cues can form
    ``"Hello world"`` when the second cue begins with a space.
    """
    t = _HTML_TAG.sub("", raw)
    t = t.replace("\n", " ").replace("\r", " ")
    return re.sub(r"[ \t]+", " ", t)


def parse_webvtt(data: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Parse WebVTT body into ``(plain_text, segments)``.

    Returns empty segments if the file has no usable cues (caller should fall back
    to writing raw bytes).
    """
    lines = data.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if lines and lines[0].startswith("\ufeff"):
        lines[0] = lines[0].lstrip("\ufeff")

    i = 0
    while i < len(lines) and not lines[i].strip().upper().startswith("WEBVTT"):
        i += 1
    if i >= len(lines):
        return "", []
    i += 1
    while i < len(lines) and lines[i].strip():
        i += 1

    segments: List[Dict[str, Any]] = []
    while i < len(lines):
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            break

        stripped = lines[i].strip()
        if stripped.startswith("NOTE"):
            i += 1
            while i < len(lines) and lines[i].strip():
                i += 1
            continue
        if stripped.startswith("STYLE") or stripped.startswith("REGION"):
            i += 1
            while i < len(lines) and lines[i].strip():
                i += 1
            continue

        if "-->" not in stripped:
            if i + 1 < len(lines) and "-->" in lines[i + 1]:
                i += 1
                if i >= len(lines):
                    break
                stripped = lines[i].strip()
            else:
                i += 1
                continue

        m = _WEBVTT_CUE_LINE.match(stripped)
        if not m:
            i += 1
            continue

        start_s = _timestamp_to_seconds(m.group(1))
        end_s = _timestamp_to_seconds(m.group(2))
        i += 1
        text_lines: List[str] = []
        while i < len(lines) and lines[i].strip():
            text_lines.append(lines[i])
            i += 1
        raw_text = "\n".join(text_lines)
        norm = _normalize_cue_text(raw_text)
        if norm.strip():
            segments.append({"start": start_s, "end": end_s, "text": norm})

    plain = "".join(s["text"] for s in segments)
    return plain, segments


def parse_srt(data: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Parse SubRip body into ``(plain_text, segments)``."""
    text = data.replace("\r\n", "\n").replace("\r", "\n")
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")
    blocks = re.split(r"\n\s*\n+", text.strip())
    segments: List[Dict[str, Any]] = []

    for block in blocks:
        block_lines = [ln for ln in block.split("\n") if ln is not None]
        if not block_lines:
            continue
        li = 0
        if re.match(r"^\d+\s*$", block_lines[0].strip()):
            li = 1
        if li >= len(block_lines):
            continue
        time_line = block_lines[li].strip()
        m = _SRT_CUE_LINE.match(time_line)
        if not m:
            continue
        start_s = _timestamp_to_seconds(m.group(1))
        end_s = _timestamp_to_seconds(m.group(2))
        raw_body = "\n".join(block_lines[li + 1 :])
        norm = _normalize_cue_text(raw_body)
        if not norm.strip():
            continue
        segments.append({"start": start_s, "end": end_s, "text": norm})

    plain = "".join(s["text"] for s in segments)
    return plain, segments
