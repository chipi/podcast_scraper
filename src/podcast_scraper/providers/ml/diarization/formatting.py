"""Format diarized Whisper segments as screenplay text."""

from __future__ import annotations

from typing import List


def format_diarized_screenplay_from_segments(segments: List[dict]) -> str:
    """Format segments that include ``speaker_label`` (or ``speaker``) as screenplay."""
    if not segments:
        return ""

    lines: List[str] = []
    previous_label: str | None = None
    buffer: List[str] = []

    def _flush() -> None:
        nonlocal buffer
        if not buffer or previous_label is None:
            return
        lines.append(f"{previous_label}: {' '.join(buffer)}")
        buffer = []

    for segment in sorted(segments, key=lambda seg: float(seg.get("start") or 0.0)):
        text = (segment.get("text") or "").strip()
        if not text:
            continue
        label = str(segment.get("speaker_label") or segment.get("speaker") or "SPEAKER")
        if label != previous_label:
            _flush()
            previous_label = label
        buffer.append(text)

    _flush()
    return "\n".join(lines) + ("\n" if lines else "")
