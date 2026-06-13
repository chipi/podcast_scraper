"""Format diarized Whisper segments as screenplay text.

The screenplay coalesces consecutive same-speaker segments into one ``Label: ...``
turn line. Because that inserts ``Label: `` prefixes and ``\n`` between turns, the
character offsets of a segment's text in the *screenplay* differ from a naive
``sum(len(seg["text"]))`` concatenation. Downstream consumers (#974) that map a
quote ``char_start`` back to a segment (speaker / timestamp) therefore need each
segment's *screenplay* char range, not a re-derived cumulative position. That is
what :func:`format_diarized_screenplay_with_offsets` emits — and the legacy
text-only formatter delegates to it so the two can never drift.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def format_diarized_screenplay_with_offsets(
    segments: List[dict],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Format segments as a screenplay AND return each segment's char range in it.

    Returns ``(screenplay_text, offset_segments)`` where ``offset_segments`` mirrors
    the input order (after the same start-time sort + blank-text drop the text
    formatter applies) and each entry carries:

    - ``start`` / ``end`` (seconds, floats — copied through)
    - ``speaker_label`` (the label used on the screenplay line)
    - ``text`` (the *stripped* text as it appears in the screenplay)
    - ``char_start`` / ``char_end`` — half-open range into ``screenplay_text`` such
      that ``screenplay_text[char_start:char_end] == text``.

    The emitted ``screenplay_text`` is byte-for-byte identical to
    :func:`format_diarized_screenplay_from_segments`.
    """
    if not segments:
        return "", []

    ordered = sorted(segments, key=lambda seg: float(seg.get("start") or 0.0))

    parts: List[str] = []  # accumulated screenplay fragments (joined → final text)
    cursor = 0  # running char offset into the eventual screenplay_text
    out_segments: List[Dict[str, Any]] = []
    previous_label: Optional[str] = None
    first_seg_on_line = True

    def _emit(fragment: str) -> int:
        """Append a fragment, return the offset at which it starts."""
        nonlocal cursor
        start = cursor
        parts.append(fragment)
        cursor += len(fragment)
        return start

    for segment in ordered:
        text = (segment.get("text") or "").strip()
        if not text:
            continue
        label = str(segment.get("speaker_label") or segment.get("speaker") or "SPEAKER")
        if label != previous_label:
            if previous_label is not None:
                _emit("\n")  # newline terminates the previous turn
            _emit(f"{label}: ")
            previous_label = label
            first_seg_on_line = True
        elif not first_seg_on_line:
            _emit(" ")  # space between coalesced segments in the same turn

        char_start = _emit(text)
        out_segments.append(
            {
                "start": float(segment.get("start") or 0.0),
                "end": float(segment.get("end") or 0.0),
                "speaker_label": label,
                "text": text,
                "char_start": char_start,
                "char_end": cursor,
            }
        )
        first_seg_on_line = False

    if previous_label is not None:
        _emit("\n")  # trailing newline (matches the text-only formatter)

    return "".join(parts), out_segments


def format_diarized_screenplay_from_segments(segments: List[dict]) -> str:
    """Format segments that include ``speaker_label`` (or ``speaker``) as screenplay."""
    text, _ = format_diarized_screenplay_with_offsets(segments)
    return text
