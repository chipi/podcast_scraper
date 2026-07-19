"""Plain (un-diarized) transcript formatting — one line per Whisper segment.

An un-diarized transcript is otherwise ``result["text"]`` — the space-joined Whisper segments
on a single unbroken line. LLM evidence extraction silently collapses to ~zero quotes on such a
blob (#1182), and a transcript should carry sentence/segment structure even without speaker turns.

:func:`format_plain_transcript_with_offsets` is the plain-transcript analogue of
``providers.ml.diarization.formatting.format_diarized_screenplay_with_offsets``: it joins segments
one-per-line (no speaker labels) AND returns each segment's char range into the joined text, so the
downstream char->segment/timestamp mapping (``gi.pipeline._segment_char_spans``) uses the segment's
explicit ``char_start``/``char_end`` rather than a cumulative ``len(text)`` sum that the inserted
newlines would throw off (#1212).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def format_plain_transcript_with_offsets(
    segments: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Join segments one-per-line and return ``(text, offset_segments)``.

    ``offset_segments`` preserves each input segment's fields (``start``/``end``/``id``/…), with
    ``text`` stripped and ``char_start``/``char_end`` added as a half-open range into ``text`` such
    that ``text[char_start:char_end] == segment_text``. Blank-text segments are dropped (they
    contribute no line), mirroring the diarized formatter.
    """
    parts: List[str] = []
    out: List[Dict[str, Any]] = []
    cursor = 0
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        seg_text = (seg.get("text") or "").strip()
        if not seg_text:
            continue
        if parts:
            cursor += 1  # the "\n" that ``"\n".join`` inserts before this segment
        char_start = cursor
        parts.append(seg_text)
        cursor += len(seg_text)
        new_seg = dict(seg)
        new_seg["text"] = seg_text
        new_seg["char_start"] = char_start
        new_seg["char_end"] = cursor
        out.append(new_seg)
    return "\n".join(parts), out
