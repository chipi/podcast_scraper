"""A document body cut into the rows of ``corpus.segments`` (CRFC-005 §3.5).

Every character of the body lands in exactly one cut, so joining the cuts in ``seq`` order gives
the body back byte for byte; an evidence range (seq, offset) then resolves against the same text
the file holds. The pipeline's writer and the importer both cut with ``cut_body``, so a document
written either way has the same segments.

A cut ends where its segment's text ends: the characters between two segments (a newline, a
speaker label) belong to the turn after them, text before the first segment to cut 0 and text
after the last to the last cut. A segment's span is its ``char_start``/``char_end`` when every
segment has a usable pair (the ad-free base), else where its text is found in the body searching
forward from the previous one; a segment whose text is not found has no cut of its own, and its
text joins the next. With no segments, or none found, the body is cut one line per segment with no
times.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence


@dataclass(frozen=True)
class Cut:
    """One ``segments`` row: its position, its text and its times in milliseconds."""

    seq: int
    text: str
    start_ms: Optional[int]
    end_ms: Optional[int]


class DocumentCutError(ValueError):
    """The cuts do not join back to the body."""


def seconds_to_ms(value: Any) -> Optional[int]:
    """``floor(s * 1000 + 0.5)``; None for a missing or non-numeric time."""
    if value is None or isinstance(value, bool):
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(seconds):
        return None
    return math.floor(seconds * 1000 + 0.5)


def _times(segment: Mapping[str, Any]) -> tuple[Optional[int], Optional[int]]:
    start = seconds_to_ms(segment.get("start"))
    end = seconds_to_ms(segment.get("end"))
    if start is not None and end is not None and end < start:
        return None, None
    return start, end


def _offset(value: Any) -> Optional[int]:
    return None if isinstance(value, bool) or not isinstance(value, int) else value


def _explicit_ends(body: str, segments: Sequence[Mapping[str, Any]]) -> Optional[list[int]]:
    ends: list[int] = []
    previous = 0
    for segment in segments:
        start = _offset(segment.get("char_start"))
        end = _offset(segment.get("char_end"))
        if start is None or end is None or not previous <= start <= end <= len(body):
            return None
        ends.append(end)
        previous = end
    return ends


def _found_ends(body: str, segments: Sequence[Mapping[str, Any]]) -> list[Optional[int]]:
    ends: list[Optional[int]] = []
    pos = 0
    for segment in segments:
        text = str(segment.get("text") or "").strip()
        at = body.find(text, pos) if text else -1
        if at < 0:
            ends.append(None)
            continue
        pos = at + len(text)
        ends.append(pos)
    return ends


def _line_cuts(body: str) -> list[Cut]:
    return [Cut(seq, line, None, None) for seq, line in enumerate(body.splitlines(keepends=True))]


def cut_body(body: str, segments: Optional[Sequence[Mapping[str, Any]]]) -> list[Cut]:
    """``body`` cut at its segments (``.segments.json`` entries), or one line per cut without.

    Raises ``DocumentCutError`` if the cuts would not join back to ``body``.
    """
    dict_segments = [s for s in (segments or []) if isinstance(s, Mapping)]
    explicit = _explicit_ends(body, dict_segments) if dict_segments else None
    found = explicit if explicit is not None else _found_ends(body, dict_segments)
    anchors = [(end, seg) for end, seg in zip(found, dict_segments) if end is not None]
    if not anchors:
        cuts = _line_cuts(body)
    else:
        anchors[-1] = (len(body), anchors[-1][1])
        cuts = []
        start = 0
        for end, segment in anchors:
            text = body[start:end]
            if text:
                start_ms, end_ms = _times(segment)
                cuts.append(Cut(len(cuts), text, start_ms, end_ms))
            start = end
    if "".join(cut.text for cut in cuts) != body:
        raise DocumentCutError("the cuts do not join back to the body")
    return cuts


def paragraph_cuts(body: str) -> list[Cut]:
    """An article body cut one paragraph per segment, the blank lines after a paragraph kept on
    it, no times."""
    cuts: list[Cut] = []
    current = ""
    for line in body.splitlines(keepends=True):
        if current and line.strip() and not current.splitlines()[-1].strip():
            cuts.append(Cut(len(cuts), current, None, None))
            current = ""
        current += line
    if current:
        cuts.append(Cut(len(cuts), current, None, None))
    if "".join(cut.text for cut in cuts) != body:
        raise DocumentCutError("the cuts do not join back to the body")
    return cuts
