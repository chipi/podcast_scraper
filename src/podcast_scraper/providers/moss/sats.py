"""Parse MOSS-Transcribe-Diarize's SATS output into segments (#1177).

MOSS emits Speaker-Attributed, Time-Stamped transcription as a single flat token stream — one
autoregressive pass produces text, speaker, and timestamps together::

    [0.48][S01]Welcome everyone[1.66][12.26][S02]The new pipeline is ready[13.81]

i.e. ``[start][speaker]text[end]`` repeated. This module turns that into the segment dicts the
rest of the pipeline already speaks (``start`` / ``end`` / ``text`` / ``speaker``), so both the
transcription provider and the diarization provider read the same parse.

It is deliberately a pure function with no I/O: the model is three days old and its output format
is the least settled thing about it, so this is the piece that must be trivially testable and
trivially fixable.

Robustness matters more than elegance here. A decoder-only ASR can emit malformed spans — a
missing end timestamp, a stray acoustic-event tag, timestamps that run backwards — and a parser
that raises on the first oddity would take down a 90-minute episode over one bad token. Anything
unparsable is skipped, and what survives is returned.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_SPEAKER = re.compile(r"\[(S\d+)\]")
_TIMESTAMP = re.compile(r"\[(\d+(?:\.\d+)?)\]")

# Acoustic-event / non-speech tags the model may interleave, e.g. [laughter], [music].
# They are not speakers and not timestamps; strip them from the transcript text.
_EVENT_TAG = re.compile(r"\[(?!S\d+\])(?![\d.]+\])[^\]]{0,32}\]")


def _clean(text: str) -> str:
    """Transcript text with acoustic-event tags removed and whitespace normalized."""
    return " ".join(_EVENT_TAG.sub(" ", text).split())


def parse_sats(raw: str) -> List[Dict[str, Any]]:
    """Parse a MOSS SATS stream into ``[{start, end, text, speaker}, ...]``.

    Anchored on the **speaker tag**, not on a single `[start][spk]text[end]` regex. That matters:
    a dangling span (a speaker tag whose text or end timestamp the model failed to emit) would
    otherwise consume the *next* segment's start timestamp, so one malformed span would silently
    swallow the good segment after it. Anchoring on the speaker means a bad span can only cost
    itself.

    For each speaker tag: ``start`` is the nearest timestamp before it, ``end`` the nearest one
    after its text. Segments with no text, no bounding timestamps, or ``end`` before ``start`` are
    dropped — losing a span is acceptable; raising would lose a 90-minute episode.
    """
    if not raw:
        return []

    timestamps = [(m.start(), float(m.group(1))) for m in _TIMESTAMP.finditer(raw)]
    segments: List[Dict[str, Any]] = []
    dropped = 0

    tags = list(_SPEAKER.finditer(raw))
    for index, tag in enumerate(tags):
        # start = the last timestamp before this speaker tag
        before = [t for pos, t in timestamps if pos < tag.start()]
        # end = the first timestamp after the tag; the text is everything up to it
        after = [(pos, t) for pos, t in timestamps if pos >= tag.end()]
        if not before or not after:
            dropped += 1
            continue

        end_pos, end = after[0]
        # Never let a segment's text run past the next speaker tag.
        limit = tags[index + 1].start() if index + 1 < len(tags) else len(raw)
        if end_pos > limit:
            dropped += 1
            continue

        text = _clean(raw[tag.end() : end_pos])
        start = before[-1]
        if not text or end < start:
            dropped += 1
            continue

        segments.append({"start": start, "end": end, "text": text, "speaker": tag.group(1)})

    if dropped:
        logger.warning("MOSS SATS: skipped %d malformed segment(s)", dropped)
    if not segments and raw.strip():
        logger.warning("MOSS SATS: no segments parsed from a non-empty response")
    return segments


def transcript_text(segments: List[Dict[str, Any]]) -> str:
    """Flat transcript — the concatenation the transcription provider returns as ``text``."""
    return " ".join(str(s.get("text", "")) for s in segments if s.get("text"))


def speakers(segments: List[Dict[str, Any]]) -> List[str]:
    """Distinct speaker labels, in first-appearance order.

    MOSS labels are **anonymous and relative** (``S01``, ``S02``) — the same semantics as
    pyannote's ``SPEAKER_00``, so the roster still has to resolve them to real people downstream.
    """
    seen: List[str] = []
    for seg in segments:
        label = str(seg.get("speaker") or "")
        if label and label not in seen:
            seen.append(label)
    return seen
