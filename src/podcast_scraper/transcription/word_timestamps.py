"""Refine segment timestamps from word-level times (#1173).

Whisper's *segment*-level timestamps drift on long audio — increasingly so, up to tens of
seconds — while its *word*-level timestamps stay accurate (measured: openai whisper-1
word-level is within ~0.05 s of an independent reference, vs ~1.6–20 s for segment-level). A
subtitle/player that seeks by these times therefore lands on the wrong line.

When a transcription provider is asked for word timestamps (``timestamp_granularities=["word"]``
/ ``word_timestamps=True``), it returns both a coarse ``segments`` list and an accurate flat
``words`` list. :func:`apply_word_timestamps` rewrites each segment's ``start``/``end`` from the
words it is composed of, keeping the segment text/grouping intact.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

_NON_WORD = re.compile(r"[^0-9a-z]+")


def _key(text: str) -> str:
    """Lowercased, punctuation/space-stripped form for length-based consumption."""
    return _NON_WORD.sub("", str(text or "").lower())


def apply_word_timestamps(
    segments: Sequence[Dict[str, Any]], words: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Return ``segments`` with ``start``/``end`` reset from ``words`` (word-accurate times).

    ``segments`` and ``words`` are both in transcription order and come from the same pass, so
    both spell the same text. Each is placed on that text by its CUMULATIVE (normalized)
    character position, and a segment takes the words overlapping its own span: the first one's
    ``start``, the last one's ``end``. Robust to punctuation/spacing differences between the two
    granularities.

    Positions are cumulative, not counted afresh per segment, because the server can cut one word
    across two segments ("and" | "ocracy") while its word list holds it whole ("andocracy"). Per
    segment, that word overshoots the first and every later segment takes its neighbour's word;
    on short segments the error grows into minutes (#2370). Cumulatively, both halves overlap the
    one word and nothing after it moves.

    No-op (returns the input segments unchanged, copied) when ``words`` is empty — so a provider
    that could not produce word timestamps degrades to the old segment-level behaviour.
    """
    if not words:
        return [dict(s) for s in segments]

    wend: List[int] = []
    pos = 0
    for w in words:
        pos += len(_key(w.get("word", "")))
        wend.append(pos)
    wstart = [e - len(_key(w.get("word", ""))) for e, w in zip(wend, words)]

    out: List[Dict[str, Any]] = []
    first, at = 0, 0
    for seg in segments:
        a = at
        at += len(_key(seg.get("text", "")))
        while first < len(words) and wend[first] <= a:
            first += 1
        last = first
        while last < len(words) and wstart[last] < at:
            last += 1
        if last == first:
            out.append(dict(seg))
            continue
        new = dict(seg)
        new["start"] = float(words[first].get("start", seg.get("start", 0.0)))
        new["end"] = float(words[last - 1].get("end", seg.get("end", 0.0)))
        out.append(new)
    return out


def apply_nested_word_timestamps(segments: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Refine segments that carry their own ``words`` list (openai-whisper's ``word_timestamps``).

    Unlike the OpenAI API — which returns one flat ``words`` list alongside the segments — the
    local ``whisper`` package nests each segment's words inside it. Each segment's times are taken
    straight from its own first/last word; segments without words keep theirs. The ``words`` key is
    dropped so the stored segment JSON keeps its existing shape.
    """
    out: List[Dict[str, Any]] = []
    for seg in segments:
        new = dict(seg)
        words = word_dicts(new.pop("words", None))
        if words:
            new["start"] = float(words[0]["start"])
            new["end"] = float(words[-1]["end"])
        out.append(new)
    return out


def word_dicts(raw_words: Any) -> List[Dict[str, Any]]:
    """Normalize provider word objects (dict or SDK object) to ``{word,start,end}`` dicts.

    Returns ``[]`` for anything that is not a list/tuple of words — a provider that omits word
    timestamps (or whose SDK exposes ``words`` as some other shape) must degrade to segment-level
    times, never crash the transcription.
    """
    if not isinstance(raw_words, (list, tuple)):
        return []
    out: List[Dict[str, Any]] = []
    for w in raw_words:
        if isinstance(w, dict):
            out.append({"word": w.get("word", ""), "start": w.get("start"), "end": w.get("end")})
        else:
            out.append(
                {
                    "word": getattr(w, "word", ""),
                    "start": getattr(w, "start", None),
                    "end": getattr(w, "end", None),
                }
            )
    return [w for w in out if w.get("start") is not None and w.get("end") is not None]
