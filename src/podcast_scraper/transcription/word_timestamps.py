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
    each segment's text maps to a contiguous run of words. For each segment we consume words
    until their combined (normalized) text covers the segment's text, then take the first
    consumed word's ``start`` and the last's ``end``. Robust to punctuation/spacing differences
    between the two granularities.

    No-op (returns the input segments unchanged, copied) when ``words`` is empty — so a provider
    that could not produce word timestamps degrades to the old segment-level behaviour.
    """
    out: List[Dict[str, Any]] = []
    if not words:
        return [dict(s) for s in segments]

    wkeys = [_key(w.get("word", "")) for w in words]
    wi = 0
    n = len(words)
    for seg in segments:
        target = _key(seg.get("text", ""))
        if not target or wi >= n:
            out.append(dict(seg))
            continue
        start_wi = wi
        acc = ""
        # Consume words until we've covered this segment's text (guard against runaway).
        while wi < n and len(acc) < len(target):
            acc += wkeys[wi]
            wi += 1
        if wi > start_wi:
            new = dict(seg)
            new["start"] = float(words[start_wi].get("start", seg.get("start", 0.0)))
            new["end"] = float(words[wi - 1].get("end", seg.get("end", 0.0)))
            out.append(new)
        else:
            out.append(dict(seg))
    return out


def word_dicts(raw_words: Sequence[Any]) -> List[Dict[str, Any]]:
    """Normalize provider word objects (dict or SDK object) to ``{word,start,end}`` dicts."""
    out: List[Dict[str, Any]] = []
    for w in raw_words or []:
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
