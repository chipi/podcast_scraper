"""Measure transcript segment-time drift against a per-turn ground-truth timeline (#1173).

The v3 fixtures carry an exact timeline: turn *i* was rendered as a separate ``say`` aiff, so RTTM
line *i*'s onset is the true audio start of turn *i* (:mod:`transcripts_to_mp3`). Transcribing the
concatenated mp3 with word timestamps and asking "where does turn *i*'s first word land?" measures
segment-start drift with no listening.

The one trap: ASR garble makes the transcript text diverge from the source, so naive
char-position alignment accumulates error and reports seconds of phantom drift. The fix is a
**word-sequence alignment** (:func:`align_word_streams`, difflib) that tolerates
insert/delete/substitute and only scores boundaries whose word actually matched.

Pure functions over abstract ``(key, time)`` streams — no whisper import, so the alignment/metric
math is unit-testable without a model. The live transcription harness lives in the fixture tests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Sequence, Tuple

_NON_WORD = re.compile(r"[^0-9a-z]+")


def normalize_key(text: str) -> str:
    """Lowercased, punctuation/space-stripped word key (matches ``word_timestamps._key``)."""
    return _NON_WORD.sub("", str(text or "").lower())


def align_word_streams(
    source_keys: Sequence[str], transcript_keys: Sequence[str], *, min_anchor: int = 1
) -> Dict[int, int]:
    """Map source-word index → transcript-word index for matched words (difflib blocks).

    ``min_anchor`` guards against a lone common word (``the``, ``i``, ``welcome``) matching the
    *wrong* occurrence: only words inside a contiguous matched block of at least ``min_anchor``
    words are mapped, so a boundary is trusted only when its neighbours align too. ``min_anchor=1``
    keeps every match (the raw difflib behaviour).
    """
    sm = SequenceMatcher(a=list(source_keys), b=list(transcript_keys), autojunk=False)
    mapping: Dict[int, int] = {}
    for i, j, size in sm.get_matching_blocks():
        if size < min_anchor:
            continue
        for d in range(size):
            mapping[i + d] = j + d
    return mapping


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile in ``[0, 100]`` (0.0 for an empty sequence)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


@dataclass
class DriftResult:
    """Boundary-drift metrics for one episode (all drifts in milliseconds)."""

    boundaries_total: int = 0
    boundaries_matched: int = 0
    mean_ms: float = 0.0
    p95_ms: float = 0.0
    max_ms: float = 0.0
    drifts_ms: List[float] = field(default_factory=list)


def measure_boundary_drift(
    source_words: Sequence[Tuple[str, int, bool]],
    transcript_words: Sequence[Tuple[str, float]],
    onsets: Sequence[float],
    *,
    min_anchor: int = 1,
) -> DriftResult:
    """Drift between each source turn-start and its aligned transcript-word time.

    ``source_words``: ``(key, turn_index, is_turn_start)`` in speaking order.
    ``transcript_words``: ``(key, start_seconds)`` in transcription order.
    ``onsets``: per-turn true onset (seconds); ``onsets[i]`` is turn *i*'s ground-truth start.
    ``min_anchor``: only score a boundary anchored in a matched block this long (see
    :func:`align_word_streams`) — 3+ removes lone-common-word mismatches on ad/garble boundaries.

    Turn 0 is skipped (its onset is trivially 0). A boundary whose word did not align (or aligned
    only as a lone common word) is not scored, and reported via ``boundaries_matched``.
    """
    mapping = align_word_streams(
        [k for k, _, _ in source_words], [k for k, _ in transcript_words], min_anchor=min_anchor
    )
    result = DriftResult()
    for si, (_key, turn_index, is_turn_start) in enumerate(source_words):
        if not is_turn_start or turn_index == 0 or turn_index >= len(onsets):
            continue
        result.boundaries_total += 1
        tj = mapping.get(si)
        if tj is None:
            continue
        result.boundaries_matched += 1
        result.drifts_ms.append(abs(transcript_words[tj][1] - onsets[turn_index]) * 1000.0)
    if result.drifts_ms:
        result.mean_ms = sum(result.drifts_ms) / len(result.drifts_ms)
        result.p95_ms = percentile(result.drifts_ms, 95)
        result.max_ms = max(result.drifts_ms)
    return result


def pool_drift(results: Sequence[DriftResult]) -> Dict[str, float]:
    """Pool per-episode drifts into corpus-level percentiles (AC2 is a pooled bar)."""
    pooled: List[float] = []
    for r in results:
        pooled.extend(r.drifts_ms)
    return {
        "boundaries": len(pooled),
        "mean_ms": (sum(pooled) / len(pooled)) if pooled else 0.0,
        "p50_ms": percentile(pooled, 50),
        "p95_ms": percentile(pooled, 95),
        "max_ms": max(pooled) if pooled else 0.0,
    }


def refined_and_segment_timelines(
    segments: Sequence[Dict],
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """Build the (refined word-level, unrefined segment-level) transcript timelines from segments.

    ``segments``: ``[{"start": float, "words": [[key, start], ...]}, ...]`` (the cached whisper
    output). The **refined** timeline uses each word's own start (the #1173 fix); the
    **unrefined** timeline maps every word to its segment's coarse ``start`` (the pre-fix bug).
    Returned streams are word-for-word parallel, so both align identically to the source.
    """
    refined: List[Tuple[str, float]] = []
    unrefined: List[Tuple[str, float]] = []
    for seg in segments:
        seg_start = float(seg.get("start", 0.0))
        for word in seg.get("words", []) or []:
            key = normalize_key(word[0])
            if not key:
                continue
            refined.append((key, float(word[1])))
            unrefined.append((key, seg_start))
    return refined, unrefined
