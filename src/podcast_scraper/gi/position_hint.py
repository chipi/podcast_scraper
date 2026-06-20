"""Insight position_hint helper — RFC-097 v3.0 / RFC-072 §2b.

``position_hint`` is a float in [0.0, 1.0]: mean Quote start time relative to
episode duration. Powers Position Tracker timeline + temporal ordering.

4-step computation waterfall:

1. ``Episode.duration_ms`` from RSS ``<itunes:duration>`` (~86% prod corpus
   coverage today)
2. Last segment's ``end × 1000`` from ``*.segments.json`` (~99.9% — every
   transcribed episode has segments)
3. ``max(Quote.timestamp_end_ms)`` across the episode's Quotes (lower-bound;
   preserves ordering)
4. Skip emission (field is Optional; < 0.1% edge case)

Steps 1, 3, 4 were already in place; step 2 (segments fallback) lands as part
of chunk 5 of RFC-097. The function returns ``(value, source)`` where source
is the step number that resolved the duration — for telemetry / debugging.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple


def _last_segment_end_ms(transcript_segments: Optional[List[Any]]) -> Optional[int]:
    """Step 2: last segment's ``end`` × 1000 from ``*.segments.json``.

    Handles dict-shaped segments (``{"end": 1963.68, ...}``) and dataclass-shaped
    segments (``segment.end``). Returns None when segments are empty/missing.
    """
    if not transcript_segments:
        return None
    last = transcript_segments[-1]
    raw: Any = None
    if isinstance(last, dict):
        raw = last.get("end")
    else:
        raw = getattr(last, "end", None)
    if raw is None:
        return None
    try:
        end_f = float(raw)
    except (TypeError, ValueError):
        return None
    if end_f <= 0:
        return None
    return int(round(end_f * 1000.0))


def compute_position_hint(
    timestamp_starts_ms: List[int],
    episode_duration_ms: Optional[int],
    *,
    transcript_segments: Optional[List[Any]] = None,
    quote_end_fallback_ms: Optional[int] = None,
) -> Tuple[Optional[float], Optional[int]]:
    """RFC-097 4-step waterfall for Insight.position_hint.

    Args:
        timestamp_starts_ms: Quote ``timestamp_start_ms`` values (one per
            supporting Quote on the Insight).
        episode_duration_ms: ``Episode.duration_ms`` from RSS (step 1 source).
        transcript_segments: Optional ``*.segments.json``-shaped sequence
            (step 2 source). Each element may be a dict with ``end`` or any
            object with an ``.end`` attribute (seconds).
        quote_end_fallback_ms: Optional ``max(Quote.timestamp_end_ms)`` across
            the episode's quotes (step 3 source — lower-bound but preserves
            ordering when the Insight clusters quotes near the episode end).

    Returns:
        ``(position_hint, step)`` where ``position_hint`` is the float in
        ``[0.0, 1.0]`` (rounded to two decimals) and ``step`` is ``1`` / ``2``
        / ``3`` / ``4`` indicating which waterfall rung resolved the duration
        (4 = skip emission). Returns ``(None, 4)`` when no duration is
        recoverable from any of the four sources or when no Quote timestamps
        exist.
    """
    if not timestamp_starts_ms:
        return (None, 4)
    # Step 1: RSS duration.
    if episode_duration_ms and episode_duration_ms > 0:
        return (_compute_ratio(timestamp_starts_ms, int(episode_duration_ms)), 1)
    # Step 2: last segment end × 1000.
    seg_dur = _last_segment_end_ms(transcript_segments)
    if seg_dur is not None and seg_dur > 0:
        return (_compute_ratio(timestamp_starts_ms, seg_dur), 2)
    # Step 3: max Quote timestamp_end_ms as a lower-bound.
    if quote_end_fallback_ms and quote_end_fallback_ms > 0:
        return (_compute_ratio(timestamp_starts_ms, int(quote_end_fallback_ms)), 3)
    # Step 4: skip emission.
    return (None, 4)


def _compute_ratio(timestamp_starts_ms: List[int], duration_ms: int) -> float:
    """``mean(starts) / duration``, clamped to ``[0.0, 1.0]`` and rounded to 2dp."""
    mean_start = sum(timestamp_starts_ms) / len(timestamp_starts_ms)
    return round(min(mean_start / float(duration_ms), 1.0), 2)
