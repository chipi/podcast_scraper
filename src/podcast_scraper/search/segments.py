"""Segment-document construction + insight linking (RFC-090 §3.8, #857).

Adapts the existing sentence-based ``chunk_transcript`` (chunker.py) into Tier-1
``SegmentDocument`` objects, and links GIL insights to the segment that contains
their grounding quote (by timestamp overlap). The segment↔insight link is what
lets the retrieval layer merge a raw segment and its synthesized insight into a
``CompoundResult`` (#856 dedup).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from .backend import SegmentDocument
from .chunker import chunk_transcript

# RFC-090 KD-5: 200–300 words, 50-word overlap.
DEFAULT_TARGET_TOKENS = 250
DEFAULT_OVERLAP_TOKENS = 50


def build_segment_documents(
    transcript: str,
    *,
    episode_id: str,
    show_id: str,
    timestamps: Optional[List[Dict]] = None,
    speaker_id: Optional[str] = None,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> List[SegmentDocument]:
    """Chunk *transcript* into Tier-1 ``SegmentDocument`` objects.

    ``timestamps`` (Whisper segment dicts with ``char_start``/``char_end`` +
    ``start_ms``/``end_ms``) drive the per-chunk time span; absent timestamps yield
    ``0.0`` start/end. ``embedding`` is left empty for the embedding step.
    """
    overlap_tokens = min(overlap_tokens, max(0, target_tokens - 1))  # overlap < target
    chunks = chunk_transcript(
        transcript,
        target_tokens=target_tokens,
        overlap_tokens=overlap_tokens,
        timestamps=timestamps,
    )
    return [
        SegmentDocument(
            id=f"{episode_id}_chunk_{ch.chunk_index}",
            text=ch.text,
            show_id=show_id,
            episode_id=episode_id,
            start_time=(ch.timestamp_start_ms or 0) / 1000.0,
            end_time=(ch.timestamp_end_ms or 0) / 1000.0,
            speaker_id=speaker_id,
        )
        for ch in chunks
    ]


def link_insights_to_segments(
    segments: Sequence[SegmentDocument],
    insight_quotes: Sequence[Tuple[str, Optional[float], Optional[float]]],
    *,
    tolerance_seconds: float = 2.0,
) -> Dict[str, str]:
    """Link insights to the segment containing their grounding quote (by time).

    ``insight_quotes`` are ``(insight_id, quote_start_s, quote_end_s)`` (convert
    GIL ms → seconds at the call site). Mutates ``segment.linked_insight_ids`` in
    place and returns ``{insight_id: segment_id}`` so the caller can set
    ``InsightDocument.source_segment_id``. One segment per insight (first match).
    """
    mapping: Dict[str, str] = {}
    for insight_id, quote_start, quote_end in insight_quotes:
        if quote_start is None:
            continue
        for seg in segments:
            end_ok = quote_end is None or quote_end <= seg.end_time + tolerance_seconds
            if seg.start_time - tolerance_seconds <= quote_start and end_ok:
                seg.linked_insight_ids.append(insight_id)
                mapping[insight_id] = seg.id
                break
    return mapping


def _normalize_for_match(text: str) -> str:
    """Lowercase + collapse whitespace so verbatim quotes match across formatting."""
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def link_insights_to_segments_by_text(
    segments: Sequence[SegmentDocument],
    insight_quote_texts: Sequence[Tuple[str, str]],
    *,
    shingle_words: int = 8,
) -> Dict[str, str]:
    """Link insights to the segment containing their grounding quote TEXT.

    The timestamp-based :func:`link_insights_to_segments` fails when segments carry
    no per-chunk timestamps (``summary.timestamps`` unpopulated → segment spans at
    ``0.0``). But GIL grounding quotes are *verbatim* transcript spans, so the
    segment whose text contains the quote's leading ``shingle_words``-word shingle
    is its source segment — no timestamps required. Mutates
    ``segment.linked_insight_ids`` in place and returns ``{insight_id: segment_id}``
    (first containing segment per insight).
    """
    norm_segs = [(seg, _normalize_for_match(seg.text)) for seg in segments]
    mapping: Dict[str, str] = {}
    for insight_id, quote_text in insight_quote_texts:
        frag = " ".join(_normalize_for_match(quote_text).split()[:shingle_words])
        if not frag:
            continue
        for seg, seg_norm in norm_segs:
            if frag in seg_norm:
                seg.linked_insight_ids.append(insight_id)
                mapping[insight_id] = seg.id
                break
    return mapping
