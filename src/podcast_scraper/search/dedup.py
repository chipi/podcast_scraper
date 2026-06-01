"""Compound-result deduplication (RFC-090 §3.4).

When a segment result and an insight result refer to the same content (the
segment contains the insight's grounding quote), merge them into one
``CompoundResult`` so consumers (viewer, MCP, autoresearch) receive one object
with both the raw segment and the synthesized insight. Done once here, not in
every consumer (RFC-090 KD-3).
"""

from __future__ import annotations

from typing import List, Set, Union

from .backend import CompoundResult, ScoredResult

Result = Union[ScoredResult, CompoundResult]


def deduplicate(results: List[ScoredResult]) -> List[Result]:
    """Merge segment+insight pairs referring to the same content into compounds.

    A pair merges when the insight's ``source_segment_id`` (or the segment's
    ``linked_insight_ids``) links them. The compound takes ``max`` score and
    ``min`` rank; output is re-sorted by score.
    """
    segment_map = {r.doc_id: r for r in results if r.source_tier == "segment"}
    insight_map = {r.doc_id: r for r in results if r.source_tier == "insight"}

    output: List[Result] = []
    consumed: Set[str] = set()

    for insight_id, insight in insight_map.items():
        seg_id = insight.payload.get("source_segment_id")
        if not seg_id or seg_id not in segment_map:
            # Fall back to the segment->insight link direction.
            seg_id = next(
                (
                    s_id
                    for s_id, seg in segment_map.items()
                    if insight_id in (seg.payload.get("linked_insight_ids") or [])
                ),
                None,
            )
        if seg_id and seg_id in segment_map and seg_id not in consumed:
            seg = segment_map[seg_id]
            output.append(
                CompoundResult(
                    doc_id=seg_id,
                    score=max(insight.score, seg.score),
                    rank=min(insight.rank, seg.rank),
                    segment=seg,
                    insight=insight,
                )
            )
            consumed.add(seg_id)
            consumed.add(insight_id)

    for result in results:
        if result.doc_id not in consumed:
            output.append(result)

    return sorted(output, key=lambda r: r.score, reverse=True)
