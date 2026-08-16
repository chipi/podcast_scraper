"""LITM-aware corpus briefing packs (RFC-093 / #861).

Reshapes ``RetrievalLayer`` output into a structured, token-budgeted document
positioned for attention: critical grounding at the start, supporting evidence in
the middle, caveats at the end (LITM — models attend poorly to the middle of long
contexts). The builder is plain Python.

**Consumers (2026-06-25):**

- The ``corpus_briefing_pack`` MCP tool
  (``mcp/tools/briefing_pack.py``) wraps this builder for agents on
  Claude Desktop / Cursor; registered in the existing RFC-095 server.
- The autoresearch loop can consume it directly when ready — the
  pack-builder is an independent Python module with no MCP coupling.

``top_contradiction`` / ``coverage_gaps`` remain empty until typed contradiction
KG edges and a corpus-impact surface exist; the fields are kept so the consumer
schema is stable when those land.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .backend import CompoundResult, ScoredResult
from .dedup import Result


@dataclass
class CorpusBriefingPack:
    """A LITM-positioned, token-budgeted briefing over retrieval results."""

    query: str
    query_type: str
    canonical_entity: Optional[Dict] = None
    top_insight: Optional[ScoredResult] = None
    top_contradiction: Optional[Dict] = None  # populated when typed KG edges exist
    supporting_segments: List[ScoredResult] = field(default_factory=list)
    coverage_summary: Dict = field(default_factory=dict)
    coverage_gaps: List[str] = field(default_factory=list)  # from corpus-impact surface (future)
    confidence_p50: float = 0.0
    token_count: int = 0

    def render(self) -> str:
        """Serialize to LITM-positioned text (critical → supporting → caveats)."""
        lines: List[str] = ["[CRITICAL GROUNDING]"]
        if self.canonical_entity:
            name = self.canonical_entity.get("name") or self.canonical_entity.get("id")
            lines.append(f"Entity: {name}")
        if self.top_insight is not None:
            lines.append(f"Top insight: {self.top_insight.payload.get('text', '')}")
        lines.append(f"Contradiction: {self.top_contradiction or 'none detected'}")

        lines.append("")
        lines.append("[SUPPORTING EVIDENCE]")
        cov = self.coverage_summary
        lines.append(
            f"Coverage: {len(cov.get('show_ids', []))} shows, "
            f"{cov.get('episode_count', 0)} episodes"
        )
        for seg in self.supporting_segments:
            lines.append(f"- {seg.payload.get('text', '')}")

        lines.append("")
        lines.append("[CAVEATS]")
        # The corpus-impact gap surface is not wired yet — an empty list means "not assessed",
        # NOT "no gaps" (#21: the old "none" read as a trustworthy all-clear when it wasn't).
        gaps = ", ".join(self.coverage_gaps) if self.coverage_gaps else "not assessed"
        lines.append(f"Coverage gaps: {gaps}")
        # Insight confidence isn't scored in the current index (all rows carry 0.0), so a "0.00"
        # p50 is misleading precision — surface it as n/a until real confidences exist (#21).
        if self.confidence_p50 and self.confidence_p50 > 0:
            lines.append(f"Confidence (p50): {self.confidence_p50:.2f}")
        else:
            lines.append("Confidence (p50): n/a (insight confidence not scored in this corpus)")
        lines.append(f"Date range: {cov.get('date_range') or 'n/a'}")
        return "\n".join(lines)


def _result_payload(result: Result) -> Dict:
    if isinstance(result, CompoundResult):
        return result.segment.payload or result.insight.payload
    return result.payload


def _result_show(result: Result) -> Optional[str]:
    """Show id for coverage — reads ``show_id`` then ``feed_id``, checking BOTH tiers of a
    compound (its segment payload may lack the field while the insight carries it, which would
    otherwise under-count shows). #21/advisor-L2."""
    payloads = (
        [result.segment.payload, result.insight.payload]
        if isinstance(result, CompoundResult)
        else [result.payload]
    )
    for p in payloads:
        v = (p or {}).get("show_id") or (p or {}).get("feed_id")
        if v:
            return str(v)
    return None


def _date_range(results: Sequence[Result]) -> Optional[str]:
    dates = sorted(
        d
        for r in results
        for d in [_result_payload(r).get("publish_date") or _result_payload(r).get("date")]
        if d
    )
    return f"{dates[0]} – {dates[-1]}" if dates else None


def _count_tokens(text: str) -> int:
    # Word-count approximation; a canonical tokenizer is RFC-093 OQ-3.
    return len(text.split())


def build_briefing_pack(
    query: str,
    query_type: str,
    results: Sequence[Result],
    *,
    canonical_entity: Optional[Dict] = None,
    max_tokens: int = 2000,
) -> CorpusBriefingPack:
    """Build a LITM-positioned, token-budgeted briefing pack from *results*.

    Compounds contribute their insight (top-grounding candidate) and segment
    (supporting evidence). Supporting segments are trimmed to fit ``max_tokens``.
    """
    insights: List[ScoredResult] = []
    segments: List[ScoredResult] = []
    for result in results:
        if isinstance(result, CompoundResult):
            insights.append(result.insight)
            segments.append(result.segment)
        elif result.source_tier == "insight":
            insights.append(result)
        elif result.source_tier == "segment":
            segments.append(result)

    # Robust show extraction: raw index rows carry ``show_id``; some result shapes carry
    # ``feed_id`` — read either so coverage can't report 0 shows while counting episodes (#21).
    show_ids = sorted({s for r in results if (s := _result_show(r))})
    episode_count = len({e for r in results if (e := _result_payload(r).get("episode_id"))})
    # Only real (>0) confidences count — the indexer currently hardcodes 0.0 on every insight,
    # and averaging those produced a fake "0.00" p50 (#21). Empty → 0.0 → rendered as n/a.
    # TODO(#21): drop the `> 0` filter once the indexer scores real per-insight confidence — until
    # then a genuine 0.0 can't be distinguished from "unscored", so excluding 0.0 is correct.
    confidences = sorted(
        c for r in insights if (c := r.payload.get("confidence")) is not None and c > 0
    )
    p50 = confidences[len(confidences) // 2] if confidences else 0.0

    pack = CorpusBriefingPack(
        query=query,
        query_type=query_type,
        canonical_entity=canonical_entity,
        top_insight=insights[0] if insights else None,
        supporting_segments=segments[:5],
        coverage_summary={
            "show_ids": show_ids,
            "episode_count": episode_count,
            "date_range": _date_range(results),
        },
        confidence_p50=float(p50),
    )

    pack.token_count = _count_tokens(pack.render())
    while pack.token_count > max_tokens and pack.supporting_segments:
        pack.supporting_segments.pop()
        pack.token_count = _count_tokens(pack.render())
    return pack
