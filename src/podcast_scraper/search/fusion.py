"""Reciprocal Rank Fusion for hybrid retrieval (RFC-090 §3.3).

Fuses independent ranked lists (BM25, dense vector, and — via RFC-091/#859 — KG
proximity) into one list. RRF uses rank, not raw score, so signals with
incomparable score scales combine cleanly. Tier weights gently boost insights
(grounded signal) over raw segments; the query router (#856 router.py) overrides
these per intent.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .backend import ScoredResult

# Default tier weights: insights are grounded signal, slight boost is appropriate.
# ``aux`` (kg/quote/summary) defaults to 1.0; unlisted tiers also fall back to 1.0.
TIER_WEIGHTS: Dict[str, float] = {"insight": 1.2, "segment": 1.0, "aux": 1.0}


def rrf_fuse(
    ranked_lists: Sequence[List[ScoredResult]],
    *,
    k: int = 60,
    signal_weights: Optional[Dict[str, float]] = None,
    tier_weights: Optional[Dict[str, float]] = None,
) -> List[ScoredResult]:
    """Fuse ranked lists via RRF: ``score(d) = Σ (signal_w · tier_w) / (k + rank_i(d))``.

    Each input list must be from a single signal (``result_list[0].signal`` names
    it). Returns one ``rrf``-signal list ordered by fused score.
    """
    signal_weights = signal_weights or {}
    tier_weights = tier_weights or TIER_WEIGHTS
    scores: Dict[str, float] = {}
    payloads: Dict[str, Dict] = {}
    tiers: Dict[str, str] = {}

    for result_list in ranked_lists:
        if not result_list:
            continue
        signal = result_list[0].signal
        sw = signal_weights.get(signal, 1.0)
        for result in result_list:
            tw = tier_weights.get(result.source_tier, 1.0)
            scores[result.doc_id] = scores.get(result.doc_id, 0.0) + (sw * tw) / (k + result.rank)
            payloads.setdefault(result.doc_id, result.payload)
            tiers.setdefault(result.doc_id, result.source_tier)

    ordered = sorted(scores, key=lambda d: scores[d], reverse=True)
    return [
        ScoredResult(
            doc_id=doc_id,
            score=scores[doc_id],
            rank=i + 1,
            payload=payloads[doc_id],
            signal="rrf",
            source_tier=tiers[doc_id],
        )
        for i, doc_id in enumerate(ordered)
    ]
