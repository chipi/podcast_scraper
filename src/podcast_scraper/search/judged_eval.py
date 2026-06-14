"""Human-judged hybrid-vs-baseline eval harness (RFC-057 / wire-live follow-up C).

The eval that unblocked FAISS removal (#995, done) and #860 (ML-router promotion). The
Stage-4 known-item proxy (``scripts/eval_two_tier_retrieval.py``) *saturated* — it
confirmed parity but could not discriminate, so it could not justify retiring the
baseline backend or promoting the ML router. This harness produces a **discriminating,
graded** eval:

1. ``build_judgment_template`` runs a real query set through both backends, unions
   the candidates, and emits a per-query record with each backend's ranking — ready
   for a human to fill in graded relevance (0/1/2…) per candidate.
2. ``score_from_judgments`` reads the filled-in judgments back and computes mean
   nDCG@k + recall@k per backend — the discriminating number the gates need.

The human grading step is the gate; everything around it (run both backends → merge
→ template → score) is here and tested. The same query set, once intent-labeled,
also seeds #860's real training data — so judging once unblocks both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

# (doc_id, text) pairs in rank order, best first.
RankFn = Callable[[str], Sequence[Tuple[str, str]]]


@dataclass
class JudgmentRecord:
    """One query's candidates + each backend's ranking, awaiting graded relevance."""

    query: str
    intent: str
    candidates: List[Dict] = field(
        default_factory=list
    )  # {doc_id, text, baseline_rank, hybrid_rank}
    baseline_ranking: List[str] = field(default_factory=list)  # doc_ids, best first
    hybrid_ranking: List[str] = field(default_factory=list)
    relevance: Dict[str, int] = field(default_factory=dict)  # doc_id -> grade (human-filled)


def build_judgment_template(
    queries: Sequence[str],
    *,
    baseline_ranks: RankFn,
    hybrid_ranks: RankFn,
    intent_of: Callable[[str], str],
    k: int = 10,
) -> List[JudgmentRecord]:
    """Run *queries* through both backends and build per-query judgment records.

    ``baseline_ranks`` / ``hybrid_ranks`` return ranked ``(doc_id, text)`` for a query;
    ``intent_of`` labels the query (seeds #860). Candidates from both backends are
    unioned with each backend's 1-based rank (or ``None`` if a backend missed it).
    """
    records: List[JudgmentRecord] = []
    for query in queries:
        baseline_list = list(baseline_ranks(query))[:k]
        hybrid_list = list(hybrid_ranks(query))[:k]
        baseline_rank = {doc_id: i + 1 for i, (doc_id, _) in enumerate(baseline_list)}
        hybrid_rank = {doc_id: i + 1 for i, (doc_id, _) in enumerate(hybrid_list)}
        texts = {doc_id: text for doc_id, text in list(baseline_list) + list(hybrid_list)}

        candidates = [
            {
                "doc_id": doc_id,
                "text": texts.get(doc_id, ""),
                "baseline_rank": baseline_rank.get(doc_id),
                "hybrid_rank": hybrid_rank.get(doc_id),
            }
            for doc_id in sorted(
                texts, key=lambda d: (baseline_rank.get(d, 999), hybrid_rank.get(d, 999))
            )
        ]
        records.append(
            JudgmentRecord(
                query=query,
                intent=intent_of(query),
                candidates=candidates,
                baseline_ranking=[doc_id for doc_id, _ in baseline_list],
                hybrid_ranking=[doc_id for doc_id, _ in hybrid_list],
            )
        )
    return records


def _dcg(grades: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(grades))


def _ndcg_at(ranking: Sequence[str], relevance: Dict[str, int], k: int) -> float:
    grades = [relevance.get(doc_id, 0) for doc_id in ranking[:k]]
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = _dcg(ideal)
    return _dcg(grades) / idcg if idcg > 0 else 0.0


def _recall_at(ranking: Sequence[str], relevance: Dict[str, int], k: int) -> float:
    relevant = {doc_id for doc_id, g in relevance.items() if g > 0}
    if not relevant:
        return 0.0
    hit = sum(1 for doc_id in ranking[:k] if doc_id in relevant)
    return hit / len(relevant)


def score_from_judgments(
    records: Sequence[JudgmentRecord], *, k: int = 10
) -> Dict[str, Dict[str, float]]:
    """Mean nDCG@k + recall@k per backend over judged *records*.

    Records with no positive grade are skipped (no signal). Returns
    ``{"baseline": {...}, "hybrid": {...}}``; compare to gate backend retirement / ML promotion.
    """
    agg = {b: {"ndcg": 0.0, "recall": 0.0} for b in ("baseline", "hybrid")}
    judged = [r for r in records if any(g > 0 for g in r.relevance.values())]
    n = len(judged)
    for rec in judged:
        agg["baseline"]["ndcg"] += _ndcg_at(rec.baseline_ranking, rec.relevance, k)
        agg["baseline"]["recall"] += _recall_at(rec.baseline_ranking, rec.relevance, k)
        agg["hybrid"]["ndcg"] += _ndcg_at(rec.hybrid_ranking, rec.relevance, k)
        agg["hybrid"]["recall"] += _recall_at(rec.hybrid_ranking, rec.relevance, k)
    for backend in agg:
        for metric in agg[backend]:
            agg[backend][metric] /= max(n, 1)
    agg["_judged_count"] = {"n": float(n)}  # for transparency / no-silent-truncation
    return agg
