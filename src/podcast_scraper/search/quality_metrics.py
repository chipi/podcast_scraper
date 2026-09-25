"""Search-oriented quality aggregates over a corpus's LanceDB index (file-based, no DB).

Use for operator reports and optional CI gates via ``scripts/tools/search_quality_metrics.py``.

The third sibling of ``podcast_scraper.gi.quality_metrics`` (GI artifacts) and
``podcast_scraper.kg.quality_metrics`` (KG artifacts). Those two measure what extraction
WROTE; this one measures what retrieval RETURNS, which is the half of the product no
artifact check can see — a corpus can be perfectly extracted and still un-findable.

It lived outside the product for a while, as a standalone script in the evaluation
research. That put the only tool that could maintain a committed fixture
(``tests/fixtures/viewer-validation-corpus/v3/search-queries.json``) somewhere this repo
could not reach: regenerating that corpus invalidated 234 of its 250 frozen relevance
anchors, and nothing here could re-freeze them or even notice (#2147). Measurement of the
shipped stack belongs with the shipped stack. Baselines, comparisons between runs, and
written-up results are research and stay with the research.

RFC-107 §T2 defines the metric set; RFC-092 defines the intent taxonomy.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

#: Dimension of the all-MiniLM-L6-v2 space the corpora are indexed in. Used only for the
#: zero-vector stand-in when embeddings are skipped.
_ZERO_VECTOR_DIM = 384


# --------------------------------------------------------------------------------------
# Ranking metrics — pure, no I/O, so they are testable without an index.
# --------------------------------------------------------------------------------------
def dcg(relevances: Sequence[float]) -> float:
    """Discounted cumulative gain over a ranked relevance list."""
    return sum(r / math.log2(i + 2) for i, r in enumerate(relevances))


def ndcg_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int = 10) -> float:
    """Binary-relevance nDCG@k. 1.0 = perfect ranking; 0.0 = nothing relevant in top-k."""
    rels = [1.0 if d in relevant_ids else 0.0 for d in retrieved_ids[:k]]
    ideal_hits = min(len(relevant_ids), k)
    idcg = dcg([1.0] * ideal_hits)
    return dcg(rels) / idcg if idcg > 0 else 0.0


def mrr_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int = 10) -> float:
    """Reciprocal rank of the first relevant hit in the top-k; 0.0 if there is none."""
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_ids:
            return 1.0 / i
    return 0.0


# --------------------------------------------------------------------------------------
# Hit accessors — the retrieval layer returns ScoredResult or CompoundResult.
# --------------------------------------------------------------------------------------
def hit_doc_id(hit: Any) -> str:
    """The doc id a hit carries, whichever result type it is."""
    return getattr(hit, "doc_id", None) or getattr(hit, "id", "") or ""


def hit_tier(hit: Any) -> str:
    """The source tier a hit reports (insight / segment / aux / compound / unknown)."""
    return getattr(hit, "source_tier", None) or "unknown"


def hit_has_lifted(hit: Any) -> bool:
    """Whether a hit carries a compound lift (both a segment and an insight)."""
    if getattr(hit, "insight", None) and getattr(hit, "segment", None):
        return True
    payload = getattr(hit, "payload", None) or {}
    return bool(payload.get("lifted"))


@dataclass
class QueryQualityResult:
    """One query's outcome. ``None`` metrics mean NOT MEASURED, never zero."""

    id: str
    q: str
    intent_expected: Optional[str]
    intent_predicted: Optional[str]
    label_status: str
    ndcg_at_k: Optional[float]
    mrr_at_k: Optional[float]
    tier_counts: Dict[str, int]
    compound_lift_hits: int
    transcript_hits: int
    hit_count: int
    top_doc_ids: List[str] = field(default_factory=list)
    consensus_pairs_in_topk: Optional[int] = None
    consensus_pairs_grounded: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly per-query record."""
        return {
            "id": self.id,
            "q": self.q,
            "intent_expected": self.intent_expected,
            "intent_predicted": self.intent_predicted,
            "label_status": self.label_status,
            "ndcg_at_k": self.ndcg_at_k,
            "mrr_at_k": self.mrr_at_k,
            "tier_counts": dict(self.tier_counts),
            "compound_lift_hits": self.compound_lift_hits,
            "transcript_hits": self.transcript_hits,
            "hit_count": self.hit_count,
            "top_doc_ids": list(self.top_doc_ids),
            "consensus_pairs_in_topk": self.consensus_pairs_in_topk,
            "consensus_pairs_grounded": self.consensus_pairs_grounded,
        }


@dataclass
class SearchQualityMetrics:
    """Aggregated metrics over the queries that were successfully run."""

    top_k: int = 10
    per_query: List[QueryQualityResult] = field(default_factory=list)
    skipped_queries: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    # -- means over the subset each metric can actually be computed on -------------------
    def labeled(self) -> List[QueryQualityResult]:
        """Queries carrying relevance labels — the only ones nDCG/MRR mean anything for."""
        return [q for q in self.per_query if q.ndcg_at_k is not None]

    def ndcg_mean(self) -> Optional[float]:
        """Mean nDCG@k over labelled queries; ``None`` when none are labelled."""
        rows = self.labeled()
        if not rows:
            return None
        return sum(q.ndcg_at_k or 0.0 for q in rows) / len(rows)

    def mrr_mean(self) -> Optional[float]:
        """Mean MRR@k over labelled queries; ``None`` when none are labelled."""
        rows = self.labeled()
        if not rows:
            return None
        return sum(q.mrr_at_k or 0.0 for q in rows) / len(rows)

    def intent_router_accuracy(self) -> Optional[float]:
        """Share of queries whose predicted intent matched the declared one.

        Needs no relevance labels — ``intent_expected`` is in the query set itself.
        """
        rows = [q for q in self.per_query if q.intent_expected is not None]
        if not rows:
            return None
        return sum(1 for q in rows if q.intent_predicted == q.intent_expected) / len(rows)

    def tier_coverage_rate(self) -> Optional[float]:
        """Share of queries returning at least one insight AND one transcript segment."""
        if not self.per_query:
            return None
        ok = sum(
            1
            for q in self.per_query
            if q.tier_counts.get("insight", 0) >= 1 and q.tier_counts.get("segment", 0) >= 1
        )
        return ok / len(self.per_query)

    def compound_lift_rate(self) -> Optional[float]:
        """Share of transcript hits that carried a compound lift."""
        transcript_total = sum(q.transcript_hits for q in self.per_query)
        if not transcript_total:
            return None
        return sum(q.compound_lift_hits for q in self.per_query) / transcript_total

    def topic_consensus_precision(self) -> Optional[float]:
        """Share of consensus pairs touching top-K whose both-grounded flag holds.

        ``None`` when the corpus has no ``enrichments/topic_consensus.json`` output, and
        also when it has one that is empty — a corpus with no people in its graph produces
        no pairs, so there is nothing to be precise about. Not a zero.
        """
        total = sum(q.consensus_pairs_in_topk for q in self.per_query if q.consensus_pairs_in_topk)
        if not total:
            return None
        grounded = sum(
            q.consensus_pairs_grounded for q in self.per_query if q.consensus_pairs_grounded
        )
        return grounded / total

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly aggregate metrics."""
        return {
            "top_k": self.top_k,
            "query_count": len(self.per_query),
            "labeled_query_count": len(self.labeled()),
            "unlabeled_query_count": len(self.per_query) - len(self.labeled()),
            "skipped_query_count": len(self.skipped_queries),
            "ndcg_at_k_mean": _round(self.ndcg_mean()),
            "mrr_at_k_mean": _round(self.mrr_mean()),
            "intent_router_accuracy": _round(self.intent_router_accuracy()),
            "tier_coverage_rate": _round(self.tier_coverage_rate()),
            "compound_lift_rate": _round(self.compound_lift_rate()),
            "topic_consensus_precision": _round(self.topic_consensus_precision()),
            "errors": list(self.errors),
        }


def _round(value: Optional[float]) -> Optional[float]:
    """Round for reporting while keeping ``None`` distinguishable from ``0.0``."""
    return None if value is None else round(value, 4)


def load_queries(path: Path) -> List[Dict[str, Any]]:
    """The labelled query set. Raises ValueError when the file is malformed."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    queries = data.get("queries", [])
    if not isinstance(queries, list):
        raise ValueError(f"queries file malformed: 'queries' is not a list ({path})")
    return queries


def load_topic_consensus_pairs(corpus: Path) -> List[Dict[str, Any]]:
    """Consensus pairs from the corpus's enricher output, or ``[]`` when there are none."""
    for name in ("topic_consensus.json", "topic_consensus_pairs.json"):
        candidate = Path(corpus) / "enrichments" / name
        if candidate.is_file():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return []
            pairs = data.get("pairs") or (data.get("data") or {}).get("pairs") or []
            return list(pairs) if isinstance(pairs, list) else []
    return []


def count_consensus_pairs_in_topk(
    doc_ids: Sequence[str], pairs: Sequence[Dict[str, Any]]
) -> Tuple[int, int]:
    """``(pairs touching top-K, of those, pairs flagged grounded)``.

    A pair "touches" top-K when either of its insight ids appears in ``doc_ids``
    (RFC-088 tuple shape per ADR-108). ``grounded`` defaults to True because the shipped
    enricher only emits pairs whose both sides are grounded.
    """
    if not pairs or not doc_ids:
        return (0, 0)
    top = set(doc_ids)
    touching = grounded = 0
    for pair in pairs:
        ids = [pair.get("insight_a_id"), pair.get("insight_b_id")]
        if not any(isinstance(i, str) and i in top for i in ids):
            continue
        touching += 1
        if pair.get("grounded", True) is True:
            grounded += 1
    return (touching, grounded)


def compute_search_quality_metrics(
    corpus: Path,
    queries: Sequence[Dict[str, Any]],
    *,
    layer: Any,
    embed_fn: Optional[Callable[[str], List[float]]] = None,
    top_k: int = 10,
    seed_labels: bool = False,
) -> SearchQualityMetrics:
    """Run every query against ``layer`` and aggregate RFC-107 §T2 metrics.

    ``layer`` is a constructed ``RetrievalLayer``; injected rather than built here so the
    metrics can be tested against a stub with no index on disk.

    ``seed_labels`` MUTATES the ``queries`` entries in place, freezing each
    ``unlabeled-seed`` query's current top-K into ``expected_top_k_doc_ids``. The caller
    owns persisting them. It is deliberately a no-op for any other ``label_status``, so
    re-running can never silently overwrite a human audit — and the run that seeds scores
    ``ndcg_at_k_mean = 1.0`` by construction, which is worth nothing on its own. The value
    is in a LATER run differing.
    """
    metrics = SearchQualityMetrics(top_k=top_k)
    consensus_pairs = load_topic_consensus_pairs(corpus)

    for entry in queries:
        qid = str(entry.get("id") or "")
        qtext = str(entry.get("q") or "")
        label_status = str(entry.get("label_status") or "unlabeled-seed")
        if not qtext or label_status == "retired":
            metrics.skipped_queries.append(qid or "<unnamed>")
            continue

        try:
            hits = list(
                layer.retrieve(
                    text=qtext,
                    embedding=(embed_fn(qtext) if embed_fn else [0.0] * _ZERO_VECTOR_DIM),
                    k=top_k,
                    signals="hybrid",
                )
            )
        except Exception as exc:  # noqa: BLE001 - one bad query must not lose the rest
            metrics.errors.append(f"{qid}: retrieval failed: {type(exc).__name__}: {exc}")
            continue

        doc_ids = [hit_doc_id(h) for h in hits]
        expected = entry.get("expected_top_k_doc_ids")
        if seed_labels and label_status == "unlabeled-seed":
            expected = [d for d in doc_ids if d][:top_k]
            entry["expected_top_k_doc_ids"] = expected
            entry["label_status"] = label_status = "regression-anchor"

        tier_counts: Dict[str, int] = {}
        for hit in hits:
            tier = hit_tier(hit)
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        relevant = set(expected) if isinstance(expected, list) and expected else None
        touching, grounded = (
            count_consensus_pairs_in_topk(doc_ids, consensus_pairs)
            if consensus_pairs
            else (None, None)
        )

        metrics.per_query.append(
            QueryQualityResult(
                id=qid,
                q=qtext,
                intent_expected=entry.get("intent_expected"),
                intent_predicted=_classify_intent(layer, qtext),
                label_status=label_status,
                ndcg_at_k=(ndcg_at_k(doc_ids, relevant, k=top_k) if relevant else None),
                mrr_at_k=(mrr_at_k(doc_ids, relevant, k=top_k) if relevant else None),
                tier_counts=tier_counts,
                compound_lift_hits=sum(1 for h in hits if hit_has_lifted(h)),
                transcript_hits=sum(1 for h in hits if hit_tier(h) in ("segment", "compound")),
                hit_count=len(hits),
                top_doc_ids=doc_ids,
                consensus_pairs_in_topk=touching,
                consensus_pairs_grounded=grounded,
            )
        )
    return metrics


def _classify_intent(layer: Any, text: str) -> Optional[str]:
    """The retrieval layer's own intent classification, or ``None`` if it exposes none."""
    classify = getattr(layer, "classify", None) or getattr(layer, "_classify", None)
    if not callable(classify):
        return None
    try:
        predicted = classify(text)
    except Exception:  # noqa: BLE001 - a router failure is not a measurement failure
        return None
    return str(predicted) if predicted is not None else None


def enforce_rfc107_thresholds(
    m: SearchQualityMetrics,
    *,
    min_queries: int = 1,
    min_ndcg: Optional[float] = None,
    min_intent_accuracy: Optional[float] = None,
    min_tier_coverage: Optional[float] = None,
) -> Tuple[bool, List[str]]:
    """Return ``(all_passed, human-readable failures)``.

    Every threshold defaults to ``None`` — OFF. That is deliberate and differs from the
    GI/KG siblings, which ship defensible defaults: this measures RELEVANCE, and a
    relevance floor is a claim about what is good enough, which nobody has made for this
    stack yet. A number nobody will defend is not a gate; it is a future argument about
    whether to lower it. Opt in explicitly when you have one.

    A metric that is ``None`` means NOT MEASURED (no labels, no queries). Asking for a
    floor on something unmeasured is a failure, not a pass — otherwise removing the labels
    would be the easiest way to go green.
    """
    failures: List[str] = []
    if len(m.per_query) < min_queries:
        failures.append(f"query_count {len(m.per_query)} < {min_queries}")
    if m.errors:
        failures.append(f"retrieval errors: {len(m.errors)}")

    for label, value, floor in (
        ("ndcg_at_k_mean", m.ndcg_mean(), min_ndcg),
        ("intent_router_accuracy", m.intent_router_accuracy(), min_intent_accuracy),
        ("tier_coverage_rate", m.tier_coverage_rate(), min_tier_coverage),
    ):
        if floor is None:
            continue
        if value is None:
            failures.append(f"{label} was NOT MEASURED but a floor of {floor} was required")
        elif value + 1e-9 < floor:
            failures.append(f"{label} {value:.3f} < {floor}")

    return len(failures) == 0, failures
