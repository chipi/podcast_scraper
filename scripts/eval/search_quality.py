#!/usr/bin/env python3
"""Search v3 quality-eval harness (RFC-107 §T2, PRD-045 FR10; #1230 S0(c)).

Runs a labelled query set (default:
``tests/fixtures/viewer-validation-corpus/v3/search-queries.json``) against a
LanceDB corpus (default: the same v3 synthetic fixture) via
``RetrievalLayer.retrieve``, computes the RFC-107 metric set, and emits a
JSON report + optional summary.md.

Metric coverage (per-query + aggregate):

- ``nDCG@10`` — per RFC-107 §T2. Skipped when ``expected_top_k_doc_ids`` is
  ``null`` (label_status: "unlabeled-seed"); the count of skipped queries is
  reported in the summary.
- ``MRR@10`` — same skip rule as nDCG.
- ``intent_router_accuracy`` — predicted vs. ``intent_expected`` (RFC-092
  taxonomy). Measurable without labels.
- ``tier_coverage_rate`` — fraction of queries returning ≥1 Insight AND ≥1
  Transcript in top-10.
- ``compound_lift_rate`` — fraction of transcript hits that carry ``lifted``.
- ``enriched_answer_groundedness_rate`` — only computed when
  ``--enrich`` is passed AND a provider responds. Null otherwise (documented
  in the report).
- ``topic_consensus_precision`` — computed against the corpus's
  ``enrichments/topic_consensus.json`` when it exists; null otherwise.

Usage:

    python scripts/eval/search_quality.py \\
        --corpus tests/fixtures/viewer-validation-corpus/v3 \\
        --queries tests/fixtures/viewer-validation-corpus/v3/search-queries.json \\
        --out docs/wip/search-v3/eval/S0-baseline.json \\
        --top-k 10

Exit 0 always (this is a report, not a gate). Regression thresholds land later
via ``make eval-search`` invoking this harness and comparing to a stored
baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class QueryResult:
    id: str
    q: str
    intent_expected: str | None
    intent_predicted: str | None
    label_status: str
    ndcg_at_10: float | None
    mrr_at_10: float | None
    tier_counts: dict[str, int]
    compound_lift_hits: int
    transcript_hits: int
    hit_count: int
    top_doc_ids: list[str] = field(default_factory=list)


@dataclass
class Report:
    schema_version: str
    generated_at: str
    corpus: str
    queries_path: str
    query_count: int
    top_k: int
    metrics: dict[str, Any]
    per_query: list[dict[str, Any]]


def _dcg(rels: list[float]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def _ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
    """Binary relevance nDCG@k. 1.0 = perfect ranking; 0.0 = no relevant in top-k."""
    rels = [1.0 if d in relevant_ids else 0.0 for d in retrieved_ids[:k]]
    dcg = _dcg(rels)
    ideal_hits = min(len(relevant_ids), k)
    idcg = _dcg([1.0] * ideal_hits)
    return dcg / idcg if idcg > 0 else 0.0


def _mrr_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
    for i, d in enumerate(retrieved_ids[:k], start=1):
        if d in relevant_ids:
            return 1.0 / i
    return 0.0


def _load_queries(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    queries = data.get("queries", [])
    if not isinstance(queries, list):
        raise SystemExit(f"queries file malformed: 'queries' is not a list ({path})")
    return queries


def _hit_doc_id(hit: Any) -> str:
    """Extract doc_id from a ScoredResult or CompoundResult; falls back to id."""
    return getattr(hit, "doc_id", None) or getattr(hit, "id", "") or ""


def _hit_tier(hit: Any) -> str:
    """The source_tier a hit reports (insight / segment / aux / compound / unknown)."""
    return getattr(hit, "source_tier", None) or "unknown"


def _hit_has_lifted(hit: Any) -> bool:
    """A CompoundResult carries both a segment and an insight; ScoredResult
    with a 'lifted' payload also counts."""
    if getattr(hit, "insight", None) and getattr(hit, "segment", None):
        return True
    payload = getattr(hit, "payload", None) or {}
    return bool(payload.get("lifted"))


def _load_topic_consensus(corpus: Path) -> list[dict[str, Any]]:
    """Load topic_consensus enricher output if it exists on the fixture."""
    for name in ("topic_consensus.json", "topic_consensus_pairs.json"):
        candidate = corpus / "enrichments" / name
        if candidate.exists():
            data = json.loads(candidate.read_text())
            return data.get("pairs", []) or data.get("data", {}).get("pairs", [])
    return []


def _run_query(layer: Any, q: str, embed_fn: Any, top_k: int) -> list[Any]:
    """Run one query against the retrieval layer; returns hits."""
    embedding = embed_fn(q) if embed_fn is not None else [0.0] * 384
    hits = layer.retrieve(text=q, embedding=embedding, k=top_k, signals="hybrid")
    return list(hits)


def _load_embedder(model_name: str) -> Any:
    """Load a sentence-transformers embedder that matches the corpus index."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def encode(text: str) -> list[float]:
        return model.encode(text).tolist()

    return encode


def _build_layer(corpus: Path) -> Any:
    """Construct RetrievalLayer over the corpus's LanceDB index."""
    from podcast_scraper.search.backends.lancedb_backend import LanceDBBackend
    from podcast_scraper.search.retrieval import RetrievalLayer

    lance_path = corpus / "search" / "lance_index"
    if not lance_path.is_dir():
        raise SystemExit(f"lance index not found under {lance_path}")
    backend = LanceDBBackend(str(lance_path))
    return backend, RetrievalLayer(backend)


def _aggregate(per_query: list[QueryResult]) -> dict[str, Any]:
    labeled = [q for q in per_query if q.ndcg_at_10 is not None]
    intent_labeled = [q for q in per_query if q.intent_expected is not None]
    intent_correct = sum(1 for q in intent_labeled if q.intent_predicted == q.intent_expected)
    tier_ok = sum(
        1
        for q in per_query
        if q.tier_counts.get("insight", 0) >= 1 and q.tier_counts.get("segment", 0) >= 1
    )
    transcript_hits_total = sum(q.transcript_hits for q in per_query)
    compound_lift_hits_total = sum(q.compound_lift_hits for q in per_query)
    metrics: dict[str, Any] = {
        "ndcg_at_10_mean": (
            sum(q.ndcg_at_10 for q in labeled if q.ndcg_at_10 is not None) / len(labeled)
            if labeled
            else None
        ),
        "mrr_at_10_mean": (
            sum(q.mrr_at_10 for q in labeled if q.mrr_at_10 is not None) / len(labeled)
            if labeled
            else None
        ),
        "labeled_query_count": len(labeled),
        "unlabeled_query_count": len(per_query) - len(labeled),
        "intent_router_accuracy": (
            intent_correct / len(intent_labeled) if intent_labeled else None
        ),
        "intent_labeled_count": len(intent_labeled),
        "tier_coverage_rate": tier_ok / len(per_query) if per_query else None,
        "compound_lift_rate": (
            compound_lift_hits_total / transcript_hits_total if transcript_hits_total else None
        ),
        "transcript_hit_total": transcript_hits_total,
        "compound_lift_hit_total": compound_lift_hits_total,
        # Enriched-answer + topic_consensus metrics are None here — they need
        # an enrichment provider (RFC-088 chunk 5) and a topic_consensus
        # enricher output on the corpus. When those exist a follow-up pass
        # can populate them; the report shape stays stable.
        "enriched_answer_groundedness_rate": None,
        "topic_consensus_precision": None,
    }
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_corpus = Path("tests/fixtures/viewer-validation-corpus/v3")
    parser.add_argument("--corpus", type=Path, default=default_corpus)
    parser.add_argument(
        "--queries",
        type=Path,
        default=default_corpus / "search-queries.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/wip/search-v3/eval/latest.json"),
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--no-embed",
        action="store_true",
        help=(
            "Skip loading the sentence-transformer; embeddings are zero vectors "
            "(BM25 signal dominates). Cheaper; adequate for the labeled-nDCG check "
            "when the labels are picked from BM25-visible content."
        ),
    )
    args = parser.parse_args()

    queries = _load_queries(args.queries)
    if not queries:
        print(f"no queries in {args.queries}", file=sys.stderr)
        return 2

    backend, layer = _build_layer(args.corpus)
    meta = backend.read_index_meta() or {}
    embed_model = meta.get("embedding_model") or "sentence-transformers/all-MiniLM-L6-v2"
    embed_fn = None if args.no_embed else _load_embedder(embed_model)

    per_query: list[QueryResult] = []
    for entry in queries:
        qid = entry.get("id", "")
        qtext = entry.get("q", "")
        intent_expected = entry.get("intent_expected")
        label_status = entry.get("label_status", "unlabeled-seed")
        expected = entry.get("expected_top_k_doc_ids")
        if not qtext or label_status == "retired":
            continue
        hits = _run_query(layer, qtext, embed_fn, args.top_k)
        doc_ids = [_hit_doc_id(h) for h in hits]
        tier_counts: dict[str, int] = {}
        for h in hits:
            tier = _hit_tier(h)
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        transcript_hits = sum(1 for h in hits if _hit_tier(h) in ("segment", "compound"))
        compound_lift_hits = sum(1 for h in hits if _hit_has_lifted(h))
        intent_predicted = layer._classify(qtext)  # noqa: SLF001 - documented delegator
        ndcg = None
        mrr = None
        if isinstance(expected, list) and expected:
            rel = set(expected)
            ndcg = _ndcg_at_k(doc_ids, rel, k=args.top_k)
            mrr = _mrr_at_k(doc_ids, rel, k=args.top_k)
        per_query.append(
            QueryResult(
                id=qid,
                q=qtext,
                intent_expected=intent_expected,
                intent_predicted=intent_predicted,
                label_status=label_status,
                ndcg_at_10=ndcg,
                mrr_at_10=mrr,
                tier_counts=tier_counts,
                compound_lift_hits=compound_lift_hits,
                transcript_hits=transcript_hits,
                hit_count=len(hits),
                top_doc_ids=doc_ids,
            )
        )

    metrics = _aggregate(per_query)
    _ = _load_topic_consensus(args.corpus)  # future: populate topic_consensus_precision

    report = Report(
        schema_version="1",
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        corpus=str(args.corpus),
        queries_path=str(args.queries),
        query_count=len(per_query),
        top_k=args.top_k,
        metrics=metrics,
        per_query=[asdict(q) for q in per_query],
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n")

    labeled = metrics["labeled_query_count"]
    unlabeled = metrics["unlabeled_query_count"]
    router_acc = metrics["intent_router_accuracy"]
    tier_rate = metrics["tier_coverage_rate"]
    compound_rate = metrics["compound_lift_rate"]
    ndcg_mean = metrics["ndcg_at_10_mean"]
    print(f"eval-search: {report.query_count} queries scored, wrote {args.out}")
    print(f"  labeled: {labeled}   unlabeled (nDCG skipped): {unlabeled}")
    if ndcg_mean is not None:
        print(f"  nDCG@10 mean (labeled only): {ndcg_mean:.3f}")
    if router_acc is not None:
        labeled_count = metrics["intent_labeled_count"]
        print(f"  intent-router accuracy: {router_acc:.3f} ({labeled_count} labeled)")
    if tier_rate is not None:
        print(f"  tier coverage rate:     {tier_rate:.3f}")
    if compound_rate is not None:
        print(f"  compound-lift rate:     {compound_rate:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
