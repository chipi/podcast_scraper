#!/usr/bin/env python3
"""Compute RFC-107 §T2 search-quality metrics over a corpus's LanceDB index.

Usage (from project root)::

    python scripts/tools/search_quality_metrics.py tests/fixtures/viewer-validation-corpus/v3
    python scripts/tools/search_quality_metrics.py path/to/corpus --json --no-embed
    python scripts/tools/search_quality_metrics.py path/to/corpus --enforce --min-ndcg 0.8

The third sibling of ``gil_quality_metrics.py`` and ``kg_quality_metrics.py``: a thin CLI
over a product module (``podcast_scraper.search.quality_metrics``), which is where the
measurement actually lives.

``--seed-labels`` re-freezes the relevance anchors in the query set from what search
returns today. It rewrites a tracked fixture, so it is opt-in and skips any query already
carrying a label.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from podcast_scraper.search.quality_metrics import (
        compute_search_quality_metrics,
        enforce_rfc107_thresholds,
        load_queries,
    )
except ImportError:
    root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(root / "src"))
    from podcast_scraper.search.quality_metrics import (
        compute_search_quality_metrics,
        enforce_rfc107_thresholds,
        load_queries,
    )


def _build_layer(corpus: Path):
    """Construct a RetrievalLayer over the corpus's LanceDB index."""
    from podcast_scraper.search.backends.lancedb_backend import LanceDBBackend
    from podcast_scraper.search.retrieval import RetrievalLayer

    lance_path = corpus / "search" / "lance_index"
    if not lance_path.is_dir():
        raise SystemExit(
            f"no LanceDB index under {lance_path}\n"
            "  build one first: make build-validation-index CORPUS=" + str(corpus)
        )
    backend = LanceDBBackend(str(lance_path))
    return backend, RetrievalLayer(backend)


def _load_embedder(model_name: str):
    """A sentence-transformers encoder matching the model the corpus was indexed with."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def encode(text: str) -> list[float]:
        return list(model.encode(text).tolist())

    return encode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="RFC-107 §T2 search quality metrics over a corpus's LanceDB index."
    )
    parser.add_argument("corpus", type=Path, help="Corpus directory containing search/")
    parser.add_argument(
        "--queries",
        type=Path,
        default=None,
        help="Labelled query set (default: <corpus>/search-queries.json)",
    )
    parser.add_argument("--json", action="store_true", help="Print metrics as JSON only")
    parser.add_argument("--out", type=Path, default=None, help="Also write the report here")
    parser.add_argument("--top-k", type=int, default=10, help="Rank depth (default: 10)")
    parser.add_argument(
        "--no-embed",
        action="store_true",
        help=(
            "Skip the sentence-transformer; embeddings are zero vectors so the BM25 signal "
            "dominates. Cheaper, and adequate when labels are BM25-visible."
        ),
    )
    parser.add_argument(
        "--seed-labels",
        action="store_true",
        help=(
            "Freeze the current top-K into expected_top_k_doc_ids for every query whose "
            "label_status is 'unlabeled-seed', then write the query set back atomically. "
            "Detects drift from this point on; says NOTHING about correctness (the seeding "
            "run scores nDCG = 1.0 by construction)."
        ),
    )
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="Exit 1 if the --min-* floors are not met (all default to OFF)",
    )
    parser.add_argument("--fail-on-errors", action="store_true", help="Exit 1 on any error")
    parser.add_argument("--min-queries", type=int, default=1)
    parser.add_argument("--min-ndcg", type=float, default=None)
    parser.add_argument("--min-intent-accuracy", type=float, default=None)
    parser.add_argument("--min-tier-coverage", type=float, default=None)
    args = parser.parse_args()

    queries_path = args.queries or (args.corpus / "search-queries.json")
    if not queries_path.is_file():
        print(f"no query set at {queries_path}", file=sys.stderr)
        return 2
    try:
        queries = load_queries(queries_path)
    except (OSError, ValueError) as exc:
        print(f"{exc}", file=sys.stderr)
        return 2
    if not queries:
        print(f"no queries in {queries_path}", file=sys.stderr)
        return 2

    backend, layer = _build_layer(args.corpus)
    meta = backend.read_index_meta() or {}
    embed_model = meta.get("embedding_model") or "sentence-transformers/all-MiniLM-L6-v2"
    embed_fn = None if args.no_embed else _load_embedder(embed_model)

    metrics = compute_search_quality_metrics(
        args.corpus,
        queries,
        layer=layer,
        embed_fn=embed_fn,
        top_k=args.top_k,
        seed_labels=args.seed_labels,
    )

    if args.seed_labels:
        _write_back(queries_path, queries)

    report = {
        "corpus": str(args.corpus),
        "queries_path": str(queries_path),
        "metrics": metrics.to_dict(),
        "per_query": [q.to_dict() for q in metrics.per_query],
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(metrics.to_dict(), indent=2))
    else:
        _print_summary(metrics, queries_path)

    exit_code = 0
    if args.enforce:
        ok, failures = enforce_rfc107_thresholds(
            metrics,
            min_queries=args.min_queries,
            min_ndcg=args.min_ndcg,
            min_intent_accuracy=args.min_intent_accuracy,
            min_tier_coverage=args.min_tier_coverage,
        )
        if not ok:
            print("\nTHRESHOLDS NOT MET:", file=sys.stderr)
            for failure in failures:
                print(f"  - {failure}", file=sys.stderr)
            exit_code = 1
    if args.fail_on_errors and metrics.errors:
        exit_code = 1
    return exit_code


def _write_back(path: Path, queries: list) -> None:
    """Persist seeded labels atomically — a crash mid-write must not truncate the set."""
    top_level = json.loads(path.read_text(encoding="utf-8"))
    top_level["queries"] = queries
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(top_level, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)
    seeded = sum(1 for q in queries if q.get("label_status") == "regression-anchor")
    print(f"--seed-labels: {seeded} regression-anchor label(s) written to {path}")


def _print_summary(metrics, queries_path: Path) -> None:
    """Human summary. Prints NOT MEASURED rather than 0.0 for anything uncomputable."""
    d = metrics.to_dict()
    print(f"search quality over {queries_path}")
    print(f"  queries scored: {d['query_count']}   labeled: {d['labeled_query_count']}")
    for key in (
        "ndcg_at_k_mean",
        "mrr_at_k_mean",
        "intent_router_accuracy",
        "tier_coverage_rate",
        "compound_lift_rate",
        "topic_consensus_precision",
    ):
        value = d[key]
        print(f"  {key:28} {'NOT MEASURED' if value is None else f'{value:.3f}'}")
    if metrics.errors:
        print(f"  errors: {len(metrics.errors)}")
        for err in metrics.errors[:5]:
            print(f"    - {err}")


if __name__ == "__main__":
    raise SystemExit(main())
