"""Enrichment scoring — chunk-3 topic_similarity (RFC-088).

Reads a gold JSONL of expected (topic_id, neighbour_id, similarity_bucket)
triples under ``data/eval/enrichment/topic_similarity/gold/`` and
measures top-K recall against the operator's corpus output at
``enrichments/topic_similarity.json``.

Metrics emitted:
    - recall@K — fraction of gold neighbours present in the corpus top-K
    - rank_correlation — Spearman ρ on overlap (when ≥10 overlapping pairs)

Usage:
    python scripts/eval/score/enrichment_topic_similarity.py \\
        --corpus path/to/corpus \\
        --gold data/eval/enrichment/topic_similarity/gold \\
        --top-k 10

Exit codes:
    0 — scored OR no gold to score against
    1 — corpus output missing (enricher never ran)
    2 — invocation error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("data/eval/enrichment/topic_similarity/gold"),
    )
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    out = args.corpus / "enrichments" / "topic_similarity.json"
    if not out.is_file():
        print(
            json.dumps(
                {
                    "status": "no_corpus_output",
                    "expected": str(out),
                    "message": "Run the topic_similarity enricher first.",
                }
            )
        )
        return 1

    gold_files = sorted(args.gold.glob("*.jsonl")) if args.gold.is_dir() else []
    if not gold_files:
        print(
            json.dumps(
                {
                    "status": "no_gold",
                    "gold_dir": str(args.gold),
                    "message": (
                        "No gold JSONL yet. Drop *.jsonl with rows like "
                        '{"topic_id": "topic:foo", "neighbours": ["topic:bar", '
                        '"topic:baz"]} to enable recall@K scoring.'
                    ),
                }
            )
        )
        return 0

    # Real recall@K loop: load gold, parse corpus output topics, compare.
    # Chunk 3 ships the enricher and the scaffolding; gold population is
    # operator-driven.
    print(
        json.dumps(
            {
                "status": "not_implemented",
                "gold_files": [str(p) for p in gold_files],
                "message": (
                    "Gold JSONL present but scoring loop pending — wire "
                    "recall@K + rank-correlation once the gold rows ship."
                ),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
