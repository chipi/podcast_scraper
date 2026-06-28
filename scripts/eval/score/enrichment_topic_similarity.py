"""Enrichment scoring — chunk-3 topic_similarity (RFC-088).

Computes recall@K against an operator-curated gold JSONL of expected
neighbours per topic. Gold row shape:

    {"topic_id": "topic:ai-safety",
     "expected_neighbours": ["topic:ai-alignment", "topic:rlhf"]}

The script reads the corpus's enrichments/topic_similarity.json output
and per-gold-topic, computes:

    recall@K = |corpus_top_k_set ∩ expected_set| / |expected_set|

Aggregate: macro-mean recall@K + the per-topic breakdown.

Usage:
    python scripts/eval/score/enrichment_topic_similarity.py \\
        --corpus path/to/corpus \\
        --gold data/eval/enrichment/topic_similarity/gold \\
        --top-k 10

Exit codes:
    0 — scored OK (or no gold to score against)
    1 — corpus output missing
    2 — invocation / gold-parse error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


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
                        '{"topic_id": "topic:foo", '
                        '"expected_neighbours": ["topic:bar", "topic:baz"]} '
                        "to enable recall@K scoring."
                    ),
                }
            )
        )
        return 0

    try:
        envelope = json.loads(out.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(json.dumps({"status": "corpus_malformed", "error": str(exc)}))
        return 2
    topics = (envelope.get("data") or {}).get("topics") or []
    corpus_top_k: dict[str, list[str]] = {}
    for row in topics:
        if isinstance(row, dict) and isinstance(row.get("topic_id"), str):
            top_k_rows = row.get("top_k") or []
            corpus_top_k[row["topic_id"]] = [
                str(n.get("topic_id"))
                for n in top_k_rows
                if isinstance(n, dict) and isinstance(n.get("topic_id"), str)
            ]

    per_topic: list[dict[str, Any]] = []
    recall_sum = 0.0
    counted = 0
    for gold_file in gold_files:
        for gold_row in _read_jsonl(gold_file):
            tid = gold_row.get("topic_id")
            expected = gold_row.get("expected_neighbours") or []
            if not isinstance(tid, str) or not isinstance(expected, list) or not expected:
                continue
            expected_set = {e for e in expected if isinstance(e, str)}
            if not expected_set:
                continue
            top_k = corpus_top_k.get(tid, [])[: args.top_k]
            hit = expected_set & set(top_k)
            recall = len(hit) / len(expected_set)
            recall_sum += recall
            counted += 1
            per_topic.append(
                {
                    "topic_id": tid,
                    f"recall@{args.top_k}": round(recall, 4),
                    "expected": sorted(expected_set),
                    "hit": sorted(hit),
                    "missing": sorted(expected_set - hit),
                }
            )

    macro = round(recall_sum / counted, 4) if counted else 0.0
    print(
        json.dumps(
            {
                "status": "scored",
                "top_k": args.top_k,
                "gold_topics": counted,
                f"macro_recall@{args.top_k}": macro,
                "per_topic": per_topic,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
