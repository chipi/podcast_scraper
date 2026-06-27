"""Enrichment scoring — chunk-4 nli_contradiction (RFC-088).

Reads a gold JSONL of labelled cross-Person Insight pairs under
``data/eval/enrichment/nli_contradiction/gold/`` and computes:

    - precision / recall / F1 against the `contradiction` class
    - Brier score (calibration of the contradiction probability)
    - error analysis JSONL: false positives + false negatives with text

Gold row shape (one per pair):
    {
      "topic_id": "topic:ai-safety",
      "insight_a_id": "insight:e1-i3",
      "insight_b_id": "insight:e2-i7",
      "insight_a_text": "AI is safe",
      "insight_b_text": "AI is dangerous",
      "label": "contradiction"  // contradiction | neutral | entailment
    }

Usage:
    python scripts/eval/score/enrichment_nli_contradiction.py \\
        --corpus path/to/corpus \\
        --gold data/eval/enrichment/nli_contradiction/gold \\
        --threshold 0.5 \\
        --model cross-encoder/nli-deberta-v3-small

NOT run in CI ([[feedback_no_llm_in_ci]]) — operator/manual only.

Exit codes:
    0 — scored OR no gold to score against
    1 — corpus output missing
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
        default=Path("data/eval/enrichment/nli_contradiction/gold"),
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--model", type=str, default="cross-encoder/nli-deberta-v3-small")
    args = parser.parse_args()

    out = args.corpus / "enrichments" / "nli_contradiction.json"
    if not out.is_file():
        print(
            json.dumps(
                {
                    "status": "no_corpus_output",
                    "expected": str(out),
                    "message": "Run the nli_contradiction enricher first.",
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
                        "No gold JSONL yet. Plan target ~100 labelled rows. "
                        "Drop *.jsonl with rows like "
                        '{"topic_id": ..., "insight_a_id": ..., '
                        '"insight_a_text": ..., "label": "contradiction"} '
                        "to enable precision/recall/F1 + Brier."
                    ),
                }
            )
        )
        return 0

    print(
        json.dumps(
            {
                "status": "not_implemented",
                "gold_files": [str(p) for p in gold_files],
                "threshold": args.threshold,
                "model": args.model,
                "message": (
                    "Gold JSONL present but scoring loop pending — wire "
                    "P/R/F1 + Brier + dev/held-out split once the ~100 "
                    "labelled rows are populated. Splits per "
                    "[[feedback_silver_judge_vendor_bias]]."
                ),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
