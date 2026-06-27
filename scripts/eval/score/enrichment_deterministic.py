"""Enrichment scoring — chunk-2 deterministic enrichers (RFC-088).

Exact-match against gold envelopes under
``data/eval/enrichment/deterministic/gold/``. One gold file per
(enricher_id, episode_stem) pair; the script runs each enricher
against the operator's corpus and compares the produced envelope
``data`` field byte-for-byte against the gold.

Usage:
    python scripts/eval/score/enrichment_deterministic.py \\
        --corpus path/to/corpus \\
        --gold data/eval/enrichment/deterministic/gold \\
        [--enricher topic_cooccurrence,grounding_rate]

Exit codes:
    0 — all enrichers match gold (or no gold to score against)
    1 — at least one mismatch
    2 — invocation error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _list_gold(gold_dir: Path) -> list[Path]:
    """Find every *.gold.json under gold_dir."""
    if not gold_dir.is_dir():
        return []
    return sorted(gold_dir.rglob("*.gold.json"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=False)
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("data/eval/enrichment/deterministic/gold"),
    )
    parser.add_argument(
        "--enricher",
        type=str,
        default="",
        help="Comma-separated subset of enricher ids to score.",
    )
    args = parser.parse_args()

    gold_files = _list_gold(args.gold)
    if not gold_files:
        # Empty gold dir — print a TODO marker and exit 0 so this script
        # can land as scaffolding and CI can wire it in once gold ships.
        print(
            json.dumps(
                {
                    "status": "no_gold",
                    "gold_dir": str(args.gold),
                    "message": (
                        "No gold files yet. Drop *.gold.json fixtures under "
                        f"{args.gold} (one per enricher_id/episode pair) "
                        "to enable exact-match scoring."
                    ),
                },
                indent=2,
            )
        )
        return 0

    # Real scoring would: load each gold file, run the enricher against
    # the corpus, diff envelope.data against gold.data. Chunk 2 ships
    # the deterministic enrichers but not yet the gold fixtures.
    #
    # Gold *files* are present but the scoring loop is not wired — this
    # is a real-but-unimplemented case (vs the empty-scaffolding above).
    # Exit 78 (EX_CONFIG, BSD sysexits) so CI distinguishes "scaffolding"
    # from "operator added gold but scorer is incomplete."
    print(
        json.dumps(
            {
                "status": "not_implemented",
                "gold_files": [str(p) for p in gold_files],
                "message": (
                    "Gold fixtures present but scoring loop pending — wire "
                    "envelope diff once the operator corpus is selected."
                ),
            },
            indent=2,
        )
    )
    return 78


if __name__ == "__main__":
    sys.exit(main())
