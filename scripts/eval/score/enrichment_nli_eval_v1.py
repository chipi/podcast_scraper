"""Two-lens precision/recall for the ``nli_contradiction`` enricher vs Opus silver (#1106).

Consumes the silver JSONL (``enrichment_nli_silver_v1.py``) and estimates the
enricher's **corpus-level** precision + recall from the stratified sample, using the
true flagged/unflagged corpus counts as stratum weights.

Why weighting: the silver sample is stratified (N flagged for precision + M unflagged
for recall), NOT a random draw. Raw sample rates would misstate corpus metrics. With
``F`` = total flagged pairs and ``U`` = total unflagged, and silver contradiction rates
``p_f`` (flagged stratum) / ``p_u`` (unflagged stratum):

    precision ≈ p_f                                  (flagged stratum ~ sample of flagged)
    recall    ≈ (F·p_f) / (F·p_f + U·p_u)            (missed = unflagged that are real)

Reported three ways via the two-lens ``contradiction_type``:
  * broad          — any contradiction counts (the Option-B product bar)
  * logical-only   — only ``logical`` contradictions (DeBERTa's fair strict-NLI bar)
  * competing-only — only ``competing_claim`` (the recall gap a strict-NLI model can't close)

Usage:
    python scripts/eval/score/enrichment_nli_eval_v1.py \\
        --silver data/eval/enrichment/nli_contradiction/gold/silver_prodv2_v1.jsonl \\
        --corpus .test_outputs/manual/prod-v2/corpus
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from enrichment_nli_harvest_v1 import (  # noqa: E402 — sibling-module import
    _all_cross_person_pairs,
    _load_flagged,
)


def _is_contra(row: dict[str, Any], lens: str) -> bool:
    if row.get("label") != "contradiction":
        return False
    ct = row.get("contradiction_type")
    if lens == "broad":
        return True
    if lens == "logical":
        return ct == "logical"
    if lens == "competing":
        return ct == "competing_claim"
    raise ValueError(lens)


def _rate(rows: list[dict[str, Any]], lens: str) -> tuple[int, int]:
    """(contradictions, total) among valid-labelled rows for a lens."""
    valid = [r for r in rows if r.get("label") in ("contradiction", "entailment", "neutral")]
    hits = sum(1 for r in valid if _is_contra(r, lens))
    return hits, len(valid)


def main() -> int:
    """Compute + print stratum-weighted precision/recall for each lens."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--silver", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True)
    args = p.parse_args()

    lines = args.silver.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(x) for x in lines if x.strip()]
    errors = [r for r in rows if r.get("label") in ("ERROR", "PARSE_ERROR")]

    # Corpus stratum weights: total flagged (F) and unflagged (U).
    flagged_map = _load_flagged(args.corpus)
    all_pairs, _ = _all_cross_person_pairs(args.corpus)
    n_flagged = len(flagged_map)
    n_unflagged = len(all_pairs) - n_flagged

    flagged_rows = [r for r in rows if r.get("deberta_flagged")]
    unflagged_rows = [r for r in rows if not r.get("deberta_flagged")]

    print(f"silver rows: {len(rows)}  (errors: {len(errors)})")
    print(f"corpus: {len(all_pairs)} cross-person pairs — F(flagged)={n_flagged} U={n_unflagged}")
    print(f"sample: flagged stratum {len(flagged_rows)}  unflagged stratum {len(unflagged_rows)}\n")

    header = (
        f"{'lens':<12} {'precision':>10} {'recall':>10}   "
        f"{'flag∩contra':>14} {'unflag∩contra':>16}"
    )
    print(header)
    print("-" * len(header))
    for lens in ("broad", "logical", "competing"):
        fh, ft = _rate(flagged_rows, lens)  # p_f numerator/denominator
        uh, ut = _rate(unflagged_rows, lens)
        p_f = fh / ft if ft else 0.0
        p_u = uh / ut if ut else 0.0
        precision = p_f
        est_flagged_true = n_flagged * p_f
        est_unflagged_true = n_unflagged * p_u
        denom = est_flagged_true + est_unflagged_true
        recall = est_flagged_true / denom if denom else 0.0
        print(
            f"{lens:<12} {precision:>10.2%} {recall:>10.2%}   "
            f"{f'{fh}/{ft}':>16} {f'{uh}/{ut}':>18}"
        )

    # Contradiction sub-type breakdown across the whole sample.
    from collections import Counter

    contra = [r for r in rows if r.get("label") == "contradiction"]
    print(
        f"\nsilver contradictions in sample: {len(contra)}  "
        f"types={dict(Counter(r.get('contradiction_type') for r in contra))}"
    )
    print("precision = of what DeBERTa flags, share that are real contradictions (silver).")
    print("recall    = of real contradictions corpus-wide (stratum-weighted), share DeBERTa flags.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
