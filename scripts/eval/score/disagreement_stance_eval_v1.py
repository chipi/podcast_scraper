"""Precision/recall for the ``stance_disagreement`` enricher vs the #1144 gold (no LLM).

Scores the injected ``NliScorer`` (real DeBERTa via the ``[ml]`` extra) against
``data/eval/enrichment/disagreement/gold_v1.jsonl`` — the 40 prod-v2 pairs plus the
designed v3 Cho-vs-Bessent disagreement (known positive) and the Cho-vs-Fischer
topic-adjacent hard negative. Reports the enricher's shipping signal
(**stance-aggregate** symmetric contradiction, matching ``StanceDisagreementEnricher``)
plus **atomic-max** as a diagnostic, then optionally writes the gate metric the accuracy
gate reads.

Measured result (2026-07-07, ``cross-encoder/nli-deberta-v3-small``):

* stance-aggregate → precision 0%, recall 0% (the real disagreement dilutes below
  threshold; ``no_shared_question`` negatives over-fire),
* atomic-max → recall 100% but precision ~10% (negatives score 0.96–0.999 too).

i.e. **no-LLM DeBERTa cannot separate genuine opposition from topic-adjacency** — the
shared-question gate needs an LLM (#1144 rules that out). This script is the durable
regression bar: any future *non-LLM* scorer must clear precision ≥ 0.5 here to auto-promote
the (currently gated-dark) enricher.

Never runs in CI — it loads the real model ([[feedback_no_llm_in_ci]]); the CI smoke path
uses ``FixedNliScorer``. Run locally on a ``.[ml]`` install:

    python scripts/eval/score/disagreement_stance_eval_v1.py \\
        --gold data/eval/enrichment/disagreement/gold_v1.jsonl \\
        --threshold 0.6 [--write-gate-metrics]
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from podcast_scraper.enrichment.scorers.nli import DeBERTaNliScorer
from podcast_scraper.enrichment.scorers.protocol import NliScorer

_POSITIVE = "disagree"
_SWEEP = (0.5, 0.6, 0.7, 0.8, 0.9)


async def _sym(scorer: NliScorer, a: str, b: str) -> float:
    """Symmetric (min of both directions) contradiction probability for a pair."""
    ab = await scorer.score(a, b)
    ba = await scorer.score(b, a)
    return min(ab.contradiction, ba.contradiction)


async def _score_rows(scorer: NliScorer, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach stance-aggregate + atomic-max symmetric contradiction to each gold row."""
    out: list[dict[str, Any]] = []
    for r in rows:
        ai, bi = r["speaker_a_insights"], r["speaker_b_insights"]
        agg = await _sym(scorer, " ".join(ai), " ".join(bi))
        mx = 0.0
        for x in ai:
            for y in bi:
                mx = max(mx, await _sym(scorer, x, y))
        out.append({"row": r, "aggregate": agg, "atomic_max": mx})
    return out


def _prf(scored: list[dict[str, Any]], key: str, thr: float) -> tuple[float, float, int, int, int]:
    """(precision, recall, tp, fp, fn) treating score>=thr as a predicted disagreement."""
    n_pos = sum(1 for s in scored if s["row"]["label"] == _POSITIVE)
    tp = sum(1 for s in scored if s[key] >= thr and s["row"]["label"] == _POSITIVE)
    fp = sum(1 for s in scored if s[key] >= thr and s["row"]["label"] != _POSITIVE)
    fn = n_pos - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall, tp, fp, fn


def main() -> int:
    """Score the gold, print stance-aggregate + atomic-max PR, optionally write gate metrics."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--gold",
        type=Path,
        default=Path("data/eval/enrichment/disagreement/gold_v1.jsonl"),
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Threshold used for the persisted gate metric (matches the enricher default).",
    )
    p.add_argument(
        "--write-gate-metrics",
        action="store_true",
        help="Persist stance-aggregate precision/recall to gate_metrics.json (the gate input).",
    )
    args = p.parse_args()

    rows = [json.loads(x) for x in args.gold.read_text(encoding="utf-8").splitlines() if x.strip()]
    scorer = DeBERTaNliScorer()
    scored = asyncio.run(_score_rows(scorer, rows))

    n_pos = sum(1 for s in scored if s["row"]["label"] == _POSITIVE)
    print(f"gold rows: {len(rows)}  (positives='disagree': {n_pos})\n")
    for key, label in (
        ("aggregate", "stance-aggregate (SHIPPING signal)"),
        ("atomic_max", "atomic-max (diagnostic)"),
    ):
        print(f"{label}:")
        print(f"  {'thr':>5} {'precision':>10} {'recall':>8} {'TP':>4} {'FP':>4} {'FN':>4}")
        for thr in _SWEEP:
            prec, rec, tp, fp, fn = _prf(scored, key, thr)
            print(f"  {thr:>5.2f} {prec:>10.2%} {rec:>8.2%} {tp:>4} {fp:>4} {fn:>4}")
        print()

    precision, recall, tp, fp, fn = _prf(scored, "aggregate", args.threshold)
    if args.write_gate_metrics:
        from podcast_scraper.enrichment.eval.admission import write_gate_metrics

        written = write_gate_metrics(
            {"stance_disagreement": {"precision": round(precision, 4), "recall": round(recall, 4)}},
            run_id=f"disagreement_stance_eval_v1@thr{args.threshold}",
        )
        for path in written:
            print(f"wrote gate metric (precision={precision:.2%}) → {path}")
        print(
            "gate: precision "
            f"{precision:.2%} {'>=' if precision >= 0.5 else '<'} 0.50 → "
            f"{'PROMOTED' if precision >= 0.5 else 'GATED DARK'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
