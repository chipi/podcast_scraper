"""Enrichment scoring — topic_consensus (RFC-088 / ADR-108).

The reimagined successor to ``enrichment_nli_contradiction.py``. Where the old
scorer measured the *contradiction* positive class, this one measures the
**consensus** (symmetric-entailment corroboration) positive class the enricher
now emits — the winnable side of the same NLI signal.

Reads operator-curated gold JSONL of cross-Person Insight pairs and computes
precision / recall / F1 against the ``consensus`` class from the corpus's
``enrichments/topic_consensus.json`` envelope.

Gold row shape (one per pair)::

    {"topic_id": "topic:ai-safety",
     "insight_a_id": "insight:ep1-i5",
     "insight_b_id": "insight:ep3-i2",
     "label": "consensus"}          # consensus | no_consensus

The corpus envelope holds the model's *positives* (pairs whose symmetric
entailment scored >= threshold). For each gold row:

    - labeled "consensus" AND in corpus envelope     → true positive
    - labeled "consensus" AND NOT in corpus envelope → false negative
    - labeled non-consensus AND in corpus envelope   → false positive

Aggregate: P / R / F1 + a head sample of the FP / FN rows for error analysis.

Brier score (calibration) requires the model's probability for every gold
pair, which the envelope only carries for positives. Pass ``--with-live-model``
to load DeBERTa locally and re-score every gold pair — the scorer then emits
Brier alongside P/R/F1. That path is intentionally OFF by default per
[[feedback_no_llm_in_ci]]; only the operator opts in.

Brier formula (consensus calibration)::

    Brier = mean over gold rows of (consensus_prob - label_int)^2

where ``consensus_prob = min(entail(a,b), entail(b,a))`` (the symmetric score the
enricher thresholds) and ``label_int = 1 if label == "consensus" else 0``.

Usage::

    python scripts/eval/score/enrichment_topic_consensus.py \\
        --corpus path/to/corpus \\
        --gold data/eval/enrichment/topic_consensus/gold \\
        --threshold 0.6

Exit codes:
    0 — scored (or no gold present)
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


def _pair_key(a: str, b: str) -> tuple[str, str]:
    """Order-independent pair key — (sorted) tuple of the two insight ids."""
    lo, hi = sorted((a, b))
    return lo, hi


def main() -> int:  # noqa: C901 — orchestration script; splitting hurts readability
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("data/eval/enrichment/topic_consensus/gold"),
    )
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--model", type=str, default="cross-encoder/nli-deberta-v3-small")
    parser.add_argument(
        "--with-live-model",
        action="store_true",
        help=(
            "Load DeBERTa locally and re-score every gold pair (both directions) "
            "so a Brier score (calibration) can be reported alongside P/R/F1. "
            "Requires the [ml] extra; downloads ~80MB on first run. "
            "Operator opt-in only — never in CI."
        ),
    )
    args = parser.parse_args()

    out = args.corpus / "enrichments" / "topic_consensus.json"
    if not out.is_file():
        print(
            json.dumps(
                {
                    "status": "no_corpus_output",
                    "expected": str(out),
                    "message": "Run the topic_consensus enricher first.",
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
                        '{"topic_id": ..., "insight_a_id": ..., '
                        '"insight_b_id": ..., "label": "consensus"} '
                        "to enable P/R/F1 scoring."
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

    consensus = (envelope.get("data") or {}).get("consensus") or []
    detected_pairs: set[tuple[str, str]] = set()
    detected_scores: dict[tuple[str, str], float] = {}
    for row in consensus:
        if not isinstance(row, dict):
            continue
        a, b = row.get("insight_a_id"), row.get("insight_b_id")
        if isinstance(a, str) and isinstance(b, str):
            key = _pair_key(a, b)
            detected_pairs.add(key)
            try:
                detected_scores[key] = float(row.get("consensus_score") or 0.0)
            except (TypeError, ValueError):
                pass

    tp = 0
    fp_rows: list[dict[str, Any]] = []
    fn_rows: list[dict[str, Any]] = []
    gold_positives = 0
    total = 0
    for gold_file in gold_files:
        for gold_row in _read_jsonl(gold_file):
            a = gold_row.get("insight_a_id")
            b = gold_row.get("insight_b_id")
            label = gold_row.get("label")
            if not (isinstance(a, str) and isinstance(b, str) and isinstance(label, str)):
                continue
            total += 1
            key = _pair_key(a, b)
            in_corpus = key in detected_pairs
            if label == "consensus":
                gold_positives += 1
                if in_corpus:
                    tp += 1
                else:
                    fn_rows.append({"pair": list(key), "label": label})
            else:
                if in_corpus:
                    fp_rows.append(
                        {
                            "pair": list(key),
                            "label": label,
                            "score": detected_scores.get(key),
                        }
                    )

    fp_count = len(fp_rows)
    fn_count = len(fn_rows)
    detected_count = tp + fp_count
    precision = tp / detected_count if detected_count else 0.0
    recall = tp / gold_positives if gold_positives else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Optional --with-live-model: load DeBERTa and compute a symmetric-consensus
    # Brier score over every gold pair. Operator opt-in only.
    brier: float | None = None
    brier_n = 0
    if args.with_live_model:
        try:
            import asyncio as _asyncio

            from podcast_scraper.enrichment.scorers.nli import DeBERTaNliScorer
        except Exception as exc:  # noqa: BLE001
            print(
                json.dumps(
                    {
                        "status": "live_model_unavailable",
                        "error": str(exc),
                        "hint": (
                            "install [ml] extra (sentence-transformers) "
                            "to enable --with-live-model"
                        ),
                    }
                )
            )
            return 2
        scorer = DeBERTaNliScorer(model_id=args.model)
        sq_err_sum = 0.0
        n = 0
        for gold_file in gold_files:
            for gold_row in _read_jsonl(gold_file):
                a_text = gold_row.get("insight_a_text")
                b_text = gold_row.get("insight_b_text")
                label = gold_row.get("label")
                if not (
                    isinstance(a_text, str) and isinstance(b_text, str) and isinstance(label, str)
                ):
                    continue
                label_int = 1.0 if label == "consensus" else 0.0
                try:
                    ab = _asyncio.run(scorer.score(a_text, b_text))
                    ba = _asyncio.run(scorer.score(b_text, a_text))
                except Exception as exc:  # noqa: BLE001
                    print(
                        json.dumps(
                            {
                                "status": "live_model_predict_failed",
                                "error": str(exc),
                                "pair": [
                                    gold_row.get("insight_a_id"),
                                    gold_row.get("insight_b_id"),
                                ],
                            }
                        )
                    )
                    return 2
                symmetric = min(ab.entailment, ba.entailment)
                sq_err_sum += (symmetric - label_int) ** 2
                n += 1
        if n > 0:
            brier = sq_err_sum / n
            brier_n = n

    payload: dict[str, Any] = {
        "status": "scored",
        "threshold": args.threshold,
        "model": args.model,
        "gold_rows": total,
        "gold_positives": gold_positives,
        "detected_positives": detected_count,
        "true_positives": tp,
        "false_positives": fp_count,
        "false_negatives": fn_count,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_positive_rows": fp_rows[:20],  # head sample for error analysis
        "false_negative_rows": fn_rows[:20],
    }
    if brier is not None:
        payload["brier"] = round(brier, 6)
        payload["brier_n"] = brier_n
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
