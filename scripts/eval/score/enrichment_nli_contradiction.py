"""Enrichment scoring — chunk-4 nli_contradiction (RFC-088).

Reads operator-curated gold JSONL of cross-Person Insight pairs and
computes precision / recall / F1 against the contradiction class from
the corpus's enrichments/nli_contradiction.json envelope.

Gold row shape (one per pair):

    {"topic_id": "topic:ai-safety",
     "insight_a_id": "insight:ep1-i3",
     "insight_b_id": "insight:ep2-i7",
     "label": "contradiction"}     # contradiction | neutral | entailment

The corpus envelope holds the model's *positives* (pairs scored
≥ threshold). For each gold row:

    - if labeled "contradiction" AND in corpus envelope → true positive
    - if labeled "contradiction" AND NOT in corpus envelope → false negative
    - if labeled non-contradiction AND in corpus envelope → false positive

Aggregate: P / R / F1 + per-row breakdown of FPs and FNs (for error
analysis).

Brier score (calibration) requires the model's probability for every
gold pair, which the envelope only carries for positives. Pass
``--with-live-model`` to load DeBERTa locally and re-score every gold
pair — the scorer then emits Brier alongside P/R/F1. That path is
intentionally OFF by default per [[feedback_no_llm_in_ci]]; only the
operator opts in.

Brier formula:
    Brier = mean over gold rows of (contradiction_prob - label_int)^2
where label_int = 1 if label=="contradiction" else 0.

Usage:
    python scripts/eval/score/enrichment_nli_contradiction.py \\
        --corpus path/to/corpus \\
        --gold data/eval/enrichment/nli_contradiction/gold \\
        --threshold 0.5

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
        default=Path("data/eval/enrichment/nli_contradiction/gold"),
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--model", type=str, default="cross-encoder/nli-deberta-v3-small")
    parser.add_argument(
        "--with-live-model",
        action="store_true",
        help=(
            "Load DeBERTa locally and re-score every gold pair so a Brier "
            "score (calibration) can be reported alongside P/R/F1. "
            "Requires the [ml] extra; downloads ~80MB on first run. "
            "Operator opt-in only — never in CI."
        ),
    )
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
                        "No gold JSONL yet. Drop *.jsonl with rows like "
                        '{"topic_id": ..., "insight_a_id": ..., '
                        '"insight_b_id": ..., "label": "contradiction"} '
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

    contradictions = (envelope.get("data") or {}).get("contradictions") or []
    detected_pairs: set[tuple[str, str]] = set()
    detected_scores: dict[tuple[str, str], float] = {}
    for row in contradictions:
        if not isinstance(row, dict):
            continue
        a, b = row.get("insight_a_id"), row.get("insight_b_id")
        if isinstance(a, str) and isinstance(b, str):
            key = _pair_key(a, b)
            detected_pairs.add(key)
            try:
                detected_scores[key] = float(row.get("contradiction_score") or 0.0)
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
            if label == "contradiction":
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

    # Optional --with-live-model: load DeBERTa and compute Brier score
    # (calibration) over every gold pair. Operator opt-in only.
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
                premise = gold_row.get("insight_a_text")
                hypothesis = gold_row.get("insight_b_text")
                label = gold_row.get("label")
                if not (
                    isinstance(premise, str)
                    and isinstance(hypothesis, str)
                    and isinstance(label, str)
                ):
                    continue
                label_int = 1.0 if label == "contradiction" else 0.0
                try:
                    score = _asyncio.run(scorer.score(premise, hypothesis))
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
                sq_err_sum += (score.contradiction - label_int) ** 2
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
