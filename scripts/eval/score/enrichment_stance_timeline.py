"""Enrichment scoring — stance_timeline (RFC-088 / ADR-108).

The reimagined successor to ``enrichment_disagreement_*``. Where the old
disagreement scorer tried to judge *cross-speaker* contested propositions (the
unwinnable shared-question problem), this one measures the **same-speaker**
signal the enricher now emits: did a given (Person, Topic) stance *shift* over
time? Same speaker deletes the shared-question gate entirely.

The unit of evaluation is a **(person, topic) trajectory**, not a pair. The
enricher classifies each trajectory as ``shifted`` (its stance range crossed the
move threshold, or the sign flipped) or not; this scorer computes precision /
recall / F1 of that ``shifted`` classification against operator gold.

Gold row shape (one per trajectory)::

    {"person_id": "person:jack-clark",
     "topic_id": "topic:ai-safety",
     "shifted": true}              # true = stance genuinely moved over time

The corpus envelope (``enrichments/stance_timeline.json``) holds every
trajectory with ``deviation.shifted``. For each gold row:

    - gold shifted=true  AND corpus shifted=true   → true positive
    - gold shifted=true  AND corpus shifted!=true  → false negative
      (includes the trajectory being absent from the envelope)
    - gold shifted=false AND corpus shifted=true   → false positive

Aggregate: P / R / F1 + a head sample of the FP / FN trajectories for error
analysis, plus a signed-error summary (predicted minus gold ``shifted`` rate)
so a systematic over/under-flagging bias is visible.

Unlike the pairwise NLI scorers there is no ``--with-live-model`` Brier path:
the ``shifted`` label is a deterministic function of the enricher's per-point
stance scores, so calibration is judged by the trajectory classification itself.

Usage::

    python scripts/eval/score/enrichment_stance_timeline.py \\
        --corpus path/to/corpus \\
        --gold data/eval/enrichment/stance_timeline/gold

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


def _as_bool(value: Any) -> bool:
    """Coerce a gold/envelope ``shifted`` value to bool, tolerating 0/1/"true"."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "shifted")
    return False


def main() -> int:  # noqa: C901 — orchestration script; splitting hurts readability
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("data/eval/enrichment/stance_timeline/gold"),
    )
    args = parser.parse_args()

    out = args.corpus / "enrichments" / "stance_timeline.json"
    if not out.is_file():
        print(
            json.dumps(
                {
                    "status": "no_corpus_output",
                    "expected": str(out),
                    "message": "Run the stance_timeline enricher first.",
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
                        '{"person_id": ..., "topic_id": ..., "shifted": true} '
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

    timelines = (envelope.get("data") or {}).get("timelines") or []
    detected_shifted: dict[tuple[str, str], bool] = {}
    for row in timelines:
        if not isinstance(row, dict):
            continue
        pid, tid = row.get("person_id"), row.get("topic_id")
        if isinstance(pid, str) and isinstance(tid, str):
            deviation = row.get("deviation")
            shifted = (
                _as_bool((deviation or {}).get("shifted")) if isinstance(deviation, dict) else False
            )
            detected_shifted[(pid, tid)] = shifted

    tp = 0
    tn = 0
    fp_rows: list[dict[str, Any]] = []
    fn_rows: list[dict[str, Any]] = []
    gold_positives = 0
    predicted_positives = 0
    total = 0
    for gold_file in gold_files:
        for gold_row in _read_jsonl(gold_file):
            pid = gold_row.get("person_id")
            tid = gold_row.get("topic_id")
            if not (isinstance(pid, str) and isinstance(tid, str) and "shifted" in gold_row):
                continue
            total += 1
            gold_shift = _as_bool(gold_row.get("shifted"))
            # Absent trajectory scores as not-shifted (a miss for a positive).
            pred_shift = detected_shifted.get((pid, tid), False)
            if gold_shift:
                gold_positives += 1
            if pred_shift:
                predicted_positives += 1
            if gold_shift and pred_shift:
                tp += 1
            elif gold_shift and not pred_shift:
                fn_rows.append(
                    {
                        "trajectory": [pid, tid],
                        "in_corpus": (pid, tid) in detected_shifted,
                    }
                )
            elif not gold_shift and pred_shift:
                fp_rows.append({"trajectory": [pid, tid]})
            else:
                tn += 1

    fp_count = len(fp_rows)
    fn_count = len(fn_rows)
    precision = tp / predicted_positives if predicted_positives else 0.0
    recall = tp / gold_positives if gold_positives else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    gold_rate = gold_positives / total if total else 0.0
    pred_rate = predicted_positives / total if total else 0.0

    payload: dict[str, Any] = {
        "status": "scored",
        "gold_rows": total,
        "gold_positives": gold_positives,
        "predicted_positives": predicted_positives,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp_count,
        "false_negatives": fn_count,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "gold_shift_rate": round(gold_rate, 4),
        "predicted_shift_rate": round(pred_rate, 4),
        "shift_rate_bias": round(pred_rate - gold_rate, 4),  # + = over-flags shifts
        "false_positive_rows": fp_rows[:20],  # head sample for error analysis
        "false_negative_rows": fn_rows[:20],
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
