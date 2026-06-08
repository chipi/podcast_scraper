"""Threshold sweep for entity canonicalization (#853).

Sweeps the tunable knobs in `kg/entity_clusters.py` + `identity/resolver.py`
against the Sonnet-labelled silver set at
`data/eval/references/silver/entity_canon_v1/labels.jsonl`. Output: per-cell
precision / recall / F1, the Pareto front, and a recommended high-precision
operating point.

We re-implement the merge predicate here as a parametrised function so the
sweep doesn't have to monkey-patch module-level constants. The implementation
mirrors `_are_xep_variants` in `src/podcast_scraper/kg/entity_clusters.py` —
keep them in sync (and re-run the sweep) when that predicate changes.

Usage:
    python scripts/eval/score/entity_canon_sweep_v1.py \
        --silver data/eval/references/silver/entity_canon_v1/labels.jsonl \
        --output data/eval/runs/entity_canon_sweep_v1/
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from podcast_scraper.kg.filters import _clean_entity_name, _is_acronymish

_VERSION_TOKEN_RE = re.compile(r"\d")


def _ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def are_variants(
    name_a: str,
    name_b: str,
    *,
    token_ratio: float,
    overall_ratio: float,
) -> bool:
    """Parametrised mirror of `entity_clusters._are_xep_variants`.

    The third knob (``same_show_required``) is applied OUTSIDE this function
    because it operates on episode/show sets, not on the name pair itself.
    """
    a, b = _clean_entity_name(name_a), _clean_entity_name(name_b)
    if not a or not b:
        return False
    if a == b:
        return True
    if _is_acronymish(name_a, a) or _is_acronymish(name_b, b):
        return False
    if a.replace(" ", "") == b.replace(" ", "") and a.replace(" ", ""):
        return True
    ta, tb = a.split(), b.split()
    if len(ta) != len(tb):
        return False
    if _ratio(a, b) < overall_ratio:
        return False
    for x, y in zip(ta, tb):
        if x == y:
            continue
        if _VERSION_TOKEN_RE.search(x) or _VERSION_TOKEN_RE.search(y):
            return False
        if _ratio(x, y) < token_ratio:
            return False
    return True


@dataclass
class CellResult:
    token_ratio: float
    overall_ratio: float
    same_show_required: bool
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        return self.tp / max(self.tp + self.fp, 1)

    @property
    def recall(self) -> float:
        return self.tp / max(self.tp + self.fn, 1)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / max(p + r, 1e-9)


def evaluate(
    pairs: list[dict[str, Any]],
    *,
    token_ratio: float,
    overall_ratio: float,
    same_show_required: bool,
) -> CellResult:
    tp = fp = fn = tn = 0
    for p in pairs:
        gold = p["label"]
        if gold not in ("SAME", "DIFFERENT"):
            continue
        predicted_merge = are_variants(
            p["a_label"],
            p["b_label"],
            token_ratio=token_ratio,
            overall_ratio=overall_ratio,
        )
        if predicted_merge and same_show_required:
            predicted_merge = bool(p.get("same_show") or p.get("shared_episode"))
        if gold == "SAME" and predicted_merge:
            tp += 1
        elif gold == "SAME" and not predicted_merge:
            fn += 1
        elif gold == "DIFFERENT" and predicted_merge:
            fp += 1
        else:
            tn += 1
    return CellResult(
        token_ratio=token_ratio,
        overall_ratio=overall_ratio,
        same_show_required=same_show_required,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
    )


def pareto_front(cells: list[CellResult]) -> list[CellResult]:
    """Cells not dominated by any other cell on (precision, recall)."""
    pareto: list[CellResult] = []
    for c in cells:
        if any(
            (o.precision >= c.precision and o.recall > c.recall)
            or (o.precision > c.precision and o.recall >= c.recall)
            for o in cells
            if o is not c
        ):
            continue
        pareto.append(c)
    return sorted(pareto, key=lambda c: (-c.precision, -c.recall))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--silver", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    pairs = [json.loads(line) for line in args.silver.read_text().splitlines() if line.strip()]
    n_total = len(pairs)
    by_label = Counter(p.get("label", "?") for p in pairs)
    n_scored = by_label["SAME"] + by_label["DIFFERENT"]

    token_grid = [0.65, 0.70, 0.74, 0.76, 0.78, 0.80, 0.82, 0.85]
    overall_grid = [0.70, 0.75, 0.80, 0.82, 0.85, 0.88, 0.90]
    same_show_grid = [True, False]

    cells: list[CellResult] = []
    for tr in token_grid:
        for ov in overall_grid:
            for ss in same_show_grid:
                cells.append(
                    evaluate(pairs, token_ratio=tr, overall_ratio=ov, same_show_required=ss)
                )

    # Baseline = repo defaults: _TOKEN_RATIO=0.78, _OVERALL_RATIO=0.85, same_show_required=True
    baseline = next(
        c
        for c in cells
        if c.token_ratio == 0.78 and c.overall_ratio == 0.85 and c.same_show_required is True
    )

    pareto = pareto_front(cells)
    # High-precision pick: highest recall among cells with precision >= 0.95
    high_p = sorted(
        [c for c in cells if c.precision >= 0.95],
        key=lambda c: (-c.recall, -c.precision),
    )
    recommendation = high_p[0] if high_p else max(cells, key=lambda c: c.f1)

    args.output.mkdir(parents=True, exist_ok=True)
    metrics = {
        "schema": "metrics_entity_canon_sweep_v1",
        "silver_total": n_total,
        "silver_scored": n_scored,
        "silver_label_distribution": dict(by_label),
        "baseline": {
            "token_ratio": baseline.token_ratio,
            "overall_ratio": baseline.overall_ratio,
            "same_show_required": baseline.same_show_required,
            "precision": round(baseline.precision, 4),
            "recall": round(baseline.recall, 4),
            "f1": round(baseline.f1, 4),
            "tp": baseline.tp,
            "fp": baseline.fp,
            "fn": baseline.fn,
            "tn": baseline.tn,
        },
        "recommendation": {
            "token_ratio": recommendation.token_ratio,
            "overall_ratio": recommendation.overall_ratio,
            "same_show_required": recommendation.same_show_required,
            "precision": round(recommendation.precision, 4),
            "recall": round(recommendation.recall, 4),
            "f1": round(recommendation.f1, 4),
            "tp": recommendation.tp,
            "fp": recommendation.fp,
            "fn": recommendation.fn,
            "tn": recommendation.tn,
            "rationale": (
                "highest recall among cells with precision >= 0.95"
                if high_p
                else "max F1 (no cell reached the 0.95 precision floor)"
            ),
        },
        "pareto_front": [
            {
                "token_ratio": c.token_ratio,
                "overall_ratio": c.overall_ratio,
                "same_show_required": c.same_show_required,
                "precision": round(c.precision, 4),
                "recall": round(c.recall, 4),
                "f1": round(c.f1, 4),
            }
            for c in pareto
        ],
        "all_cells": [
            {
                "token_ratio": c.token_ratio,
                "overall_ratio": c.overall_ratio,
                "same_show_required": c.same_show_required,
                "precision": round(c.precision, 4),
                "recall": round(c.recall, 4),
                "f1": round(c.f1, 4),
                "tp": c.tp,
                "fp": c.fp,
                "fn": c.fn,
                "tn": c.tn,
            }
            for c in cells
        ],
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    report_lines = [
        "# Entity-canon threshold sweep — #853",
        "",
        f"**Silver pairs:** {n_total} ({n_scored} scored; "
        f"{by_label.get('SAME', 0)} SAME, {by_label.get('DIFFERENT', 0)} DIFFERENT, "
        f"{by_label.get('BORDERLINE', 0)} BORDERLINE skipped).",
        "",
        "## Baseline (current repo defaults: token=0.78, overall=0.85, same_show=True)",
        "",
        f"- precision={baseline.precision:.4f}  recall={baseline.recall:.4f}  "
        f"f1={baseline.f1:.4f}  (tp={baseline.tp} fp={baseline.fp} "
        f"fn={baseline.fn} tn={baseline.tn})",
        "",
        "## Recommendation",
        "",
        f"- **token={recommendation.token_ratio}  overall={recommendation.overall_ratio}  "
        f"same_show={recommendation.same_show_required}**",
        f"- precision={recommendation.precision:.4f}  recall={recommendation.recall:.4f}  "
        f"f1={recommendation.f1:.4f}",
        f"- {metrics['recommendation']['rationale']}",
        "",
        "## Pareto front (precision-recall non-dominated)",
        "",
        "| token | overall | same_show | precision | recall | f1 |",
        "| ---: | ---: | :---: | ---: | ---: | ---: |",
    ]
    for c in pareto:
        report_lines.append(
            f"| {c.token_ratio} | {c.overall_ratio} | {c.same_show_required} | "
            f"{c.precision:.4f} | {c.recall:.4f} | {c.f1:.4f} |"
        )
    (args.output / "metrics_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"silver pairs: {n_total} ({n_scored} scored)")
    print(f"baseline: P={baseline.precision:.3f} R={baseline.recall:.3f} F1={baseline.f1:.3f}")
    print(
        f"recommendation: token={recommendation.token_ratio} "
        f"overall={recommendation.overall_ratio} "
        f"same_show={recommendation.same_show_required} "
        f"-> P={recommendation.precision:.3f} "
        f"R={recommendation.recall:.3f} F1={recommendation.f1:.3f}"
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
