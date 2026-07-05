"""Data-quality cross-check for #382.

Two comparisons the branch-internal parity gate didn't do:

1) v5-post vs shipped historical baseline (baseline_ml_bart_authority_smoke_v1).
   Validates that the v5 code produces output consistent with what we've
   been shipping for months on the same recipe.

2) v5-post vs silver_opus47_smoke_v1 reference (Claude Opus 4.7 silver
   labels). Validates absolute quality against a semantic ground truth,
   not just self-consistency. Comparable to what ADR-068 measured on v4.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from rouge_score import rouge_scorer  # type: ignore

REPO = Path(__file__).resolve().parents[2]
SCORER = rouge_scorer.RougeScorer(["rougeL", "rouge1", "rouge2"], use_stemmer=True)


def load_predictions(path: Path) -> Dict[str, str]:
    """Extract {episode_id: summary_text} from a predictions.jsonl.

    Handles the eval-harness shape: {"episode_id": ..., "output": {"summary_final": ...}}.
    Falls back to legacy shapes for robustness.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    out: Dict[str, str] = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        ep = row.get("episode_id") or row.get("id")
        out_field = row.get("output")
        if isinstance(out_field, dict):
            pred = out_field.get("summary_final") or out_field.get("summary")
        else:
            pred = out_field or row.get("prediction") or row.get("summary") or row.get("pred")
        if ep and pred:
            out[str(ep)] = str(pred)
    return out


def rouge_row(pre: str, post: str) -> Dict[str, float]:
    r = SCORER.score(pre, post)
    return {
        "rougeL": r["rougeL"].fmeasure,
        "rouge1": r["rouge1"].fmeasure,
        "rouge2": r["rouge2"].fmeasure,
    }


def compare(name: str, a_path: Path, b_path: Path) -> Dict[str, object]:
    a = load_predictions(a_path)
    b = load_predictions(b_path)
    shared = sorted(set(a) & set(b))
    print(f"\n=== {name} ({len(shared)} shared episodes) ===")
    print(f"  A: {a_path}")
    print(f"  B: {b_path}")
    per_ep: List[Dict[str, object]] = []
    for ep in shared:
        r = rouge_row(a[ep], b[ep])
        per_ep.append({"episode_id": ep, **r})
        print(
            f"  {ep}: rougeL={r['rougeL']:.4f}  rouge1={r['rouge1']:.4f}  rouge2={r['rouge2']:.4f}"
        )
    if per_ep:
        agg = {
            "min_rougeL": min(row["rougeL"] for row in per_ep),
            "mean_rougeL": sum(row["rougeL"] for row in per_ep) / len(per_ep),
            "mean_rouge1": sum(row["rouge1"] for row in per_ep) / len(per_ep),
            "mean_rouge2": sum(row["rouge2"] for row in per_ep) / len(per_ep),
        }
        print(
            f"  aggregate: min_L={agg['min_rougeL']:.4f}  "
            f"mean_L={agg['mean_rougeL']:.4f}  "
            f"mean_1={agg['mean_rouge1']:.4f}  "
            f"mean_2={agg['mean_rouge2']:.4f}"
        )
    else:
        agg = {}
    return {"name": name, "per_episode": per_ep, "aggregate": agg}


def main() -> None:
    reports = []

    # 1) v5-post vs shipped historical baseline (April 2026)
    reports.append(
        compare(
            "v5-post vs shipped_v1 (production authoritative baseline)",
            REPO / "data/eval/baselines/baseline_ml_bart_authority_smoke_v1/predictions.jsonl",
            REPO / "data/eval/baselines/baseline_ml_bart_authority_smoke_v5_post/predictions.jsonl",
        )
    )

    # 2) v5-post vs silver_opus47 reference (semantic ground truth)
    silver = REPO / "data/eval/references/silver/silver_opus47_smoke_v1/predictions.jsonl"
    if silver.exists():
        reports.append(
            compare(
                "v5-post vs silver_opus47_smoke_v1 (quality vs Opus 4.7 reference)",
                REPO
                / "data/eval/baselines/baseline_ml_bart_authority_smoke_v5_post/predictions.jsonl",
                silver,
            )
        )
    else:
        print(f"\nSKIP: silver reference missing at {silver}")

    # 3) Historical anchor: shipped_v1 vs the same silver — confirms our
    # comparison methodology matches what ADR-068 originally used.
    if silver.exists():
        reports.append(
            compare(
                "shipped_v1 vs silver_opus47_smoke_v1 (historical anchor for methodology)",
                REPO / "data/eval/baselines/baseline_ml_bart_authority_smoke_v1/predictions.jsonl",
                silver,
            )
        )

    out = REPO / "data/eval/runs/v5_data_quality_recheck_2026-07-05.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
