#!/usr/bin/env python3
"""Rescore one or more existing run dirs against a new silver reference.

Reads `predictions.jsonl` from each run dir, computes ROUGE / BLEU / WER /
embedding-cosine / coverage / numbers-retained vs the supplied silver
reference, and writes the result to `<run_dir>/metrics_vs_<reference_id>.json`
(non-destructive — does NOT overwrite `metrics.json`).

Existence-of-predictions is the only requirement; no LLM call. Useful when a
silver reference has been replaced and you want to re-score historical sweep
output without re-running inference.

Usage:

    python scripts/eval/score/rescore_against_silver.py \\
        --reference silver_opus47_smoke_v1 \\
        --runs-glob 'data/eval/runs/autoresearch_prompt_*_curated_5feeds_smoke_v1' \\
        --output-suffix opus47

This writes `metrics_vs_silver_opus47_smoke_v1.json` per run dir.

Each output JSON has shape:

    {
      "reference_id": "silver_opus47_smoke_v1",
      "reference_quality": "silver",
      "dataset_id": "curated_5feeds_smoke_v1",
      "run_id": "...",
      "vs_reference": { ...rouge/bleu/embedding/coverage/numbers... }
    }
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from podcast_scraper.evaluation.scorer import (  # noqa: E402
    compute_vs_reference_metrics,
    load_predictions,
)

logger = logging.getLogger(__name__)


def _resolve_reference(reference_id: str) -> Path:
    """Resolve a reference id to its directory, searching silver and gold."""
    for kind in ("silver", "gold/summarization", "gold"):
        candidate = REPO_ROOT / "data" / "eval" / "references" / kind / reference_id
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Reference not found under data/eval/references/{{silver,gold}}/{reference_id}/"
    )


def rescore_run(run_dir: Path, reference_id: str, reference_path: Path) -> dict | None:
    """Rescore one run dir; returns the vs_reference dict (or None if skipped)."""
    predictions_path = run_dir / "predictions.jsonl"
    if not predictions_path.exists():
        logger.warning("Skip %s — no predictions.jsonl", run_dir.name)
        return None

    predictions = load_predictions(predictions_path)
    if not predictions:
        logger.warning("Skip %s — predictions.jsonl empty", run_dir.name)
        return None

    dataset_id = predictions[0].get("dataset_id", "unknown")
    run_id = run_dir.name

    vs_ref = compute_vs_reference_metrics(predictions, reference_id, reference_path)
    if not vs_ref:
        logger.warning("Rescore returned empty for %s", run_dir.name)
        return None

    return {
        "reference_id": reference_id,
        "reference_quality": vs_ref.get("reference_quality", "silver"),
        "dataset_id": dataset_id,
        "run_id": run_id,
        "vs_reference": vs_ref,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        required=True,
        help="Reference id (e.g. silver_opus47_smoke_v1)",
    )
    parser.add_argument(
        "--runs-glob",
        action="append",
        required=True,
        help="Glob pattern (relative to repo root) for run dirs. May be repeated.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Output filename inside each run dir. Default: metrics_vs_<reference_id>.json",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    reference_path = _resolve_reference(args.reference)
    logger.info("Reference: %s at %s", args.reference, reference_path)

    output_name = args.output_name or f"metrics_vs_{args.reference}.json"
    run_dirs: list[Path] = []
    for pattern in args.runs_glob:
        matched = sorted(Path(p) for p in glob.glob(str(REPO_ROOT / pattern)))
        run_dirs.extend(d for d in matched if d.is_dir())
    if not run_dirs:
        logger.error("No run dirs matched any of the patterns: %s", args.runs_glob)
        return 1

    successes = 0
    failures = 0
    skipped = 0
    for run_dir in run_dirs:
        try:
            result = rescore_run(run_dir, args.reference, reference_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("FAIL %s: %s", run_dir.name, exc)
            failures += 1
            continue
        if result is None:
            skipped += 1
            continue
        out_path = run_dir / output_name
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        vs = result["vs_reference"]
        logger.info(
            "OK   %s rougeL=%.4f rouge1=%.4f cosine=%.4f cov=%.4f",
            run_dir.name,
            (vs.get("rougeL_f1") or 0.0),
            (vs.get("rouge1_f1") or 0.0),
            (vs.get("embedding_cosine") or 0.0),
            (vs.get("coverage_ratio") or 0.0),
        )
        successes += 1

    logger.info("Done. ok=%d skipped=%d failures=%d", successes, skipped, failures)
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
