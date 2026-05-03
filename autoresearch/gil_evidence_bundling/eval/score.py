#!/usr/bin/env python3
"""GIL evidence bundling: score a variant run against the staged baseline (#698).

Reads ``metrics.json`` from two pipeline runs (baseline + variant) plus the
GI prediction stats, and emits a single scalar combining cost reduction,
grounding preservation, and latency reduction.

This script does NOT spawn paid LLM calls itself — it only reads existing
``metrics.json`` outputs. Run the actual pipeline twice (once per cell)
externally first, then point this script at the two run directories.

Score formula:

    score = 0.5 * cost_reduction
          + 0.3 * grounding_preservation
          + 0.2 * latency_reduction

A positive score means the variant beat the baseline on a weighted blend.
Aborts loudly if quality gates fail (grounding regression > 5pp absolute,
fallback rate > 20%, token explosion > 50k input tokens/ep on bundled).

Usage:

    python autoresearch/gil_evidence_bundling/eval/score.py \\
        --baseline data/eval/runs/<baseline_run_id> \\
        --variant  data/eval/runs/<variant_run_id> \\
        [--per-feed-floor-grounding 0.50]

Exits non-zero on quality-gate violation; prints the scalar to stdout on
success so an operator can pipe it into ``results.tsv``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from podcast_scraper.evaluation.gi_scorer import (  # noqa: E402
    compute_gil_prediction_stats,
)
from podcast_scraper.evaluation.scorer import load_predictions  # noqa: E402

logger = logging.getLogger(__name__)


# Quality gates (pre-Phase-5; tunable per RFC-073 dev/held-out rule).
GROUNDING_DROP_MAX_ABS = 0.05  # 5 pp
FALLBACK_RATE_MAX = 0.20  # 20%
INPUT_TOKENS_PER_EP_MAX = 50_000  # token-explosion guard
DEFAULT_PER_FEED_GROUNDING_FLOOR = 0.50  # omnycontent canary


def _read_metrics(run_dir: Path) -> Dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"metrics.json not found at {metrics_path}")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _gil_costs(metrics: Dict[str, Any]) -> Tuple[float, int]:
    """Return (combined_gil_cost_usd, combined_gil_calls) from metrics.json."""
    extract_cost = float(metrics.get("llm_gi_extract_quotes_cost_usd", 0.0))
    nli_cost = float(metrics.get("llm_gi_score_entailment_cost_usd", 0.0))
    extract_calls = int(metrics.get("llm_gi_extract_quotes_calls", 0))
    nli_calls = int(metrics.get("llm_gi_score_entailment_calls", 0))
    return extract_cost + nli_cost, extract_calls + nli_calls


def _input_tokens_per_episode(metrics: Dict[str, Any]) -> float:
    """Extract input tokens per episode (cap on prompt-size explosion)."""
    eps = max(1, int(metrics.get("episodes_with_gil") or metrics.get("episodes", 1)))
    extract_in = int(metrics.get("llm_gi_extract_quotes_input_tokens", 0))
    nli_in = int(metrics.get("llm_gi_score_entailment_input_tokens", 0))
    return float(extract_in + nli_in) / eps


def _fallback_rate(metrics: Dict[str, Any]) -> float:
    """Combined Layer A + Layer B fallback rate vs total bundled+fallback events."""
    a_calls = int(metrics.get("gi_evidence_extract_quotes_bundled_calls", 0))
    a_fb = int(metrics.get("gi_evidence_extract_quotes_bundled_fallbacks", 0))
    b_calls = int(metrics.get("gi_evidence_score_entailment_bundled_calls", 0))
    b_fb = int(metrics.get("gi_evidence_score_entailment_bundled_fallbacks", 0))
    total = a_calls + a_fb + b_calls + b_fb
    if total == 0:
        return 0.0
    return float(a_fb + b_fb) / total


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _grounding_stats(run_dir: Path) -> Dict[str, Any]:
    """Compute grounding-quality stats from predictions.jsonl in ``run_dir``."""
    preds_path = run_dir / "predictions.jsonl"
    if not preds_path.is_file():
        logger.warning("predictions.jsonl missing at %s — grounding stats = 0", preds_path)
        return {"grounding_rate": 0.0, "quotes_per_insight_mean": 0.0, "mean_nli_score": 0.0}
    preds = load_predictions(preds_path)
    return compute_gil_prediction_stats(preds)


def _check_gates(
    *,
    baseline_grounding: float,
    variant_grounding: float,
    variant_metrics: Dict[str, Any],
) -> None:
    """Raise SystemExit on any gate violation."""
    drop = baseline_grounding - variant_grounding
    if drop > GROUNDING_DROP_MAX_ABS:
        raise SystemExit(
            f"GATE FAIL: grounding dropped {drop:.3f} (> {GROUNDING_DROP_MAX_ABS}) "
            f"baseline={baseline_grounding:.3f} variant={variant_grounding:.3f}"
        )
    fb = _fallback_rate(variant_metrics)
    if fb > FALLBACK_RATE_MAX:
        raise SystemExit(
            f"GATE FAIL: bundled fallback rate {fb:.3f} > {FALLBACK_RATE_MAX} — "
            f"prompts unreliable, reject this cell"
        )
    tokens = _input_tokens_per_episode(variant_metrics)
    if tokens > INPUT_TOKENS_PER_EP_MAX:
        raise SystemExit(
            f"GATE FAIL: input tokens/ep {tokens:.0f} > {INPUT_TOKENS_PER_EP_MAX} — "
            f"bundled prompt too large; lower chunk_size or split"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline run dir")
    parser.add_argument("--variant", type=Path, required=True, help="Variant run dir")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    baseline = _read_metrics(args.baseline)
    variant = _read_metrics(args.variant)

    base_cost, base_calls = _gil_costs(baseline)
    var_cost, var_calls = _gil_costs(variant)

    # Latency proxy: total run wall-clock seconds (best-effort).
    base_secs = float(baseline.get("run_duration_seconds", 0.0)) or 1.0
    var_secs = float(variant.get("run_duration_seconds", 0.0)) or 1.0

    base_g = _grounding_stats(args.baseline)
    var_g = _grounding_stats(args.variant)
    base_grounding = float(base_g.get("grounding_rate", 0.0))
    var_grounding = float(var_g.get("grounding_rate", 0.0))

    _check_gates(
        baseline_grounding=base_grounding,
        variant_grounding=var_grounding,
        variant_metrics=variant,
    )

    cost_reduction = _clamp01(1.0 - (var_cost / base_cost)) if base_cost > 0 else 0.0
    grounding_preservation = _clamp01(var_grounding / base_grounding) if base_grounding > 0 else 1.0
    latency_reduction = _clamp01(1.0 - (var_secs / base_secs)) if base_secs > 0 else 0.0

    score = 0.5 * cost_reduction + 0.3 * grounding_preservation + 0.2 * latency_reduction

    logger.info(
        "calls: %d -> %d | cost: $%.4f -> $%.4f | grounding: %.3f -> %.3f "
        "| secs: %.1f -> %.1f | score: %.4f",
        base_calls,
        var_calls,
        base_cost,
        var_cost,
        base_grounding,
        var_grounding,
        base_secs,
        var_secs,
        score,
    )

    print(f"{score:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
