#!/usr/bin/env python3
"""#940 Track 1 — DeepSeek-R1 as G-Eval judge agreement harness.

Goal: decide whether R1:32b (local on DGX, $0 marginal) can replace one of
the paid judges in the finale tier. We measure agreement vs. Sonnet 4.6 on
the same (summary, dimension) pairs. The decision rule:

- >= 75% exact-or-adjacent agreement on a 1-5 scale  -> integrate R1 as a
  third judge / cheap-slot substitute in the finale
- < 75% agreement                                    -> R1 not yet
  reliable; stick with Sonnet + Gemini

We sample (summary, dimension) pairs from the existing qualifier matrix
without re-running anything: a fixed set of finalist run dirs from the
finale config provides the summaries; we draw a deterministic ``N`` pairs
across (run, episode, dimension) for both judges to score.

Usage:

    python scripts/eval/explore_r1_as_judge.py \\
        --config data/eval/configs/finale/finale_smoke_v2_2026_06.yaml \\
        --n-pairs 24 \\
        --output-dir data/eval/runs/finale/r1_as_judge_2026_06

Output:
    - ``pair_scores.jsonl`` — one row per (run, episode, dim, judge) with score + cost
    - ``agreement_report.json`` — overall + per-dimension agreement rate
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from podcast_scraper.evaluation.finale_runner import (  # noqa: E402
    load_predictions,
    load_run_candidate,
    load_transcript,
    promote_finalists,
)
from podcast_scraper.evaluation.g_eval import (  # noqa: E402
    DIMENSIONS,
    build_dimension_prompt,
    parse_dimension_response,
)
from podcast_scraper.evaluation.judges import (  # noqa: E402
    DeepSeekR1Judge,
    JudgeUnavailableError,
    Sonnet46Judge,
)

logger = logging.getLogger("explore_r1_as_judge")

# Decision threshold per the #940 brief.
INTEGRATION_THRESHOLD = 0.75


def _select_pairs(
    finalists: List[Any],
    *,
    n_pairs: int,
    eval_root: Path,
) -> List[Dict[str, Any]]:
    """Pick ``n_pairs`` (run, episode, dim) tuples deterministically.

    The selection round-robins across (finalist, episode, dimension) so a
    single finalist or single episode doesn't dominate the sample — a more
    uniform spread gives a tighter agreement-rate estimate.
    """
    # Build full (finalist, episode, dim) catalogue first so we can interleave.
    catalogue: List[Dict[str, Any]] = []
    for finalist in finalists:
        preds = load_predictions(finalist.run_dir / "predictions.jsonl")
        for pred in preds:
            episode_id = pred.get("episode_id")
            dataset_id = pred.get("dataset_id") or ""
            if not episode_id or not dataset_id:
                continue
            transcript = load_transcript(
                dataset_id=dataset_id, episode_id=episode_id, eval_root=eval_root
            )
            if transcript is None:
                continue
            summary = ""
            out = pred.get("output") or {}
            if isinstance(out, dict):
                summary = out.get("summary_final") or out.get("summary_long") or ""
            elif isinstance(out, str):
                summary = out
            if not summary.strip():
                continue
            for dim in DIMENSIONS:
                catalogue.append(
                    {
                        "run_id": finalist.run_id,
                        "stratum": finalist.stratum,
                        "episode_id": str(episode_id),
                        "dimension": dim,
                        "transcript": transcript,
                        "summary": summary,
                    }
                )
    if not catalogue:
        return []
    # Round-robin: stride through the catalogue with a fixed step to spread the
    # sample evenly across run/episode/dim. Step is len/n; falls back to 1.
    step = max(1, len(catalogue) // n_pairs)
    selected = list(itertools.islice(itertools.cycle(catalogue[::step]), n_pairs))
    return selected


def _score_pair(judge: Any, *, pair: Dict[str, Any]) -> Dict[str, Any]:
    """Score one (judge, pair) and return a flat row for the JSONL output."""
    prompt = build_dimension_prompt(
        dimension=pair["dimension"], transcript=pair["transcript"], summary=pair["summary"]
    )
    try:
        jr = judge.score(prompt, max_tokens=512)
    except JudgeUnavailableError as exc:
        return {
            "run_id": pair["run_id"],
            "stratum": pair["stratum"],
            "episode_id": pair["episode_id"],
            "dimension": pair["dimension"],
            "judge_model": getattr(judge, "model", "unknown"),
            "error": f"transport: {exc}",
        }
    try:
        ds = parse_dimension_response(
            jr.text,
            expected_dimension=pair["dimension"],
            judge_model=getattr(judge, "model", "unknown"),
        )
    except ValueError as exc:
        return {
            "run_id": pair["run_id"],
            "stratum": pair["stratum"],
            "episode_id": pair["episode_id"],
            "dimension": pair["dimension"],
            "judge_model": getattr(judge, "model", "unknown"),
            "error": f"parse: {exc}",
        }
    return {
        "run_id": pair["run_id"],
        "stratum": pair["stratum"],
        "episode_id": pair["episode_id"],
        "dimension": pair["dimension"],
        "judge_model": getattr(judge, "model", "unknown"),
        "score": ds.score,
        "explanation": ds.explanation,
        "cost_usd": jr.cost_usd,
        "prompt_tokens": jr.prompt_tokens,
        "completion_tokens": jr.completion_tokens,
        "latency_seconds": jr.latency_seconds,
    }


def _agreement_summary(rows: List[Dict[str, Any]], *, tolerance: int = 1) -> Dict[str, Any]:
    """Compute exact-or-adjacent agreement between the two judges across pairs.

    Returns overall + per-dimension breakdown + per-stratum breakdown, plus a
    ``recommendation`` field per the integration threshold.
    """
    # Index by (run_id, episode_id, dim) -> {judge_model: score}
    indexed: Dict[tuple, Dict[str, int]] = {}
    for r in rows:
        if "score" not in r:
            continue
        key = (r["run_id"], r["episode_id"], r["dimension"])
        indexed.setdefault(key, {})[r["judge_model"]] = int(r["score"])

    overall_total = 0
    overall_agree = 0
    per_dim: Dict[str, Dict[str, int]] = {d: {"total": 0, "agree": 0} for d in DIMENSIONS}
    per_stratum: Dict[str, Dict[str, int]] = {}
    for (run_id, ep, dim), per_judge in indexed.items():
        if len(per_judge) != 2:
            continue
        a, b = list(per_judge.values())
        is_agree = abs(a - b) <= tolerance
        overall_total += 1
        overall_agree += int(is_agree)
        if dim in per_dim:
            per_dim[dim]["total"] += 1
            per_dim[dim]["agree"] += int(is_agree)
        stratum = next(
            (
                r["stratum"]
                for r in rows
                if (r["run_id"], r["episode_id"], r["dimension"]) == (run_id, ep, dim)
            ),
            "unknown",
        )
        slot = per_stratum.setdefault(stratum, {"total": 0, "agree": 0})
        slot["total"] += 1
        slot["agree"] += int(is_agree)

    overall_rate = overall_agree / overall_total if overall_total else 0.0
    return {
        "tolerance": tolerance,
        "n_pairs": overall_total,
        "overall_agreement_rate": overall_rate,
        "per_dimension": {
            d: {
                **per_dim[d],
                "rate": (
                    (per_dim[d]["agree"] / per_dim[d]["total"]) if per_dim[d]["total"] else None
                ),
            }
            for d in DIMENSIONS
        },
        "per_stratum": {
            name: {
                **vals,
                "rate": (vals["agree"] / vals["total"]) if vals["total"] else None,
            }
            for name, vals in per_stratum.items()
        },
        "threshold": INTEGRATION_THRESHOLD,
        "recommendation": (
            "integrate" if overall_rate >= INTEGRATION_THRESHOLD else "do_not_integrate_yet"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--n-pairs", type=int, default=24)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--skip-sonnet",
        action="store_true",
        help="Read Sonnet scores from an existing pair_scores.jsonl (re-run R1 only).",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        logger.error("Bad config: %s", args.config)
        return 1

    metrics_filename = cfg.get("metrics_filename", "metrics_vs_silver_opus47_smoke_v1.json")
    run_dirs: List[Path] = []
    for pattern in cfg.get("runs_glob", []):
        for p in sorted(glob.glob(str(REPO_ROOT / pattern))):
            path = Path(p)
            if path.is_dir():
                run_dirs.append(path)

    candidates = []
    for d in run_dirs:
        c = load_run_candidate(d, stratum_map=cfg["strata"], metrics_filename=metrics_filename)
        if c is not None:
            candidates.append(c)
    logger.info("Loaded %d run candidates", len(candidates))

    promo_cfg = cfg.get("promotion", {})
    finalists, _ = promote_finalists(
        candidates,
        per_stratum_top_k=int(promo_cfg.get("per_stratum_top_k", 3)),
        floor_fraction=float(promo_cfg.get("floor_fraction", 0.8)),
        overall_cap=int(promo_cfg.get("overall_cap", 12)),
    )
    logger.info("Selected %d finalists for the R1 agreement pool", len(finalists))

    eval_root = REPO_ROOT / "data" / "eval"
    pairs = _select_pairs(finalists, n_pairs=args.n_pairs, eval_root=eval_root)
    logger.info("Sampled %d (run, episode, dim) pairs", len(pairs))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "pair_scores.jsonl"

    rows: List[Dict[str, Any]] = []
    if args.skip_sonnet and rows_path.is_file():
        with rows_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                rows.append(json.loads(line))
        logger.info("Loaded %d existing rows from %s", len(rows), rows_path)
    else:
        sonnet = Sonnet46Judge()
        for i, pair in enumerate(pairs):
            row = _score_pair(sonnet, pair=pair)
            rows.append(row)
            logger.info(
                "Sonnet pair %d/%d %s/%s/%s -> %s",
                i + 1,
                len(pairs),
                pair["run_id"][:30],
                pair["episode_id"],
                pair["dimension"],
                row.get("score", row.get("error")),
            )

    r1 = DeepSeekR1Judge()
    for i, pair in enumerate(pairs):
        row = _score_pair(r1, pair=pair)
        rows.append(row)
        logger.info(
            "R1 pair %d/%d %s/%s/%s -> %s",
            i + 1,
            len(pairs),
            pair["run_id"][:30],
            pair["episode_id"],
            pair["dimension"],
            row.get("score", row.get("error")),
        )

    with rows_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = _agreement_summary(rows, tolerance=1)
    (args.output_dir / "agreement_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "Agreement (tol=1): overall=%.2f%% (n=%d) -> %s",
        report["overall_agreement_rate"] * 100,
        report["n_pairs"],
        report["recommendation"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
