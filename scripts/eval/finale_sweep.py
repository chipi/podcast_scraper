#!/usr/bin/env python3
"""CLI entry point for the autoresearch finale tier (#932).

Reads a finale YAML config, loads matching qualifier run dirs, promotes the
top-3-per-stratum finalists, scores them on G-Eval (Sonnet 4.6 primary +
Gemini 2.5 Pro cross-check on top-N), and writes the report artifacts under
``data/eval/runs/finale/<tag>/``.

Usage:

    python scripts/eval/finale_sweep.py \\
        --config data/eval/configs/finale/finale_smoke_v2_2026_06.yaml

Cost guard: ``cost_cap_usd`` in the config is enforced — partial results are
persisted on abort so a budget-blown sweep still leaves a usable report.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from podcast_scraper.evaluation.finale_runner import (  # noqa: E402
    FinalistAggregate,
    aggregate_finalist,
    judge_finalist,
    load_run_candidate,
    promote_finalists,
    write_finale_artifacts,
)
from podcast_scraper.evaluation.judges import (  # noqa: E402
    DeepSeekR1Judge,
    Gemini25ProJudge,
    Sonnet46Judge,
)

logger = logging.getLogger("finale_sweep")


def _build_judge(spec: Dict[str, Any]) -> Any:
    """Instantiate a judge from a config spec like ``{kind: sonnet46, model: ...}``.

    Adding a new judge: drop a new ``kind`` case + import its client here.
    """
    kind = spec.get("kind", "").lower()
    model = spec.get("model")
    if kind == "sonnet46":
        return Sonnet46Judge(model=model) if model else Sonnet46Judge()
    if kind == "gemini25pro":
        return Gemini25ProJudge(model=model) if model else Gemini25ProJudge()
    if kind == "deepseek_r1":
        return DeepSeekR1Judge(model=model) if model else DeepSeekR1Judge()
    raise ValueError(f"Unknown judge kind {kind!r}; supported: sonnet46, gemini25pro, deepseek_r1")


def _load_config(path: Path) -> Dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Finale config must be a mapping: {path}")
    for key in ("tag", "runs_glob", "strata", "judges"):
        if key not in raw:
            raise ValueError(f"Finale config missing required key: {key!r}")
    return raw


def _discover_run_dirs(runs_glob: List[str], *, repo_root: Path) -> List[Path]:
    matched: List[Path] = []
    for pattern in runs_glob:
        for p in sorted(glob.glob(str(repo_root / pattern))):
            path = Path(p)
            if path.is_dir():
                matched.append(path)
    return matched


def _per_episode_record(
    *,
    finalist_run_id: str,
    stratum: str,
    judge_role: str,
    scores: List[Any],  # List[SummaryScore]
) -> List[Dict[str, Any]]:
    """Flatten per-episode SummaryScores into JSONL records for the report bundle."""
    out: List[Dict[str, Any]] = []
    for s in scores:
        out.append(
            {
                "finalist_run_id": finalist_run_id,
                "stratum": stratum,
                "judge_role": judge_role,
                **s.as_dict(),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run promotion only — no judge calls, no API cost.",
    )
    parser.add_argument(
        "--max-finalists",
        type=int,
        default=None,
        help="Override config overall_cap (smoke testing).",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Override config max_episodes_per_finalist (smoke testing).",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = _load_config(args.config)
    tag = cfg["tag"]
    cost_cap = float(cfg.get("cost_cap_usd", 50.0))
    max_eps = args.max_episodes or int(cfg.get("max_episodes_per_finalist", 5))
    metrics_filename = cfg.get("metrics_filename", "metrics_vs_silver_opus47_smoke_v1.json")

    run_dirs = _discover_run_dirs(cfg["runs_glob"], repo_root=REPO_ROOT)
    logger.info("Discovered %d candidate run dirs", len(run_dirs))

    candidates = []
    for d in run_dirs:
        c = load_run_candidate(d, stratum_map=cfg["strata"], metrics_filename=metrics_filename)
        if c is not None:
            candidates.append(c)
    logger.info("Loaded %d run candidates with metrics", len(candidates))
    if not candidates:
        logger.error("No candidates loaded — abort. Check runs_glob + metrics_filename.")
        return 1

    promo_cfg = cfg.get("promotion", {})
    per_stratum_top_k = int(promo_cfg.get("per_stratum_top_k", 3))
    floor_fraction = float(promo_cfg.get("floor_fraction", 0.8))
    overall_cap = args.max_finalists or int(promo_cfg.get("overall_cap", 12))

    finalists, promotion = promote_finalists(
        candidates,
        per_stratum_top_k=per_stratum_top_k,
        floor_fraction=floor_fraction,
        overall_cap=overall_cap,
    )
    logger.info("Promoted %d finalists across %d strata", len(finalists), len(promotion))
    for p in promotion:
        logger.info(
            "  stratum=%s leader=%.4f floor=%.4f promoted=%s",
            p.name,
            p.leader_rouge_l,
            p.floor,
            p.promoted,
        )

    output_root = Path(cfg.get("output_root", "data/eval/runs/finale"))
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    output_dir = output_root / tag

    if args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        promo_path = output_dir / "promotion.json"
        promo_path.write_text(
            json.dumps(
                {
                    "tag": tag,
                    "strata": [
                        {
                            "name": p.name,
                            "leader_rouge_l": p.leader_rouge_l,
                            "floor": p.floor,
                            "promoted": p.promoted,
                            "rejected": p.rejected,
                        }
                        for p in promotion
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("DRY RUN — promotion written to %s; no judges invoked.", promo_path)
        return 0

    eval_root = REPO_ROOT / "data" / "eval"
    primary_judge = _build_judge(cfg["judges"]["primary"])
    cross_cfg = cfg["judges"].get("cross_check") or {}
    cross_judge = _build_judge(cross_cfg) if cross_cfg.get("kind") else None
    top_n_per_stratum = int(cross_cfg.get("top_n_per_stratum", 2))

    aggregates: List[FinalistAggregate] = []
    per_episode_records: List[Dict[str, Any]] = []
    total_cost = 0.0
    start = time.monotonic()

    # PHASE 1 — primary judge on every finalist
    primary_scores_by_run: Dict[str, List[Any]] = {}
    for finalist in finalists:
        remaining = cost_cap - total_cost
        if remaining <= 0:
            logger.error("Cost cap $%.2f exhausted; skipping remaining primary judging", cost_cap)
            break
        logger.info("Primary judging %s (stratum=%s)", finalist.run_id, finalist.stratum)
        scores, cost, errs = judge_finalist(
            finalist=finalist,
            judge=primary_judge,
            eval_root=eval_root,
            max_episodes=max_eps,
            cost_budget_remaining=remaining,
        )
        primary_scores_by_run[finalist.run_id] = scores
        total_cost += cost
        per_episode_records.extend(
            _per_episode_record(
                finalist_run_id=finalist.run_id,
                stratum=finalist.stratum,
                judge_role="primary",
                scores=scores,
            )
        )
        logger.info(
            "  primary scored %d episodes, errs=%d, cost=$%.4f, total=$%.4f",
            len(scores),
            errs,
            cost,
            total_cost,
        )

    # PHASE 2 — cross-check judge on top-N per stratum (by primary overall mean)
    cross_scores_by_run: Dict[str, List[Any]] = {}
    if cross_judge is not None:
        # Rank within stratum by primary overall mean.
        scored_by_stratum: Dict[str, List[tuple[str, float]]] = {}
        for finalist in finalists:
            ps = primary_scores_by_run.get(finalist.run_id, [])
            means = []
            for s in ps:
                if s.mean is not None:
                    means.append(s.mean)
            overall = sum(means) / len(means) if means else 0.0
            scored_by_stratum.setdefault(finalist.stratum, []).append((finalist.run_id, overall))
        cross_targets = set()
        for stratum_name, items in scored_by_stratum.items():
            items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
            for run_id, _ in items_sorted[:top_n_per_stratum]:
                cross_targets.add(run_id)
        finalist_by_id = {f.run_id: f for f in finalists}
        for run_id in cross_targets:
            remaining = cost_cap - total_cost
            if remaining <= 0:
                logger.error("Cost cap $%.2f exhausted; skipping remaining cross-check", cost_cap)
                break
            finalist = finalist_by_id[run_id]
            logger.info("Cross-check judging %s", run_id)
            scores, cost, errs = judge_finalist(
                finalist=finalist,
                judge=cross_judge,
                eval_root=eval_root,
                max_episodes=max_eps,
                cost_budget_remaining=remaining,
            )
            cross_scores_by_run[run_id] = scores
            total_cost += cost
            per_episode_records.extend(
                _per_episode_record(
                    finalist_run_id=run_id,
                    stratum=finalist.stratum,
                    judge_role="cross_check",
                    scores=scores,
                )
            )
            logger.info(
                "  cross scored %d episodes, errs=%d, cost=$%.4f, total=$%.4f",
                len(scores),
                errs,
                cost,
                total_cost,
            )

    # PHASE 3 — aggregate
    primary_model = getattr(primary_judge, "model", "")
    cross_model = getattr(cross_judge, "model", None) if cross_judge else None
    for finalist in finalists:
        ps = primary_scores_by_run.get(finalist.run_id, [])
        cs = cross_scores_by_run.get(finalist.run_id) if cross_judge else None
        aggregates.append(
            aggregate_finalist(
                finalist=finalist,
                primary_judge_model=primary_model,
                primary_scores=ps,
                cross_judge_model=cross_model,
                cross_scores=cs,
            )
        )

    artifacts = write_finale_artifacts(
        output_dir=output_dir,
        tag=tag,
        promotion=promotion,
        aggregates=aggregates,
        per_episode_records=per_episode_records,
        total_cost_usd=total_cost,
        cost_cap_usd=cost_cap,
    )
    elapsed = time.monotonic() - start
    logger.info(
        "Done in %.1fs. Total spend: $%.4f (cap $%.2f). Artifacts:",
        elapsed,
        total_cost,
        cost_cap,
    )
    for name, path in artifacts.items():
        logger.info("  %s -> %s", name, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
