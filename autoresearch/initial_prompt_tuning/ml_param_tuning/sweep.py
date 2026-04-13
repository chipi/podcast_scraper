#!/usr/bin/env python3
"""RFC-057 Track B: ML parameter autoresearch ratchet loop.

Sweeps map_params / reduce_params in an experiment config YAML, running each
candidate against a silver reference and accepting if ROUGE-L improves >= 1%.

Unlike Track A (prompt text mutation), this loop:
- Requires no LLM API keys for the experiment itself (local ML models)
- Scores via ROUGE-L F1 + embedding cosine (no judge — params affect structure
  and length, not semantic framing)
- Tries candidates one-at-a-time (greedy) from param_space.yaml, group by group

Usage:
    python autoresearch/ml_param_tuning/sweep.py \\
        --model bart_led \\
        [--max-fails 3] \\
        [--min-gain 0.01] \\
        [--results-dir autoresearch/ml_param_tuning/results] \\
        [--dry-run]

Outputs:
    results/<model>_sweep_<timestamp>.tsv  — per-experiment log
    results/<model>_best_config.yaml       — winning config (copy of base with best params)
    Stdout: promotion snippet for model_registry.py

Environment:
    AUTORESEARCH_EVAL_N   — episodes per run (default 5)
    HF_HUB_CACHE          — override HuggingFace cache dir (optional)
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PARAM_SPACE_PATH = Path(__file__).resolve().parent / "param_space.yaml"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

logger = logging.getLogger(__name__)


# ─── Scoring ─────────────────────────────────────────────────────────────────


def _run_experiment(config_path: Path, reference_id: str, force: bool = True) -> int:
    """Run scripts/eval/run_experiment.py. Returns exit code."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/eval/run_experiment.py"),
        str(config_path),
        "--reference",
        reference_id,
        "--log-level",
        "WARNING",
    ]
    if force:
        cmd.append("--force")
    logger.debug("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    return proc.returncode


def _read_scores(run_id: str, reference_id: str) -> Tuple[Optional[float], Optional[float]]:
    """Read (rouge_l_f1, embedding_cosine) from metrics.json. Returns (None, None) on miss."""
    metrics_path = REPO_ROOT / "data/eval/runs" / run_id / "metrics.json"
    if not metrics_path.is_file():
        return None, None
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        ref_scores = (metrics.get("vs_reference") or {}).get(reference_id, {})
        rouge = ref_scores.get("rougeL_f1")
        embed = ref_scores.get("embedding_cosine")
        return rouge, embed
    except Exception as e:
        logger.warning("Could not read metrics: %s", e)
        return None, None


def _score_config(
    config_path: Path,
    run_id: str,
    reference_id: str,
) -> Tuple[Optional[float], Optional[float]]:
    """Run experiment and return (rouge_l, embedding). Returns (None, None) on failure."""
    rc = _run_experiment(config_path, reference_id)
    if rc not in (0, -10, 138, 139):  # -10/138/139 = SIGBUS on MPS cleanup (non-critical)
        logger.error("run_experiment failed with code %d", rc)
        return None, None
    return _read_scores(run_id, reference_id)


# ─── Config mutation ──────────────────────────────────────────────────────────


def _mutate_config(
    base_cfg: Dict[str, Any],
    param_group: str,
    param_name: str,
    value: Any,
    exp_id: str,
) -> Dict[str, Any]:
    """Return a deep-copied config dict with one param mutated and a new unique id."""
    cfg = copy.deepcopy(base_cfg)
    cfg["id"] = exp_id
    group = cfg.get(param_group)
    if group is None:
        cfg[param_group] = {}
        group = cfg[param_group]
    group[param_name] = value
    return cfg


def _write_temp_config(cfg: Dict[str, Any], dest: Path) -> None:
    dest.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False), encoding="utf-8")


# ─── TSV logging ──────────────────────────────────────────────────────────────

TSV_HEADER = "\t".join(
    [
        "exp_id",
        "status",
        "rouge_l",
        "embed",
        "delta_pct",
        "param_group",
        "param_name",
        "value",
        "notes",
    ]
)


def _tsv_row(
    exp_id: str,
    status: str,
    rouge_l: Optional[float],
    embed: Optional[float],
    delta_pct: Optional[float],
    param_group: str,
    param_name: str,
    value: Any,
    notes: str = "",
) -> str:
    rouge_str = f"{rouge_l:.4f}" if rouge_l is not None else "n/a"
    embed_str = f"{embed:.4f}" if embed is not None else "n/a"
    delta_str = f"{delta_pct:+.2f}%" if delta_pct is not None else "n/a"
    return "\t".join(
        [
            exp_id,
            status,
            rouge_str,
            embed_str,
            delta_str,
            param_group,
            param_name,
            str(value),
            notes,
        ]
    )


# ─── Promotion snippet ────────────────────────────────────────────────────────


def _print_promotion_snippet(
    model_key: str,
    best_cfg: Dict[str, Any],
    baseline_id: str,
    rouge_l: float,
    embed: float,
    reference_id: str,
) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    mode_id = f"ml_{model_key}_autoresearch_v1"
    print("\n" + "=" * 72)
    print("PROMOTION SNIPPET — paste into model_registry.py _mode_registry:")
    print("=" * 72)

    def _to_py(obj: Any) -> str:
        """json.dumps but with Python bool/None literals."""
        return (
            json.dumps(obj, indent=16)
            .replace(": true", ": True")
            .replace(": false", ": False")
            .replace(": null", ": None")
        )

    backend = best_cfg.get("backend", {})
    is_direct = backend.get("type") == "ollama" and "map_model" not in backend
    if is_direct:
        snippet = f"""
        # Direct Ollama config — register in data/eval/configs/summarization/ not model_registry.py
        # Canonical config: data/eval/configs/summarization/{mode_id}.yaml
        # params: {_to_py(best_cfg.get("params", {}))}
        # rouge_l_f1={rouge_l:.4f}  embed={embed:.4f}
        # promoted_from="{baseline_id}"  promoted_at="{now}"
        """
    else:
        snippet = f"""
        "{mode_id}": ModeConfiguration(
            mode_id="{mode_id}",
            map_model="{backend.get("map_model", "")}",
            reduce_model="{backend.get("reduce_model", "")}",
            preprocessing_profile="{best_cfg.get("preprocessing_profile", "cleaning_v4")}",
            map_params={_to_py(best_cfg.get("map_params", {}))},
            reduce_params={_to_py(best_cfg.get("reduce_params", {}))},
            tokenize={_to_py(best_cfg.get("tokenize", {}))},
            chunking={_to_py(best_cfg.get("chunking"))},
            promoted_from="{baseline_id}",
            promoted_at="{now}",
            metrics_summary={{
                "dataset_id": "{best_cfg["data"]["dataset_id"]}",
                "reference_id": "{reference_id}",
                "rouge_l_f1": {rouge_l:.4f},
                "embedding_cosine": {embed:.4f},
            }},
        ),"""
    print(snippet)
    print("=" * 72 + "\n")


# ─── Ratchet loop ─────────────────────────────────────────────────────────────


def run_sweep(
    model_key: str,
    space: Dict[str, Any],
    max_fails: int = 3,
    min_gain: float = 0.01,
    results_dir: Path = RESULTS_DIR,
    dry_run: bool = False,
) -> int:
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tsv_path = results_dir / f"{model_key}_sweep_{timestamp}.tsv"
    best_config_path = results_dir / f"{model_key}_best_config.yaml"

    base_config_rel = space["base_config"]
    reference_id = space["reference"]
    base_config_path = REPO_ROOT / base_config_rel

    if not base_config_path.is_file():
        logger.error("Base config not found: %s", base_config_path)
        return 1

    base_cfg = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    baseline_id = base_cfg["id"]

    # ── Score baseline ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"ML PARAM SWEEP: {model_key}")
    print(f"Base config:    {base_config_rel}")
    print(f"Reference:      {reference_id}")
    print(f"Min gain:       {min_gain*100:.1f}%  Max fails: {max_fails}")
    print(f"{'='*60}\n")

    print("Scoring baseline...")
    baseline_rouge, baseline_embed = _score_config(
        config_path=base_config_path,
        run_id=baseline_id,
        reference_id=reference_id,
    )
    if baseline_rouge is None:
        logger.error("Baseline scoring failed — cannot proceed.")
        return 1

    print(
        f"Baseline: rouge_l={baseline_rouge:.4f}  embed={baseline_embed or 0:.4f}  "
        f"(id={baseline_id})"
    )

    tsv_rows: List[str] = [TSV_HEADER]
    tsv_rows.append(
        _tsv_row(
            baseline_id,
            "baseline",
            baseline_rouge,
            baseline_embed,
            0.0,
            "",
            "",
            "",
            "initial baseline",
        )
    )

    # Working state: best params accumulated from accepted experiments
    best_cfg = copy.deepcopy(base_cfg)
    best_rouge = baseline_rouge
    best_embed = baseline_embed

    exp_counter = 0
    consecutive_fails = 0

    # ── Iterate param groups ────────────────────────────────────────────────
    for group_name in ("ollama_reduce_params", "params", "reduce_params", "map_params"):
        group_space = space.get(group_name, {})
        if not group_space:
            continue

        print(f"\n── {group_name} ──")

        for param_name, param_cfg in group_space.items():
            if not param_cfg.get("enabled", True):
                print(f"  {param_name}: skipped (disabled)")
                continue

            candidates = param_cfg.get("candidates", [])
            if not candidates:
                continue

            desc = param_cfg.get("description", "")
            print(f"  {param_name}: {len(candidates)} candidates  [{desc}]")

            for value in candidates:
                if consecutive_fails >= max_fails:
                    print(f"\nEarly stop: {consecutive_fails} consecutive fails reached.")
                    break

                exp_counter += 1
                exp_id = f"{model_key}_sweep_exp{exp_counter:03d}"

                # Build mutated config from current best
                candidate_cfg = _mutate_config(best_cfg, group_name, param_name, value, exp_id)

                with tempfile.NamedTemporaryFile(
                    suffix=".yaml",
                    delete=False,
                    dir=str(base_config_path.parent),
                    prefix=f"_sweep_{exp_id}_",
                ) as tmp:
                    tmp_path = Path(tmp.name)

                try:
                    _write_temp_config(candidate_cfg, tmp_path)

                    rouge: Optional[float]
                    embed: Optional[float]
                    if dry_run:
                        print(f"    [DRY RUN] exp={exp_id}  {param_name}={value}")
                        rouge, embed = best_rouge + 0.001, best_embed
                    else:
                        rouge, embed = _score_config(
                            config_path=tmp_path,
                            run_id=exp_id,
                            reference_id=reference_id,
                        )

                    if rouge is None:
                        status = "error"
                        delta_pct = None
                        consecutive_fails += 1
                        notes = "run_experiment failed"
                    else:
                        delta_pct = (rouge - best_rouge) / best_rouge * 100
                        if delta_pct >= min_gain * 100:
                            status = "accepted"
                            # Commit mutation into best_cfg
                            best_cfg = candidate_cfg
                            best_cfg["id"] = best_cfg["id"]  # keep exp_id for tracing
                            best_rouge = rouge
                            best_embed = embed
                            consecutive_fails = 0
                            notes = f"new best rouge_l={rouge:.4f}"
                        else:
                            status = "rejected"
                            consecutive_fails += 1
                            notes = (
                                f"below threshold ({delta_pct:+.2f}%), fails={consecutive_fails}"
                            )

                    symbol = {"accepted": "✓", "rejected": "✗", "error": "!"}.get(status, "?")
                    rouge_str = f"{rouge:.4f}" if rouge is not None else "n/a"
                    delta_str = f"  delta={delta_pct:+.2f}%" if delta_pct is not None else ""
                    print(
                        f"    {symbol} exp={exp_id}  {param_name}={value}"
                        f"  rouge_l={rouge_str}{delta_str}  [{status}]"
                    )

                    tsv_rows.append(
                        _tsv_row(
                            exp_id,
                            status,
                            rouge,
                            embed,
                            delta_pct,
                            group_name,
                            param_name,
                            value,
                            notes,
                        )
                    )

                finally:
                    tmp_path.unlink(missing_ok=True)
                    # Clean up run dir for rejected/error experiments to save disk
                    if status in ("rejected", "error"):
                        run_dir = REPO_ROOT / "data/eval/runs" / exp_id
                        if run_dir.is_dir():
                            import shutil

                            shutil.rmtree(run_dir, ignore_errors=True)

            if consecutive_fails >= max_fails:
                break

        if consecutive_fails >= max_fails:
            print(f"\nStopping early: {consecutive_fails} consecutive fails reached.")
            break

    # ── Write outputs ───────────────────────────────────────────────────────
    tsv_path.write_text("\n".join(tsv_rows) + "\n", encoding="utf-8")
    print(f"\nResults TSV: {tsv_path}")

    # Save best config with canonical id
    best_cfg_out = copy.deepcopy(best_cfg)
    best_cfg_out["id"] = f"{baseline_id}_autoresearch_v1"
    _write_temp_config(best_cfg_out, best_config_path)
    print(f"Best config:  {best_config_path}")

    total_gain = (best_rouge - baseline_rouge) / baseline_rouge * 100
    print("\nSummary:")
    print(f"  Baseline rouge_l: {baseline_rouge:.4f}")
    print(f"  Best rouge_l:     {best_rouge:.4f}  ({total_gain:+.2f}%)")
    print(f"  Best embed:       {best_embed or 0:.4f}")
    print(f"  Experiments run:  {exp_counter}")
    accepted_count = sum(1 for r in tsv_rows if "\taccepted\t" in r)
    print(f"  Accepted:         {accepted_count}")

    if best_rouge > baseline_rouge:
        _print_promotion_snippet(
            model_key=model_key,
            best_cfg=best_cfg_out,
            baseline_id=baseline_id,
            rouge_l=best_rouge,
            embed=best_embed or 0.0,
            reference_id=reference_id,
        )
    else:
        print("\nNo improvement found — baseline params remain optimal.")

    return 0


# ─── CLI ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Model key from param_space.yaml (e.g. bart_led, pegasus_led)",
        metavar="MODEL",
    )
    parser.add_argument(
        "--max-fails",
        type=int,
        default=3,
        help="Consecutive experiments without ≥min-gain before stopping (default: 3)",
    )
    parser.add_argument(
        "--min-gain",
        type=float,
        default=0.01,
        help="Minimum relative ROUGE-L gain to accept (default: 0.01 = 1%%)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help=f"Directory for TSV output (default: {RESULTS_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip actual experiment runs (for testing the loop logic)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if not PARAM_SPACE_PATH.is_file():
        logger.error("param_space.yaml not found: %s", PARAM_SPACE_PATH)
        return 1

    space_all = yaml.safe_load(PARAM_SPACE_PATH.read_text(encoding="utf-8"))
    valid_models = list(space_all.keys())

    if args.model not in valid_models:
        logger.error("Unknown model key %r. Valid: %s", args.model, valid_models)
        return 1

    return run_sweep(
        model_key=args.model,
        space=space_all[args.model],
        max_fails=args.max_fails,
        min_gain=args.min_gain,
        results_dir=args.results_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
