#!/usr/bin/env python3
"""Autoresearch Track A: run summarization eval + optional dual LLM judges, emit one scalar.

Runs ``scripts/eval/experiment/run_experiment.py`` from the repo root (reuse metrics stack).
``--dry-run`` skips judge API calls and re-scores an existing run (``--score-only``).

Environment (see ``config/examples/.env.example``):
  Loads project-root ``.env`` then optional ``.env.autoresearch`` (override) via python-dotenv.

  AUTORESEARCH_EXPERIMENT_OPENAI_API_KEY — OpenAI key for summarization in subprocess
  AUTORESEARCH_JUDGE_OPENAI_API_KEY / AUTORESEARCH_JUDGE_ANTHROPIC_API_KEY — judges
  AUTORESEARCH_ALLOW_PRODUCTION_KEYS=1 — fall back to OPENAI_API_KEY / ANTHROPIC_API_KEY
  AUTORESEARCH_EVAL_N — max episodes (default 5)
  AUTORESEARCH_SCORE_ROUGE_WEIGHT — weight for ROUGE in [0,1] (default 0.4)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Repo root: autoresearch/initial_prompt_tuning/prompt_tuning/eval/score.py -> parents[4].
# (Was parents[3] before the prompt_tuning reorg; the file moved one level
# deeper and the offset was never updated — every REPO_ROOT-relative path
# silently resolved under autoresearch/ instead of repo root.)
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from podcast_scraper.evaluation.autoresearch_track_a import (  # noqa: E402
    AutoresearchConfigError,
    combine_track_a_scalar,
    eval_n_from_env,
    extract_mean_rouge_l_f1,
    load_judge_config,
    load_local_dotenv_files,
    mean_judge_scores,
    merge_max_episodes_into_config_yaml,
    provider_runtime_key_env,
    resolve_experiment_openai_key,
    resolve_experiment_provider_key,
    resolve_judge_anthropic_key,
    resolve_judge_openai_key,
    rouge_weight_from_env,
)
from podcast_scraper.evaluation.experiment_config import load_experiment_config  # noqa: E402
from podcast_scraper.evaluation.scorer import load_predictions  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_BASE_CONFIG = (
    REPO_ROOT
    / "data/eval/configs/summarization_bullets/autoresearch_prompt_openai_smoke_bullets_v1.yaml"
)
# ROUGE target for bullet JSON (must exist under data/eval/references/silver/...).
# Create by running experiment_openai_gpt4o_smoke_bullets_v1
# (CONFIG=data/eval/issue-477/experiment_openai_gpt4o_smoke_bullets_v1.yaml), then promote; see
# data/eval/issue-477/README.md and data/eval/configs/README.md.
# Interim (mismatched ROUGE): REFERENCE=silver_gpt4o_smoke_v1
DEFAULT_REFERENCE_ID = "silver_gpt4o_smoke_bullets_v1"
EVAL_DIR = Path(__file__).resolve().parent


def _run_subprocess(
    *,
    merged_config: Path,
    dry_run: bool,
    reference_id: str,
    backend_type: str = "openai",
) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/eval/experiment/run_experiment.py"),
        str(merged_config),
        "--reference",
        reference_id,
        "--log-level",
        "INFO",
    ]
    if dry_run:
        cmd.append("--score-only")
    else:
        cmd.append("--force")

    env = os.environ.copy()
    # run_experiment.py imports ``scripts.eval.data.materialize_baseline``
    # — a package-style import that needs the repo root on PYTHONPATH.
    # Subprocess inherits os.environ but cwd is set to REPO_ROOT below;
    # cwd alone isn't enough for ``from scripts.* import ...`` to resolve.
    env["PYTHONPATH"] = (
        f"{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
        if env.get("PYTHONPATH")
        else str(REPO_ROOT)
    )
    if backend_type == "ollama":
        # Ollama backend reads OLLAMA_API_BASE (inherited from os.environ above).
        # No API key injection needed; judges are called from the parent process
        # (not this subprocess) per their own judge_config_*.yaml provider.
        pass
    else:
        # Cloud-API backends — inject the experiment-specific key, overriding
        # any production key loaded from .env into os.environ. Also set
        # OPENAI_API_KEY for ancillary OpenAI calls inside run_experiment
        # (e.g. silver materialization paths), best-effort.
        runtime_env = provider_runtime_key_env(backend_type)
        env[runtime_env] = resolve_experiment_provider_key(backend_type)
        try:
            env["OPENAI_API_KEY"] = resolve_experiment_openai_key()
        except AutoresearchConfigError:
            # Primary provider key is set above; OPENAI fallback is optional.
            pass

    logger.info("Running: %s", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"run_experiment exited with code {proc.returncode}")


def main() -> int:
    load_local_dotenv_files(REPO_ROOT)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_BASE_CONFIG,
        help="Base experiment YAML (max_episodes applied from AUTORESEARCH_EVAL_N)",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=DEFAULT_REFERENCE_ID,
        help="Silver / reference id for ROUGE (passed to run_experiment)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip judges; use --score-only (requires existing predictions for run id)",
    )
    parser.add_argument(
        "--judge-config",
        type=Path,
        default=None,
        help=(
            "Path to judge YAML (judge_a/judge_b provider+model). Defaults to "
            "judge_config.yaml next to this script. Use judge_config_ollama.yaml "
            "for the DGX Ollama-only smoke (provider: ollama for both judges)."
        ),
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

    base_config = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    if not base_config.is_file():
        logger.error("Config not found: %s", base_config)
        return 1

    try:
        n = eval_n_from_env(5)
        rw = rouge_weight_from_env()
    except AutoresearchConfigError as e:
        logger.error("%s", e)
        return 1

    rubric_path = EVAL_DIR / "rubric.md"
    if args.judge_config is not None:
        judge_path = (
            args.judge_config if args.judge_config.is_absolute() else REPO_ROOT / args.judge_config
        )
    else:
        judge_path = EVAL_DIR / "judge_config.yaml"
    if not rubric_path.is_file() or not judge_path.is_file():
        logger.error("Missing rubric.md (%s) or judge config (%s)", rubric_path, judge_path)
        return 1

    rubric_text = rubric_path.read_text(encoding="utf-8")
    try:
        judge_cfg = load_judge_config(judge_path)
    except Exception as e:
        logger.error("Invalid judge config %s: %s", judge_path, e)
        return 1
    # Surface the chosen judge models — useful in CI job summaries to see at
    # a glance which models judged this run, especially when juggling several
    # Ollama variants for the DGX smoke.
    ja = judge_cfg.get("judge_a") or {}
    jb = judge_cfg.get("judge_b") or {}
    logger.info(
        "Judge A: provider=%s model=%s | Judge B: provider=%s model=%s",
        ja.get("provider", "?"),
        ja.get("model", "?"),
        jb.get("provider", "?"),
        jb.get("model", "?"),
    )

    with tempfile.NamedTemporaryFile(
        suffix=".yaml", delete=False, dir=str(EVAL_DIR), prefix="autoresearch_cfg_"
    ) as tmp:
        merged_path = Path(tmp.name)

    try:
        merge_max_episodes_into_config_yaml(base_config, merged_path, n)
        cfg = load_experiment_config(merged_path)
        run_id = cfg.id
        results_dir = REPO_ROOT / "data/eval/runs" / run_id

        if args.dry_run and not (results_dir / "predictions.jsonl").is_file():
            logger.error(
                "Dry-run requires existing predictions at %s — run once without --dry-run",
                results_dir / "predictions.jsonl",
            )
            return 1

        _run_subprocess(
            merged_config=merged_path,
            dry_run=args.dry_run,
            reference_id=args.reference,
            backend_type=cfg.backend.type,
        )

        metrics_path = results_dir / "metrics.json"
        if not metrics_path.is_file():
            logger.error("metrics.json missing: %s", metrics_path)
            return 1

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rouge = extract_mean_rouge_l_f1(metrics)
        if rouge is None:
            logger.error("Could not read rougeL_f1 from metrics vs_reference")
            return 1

        judge_mean = None
        contested = False
        if not args.dry_run:
            try:
                oai_j = resolve_judge_openai_key()
                ant_j = resolve_judge_anthropic_key()
            except AutoresearchConfigError as e:
                logger.error("%s", e)
                return 1

            preds = load_predictions(results_dir / "predictions.jsonl")
            dataset_id = str(metrics.get("dataset_id") or cfg.data.dataset_id)
            try:
                judge_mean, contested, _outs = mean_judge_scores(
                    predictions=preds,
                    rubric=rubric_text,
                    judge_cfg=judge_cfg,
                    dataset_id=dataset_id,
                    eval_root=REPO_ROOT / "data/eval",
                    openai_key=oai_j,
                    anthropic_key=ant_j,
                )
            except Exception as e:
                logger.error("Judge scoring failed: %s", e, exc_info=True)
                return 1
            if contested:
                logger.warning(
                    "Judges diverged by > %.2f on at least one episode — using ROUGE-only blend",
                    0.15,
                )

        final = combine_track_a_scalar(
            rouge_l_f1=rouge,
            judge_mean=judge_mean,
            contested=contested,
            rouge_weight=rw,
        )
        logger.info(
            "rougeL_f1=%.4f judge_mean=%s contested=%s rouge_weight=%.2f final=%.6f",
            rouge,
            f"{judge_mean:.4f}" if judge_mean is not None else "n/a",
            contested,
            rw,
            final,
        )
        print(f"{final:.6f}")
        return 0
    finally:
        merged_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
