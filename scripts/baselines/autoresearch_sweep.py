#!/usr/bin/env python3
"""Weekly autoresearch sweep driver — runs the cohort + writes the ledger.

Reads ``data/autoresearch_baselines/cohort.yaml`` for the candidate list,
runs ``make autoresearch-score`` per candidate (overriding ``backend.model``
in a temp config), collects the per-candidate breakdown JSONs, and emits
one weekly ledger ``data/autoresearch_baselines/autoresearch-YYYY-WNN.json``.

The ledger is what ``check_autoresearch_drift.py`` consumes to detect
week-over-week regressions per candidate.

Each candidate's per-model tuned paragraph templates
(``src/podcast_scraper/prompts/ollama/<model>/summarization/{system,long}_v1.j2``)
are wired into the materialized config — the sweep measures every candidate
on the prompt we'd actually ship it with, not on someone else's shared prompt.
A candidate that lacks tuned prompts fails fast (``status: missing_prompts``)
rather than silently degrading on a foreign prompt.

CLI:
  --cohort PATH            Cohort YAML (default: data/autoresearch_baselines/cohort.yaml)
  --base-config PATH       Base experiment YAML (default: Ollama smoke paragraph sweep v1)
  --judge-config PATH      Judge config (default: judge_config_ollama.yaml)
  --reference ID           Silver reference id (default: silver_sonnet46_smoke_v2)
  --output PATH            Weekly ledger output (default: autoresearch-<week>.json)
  --limit N                Run only the first N candidates (overrides cohort.default_limit)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _resolve_per_model_prompts(candidate_model: str) -> tuple[str, str] | None:
    """Return (system_prompt_id, user_prompt_id) for the candidate's tuned
    paragraph templates if both exist on disk, else None.

    Convention: ``src/podcast_scraper/prompts/ollama/<model_safe>/summarization/
    {system_v1,long_v1}.j2`` — same dir naming the production summarization
    factory uses. ``model_safe`` replaces ``:`` with ``_`` but preserves dots
    so it matches the existing dir layout (e.g. ``qwen3.5:9b`` → ``qwen3.5_9b``).

    See [[project_autoresearch]] for the harness contract.
    """
    safe = candidate_model.replace(":", "_").replace("/", "_")
    prompt_dir = REPO_ROOT / "src/podcast_scraper/prompts/ollama" / safe / "summarization"
    system_file = prompt_dir / "system_v1.j2"
    user_file = prompt_dir / "long_v1.j2"
    if system_file.is_file() and user_file.is_file():
        return (
            f"ollama/{safe}/summarization/system_v1",
            f"ollama/{safe}/summarization/long_v1",
        )
    return None


def _materialize_candidate_config(
    base_config_path: Path, candidate_model: str, out_dir: Path
) -> tuple[Path, str] | tuple[None, str]:
    """Read base config, override backend.model + prompts + id, write to out_dir.

    Returns ``(config_path, "tuned")`` when the candidate's per-model paragraph
    prompts exist (the normal sweep case — see _resolve_per_model_prompts).
    Returns ``(None, "missing_prompts")`` when they don't, so the caller can
    record a clean failure row instead of silently scoring the candidate on
    someone else's prompt (the W27 problem).
    """
    resolved = _resolve_per_model_prompts(candidate_model)
    if resolved is None:
        return None, "missing_prompts"
    system_id, user_id = resolved

    base = load_yaml(base_config_path)
    base["backend"]["model"] = candidate_model
    base["prompts"] = {"system": system_id, "user": user_id}
    safe = candidate_model.replace(":", "_").replace("/", "_").replace(".", "_")
    base["id"] = f"autoresearch_prompt_ollama_{safe}_smoke_paragraph_sweep"
    out_path = out_dir / f"config_{safe}.yaml"
    out_path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    return out_path, "tuned"


def _run_candidate(
    *,
    candidate: dict[str, Any],
    base_config_path: Path,
    judge_config_path: Path,
    reference_id: str,
    tmp_dir: Path,
) -> dict[str, Any] | None:
    """Run one candidate; return its breakdown dict (or None on failure)."""
    model = candidate["model"]
    family = candidate["family"]
    logger.info("=" * 70)
    logger.info("Candidate: %s (%s)", model, family)
    logger.info("=" * 70)

    cfg_path, prompts_source = _materialize_candidate_config(base_config_path, model, tmp_dir)
    if cfg_path is None:
        # Per-model tuned prompts missing — fail fast so the operator gets a
        # clean signal instead of a quietly-degraded score. New cohort members
        # should ship their tuned templates first; see [[project_autoresearch]].
        logger.error(
            "Candidate %s: no tuned prompts under "
            "src/podcast_scraper/prompts/ollama/%s/summarization/; "
            "skipping (status=missing_prompts)",
            model,
            model.replace(":", "_").replace("/", "_"),
        )
        return {
            "model": model,
            "family": family,
            "status": "missing_prompts",
            "prompts_source": prompts_source,
            "wall_clock_s": 0.0,
        }
    output_json = tmp_dir / f"output_{model.replace(':', '_')}.json"

    started = time.time()
    # Invoke through the Makefile target — same surface anyone would use
    # manually (``make autoresearch-score CONFIG=... OUTPUT_JSON=...``)
    # so debugging the sweep is "just" running the same make invocation.
    # Use absolute paths: temp dir is under /tmp (outside REPO_ROOT), so
    # .relative_to(REPO_ROOT) would raise. score.py already accepts both
    # absolute and relative paths for --config / --judge-config /
    # --output-json.
    proc = subprocess.run(
        [
            "make",
            "autoresearch-score",
            f"CONFIG={cfg_path}",
            f"JUDGE_CONFIG={judge_config_path}",
            f"REFERENCE={reference_id}",
            f"OUTPUT_JSON={output_json}",
        ],
        cwd=str(REPO_ROOT),
        check=False,
    )
    elapsed = time.time() - started

    if proc.returncode != 0:
        logger.error("Candidate %s failed (exit %d)", model, proc.returncode)
        return {
            "model": model,
            "family": family,
            "status": "failed",
            "exit_code": proc.returncode,
            "prompts_source": prompts_source,
            "wall_clock_s": round(elapsed, 1),
        }

    if not output_json.is_file():
        logger.error("Candidate %s exited 0 but no output JSON at %s", model, output_json)
        return {
            "model": model,
            "family": family,
            "status": "no_output",
            "prompts_source": prompts_source,
            "wall_clock_s": round(elapsed, 1),
        }

    breakdown = json.loads(output_json.read_text(encoding="utf-8"))
    breakdown["status"] = "ok"
    breakdown["wall_clock_s"] = round(elapsed, 1)
    breakdown["family"] = family
    breakdown["prompts_source"] = prompts_source
    return breakdown


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    parser = argparse.ArgumentParser(description="Weekly autoresearch sweep driver.")
    parser.add_argument(
        "--cohort",
        type=Path,
        default=REPO_ROOT / "data/autoresearch_baselines/cohort.yaml",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=(
            REPO_ROOT
            / "data/eval/configs/summarization"
            / "autoresearch_prompt_ollama_smoke_paragraph_sweep_v1.yaml"
        ),
    )
    parser.add_argument(
        "--judge-config",
        type=Path,
        default=REPO_ROOT
        / "autoresearch/initial_prompt_tuning/prompt_tuning/eval/judge_config_ollama.yaml",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="silver_sonnet46_smoke_v2",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: data/autoresearch_baselines/autoresearch-YYYY-WNN.json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only first N candidates (overrides cohort.default_limit)",
    )
    args = parser.parse_args()

    cohort_doc = load_yaml(args.cohort)
    candidates = cohort_doc.get("candidates") or []
    if not candidates:
        logger.error("No candidates in %s", args.cohort)
        return 1

    limit = args.limit
    if limit is None:
        limit = cohort_doc.get("default_limit")
    if limit is not None and limit > 0:
        candidates = candidates[: int(limit)]
        logger.info("Running first %d candidate(s) (limit=%d)", len(candidates), limit)
    else:
        logger.info("Running full cohort (%d candidate(s))", len(candidates))

    judge_families = cohort_doc.get("judge_families") or []

    week_id = dt.datetime.utcnow().strftime("%G-W%V")
    out_path = (
        args.output or REPO_ROOT / "data/autoresearch_baselines" / f"autoresearch-{week_id}.json"
    )

    judge_cfg = load_yaml(args.judge_config)
    base_cfg = load_yaml(args.base_config)

    sweep_results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="autoresearch_sweep_") as tmp:
        tmp_dir = Path(tmp)
        for candidate in candidates:
            result = _run_candidate(
                candidate=candidate,
                base_config_path=args.base_config,
                judge_config_path=args.judge_config,
                reference_id=args.reference,
                tmp_dir=tmp_dir,
            )
            if result is not None:
                # Attach the same-family bias flag for the report.
                result["same_family_judge"] = candidate["family"] in judge_families
                sweep_results.append(result)

    ledger = {
        "schema_version": 1,
        "week_id": week_id,
        "captured_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "judges": {
            "judge_a": judge_cfg.get("judge_a") or {},
            "judge_b": judge_cfg.get("judge_b") or {},
        },
        "silver": args.reference,
        "dataset": (base_cfg.get("data") or {}).get("dataset_id"),
        "cohort": sweep_results,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    logger.info("Wrote weekly ledger: %s", out_path)

    ok_count = sum(1 for r in sweep_results if r.get("status") == "ok")
    logger.info("Cohort summary: %d/%d candidates OK", ok_count, len(sweep_results))
    # Exit 0 even on per-candidate failures — the ledger captures them as
    # ``status != ok`` rows and the drift check / issue management handles
    # surfacing them. Sweep exits non-zero only on driver-level errors
    # (no cohort, missing configs, etc.).
    return (
        0
        if os.environ.get("AUTORESEARCH_SWEEP_STRICT") != "1"
        else (0 if ok_count == len(sweep_results) else 2)
    )


if __name__ == "__main__":
    sys.exit(main())
