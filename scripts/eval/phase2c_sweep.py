#!/usr/bin/env python3
"""#1016 Phase 2c — GI + KG prediction sweep across the 6-candidate cohort.

For each candidate model, runs the GI experiment then the KG experiment
against the autoresearch vLLM endpoint, using the per-candidate dedicated
configs at ``data/eval/configs/{gi,kg}_autoresearch_prompt_*_dev_v1.yaml``.

For vLLM-backed candidates, the operator must first swap the homelab
``autoresearch`` vLLM compose to the candidate model (see
``/tmp/swap_vllm_model.py`` for the DGX-side helper). This runner does NOT
ssh into the DGX — it only orchestrates the laptop-side experiment runs and
checks ``/health`` to confirm the right model is live before starting.

Usage:

    # Single-candidate (most common — operator-driven loop):
    PYTHONPATH=. .venv/bin/python scripts/eval/phase2c_sweep.py \\
        --candidate qwen3_30b \\
        --vllm-base-url "$VLLM_API_BASE"

    # All-cloud candidates only (Gemini-Flash-Lite, no swap needed):
    PYTHONPATH=. .venv/bin/python scripts/eval/phase2c_sweep.py \\
        --candidate gemini_flash_lite

The script writes predictions.jsonl per task per candidate under the
standard ``data/eval/runs/<config-id>/`` layout (same as Phase 2a).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


# Candidate registry — maps a short candidate key to:
#   - gi_config: GI experiment YAML
#   - kg_config: KG experiment YAML
#   - vllm: True if vLLM-backed (requires base URL + health check)
CANDIDATES: dict[str, dict] = {
    "qwen3_30b": {
        "gi": (
            "data/eval/configs/gi_autoresearch_prompt_vllm_qwen3_30b_a3b_instruct_2507_dev_v1.yaml"
        ),
        "kg": (
            "data/eval/configs/kg_autoresearch_prompt_vllm_qwen3_30b_a3b_instruct_2507_dev_v1.yaml"
        ),
        "vllm": True,
    },
    "qwen3_5_35b": {
        "gi": "data/eval/configs/gi_autoresearch_prompt_vllm_qwen3_5_35b_a3b_dev_v1.yaml",
        "kg": "data/eval/configs/kg_autoresearch_prompt_vllm_qwen3_5_35b_a3b_dev_v1.yaml",
        "vllm": True,
    },
    "r1distill_32b": {
        "gi": "data/eval/configs/gi_autoresearch_prompt_vllm_r1distill_32b_dev_v1.yaml",
        "kg": "data/eval/configs/kg_autoresearch_prompt_vllm_r1distill_32b_dev_v1.yaml",
        "vllm": True,
    },
    "magistral": {
        "gi": "data/eval/configs/gi_autoresearch_prompt_vllm_magistral_small_2509_dev_v1.yaml",
        "kg": "data/eval/configs/kg_autoresearch_prompt_vllm_magistral_small_2509_dev_v1.yaml",
        "vllm": True,
    },
    "mistral_small_3_2": {
        "gi": "data/eval/configs/gi_autoresearch_prompt_vllm_mistral_small_3_2_24b_dev_v1.yaml",
        "kg": "data/eval/configs/kg_autoresearch_prompt_vllm_mistral_small_3_2_24b_dev_v1.yaml",
        "vllm": True,
    },
    "gemini_flash_lite": {
        "gi": "data/eval/configs/gi_autoresearch_prompt_gemini25_gemini25_flash_lite_dev_v1.yaml",
        "kg": "data/eval/configs/kg_autoresearch_prompt_gemini25_gemini25_flash_lite_dev_v1.yaml",
        "vllm": False,
    },
}


def _check_vllm_health(base_url: str, *, timeout: int = 30) -> bool:
    """Probe the vLLM /health and /v1/models endpoints. Return True if both OK."""
    base = base_url.rstrip("/").removesuffix("/v1")
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=timeout) as r:
            if r.status != 200:
                return False
        with urllib.request.urlopen(f"{base}/v1/models", timeout=timeout) as r:
            if r.status != 200:
                return False
            data = r.read().decode("utf-8", errors="replace")
            print(f"  /v1/models -> {data[:200]}")
        return True
    except Exception as e:
        print(f"  health check failed: {e}", file=sys.stderr)
        return False


def _run_experiment(
    config_path: Path,
    *,
    vllm_base_url: str | None,
    force: bool,
) -> int:
    """Invoke run_experiment.py with the given config + optional vLLM override."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval" / "experiment" / "run_experiment.py"),
        str(config_path),
        "--dry-run",  # predictions only — scoring done later via rescore_against_silver.py
    ]
    if vllm_base_url:
        cmd.extend(["--vllm-base-url", vllm_base_url])
    if force:
        cmd.append("--force")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    print(f"  running: {' '.join(cmd[2:])}")  # skip python+script
    proc = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT))
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        required=True,
        choices=sorted(CANDIDATES.keys()),
        help="Candidate short key (the operator picks which one is currently live on vLLM).",
    )
    parser.add_argument(
        "--vllm-base-url",
        default=os.environ.get("VLLM_API_BASE"),
        help="Override the vLLM base URL (tailnet FQDN). Defaults to VLLM_API_BASE env. "
        "Ignored for cloud candidates.",
    )
    parser.add_argument(
        "--skip-gi",
        action="store_true",
        help="Run only the KG experiment (skip GI). Useful when GI is already done.",
    )
    parser.add_argument(
        "--skip-kg",
        action="store_true",
        help="Run only the GI experiment (skip KG).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Pass --force to run_experiment.py (wipe existing predictions).",
    )
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="Skip the vLLM /health probe before starting. Use with caution.",
    )
    args = parser.parse_args()

    spec = CANDIDATES[args.candidate]
    print(f"=== Phase 2c sweep — candidate {args.candidate} ===")
    print(f"  gi config: {spec['gi']}")
    print(f"  kg config: {spec['kg']}")

    if spec["vllm"]:
        if not args.vllm_base_url:
            print(
                "ERROR: vLLM candidate requires --vllm-base-url or VLLM_API_BASE env",
                file=sys.stderr,
            )
            return 2
        print(f"  vllm base: {args.vllm_base_url}")
        if not args.skip_health_check:
            print("  probing vLLM /health and /v1/models...")
            if not _check_vllm_health(args.vllm_base_url):
                print(
                    "ERROR: vLLM health probe failed. Confirm the compose is up + "
                    "the right model is pinned.",
                    file=sys.stderr,
                )
                return 3

    failures = []
    if not args.skip_gi:
        print(f"\n--- {args.candidate}: GI experiment ---")
        rc = _run_experiment(
            REPO_ROOT / spec["gi"],
            vllm_base_url=args.vllm_base_url if spec["vllm"] else None,
            force=args.force,
        )
        if rc != 0:
            failures.append(("gi", rc))

    if not args.skip_kg:
        print(f"\n--- {args.candidate}: KG experiment ---")
        rc = _run_experiment(
            REPO_ROOT / spec["kg"],
            vllm_base_url=args.vllm_base_url if spec["vllm"] else None,
            force=args.force,
        )
        if rc != 0:
            failures.append(("kg", rc))

    if failures:
        print(f"\nFAIL — {args.candidate}: {failures}", file=sys.stderr)
        return 1
    print(f"\nOK — {args.candidate}: GI + KG predictions written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
