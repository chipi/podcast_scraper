#!/usr/bin/env python3
"""Retro re-fingerprint chunk-7 v2 scoreboard runs (extraordinary measure).

RFC-097 fingerprint gap-closure landed today (FINGERPRINT_GAPS_ANALYSIS_2026-06-22.md
+ commits b1ef7046..a31b4e9a). Existing ``fingerprint.json`` files written
BEFORE today's commits have:

- ``generation_params: {}`` for every GI/KG run (headline §1 bug)
- ``model_name: "autoresearch"`` alias hiding the backing HF id (§2)
- no ``task_pipeline`` / ``inference_args`` / ``inference_image`` (§3)
- no ``runtime.inference_target`` (§4)
- no ``podcast_scraper_config`` (§5)
- no ``dataset_content_hash`` (§6)
- weak ``fingerprint_hash`` over a small subset (§7)

The chunk-7 v2 scoreboard published in
``docs/guides/eval-reports/EVAL_RFC097_V2_BASELINE_2026_06_20.md`` is not
auditable today because the per-row fingerprints don't distinguish materially
different configs. Operator authorised in-place replacement of these specific
fingerprints as an extraordinary measure (per
``feedback_never_mutate_historical_artifacts`` memory rule).

This tool:

1. Reads ``data/eval/runs/<run_id>/fingerprint.json`` + the eval YAML at
   ``data/eval/configs/<task>_<run_id_stem>.yaml`` (or a sibling).
2. Reconstructs every field the new shape captures, marking the ones it
   can't recover from disk with explicit ``null`` + a ``_retro_unknown``
   entry in the ``_retro_audit`` section.
3. Computes ``dataset_content_hash`` from today's
   ``data/eval/materialized/<dataset_id>/`` and timestamps that hash with
   ``dataset_content_hash_audited_at`` so a reader can distinguish "the
   dataset content at the time of the original run" (lost) from "the
   dataset content right now" (what we have).
4. Recomputes ``fingerprint_hash`` v2 over the full retro fingerprint.
5. Overwrites the existing ``fingerprint.json`` in place. Original is
   archived as ``fingerprint.v1.original.json`` before replacement so the
   pre-mutation state is recoverable if needed.

Usage::

    .venv/bin/python scripts/eval/fingerprint/refingerprint_from_run.py \\
        --run-id autoresearch_prompt_vllm_mistral_small_3_2_24b_dev_knowledge_graph_v1

    .venv/bin/python scripts/eval/fingerprint/refingerprint_from_run.py --all-chunk7

The ``--all-chunk7`` flag runs against the 6 headline candidates first
(Mistral-3.2, Qwen3.5, Ministral, Gemma, Magistral, Moonlight × KG + GI = 12
runs) then the rest (Qwen3-30B, Gemini, DeepSeek × KG + GI = 6 more = 18 total).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS_ROOT = REPO_ROOT / "data" / "eval" / "runs"
CONFIGS_ROOT = REPO_ROOT / "data" / "eval" / "configs"
MATERIALIZED_ROOT = REPO_ROOT / "data" / "eval" / "materialized"

# Chunk-7 v2 scoreboard candidates, ordered: 6 headlines first, then the rest.
# Each name corresponds to a {kg,gi}_<name>_dev_v1 config + a
# {knowledge_graph,grounded_insights}_v1 run.
CHUNK7_HEADLINES = [
    "mistral_small_3_2_24b",
    "qwen3_5_35b_a3b",
    "ministral_3_14b",
    "gemma_4_26b_a4b",
    "magistral_small_2509",
    "moonlight_16b_a3b",
]
CHUNK7_REMAINDER = [
    "qwen3_30b_a3b_instruct_2507",
    "deepseek_v2_lite_chat",
]
# Gemini is cloud (different config naming) — handled separately below.
GEMINI_CANDIDATE = "gemini25_flash_lite"

# Chunk-7 recovery table — for the 2026-06-21 sweep + workaround retries.
# Sourced from autoresearch/PER_MODEL_OPTIMAL_PARAMS.md (Phase 2c flags) +
# docs/guides/eval-reports/EVAL_RFC097_CHUNK7_VLLM_WORKAROUNDS_2026_06_21.md
# (per-model workaround flags applied during the retry sweep). Used when the
# run.log + chunk-7 sweep logs don't carry the literal docker run command
# (which is the common case — the swap scripts ssh'd to DGX and only the
# server-side stdout was tailed back to the log).
#
# Each value is the canonical CHUNK-7 backing_model_id + the args string that
# WAS effectively applied. inference_image was nvcr.io/nvidia/vllm:26.05-py3
# for the entire chunk-7 sweep.
CHUNK7_RECOVERY: Dict[str, Dict[str, str]] = {
    "qwen3_30b_a3b_instruct_2507": {
        "backing_model_id": "NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4",
        "args": (
            "vllm serve NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4 "
            "--host 0.0.0.0 --port 8003 --api-key buddy-is-the-king "
            "--gpu-memory-utilization 0.60 --max-model-len 32768 "
            "--served-model-name autoresearch --dtype bfloat16"
        ),
    },
    "qwen3_5_35b_a3b": {
        "backing_model_id": "Qwen/Qwen3.5-35B-A3B",
        "args": (
            "vllm serve Qwen/Qwen3.5-35B-A3B "
            "--host 0.0.0.0 --port 8003 --api-key buddy-is-the-king "
            "--gpu-memory-utilization 0.65 --max-model-len 16384 "
            "--served-model-name autoresearch --dtype bfloat16"
        ),
    },
    "gemma_4_26b_a4b": {
        "backing_model_id": "google/gemma-4-26B-A4B-it",
        "args": (
            "vllm serve google/gemma-4-26B-A4B-it "
            "--host 0.0.0.0 --port 8003 --api-key buddy-is-the-king "
            "--gpu-memory-utilization 0.65 --max-model-len 32768 "
            "--max-num-batched-tokens 4096 --max-num-seqs 4 --enforce-eager "
            "--served-model-name autoresearch --dtype bfloat16"
        ),
    },
    "mistral_small_3_2_24b": {
        "backing_model_id": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "args": (
            "vllm serve mistralai/Mistral-Small-3.2-24B-Instruct-2506 "
            "--host 0.0.0.0 --port 8003 --api-key buddy-is-the-king "
            "--gpu-memory-utilization 0.55 --max-model-len 32768 "
            "--max-num-batched-tokens 4096 --max-num-seqs 4 --enforce-eager "
            "--served-model-name autoresearch --dtype bfloat16 "
            "--tokenizer-mode mistral --config-format mistral "
            "--load-format mistral --language-model-only"
        ),
    },
    "magistral_small_2509": {
        "backing_model_id": "mistralai/Magistral-Small-2509",
        "args": (
            "vllm serve mistralai/Magistral-Small-2509 "
            "--host 0.0.0.0 --port 8003 --api-key buddy-is-the-king "
            "--gpu-memory-utilization 0.55 --max-model-len 32768 "
            "--max-num-batched-tokens 4096 --max-num-seqs 4 --enforce-eager "
            "--served-model-name autoresearch --dtype bfloat16 "
            "--reasoning-parser mistral --tokenizer-mode mistral "
            "--config-format mistral --tool-call-parser mistral "
            "--language-model-only"
        ),
    },
    "ministral_3_14b": {
        "backing_model_id": "mistralai/Ministral-3-14B-Instruct-2512",
        "args": (
            "vllm serve mistralai/Ministral-3-14B-Instruct-2512 "
            "--host 0.0.0.0 --port 8003 --api-key buddy-is-the-king "
            "--gpu-memory-utilization 0.65 --max-model-len 32768 "
            "--max-num-batched-tokens 4096 --max-num-seqs 4 --enforce-eager "
            "--served-model-name autoresearch --dtype bfloat16 "
            "--tokenizer-mode mistral --config-format mistral "
            "--load-format mistral --language-model-only"
        ),
    },
    "moonlight_16b_a3b": {
        "backing_model_id": "moonshotai/Moonlight-16B-A3B-Instruct",
        "args": (
            "vllm serve moonshotai/Moonlight-16B-A3B-Instruct "
            "--host 0.0.0.0 --port 8003 --api-key buddy-is-the-king "
            "--gpu-memory-utilization 0.55 --max-model-len 8192 "
            "--max-num-batched-tokens 4096 --max-num-seqs 4 --enforce-eager "
            "--served-model-name autoresearch --dtype bfloat16"
        ),
    },
    "deepseek_v2_lite_chat": {
        "backing_model_id": "deepseek-ai/DeepSeek-V2-Lite-Chat",
        "args": (
            "vllm serve deepseek-ai/DeepSeek-V2-Lite-Chat "
            "--host 0.0.0.0 --port 8003 --api-key buddy-is-the-king "
            "--gpu-memory-utilization 0.60 --max-model-len 32768 "
            "--served-model-name autoresearch --dtype bfloat16"
        ),
    },
}
CHUNK7_INFERENCE_IMAGE = "nvcr.io/nvidia/vllm:26.05-py3"
# All chunk-7 runs actually went to DGX even when YAML base_url said
# localhost:8003 (overridden via --vllm-base-url CLI). For these specific
# candidates, override the inference_target.
CHUNK7_INFERENCE_TARGET = "dgx-vllm"
CHUNK7_BASE_URL_ACTUAL = "http://dgx-llm-1.tail6d0ed4.ts.net:8003/v1"


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _find_eval_yaml(short: str, task_kind: str) -> Optional[Path]:
    """task_kind is 'kg' or 'gi'. Looks for the dev_v1 config."""
    # vLLM candidate naming
    p = CONFIGS_ROOT / f"{task_kind}_autoresearch_prompt_vllm_{short}_dev_v1.yaml"
    if p.exists():
        return p
    # Gemini naming (double-prefix variant: gemini25_<short>)
    p = CONFIGS_ROOT / f"{task_kind}_autoresearch_prompt_gemini25_{short}_dev_v1.yaml"
    if p.exists():
        return p
    # Plain prefix (older / other cloud providers)
    p = CONFIGS_ROOT / f"{task_kind}_autoresearch_prompt_{short}_dev_v1.yaml"
    if p.exists():
        return p
    return None


def _parse_eval_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _extract_generation_params(yaml_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Mirror the post-gap-closure logic in materialize_baseline.py."""
    params = yaml_cfg.get("params") or {}
    out: Dict[str, Any] = {
        "temperature": params.get("temperature"),
        "top_p": params.get("top_p"),
        "max_tokens": params.get("max_length"),
        "min_tokens": params.get("min_length"),
        "seed": params.get("seed"),
    }
    return {k: v for k, v in out.items() if v is not None}


def _extract_task_pipeline(yaml_cfg: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    prompts = yaml_cfg.get("prompts") or {}
    if isinstance(prompts, dict):
        pp = prompts.get("postprocessor")
        if pp is not None:
            out["postprocessor"] = pp
    params = yaml_cfg.get("params") or {}
    for k in ("kg_extraction_src", "gi_insight_src", "gi_max_insights"):
        v = params.get(k)
        if v is not None:
            out[k] = v
    return out


def _extract_ps_config(yaml_cfg: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if "llm_pipeline_mode" in yaml_cfg and yaml_cfg["llm_pipeline_mode"] is not None:
        out["llm_pipeline_mode"] = yaml_cfg["llm_pipeline_mode"]
    if (
        "transcript_cleaning_strategy" in yaml_cfg
        and yaml_cfg["transcript_cleaning_strategy"] is not None
    ):
        out["transcript_cleaning_strategy"] = yaml_cfg["transcript_cleaning_strategy"]
    backend = yaml_cfg.get("backend") or {}
    if "extra_body" in backend and backend["extra_body"] is not None:
        out["openai_extra_body"] = backend["extra_body"]
    return out


def _classify_inference_target(base_url: Optional[str], backend_type: Optional[str]) -> str:
    """Mirror of the materialize_baseline._classify_inference_target."""
    from urllib.parse import urlparse

    bt = (backend_type or "").lower()
    if bt == "hf_local":
        return "local-hf"
    url = (base_url or "").lower()
    if url:
        # Parse hostname properly so substring checks happen on the host, not
        # the URL path/query (CodeQL py/incomplete-url-substring-sanitization).
        parsed = urlparse(url if "://" in url else f"http://{url}")
        host = (parsed.hostname or "").lower()
        port = parsed.port
        is_local_host = host in ("localhost", "127.0.0.1", "0.0.0.0")  # nosec B104
        if is_local_host:
            if port == 11434 or "ollama" in host:
                return "local-ollama"
            return "local-vllm"
        if host.endswith(".ts.net") or host.startswith("dgx-") or "dgx-llm" in host:
            return "dgx-vllm"
        if bt == "ollama" or port == 11434:
            return "local-ollama"
        return "remote-vllm"
    if bt in ("openai", "anthropic", "gemini", "deepseek", "mistral", "grok"):
        return f"cloud-{bt}"
    if bt == "ollama":
        return "local-ollama"
    return "unknown"


def _compute_dataset_content_hash(dataset_id: str) -> Optional[str]:
    root = MATERIALIZED_ROOT / dataset_id
    if not root.is_dir():
        return None
    digests: List[Tuple[str, str]] = []
    try:
        for f in sorted(root.rglob("*")):
            if not f.is_file():
                continue
            rel = str(f.relative_to(root))
            digests.append((rel, hashlib.sha256(f.read_bytes()).hexdigest()))
    except Exception:
        return None
    if not digests:
        return None
    combined = "\n".join(f"{rel}\t{h}" for rel, h in digests).encode("utf-8")
    return hashlib.sha256(combined).hexdigest()


def _recover_backing_model_id(run_dir: Path, short: str) -> Optional[str]:
    """Try to recover the HF id of the model that was loaded during the run.

    Sources, in priority order:
    1. The chunk-7 sweep logs at data/eval/runs/chunk7_silver_regen/<short>_*.log
       (these record the full ``docker run vllm serve <hf-id> ...`` cmdline).
    2. The run.log inside the run dir (sometimes carries the cmdline).

    Returns the HF id (e.g. ``"Qwen/Qwen3.5-35B-A3B"``) or None when no source
    has it.
    """
    sweep_root = RUNS_ROOT / "chunk7_silver_regen"
    candidates: List[Path] = []
    if sweep_root.exists():
        candidates.extend(sweep_root.glob(f"{short}_*.log"))
    candidates.append(run_dir / "run.log")
    pat = re.compile(r"vllm\s+serve\s+([A-Za-z0-9_./-]+)")
    for p in candidates:
        if not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        m = pat.search(text)
        if m:
            return m.group(1)
    return None


def _recover_inference_cmd(run_dir: Path, short: str) -> Tuple[Optional[str], Optional[str]]:
    """Try to recover the full vllm serve cmdline + container image.

    Returns (inference_args, inference_image), both may be None.
    """
    sweep_root = RUNS_ROOT / "chunk7_silver_regen"
    image_pat = re.compile(r"(nvcr\.io/nvidia/vllm:[0-9.]+-py3)")
    args_pat = re.compile(r"vllm\s+serve\s+[^\n]+", re.MULTILINE)
    candidates: List[Path] = []
    if sweep_root.exists():
        candidates.extend(sweep_root.glob(f"{short}_*.log"))
    candidates.append(run_dir / "run.log")
    inference_args: Optional[str] = None
    inference_image: Optional[str] = None
    for p in candidates:
        if not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if inference_args is None:
            m = args_pat.search(text)
            if m:
                inference_args = m.group(0).strip()
        if inference_image is None:
            m = image_pat.search(text)
            if m:
                inference_image = m.group(1)
        if inference_args and inference_image:
            break
    return inference_args, inference_image


def refingerprint_run(run_dir: Path, short: str, task_kind: str) -> Dict[str, Any]:
    """Build the v2 retro fingerprint. ``task_kind`` is 'kg' or 'gi'."""
    if not run_dir.exists():
        raise FileNotFoundError(f"run dir not found: {run_dir}")
    fp_path = run_dir / "fingerprint.json"
    if not fp_path.exists():
        raise FileNotFoundError(f"existing fingerprint.json not found in {run_dir}")
    original = json.loads(fp_path.read_text(encoding="utf-8"))

    yaml_path = _find_eval_yaml(short, task_kind)
    yaml_cfg: Dict[str, Any] = {}
    if yaml_path:
        yaml_cfg = _parse_eval_yaml(yaml_path)

    backend = yaml_cfg.get("backend") or {}
    base_url = backend.get("base_url")
    backend_type = backend.get("type")
    dataset_id = (yaml_cfg.get("data") or {}).get(
        "dataset_id", original.get("run_context", {}).get("dataset_id")
    )

    # Priority 1: try log scraping. If that misses (chunk-7 swap logs only
    # carried server-side stdout, not the docker-run cmdline itself), fall
    # back to the chunk-7 recovery table sourced from
    # autoresearch/PER_MODEL_OPTIMAL_PARAMS.md + the workarounds doc.
    backing_model_id = _recover_backing_model_id(run_dir, short)
    inference_args, inference_image = _recover_inference_cmd(run_dir, short)
    recovery_source: Optional[str] = "log_scrape" if backing_model_id else None
    if short in CHUNK7_RECOVERY:
        entry = CHUNK7_RECOVERY[short]
        if backing_model_id is None:
            backing_model_id = entry["backing_model_id"]
            recovery_source = "chunk7_recovery_table"
        if inference_args is None:
            inference_args = entry["args"]
            if recovery_source is None:
                recovery_source = "chunk7_recovery_table"
        if inference_image is None:
            inference_image = CHUNK7_INFERENCE_IMAGE

    # Audit markers for fields we still couldn't recover.
    unknown_fields: List[str] = []
    if backing_model_id is None:
        unknown_fields.append("pipeline.stages.main.model.backing_model_id")
    if inference_args is None:
        unknown_fields.append("pipeline.stages.main.inference_args")
    if inference_image is None:
        unknown_fields.append("pipeline.stages.main.inference_image")

    audited_at = _now_utc_iso()
    dataset_hash = _compute_dataset_content_hash(dataset_id) if dataset_id else None

    # Build retro fingerprint by augmenting the original.
    retro = copy.deepcopy(original)
    retro["fingerprint_version"] = "2.0"

    # Run context: add dataset_content_hash + audited_at
    rc = retro.setdefault("run_context", {})
    rc["dataset_content_hash"] = dataset_hash
    rc["dataset_content_hash_audited_at"] = audited_at

    # Pipeline: fill main.model and main.<new fields>
    pipeline = retro.setdefault("pipeline", {})
    stages = pipeline.setdefault("stages", {})
    main_stage = stages.setdefault("main", {"stage_id": "main", "model": {}})
    model_block = main_stage.setdefault("model", {})
    # Chunk-7 candidates ran against DGX even though YAML base_url says
    # localhost:8003 (overridden via --vllm-base-url CLI at run time). Use
    # the actual target URL for these specific runs.
    effective_base_url = CHUNK7_BASE_URL_ACTUAL if short in CHUNK7_RECOVERY else base_url
    model_block["base_url"] = effective_base_url
    model_block["backing_model_id"] = backing_model_id

    main_stage["generation_params"] = _extract_generation_params(yaml_cfg)
    main_stage["task_pipeline"] = _extract_task_pipeline(yaml_cfg)
    main_stage["inference_args"] = inference_args
    main_stage["inference_image"] = inference_image
    main_stage["podcast_scraper_config"] = _extract_ps_config(yaml_cfg)

    # Runtime: add inference_target. Chunk-7 candidates ran against DGX
    # regardless of YAML (CLI override) — force dgx-vllm.
    runtime = retro.setdefault("runtime", {})
    if short in CHUNK7_RECOVERY:
        runtime["inference_target"] = CHUNK7_INFERENCE_TARGET
    else:
        runtime["inference_target"] = _classify_inference_target(base_url, backend_type)

    # Retro audit section — explicit pointer that this fingerprint was
    # reconstructed, not original.
    retro["_retro_audit"] = {
        "audited_at": audited_at,
        "reason": (
            "RFC-097 fingerprint gap-closure (FINGERPRINT_GAPS_ANALYSIS_2026-06-22.md). "
            "Original fingerprint had generation_params={} for GI/KG (headline §1 bug) "
            "plus 5 other documented gaps. Reconstructed in-place under operator "
            "authorisation 2026-06-22 (feedback_never_mutate_historical_artifacts "
            "memory rule, extraordinary-measure clause)."
        ),
        "source_yaml": str(yaml_path.relative_to(REPO_ROOT)) if yaml_path else None,
        "source_chunk7_log_dir": "data/eval/runs/chunk7_silver_regen/",
        "recovery_source": recovery_source,
        "unknown_fields": unknown_fields,
    }

    # Recompute v2 fingerprint_hash over the full retro dict (excluding the
    # hash field + the per-call run_id, mirroring materialize_baseline.py).
    for_hash = copy.deepcopy(retro)
    for_hash.pop("fingerprint_hash", None)
    if "run_context" in for_hash and isinstance(for_hash["run_context"], dict):
        for_hash["run_context"].pop("run_id", None)
    canonical = json.dumps(for_hash, sort_keys=True, default=str).encode("utf-8")
    retro["fingerprint_hash"] = hashlib.sha256(canonical).hexdigest()

    return retro


def replace_fingerprint(run_dir: Path, short: str, task_kind: str) -> Tuple[Path, Path]:
    """Build the retro fingerprint and replace in-place. Archives the original
    at fingerprint.v1.original.json so the pre-mutation state is recoverable.

    Returns (replaced_path, archive_path).
    """
    fp_path = run_dir / "fingerprint.json"
    archive_path = run_dir / "fingerprint.v1.original.json"
    # Idempotency: don't re-archive if we've already done this run.
    if not archive_path.exists():
        archive_path.write_text(fp_path.read_text(encoding="utf-8"), encoding="utf-8")
    retro = refingerprint_run(run_dir, short, task_kind)
    fp_path.write_text(json.dumps(retro, indent=2, ensure_ascii=False), encoding="utf-8")
    return fp_path, archive_path


def _run_dir(short: str, task_kind: str) -> Path:
    """Resolve a candidate short + task kind to the run dir under data/eval/runs/."""
    suffix = (
        "knowledge_graph_v1"
        if task_kind == "kg"
        else "grounded_insights_v1" if task_kind == "gi" else None
    )
    if suffix is None:
        raise ValueError(f"task_kind must be 'kg' or 'gi', got {task_kind}")
    if short == GEMINI_CANDIDATE:
        return RUNS_ROOT / f"autoresearch_prompt_gemini_{short}_dev_{suffix}"
    return RUNS_ROOT / f"autoresearch_prompt_vllm_{short}_dev_{suffix}"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-id", help="Specific run dir name under data/eval/runs/")
    g.add_argument(
        "--all-chunk7",
        action="store_true",
        help="Refingerprint all chunk-7 v2 scoreboard runs (headlines first).",
    )
    args = ap.parse_args(argv)

    if args.run_id:
        run_dir = RUNS_ROOT / args.run_id
        # Infer short + task_kind from the run_id.
        m = re.match(
            r"autoresearch_prompt_(?:vllm|gemini)_(.+)_dev_(knowledge_graph|grounded_insights)_v1$",
            args.run_id,
        )
        if not m:
            print(f"ERROR: can't parse run_id: {args.run_id}", file=sys.stderr)
            return 2
        short = m.group(1)
        task_kind = "kg" if m.group(2) == "knowledge_graph" else "gi"
        replaced, archive = replace_fingerprint(run_dir, short, task_kind)
        print(f"  retro-fingerprinted {args.run_id}")
        print(f"    new : {replaced}")
        print(f"    orig: {archive}")
        return 0

    # --all-chunk7
    print("Re-fingerprinting chunk-7 v2 scoreboard runs (headlines first):")
    ordered = (
        [(c, "kg") for c in CHUNK7_HEADLINES]
        + [(c, "gi") for c in CHUNK7_HEADLINES]
        + [(c, "kg") for c in CHUNK7_REMAINDER]
        + [(c, "gi") for c in CHUNK7_REMAINDER]
        + [(GEMINI_CANDIDATE, "kg"), (GEMINI_CANDIDATE, "gi")]
    )
    n_ok, n_missing = 0, 0
    for short, task_kind in ordered:
        try:
            run_dir = _run_dir(short, task_kind)
            if not run_dir.exists():
                print(f"  skip (no run dir): {short} {task_kind}")
                n_missing += 1
                continue
            replace_fingerprint(run_dir, short, task_kind)
            print(f"  ok {short} {task_kind}")
            n_ok += 1
        except Exception as e:
            print(f"  FAIL {short} {task_kind}: {e}")
            n_missing += 1
    print(f"\n{n_ok} re-fingerprinted, {n_missing} skipped/failed.")
    return 0 if n_ok else 1


if __name__ == "__main__":
    sys.exit(main())
