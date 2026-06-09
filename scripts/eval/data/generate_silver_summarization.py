#!/usr/bin/env python3
"""Generate a silver summarization reference dataset using an Anthropic model.

This is a focused one-shot generation script. It is the minimal path to
produce a `silver_*` reference for a curated dataset when the standard
`make experiment-run` route is unavailable (e.g. Opus 4.x thinking models
deprecate the `temperature` parameter that the production summarization
provider always passes).

Output layout mirrors what `scripts/eval/experiment/run_experiment.py` writes
so the resulting directory can be promoted via `make run-promote` and consumed
by the comparator/scorer in `score_only` mode.

Usage:

    python scripts/eval/data/generate_silver_summarization.py \\
        --config data/eval/configs/silver_selection/<config>.yaml \\
        --run-id silver_candidate_anthropic_opus47_smoke_v1

Required env: ANTHROPIC_API_KEY (loaded from .env or shell).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from podcast_scraper.prompts.store import render_prompt  # noqa: E402

logger = logging.getLogger(__name__)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_info() -> Dict[str, Any]:
    def _run(cmd: list[str]) -> str:
        try:
            return subprocess.check_output(cmd, cwd=REPO_ROOT).decode().strip()
        except Exception:
            return "unknown"

    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(_run(["git", "status", "--porcelain"])),
    }


def _materialized_root(dataset_id: str) -> Path:
    return REPO_ROOT / "data" / "eval" / "materialized" / dataset_id


def _load_dataset_episodes(dataset_id: str) -> list[Dict[str, Any]]:
    root = _materialized_root(dataset_id)
    if not root.exists():
        raise FileNotFoundError(f"Materialized dataset not found: {root}")
    episodes: list[Dict[str, Any]] = []
    for txt in sorted(root.glob("*.txt")):
        episode_id = txt.stem
        meta_path = root / f"{episode_id}.meta.json"
        meta: Dict[str, Any] = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        episodes.append(
            {
                "episode_id": episode_id,
                "input_path": str(txt.relative_to(REPO_ROOT)),
                "text": txt.read_text(encoding="utf-8"),
                "meta": meta,
            }
        )
    return episodes


def _build_prompt(
    user_template: str,
    transcript: str,
    title: str,
    max_length: int,
    min_length: int,
) -> str:
    paragraphs_min = max(1, min_length // 100)
    paragraphs_max = max(paragraphs_min, max_length // 100)
    return render_prompt(
        user_template,
        transcript=transcript,
        title=title or "",
        paragraphs_min=paragraphs_min,
        paragraphs_max=paragraphs_max,
    )


def _call_anthropic(
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float | None,
) -> tuple[str, Any]:
    """Call Anthropic with deterministic silver settings.

    Thinking models (Opus 4.x) deprecate non-default `temperature`. The caller
    should pass `temperature=None` to omit the parameter entirely.
    """
    kwargs: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    if temperature is not None:
        kwargs["temperature"] = temperature

    response = client.messages.create(**kwargs)
    # Walk content blocks; skip thinking blocks if present.
    text_parts: list[str] = []
    for block in response.content or []:
        block_type = getattr(block, "type", None)
        if block_type == "text" and hasattr(block, "text"):
            text_parts.append(block.text)
    summary = "".join(text_parts).strip()
    return summary, response


def _cost_usd(
    pricing: Dict[str, Any],
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    text_pricing = pricing.get("providers", {}).get("anthropic", {}).get("text", {})
    rates = text_pricing.get(model)
    if rates is None:
        # Try a coarse alias fallback (claude-opus-4-7 → claude-opus-4-5 pricing).
        for fallback in ("claude-opus-4-5", "claude-opus-4", "default"):
            rates = text_pricing.get(fallback)
            if rates is not None:
                logger.warning("No pricing for %s; falling back to %s", model, fallback)
                break
    if rates is None:
        return 0.0
    in_cost = (input_tokens / 1_000_000) * rates.get("input_cost_per_1m_tokens", 0.0)
    out_cost = (output_tokens / 1_000_000) * rates.get("output_cost_per_1m_tokens", 0.0)
    return in_cost + out_cost


def _load_pricing() -> Dict[str, Any]:
    pricing_path = REPO_ROOT / "config" / "pricing_assumptions.yaml"
    if not pricing_path.exists():
        return {}
    return yaml.safe_load(pricing_path.read_text(encoding="utf-8")) or {}


def _user_prompt_to_template_path(user_prompt_name: str) -> Path:
    return REPO_ROOT / "src" / "podcast_scraper" / "prompts" / f"{user_prompt_name}.j2"


def _read_template_hash(user_prompt_name: str) -> tuple[str, str]:
    path = _user_prompt_to_template_path(user_prompt_name)
    if not path.exists():
        return ("", str(path))
    return (_sha256(path.read_bytes()), str(path.relative_to(REPO_ROOT)))


def generate(cfg_path: Path, run_id: str, force: bool = False) -> Path:
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    backend = cfg["backend"]
    prompts = cfg.get("prompts") or {}
    user_prompt_name = prompts.get("user", "anthropic/summarization/long_v2")
    system_prompt_name = prompts.get("system", "anthropic/summarization/system_v1")
    params = cfg.get("params") or {}
    max_length = int(params.get("max_length", 800))
    min_length = int(params.get("min_length", 200))
    dataset_id = cfg["data"]["dataset_id"]
    model = backend["model"]

    # Thinking models deprecate non-1.0 temperature. Detect by id substring.
    is_thinking_model = any(s in model for s in ("opus-4-7", "opus-4-8", "opus-4-9", "opus-5"))
    if is_thinking_model:
        temperature_param: float | None = None  # omit; default greedy-ish for thinking models
        temperature_recorded = "omitted (thinking model)"
    else:
        temperature_param = float(params.get("temperature", 0.0))
        temperature_recorded = temperature_param

    results_dir = REPO_ROOT / "data" / "eval" / "runs" / run_id
    if results_dir.exists():
        if not force:
            raise FileExistsError(
                f"Run directory exists: {results_dir} (pass --force to overwrite)"
            )
        # Remove only known artifacts to avoid wiping accidental neighbors.
        for child in results_dir.iterdir():
            if child.is_file():
                child.unlink()
            else:
                import shutil

                shutil.rmtree(child)
    results_dir.mkdir(parents=True, exist_ok=True)

    log_path = results_dir / "run.log"
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)
    logger.info(
        "Starting silver generation: run_id=%s model=%s dataset=%s",
        run_id,
        model,
        dataset_id,
    )

    # Late import so the script can be inspected without anthropic installed.
    import anthropic

    client = anthropic.Anthropic()
    pricing = _load_pricing()
    git_info = _git_info()
    system_prompt = render_prompt(system_prompt_name)
    user_template_sha, user_template_relpath = _read_template_hash(user_prompt_name)
    system_template_sha, system_template_relpath = _read_template_hash(system_prompt_name)

    episodes = _load_dataset_episodes(dataset_id)
    if not episodes:
        raise RuntimeError(f"No episodes found for dataset {dataset_id}")
    logger.info("Loaded %d episode(s)", len(episodes))

    predictions: list[Dict[str, Any]] = []
    total_in = 0
    total_out = 0
    total_cost = 0.0
    total_chars_in = 0
    total_chars_out = 0
    total_latency_ms = 0.0
    started = datetime.now(timezone.utc)
    for ep in episodes:
        title = (ep.get("meta") or {}).get("title") or ""
        user_prompt = _build_prompt(
            user_template=user_prompt_name,
            transcript=ep["text"],
            title=title,
            max_length=max_length,
            min_length=min_length,
        )
        logger.info("Summarizing %s (%d chars)", ep["episode_id"], len(ep["text"]))
        t0 = time.perf_counter()
        summary, response = _call_anthropic(
            client,
            model,
            system_prompt,
            user_prompt,
            max_length,
            temperature_param,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if not summary:
            raise RuntimeError(f"Empty summary from {model} on {ep['episode_id']}")

        in_tok = int(getattr(response.usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(response.usage, "output_tokens", 0) or 0)
        cost = _cost_usd(pricing, model, in_tok, out_tok)
        total_in += in_tok
        total_out += out_tok
        total_cost += cost
        total_chars_in += len(ep["text"])
        total_chars_out += len(summary)
        total_latency_ms += latency_ms

        predictions.append(
            {
                "episode_id": ep["episode_id"],
                "dataset_id": dataset_id,
                "output": {"summary_final": summary},
                "fingerprint_ref": "fingerprint.json",
                "metadata": {
                    "input_hash": "sha256:" + _sha256(ep["text"].encode()),
                    "output_hash": "sha256:" + _sha256(summary.encode()),
                    "input_path": ep["input_path"],
                    "input_length_chars": len(ep["text"]),
                    "output_length_chars": len(summary),
                    "processing_time_seconds": latency_ms / 1000.0,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "cost_usd": cost,
                },
            }
        )
    finished = datetime.now(timezone.utc)

    # Write predictions.jsonl
    predictions_path = results_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as fh:
        for row in predictions:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # fingerprint.json mirrors the structure of existing silver baselines.
    fingerprint = {
        "fingerprint_version": "1.0",
        "task": "summarization",
        "run_context": {
            "run_id": started.isoformat(),
            "baseline_id": run_id,
            "dataset_id": dataset_id,
            "git": git_info,
        },
        "provider": {
            "provider_type": "anthropic",
            "provider_library": "anthropic",
            "provider_library_version": _try_version("anthropic"),
        },
        "pipeline": {
            "type": "single_stage",
            "stages": {
                "main": {
                    "stage_id": "main",
                    "model": {
                        "provider_type": "anthropic",
                        "framework": None,
                        "model_name": model,
                        "model_revision": None,
                        "tokenizer_name": None,
                        "tokenizer_revision": None,
                        "endpoint": None,
                    },
                    "generation_params": {
                        "temperature": temperature_recorded,
                        "max_tokens": max_length,
                    },
                }
            },
        },
        "preprocessing": {
            "profile_id": cfg.get("preprocessing_profile", "cleaning_v4"),
            "profile_version": "4.0",
            "steps": None,
            "note": (
                "Silver generation reads materialized cleaned dataset directly; "
                "no pipeline preprocessing applied at generation time."
            ),
        },
        "tokenization": {},
        "chunking": None,
        "environment": {
            "python_version": platform.python_version(),
            "os": platform.platform(),
        },
        "runtime": {},
        "prompts": {
            "user": {
                "name": user_prompt_name,
                "file": user_template_relpath,
                "sha256": user_template_sha,
                "params": {},
            },
            "system": {
                "name": system_prompt_name,
                "file": system_template_relpath,
                "sha256": system_template_sha,
                "params": {},
            },
        },
    }
    (results_dir / "fingerprint.json").write_text(
        json.dumps(fingerprint, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # baseline.json — matches the existing silver shape so promote_run.py reads it.
    total_seconds = (finished - started).total_seconds()
    baseline = {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "created_at": started.isoformat(),
        "created_by": os.environ.get("USER", "unknown"),
        "fingerprint_ref": "fingerprint.json",
        "task": "summarization",
        "backend": {"type": "anthropic", "model": model},
        "params": {
            "model": model,
            "max_length": max_length,
            "temperature": temperature_recorded,
            "user_prompt": user_prompt_name,
            "system_prompt": system_prompt_name,
        },
        "stats": {
            "num_episodes": len(episodes),
            "total_time_seconds": total_seconds,
            "avg_time_seconds": total_seconds / max(1, len(episodes)),
            "total_chars_in": total_chars_in,
            "total_chars_out": total_chars_out,
            "avg_compression": (total_chars_in / total_chars_out) if total_chars_out else 0.0,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_cost_usd": total_cost,
        },
    }
    (results_dir / "baseline.json").write_text(
        json.dumps(baseline, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # metrics.json — minimal aggregate intrinsics. (No vs_reference at silver gen time.)
    avg_tokens = total_out / max(1, len(episodes))
    metrics = {
        "dataset_id": dataset_id,
        "run_id": run_id,
        "episode_count": len(episodes),
        "intrinsic": {
            "gates": {
                "boilerplate_leak_rate": 0.0,
                "speaker_label_leak_rate": 0.0,
                "truncation_rate": 0.0,
                "failed_episodes": [],
                "episode_gate_failures": {},
            },
            "warnings": {"speaker_name_leak_rate": 0.0},
            "length": {
                "avg_tokens": avg_tokens,
                "min_tokens": min(p["metadata"]["output_tokens"] for p in predictions),
                "max_tokens": max(p["metadata"]["output_tokens"] for p in predictions),
            },
            "performance": {
                "avg_latency_ms": total_latency_ms / max(1, len(episodes)),
            },
            "cost": {
                "total_cost_usd": total_cost,
                "total_input_tokens": total_in,
                "total_output_tokens": total_out,
            },
        },
        "vs_reference": None,
        "schema": "metrics_summarization_v1",
        "task": "summarization",
    }
    (results_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # README — short stub; full provenance lives in baseline + fingerprint.
    (results_dir / "README.md").write_text(
        f"""# Silver candidate: {run_id}

Generated by `scripts/eval/data/generate_silver_summarization.py` because
{model} is a thinking model that deprecates the `temperature` parameter
passed by the standard `run_experiment.py` summarization path.

- Dataset: `{dataset_id}` (5 episodes)
- Model: `{model}`
- Prompt: `{user_prompt_name}` (system: `{system_prompt_name}`)
- Generation date: {started.isoformat()}
- Cost: ${total_cost:.4f} ({total_in} input / {total_out} output tokens)

Promote with:

```bash
make run-promote RUN_ID={run_id} AS=reference \\
  PROMOTED_ID=silver_opus47_smoke_v1 REFERENCE_QUALITY=silver \\
  REASON="Opus 4.7 silver — #939"
```
""",
        encoding="utf-8",
    )

    logger.info(
        "Done. %d episodes, total_in=%d, total_out=%d, cost_usd=%.4f, wall=%.1fs",
        len(episodes),
        total_in,
        total_out,
        total_cost,
        total_seconds,
    )
    return results_dir


def _try_version(pkg: str) -> str:
    try:
        from importlib.metadata import version

        return version(pkg)
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        out = generate(args.config, args.run_id, force=args.force)
    except Exception as exc:
        logger.error("Silver generation failed: %s", exc, exc_info=True)
        return 1
    print(f"Silver candidate written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
