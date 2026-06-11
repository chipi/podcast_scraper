"""Generate summary predictions via the vLLM autoresearch endpoint (#928).

DEPRECATED in #960. Use the native ``openai`` backend with ``base_url``
+ ``extra_body`` + ``api_key_env`` set in the experiment YAML — see
``data/eval/configs/summarization/autoresearch_prompt_vllm_qwen36_35b_smoke_paragraph_v1.yaml``
for the canonical example. The native path goes through
``scripts/eval/experiment/run_experiment.py`` and emits the same
predictions.jsonl shape with no side-channel script.

This script is retained for historical reproducibility of #928 batch-3
results only; new code should not call it.

Produces a predictions.jsonl in the same shape the autoresearch pipeline
produces for Ollama / OpenAI / Gemini runs, so the finale framework
(``scripts/eval/finale_sweep.py``) can score it side-by-side with the
existing Ollama qwen3.5:35b results from #949.

Why this script was originally needed:

The autoresearch experiment-run path (``autoresearch_track_a.py``) had
explicit backend types: ``openai``, ``gemini``, ``ollama``, etc. — no
``vllm`` type. vLLM exposes an OpenAI-compatible API, so the clean fix
was either (a) add a ``vllm`` backend type or (b) extend ``openai``
to accept a per-experiment ``base_url`` override. #960 picked (b).

Output:
    ``data/eval/runs/<run_id>/predictions.jsonl``
    ``data/eval/runs/<run_id>/fingerprint.json``
    ``data/eval/runs/<run_id>/run.log``

Usage:
    python scripts/eval/score/summary_vllm_predict_v1.py \\
        --vllm-url http://dgx-llm-1.tail6d0ed4.ts.net:8003/v1 \\
        --model "Qwen/Qwen3-Coder-Next-FP8" \\
        --prompt-system src/podcast_scraper/prompts/ollama/qwen3.5_35b/summarization/system_v1.j2 \\
        --prompt-user src/podcast_scraper/prompts/ollama/qwen3.5_35b/summarization/long_v1.j2 \\
        --materialized-dir data/eval/materialized/curated_5feeds_smoke_v1 \\
        --episodes p01_e01 p02_e01 p03_e01 p04_e01 p05_e01 \\
        --output-dir data/eval/runs/autoresearch_prompt_vllm_<tag>_curated_5feeds_smoke_v1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _path_for_log(p: Path) -> str:
    """Return a repo-relative path when possible, otherwise an absolute string."""
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p.resolve())


def _render_jinja(template_text: str, **vars: Any) -> str:
    """Minimal Jinja2 stand-in. Only supports ``{{ name }}`` substitution."""
    from jinja2 import Template

    return Template(template_text).render(**vars)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vllm-url", required=True, help="vLLM /v1 endpoint URL")
    p.add_argument("--model", required=True, help="Model id served by vLLM")
    p.add_argument("--prompt-system", required=True, type=Path)
    p.add_argument("--prompt-user", required=True, type=Path)
    p.add_argument("--materialized-dir", required=True, type=Path)
    p.add_argument("--episodes", nargs="+", required=True)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--max-tokens", type=int, default=800)
    p.add_argument("--min-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--disable-thinking",
        action="store_true",
        help=(
            "Pass chat_template_kwargs={'enable_thinking': False} to vLLM. Required for "
            "Qwen3.5/3.6 family to produce a clean summary instead of leaking reasoning prose."
        ),
    )
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "run.log"
    preds_path = args.output_dir / "predictions.jsonl"

    system_tpl = args.prompt_system.read_text(encoding="utf-8")
    user_tpl = args.prompt_user.read_text(encoding="utf-8")

    from openai import OpenAI

    # vLLM doesn't validate api_key; any non-empty string works.
    client = OpenAI(base_url=args.vllm_url, api_key="vllm-no-auth-needed")

    dataset_id = args.materialized_dir.name
    n_total = len(args.episodes)
    rows: list[dict[str, Any]] = []
    total_cost = 0.0  # always 0 for local DGX serving
    started = time.time()

    log_lines: list[str] = []

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, file=sys.stderr)
        log_lines.append(line)

    log(f"vllm url={args.vllm_url}")
    log(f"model={args.model}")
    log(f"dataset={dataset_id} n_episodes={n_total}")

    for i, ep in enumerate(args.episodes, 1):
        transcript_path = args.materialized_dir / f"{ep}.txt"
        if not transcript_path.exists():
            log(f"  SKIP {ep}: materialized transcript missing at {transcript_path}")
            continue
        transcript = transcript_path.read_text(encoding="utf-8")
        system_msg = _render_jinja(system_tpl, transcript=transcript)
        user_msg = _render_jinja(user_tpl, transcript=transcript)

        log(f"  [{i}/{n_total}] {ep} — transcript {len(transcript)}c → calling vLLM…")
        t0 = time.time()
        extra_body: dict[str, Any] = {}
        if args.disable_thinking:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout=600.0,
                extra_body=extra_body or None,
            )
            summary = (resp.choices[0].message.content or "").strip()
            usage = resp.usage
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            err = ""
        except Exception as exc:  # noqa: BLE001
            summary = ""
            prompt_tokens = completion_tokens = 0
            err = str(exc)[:300]
        elapsed = time.time() - t0
        log(
            f"    {ep}: {elapsed:.1f}s — out {len(summary)}c "
            f"({prompt_tokens}+{completion_tokens} tok) {f'ERR={err}' if err else ''}"
        )

        rows.append(
            {
                "episode_id": ep,
                "dataset_id": dataset_id,
                "output": {"summary_final": summary},
                "metadata": {
                    "model": args.model,
                    "backend": "vllm",
                    "endpoint": args.vllm_url,
                    "elapsed_seconds": round(elapsed, 2),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cost_usd": 0.0,
                    "error": err,
                },
            }
        )

    total_elapsed = time.time() - started
    with preds_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    fingerprint = {
        "schema": "predictions_v1",
        "backend": "vllm",
        "endpoint": args.vllm_url,
        "model": args.model,
        "dataset_id": dataset_id,
        "n_episodes": len(rows),
        "total_elapsed_seconds": round(total_elapsed, 1),
        "total_cost_usd": round(total_cost, 4),
        "prompts": {
            # Store as absolute paths if they're outside PROJECT_ROOT; otherwise
            # use repo-relative for readability. The path comparison guards against
            # ValueError on relative_to when the caller passed a CLI-relative path.
            "system": _path_for_log(args.prompt_system),
            "user": _path_for_log(args.prompt_user),
        },
        "params": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "disable_thinking": args.disable_thinking,
        },
    }
    (args.output_dir / "fingerprint.json").write_text(
        json.dumps(fingerprint, indent=2), encoding="utf-8"
    )
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    print(f"\nwrote {len(rows)} predictions in {total_elapsed:.1f}s")
    print(f"  {preds_path}")
    print(f"  {args.output_dir / 'fingerprint.json'}")
    print(f"  {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
