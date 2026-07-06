#!/usr/bin/env python3
"""Capture TransformersReduceBackend (FLAN-T5) output for #382 regression guard.

The hybrid_ml tier-1 REDUCE path uses ``TransformersReduceBackend`` (FLAN-T5).
There is no shipped production mode for it (registry has Ollama-reduce
hybrid modes only), so we can't run ``make baseline-create`` on it.
Instead: fixed (instruction, notes) prompts → frozen generation output.

Same design as the QA / NLI / embedding baselines: fixed inputs,
deterministic device=cpu, capture output text. Emits JSONL:
``data/eval/references/flant5_reduce_baseline_v5.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# 4 fixed (instruction, notes) prompts. Instruction wording matches the
# HybridMLProvider._build_reduce_instruction shapes so we're exercising
# the realistic call surface.
FIXTURES: list[dict[str, str]] = [
    {
        "id": "reduce_01_paragraph",
        "instruction": "Summarize these notes as one concise paragraph.",
        "notes": (
            "Trail building basics. Water management is critical. "
            "Machine-built jump lines. Sustainable tread widths. "
            "Riders should test features slowly before committing."
        ),
    },
    {
        "id": "reduce_02_bullets",
        "instruction": ("Turn these notes into 3 bullet points capturing the key takeaways."),
        "notes": (
            "Attention mechanism replaces recurrence. "
            "Parallelizable training. "
            "Positional encoding via sinusoids. "
            "Multi-head attention captures different representation subspaces. "
            "Layer norm applied post-residual."
        ),
    },
    {
        "id": "reduce_03_short",
        "instruction": "Rewrite in one sentence.",
        "notes": "The market closed higher today after the Fed announcement.",
    },
    {
        "id": "reduce_04_extract",
        "instruction": "Extract the guest's main recommendation.",
        "notes": (
            "Marco: 'I always dive with two independent light sources on wreck penetrations. "
            "Redundancy matters more than power when you're inside a wreck. "
            "Even a 50-lumen backup can guide you out if your primary fails.'"
        ),
    },
]

DEFAULT_MODEL = "google/flan-t5-base"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("capture_flant5_reduce_baseline")

    import transformers

    from podcast_scraper.providers.ml.hybrid_ml_provider import (
        TransformersReduceBackend,
    )

    log.info(
        "Capturing FLAN-T5 reduce baseline: model=%s device=%s transformers=%s prompts=%d",
        args.model,
        args.device,
        transformers.__version__,
        len(FIXTURES),
    )

    backend = TransformersReduceBackend(model_name=args.model, device=args.device, cache_dir=None)
    backend.initialize()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.out.open("w") as f:
            for fx in FIXTURES:
                t0 = time.perf_counter()
                result = backend.reduce(
                    notes=fx["notes"],
                    instruction=fx["instruction"],
                    # Deterministic: greedy beam search, seed-free.
                    params={"max_new_tokens": 120, "num_beams": 4, "do_sample": False},
                )
                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                row = {
                    "id": fx["id"],
                    "instruction": fx["instruction"],
                    "notes": fx["notes"],
                    "output_text": result.text,
                    "backend": result.backend,
                    "model": result.model,
                    "device": args.device,
                    "transformers_version": transformers.__version__,
                    "elapsed_ms": elapsed_ms,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                display = result.text[:80] + ("…" if len(result.text) > 80 else "")
                log.info("  %-20s → %r (%dms)", fx["id"], display, elapsed_ms)
    finally:
        backend.cleanup()

    log.info("wrote %s (%d rows)", args.out, len(FIXTURES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
