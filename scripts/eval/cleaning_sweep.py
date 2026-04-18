"""Transcript cleaning model sweep: compare cleaning models on quality + cost.

For each provider's cleaning model, runs the cleaning stage on held-out
transcripts and measures:
1. Cleaning diff (chars removed, lines changed)
2. Downstream summary quality (chain test)
3. Wall-clock time per episode

Usage:
    python scripts/eval/cleaning_sweep.py \
        --dataset curated_5feeds_benchmark_v2 \
        --providers openai,gemini

    python scripts/eval/cleaning_sweep.py \
        --dataset curated_5feeds_benchmark_v2 \
        --providers openai --models gpt-4o,gpt-4o-mini
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Default cleaning models per provider (from config defaults)
CLEANING_CONFIGS = {
    "openai/gpt-4o": {
        "provider": "openai",
        "model_field": "openai_cleaning_model",
        "model": "gpt-4o",
        "temp_field": "openai_cleaning_temperature",
        "temp": 0.2,
    },
    "openai/gpt-4o-mini": {
        "provider": "openai",
        "model_field": "openai_cleaning_model",
        "model": "gpt-4o-mini",
        "temp_field": "openai_cleaning_temperature",
        "temp": 0.2,
    },
    "gemini/gemini-2.5-flash-lite": {
        "provider": "gemini",
        "model_field": "gemini_cleaning_model",
        "model": "gemini-2.5-flash-lite",
        "temp_field": "gemini_cleaning_temperature",
        "temp": 0.2,
    },
    "ollama/qwen3.5:9b": {
        "provider": "ollama",
        "model_field": "ollama_cleaning_model",
        "model": "qwen3.5:9b",
        "temp_field": "ollama_cleaning_temperature",
        "temp": 0.2,
    },
}


def measure_cleaning_diff(original: str, cleaned: str) -> Dict[str, Any]:
    """Measure what the cleaning stage changed."""
    orig_lines = original.splitlines()
    clean_lines = cleaned.splitlines()
    chars_removed = len(original) - len(cleaned)
    lines_removed = len(orig_lines) - len(clean_lines)
    pct_removed = chars_removed / max(len(original), 1) * 100

    return {
        "original_chars": len(original),
        "cleaned_chars": len(cleaned),
        "chars_removed": chars_removed,
        "pct_removed": round(pct_removed, 1),
        "original_lines": len(orig_lines),
        "cleaned_lines": len(clean_lines),
        "lines_removed": lines_removed,
    }


def main():
    parser = argparse.ArgumentParser(description="Transcript cleaning model sweep")
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--providers",
        default="openai,gemini",
        help="Comma-sep providers to test",
    )
    args = parser.parse_args()

    print(f"Dataset: {args.dataset}")
    print(f"Providers: {args.providers}")
    print()
    print("NOTE: This script is a scaffold. Full implementation requires")
    print("calling each provider's cleaning method on raw transcripts")
    print("and comparing output quality. See #594 for full plan.")
    print()

    # List available configs
    requested = set(args.providers.split(","))
    for label, cfg in sorted(CLEANING_CONFIGS.items()):
        prov = cfg["provider"]
        if prov in requested:
            print(f"  {label}: model={cfg['model']}, " f"temp={cfg['temp']}")


if __name__ == "__main__":
    main()
