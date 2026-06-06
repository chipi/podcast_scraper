#!/usr/bin/env python3
"""CLI driver for embedding-provider A/B eval (#897, ADR-098).

Usage:
    python scripts/eval_embedding_providers.py \\
        --corpus ./output \\
        --output-root eval/embedding_provider_comparison \\
        [--max-pairs 500] \\
        [--ollama-base-url http://127.0.0.1:11434] \\
        [--ollama-model nomic-embed-text] \\
        [--st-model sentence-transformers/all-MiniLM-L6-v2]

Requires `ollama pull nomic-embed-text` on whichever host the URL points to.
On the operator's laptop with Ollama installed, the default
`http://127.0.0.1:11434` works after pulling the model locally.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from podcast_scraper.evaluation.embedding_provider_eval import (
    ProviderConfig,
    run_comparison,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare embedding providers on the operator's corpus using "
            "gi.json SUPPORTED_BY edges as ground truth."
        ),
    )
    parser.add_argument("--corpus", required=True, help="Corpus root (parent of feeds/).")
    parser.add_argument(
        "--output-root",
        default="eval/embedding_provider_comparison",
        help="Where to write the timestamped run dir.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Cap on insight→quote pairs (default: all).",
    )
    parser.add_argument(
        "--ollama-base-url",
        default="http://127.0.0.1:11434",
        help="Ollama base URL. Default: localhost (laptop-side Ollama).",
    )
    parser.add_argument(
        "--ollama-model",
        default="nomic-embed-text",
        help="Ollama embedding tag (default: nomic-embed-text).",
    )
    parser.add_argument(
        "--st-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="sentence-transformers model for the baseline arm.",
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        help="Override the run-dir timestamp (default: UTC now). For deterministic tests.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    corpus_root = Path(args.corpus).resolve()
    output_root = Path(args.output_root).resolve()
    timestamp = args.timestamp or datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    matrix = [
        ProviderConfig(
            label="MiniLM (current default)",
            provider="sentence_transformers",
            model_id=args.st_model,
            endpoint=None,
        ),
        ProviderConfig(
            label="nomic-embed-text (Ollama)",
            provider="ollama",
            model_id=args.ollama_model,
            endpoint=args.ollama_base_url,
        ),
    ]
    run_dir = run_comparison(
        corpus_root,
        output_root,
        matrix,
        timestamp=timestamp,
        max_pairs=args.max_pairs,
    )
    print(f"Report written to: {run_dir}")
    print(f"  - {run_dir / 'report.md'}")
    print(f"  - {run_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
