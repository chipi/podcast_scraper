#!/usr/bin/env python3
"""Capture sentence-embedding vectors for #382 regression guard.

Freezes dim + L2-norm + first-8 dims of the embedding produced for
each fixed input string. Deliberately NOT the full vector or a SHA of
it — sentence-transformers on CPU has BLAS-thread-order non-determinism
that would flip a bit-hash without changing semantics. The first-8 dims
+ norm still catch any real drift (wrong model / wrong pool / wrong
normalization). Regression tolerance lives in
``scripts/eval/full_ml_recheck.py``.

Deterministic: fixed model_id, device=cpu, ``local_files_only=True``,
``normalize_embeddings=True`` (L2-normalized output). Emits JSONL:
``data/eval/references/embedding_baseline_v5.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# 10 diverse strings — short/long, English/technical/proper-noun, empty edge.
FIXTURES: list[dict[str, str]] = [
    {"id": "short_01", "text": "Hello world."},
    {"id": "short_02", "text": "The quick brown fox jumps over the lazy dog."},
    {"id": "tech_01", "text": "Backpropagation computes gradients via the chain rule."},
    {"id": "tech_02", "text": "Attention is all you need — the Transformer architecture."},
    {
        "id": "podcast_01",
        "text": (
            "In this episode we discuss trail building, drainage grading, "
            "and machine-built jump lines."
        ),
    },
    {
        "id": "podcast_02",
        "text": "The guest recommends dollar-cost averaging into broad-market index funds.",
    },
    {"id": "person_01", "text": "Ada Lovelace, mathematician and pioneer of computation."},
    {"id": "single_01", "text": "cat"},
    {
        "id": "long_01",
        "text": (
            "This is a much longer passage discussing multiple topics in sequence. "
            "It touches on machine learning, distributed systems, and the challenges "
            "of maintaining consistency across replicas. The final section considers "
            "how these ideas connect to podcast summarization workflows."
        ),
    },
    {
        "id": "quote_01",
        "text": '"The only way to do great work is to love what you do." — apocryphal.',
    },
]

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _vec_signature(vec: list[float]) -> dict:
    """Compact fingerprint of a vector: dim, L2-norm, first-8 dims.

    Deliberately NOT a full-vector hash. sentence-transformers on CPU has
    BLAS-thread-order non-determinism that flips bit-level hashes even
    when the vector is semantically identical to ~1e-6. Dim + L2 + first-8
    catch any real drift (wrong model / wrong pool / wrong normalization)
    without false-positive'ing on floating-point noise. See
    ``scripts/eval/full_ml_recheck.py::_check_embedding`` for the
    enforced tolerance.
    """
    n = sum(x * x for x in vec) ** 0.5
    return {
        "dim": len(vec),
        "l2_norm": round(n, 6),
        "first_8": [round(x, 6) for x in vec[:8]],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("capture_embedding_baseline")

    import transformers

    from podcast_scraper.providers.ml import embedding_loader

    log.info(
        "Capturing embedding baseline: model=%s device=%s transformers=%s inputs=%d",
        args.model,
        args.device,
        transformers.__version__,
        len(FIXTURES),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for fx in FIXTURES:
            t0 = time.perf_counter()
            vec = embedding_loader.encode(
                fx["text"],
                model_id=args.model,
                device=args.device,
                normalize=True,
            )
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            # encode(single-string, ...) returns List[float]
            sig = _vec_signature(vec)
            row = {
                "id": fx["id"],
                "text": fx["text"],
                "model": args.model,
                "device": args.device,
                "transformers_version": transformers.__version__,
                "elapsed_ms": elapsed_ms,
                **sig,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.info(
                "  %-12s dim=%d L2=%.4f d0=%+.4f (%dms)",
                fx["id"],
                sig["dim"],
                sig["l2_norm"],
                sig["first_8"][0],
                elapsed_ms,
            )
    log.info("wrote %s (%d rows)", args.out, len(FIXTURES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
