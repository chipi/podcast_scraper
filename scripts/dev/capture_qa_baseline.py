#!/usr/bin/env python3
"""Capture extractive-QA span outputs for #382 parity gate.

Runs ``extractive_qa.answer_candidates`` against a fixed set of
(context, question) pairs derived from the ``curated_5feeds_smoke_v1``
transcripts. Emits JSONL that Phase 7 compares against post-upgrade to
detect any drift introduced by the ``pipeline("question-answering")``
→ ``AutoModelForQuestionAnswering`` refactor.

Deterministic: fixed model id + revision + local_files_only=True. No
network. Written to ``data/eval/references/qa_baseline_v5_{pre,post}.jsonl``.

Usage:
    python scripts/dev/capture_qa_baseline.py \\
        --out data/eval/references/qa_baseline_v5_pre.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Repo root on sys.path so scripts imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


REPO = Path(__file__).resolve().parents[2]
SOURCES = REPO / "data" / "eval" / "sources" / "curated_5feeds_raw_v1"

# Fixed set of (episode_id, context_slice, question) pairs. Contexts are
# ~500-char slices from the smoke episodes. Questions are hand-picked
# to hit answers that live in the slice, so top-1 spans should be
# stable across the migration (identical checkpoint + deterministic).
FIXTURES: list[dict[str, str]] = [
    {
        "id": "p01_e01_q1_guest",
        "episode_id": "p01_e01",
        "context_line_range": (1, 60),
        "question": "Who is the guest?",
    },
    {
        "id": "p01_e01_q2_topic",
        "episode_id": "p01_e01",
        "context_line_range": (1, 60),
        "question": "What is the episode topic?",
    },
    {
        "id": "p02_e01_q1_guest",
        "episode_id": "p02_e01",
        "context_line_range": (1, 60),
        "question": "Who is the guest?",
    },
    {
        "id": "p02_e01_q2_topic",
        "episode_id": "p02_e01",
        "context_line_range": (1, 60),
        "question": "What is the topic of this episode?",
    },
    {
        "id": "p03_e01_q1_guest",
        "episode_id": "p03_e01",
        "context_line_range": (1, 60),
        "question": "Who is the guest?",
    },
    {
        "id": "p04_e01_q1_guest",
        "episode_id": "p04_e01",
        "context_line_range": (1, 60),
        "question": "Who is the guest?",
    },
    {
        "id": "p05_e01_q1_guest",
        "episode_id": "p05_e01",
        "context_line_range": (1, 60),
        "question": "Who is the guest?",
    },
    {
        "id": "p05_e01_q2_topic",
        "episode_id": "p05_e01",
        "context_line_range": (1, 60),
        "question": "What is the topic?",
    },
]

DEFAULT_MODEL = "deepset/roberta-base-squad2"


def load_context(episode_id: str, line_range: tuple[int, int]) -> str:
    """Return joined lines [start, end) from the fixture transcript."""
    feed = episode_id.split("_")[0]  # p01 → feed-p01
    txt = SOURCES / f"feed-{feed}" / f"{episode_id}.txt"
    if not txt.exists():
        raise FileNotFoundError(f"transcript not found: {txt}")
    lines = txt.read_text().splitlines()
    start, end = line_range
    return "\n".join(lines[start - 1 : end])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu / mps / cuda (default cpu for cross-machine determinism)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("capture_qa_baseline")

    # Late import so the script can be discovered without ML deps loaded.
    import transformers

    from podcast_scraper.providers.ml import extractive_qa

    log.info(
        "Capturing QA baseline: model=%s device=%s transformers=%s pairs=%d",
        args.model,
        args.device,
        transformers.__version__,
        len(FIXTURES),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for fixture in FIXTURES:
            ctx = load_context(
                fixture["episode_id"],
                tuple(fixture["context_line_range"]),  # type: ignore[arg-type]
            )
            t0 = time.perf_counter()
            spans = extractive_qa.answer_candidates(
                context=ctx,
                question=fixture["question"],
                model_id=args.model,
                device=args.device,
                top_k=args.top_k,
            )
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            row = {
                "id": fixture["id"],
                "episode_id": fixture["episode_id"],
                "question": fixture["question"],
                "context_line_range": list(fixture["context_line_range"]),
                "context_hash_first120": ctx[:120],
                "model": args.model,
                "device": args.device,
                "transformers_version": transformers.__version__,
                "elapsed_ms": elapsed_ms,
                "top_k_spans": [asdict(s) for s in spans],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            best = spans[0] if spans else None
            best_answer = best.answer if best else ""
            display = (best_answer[:60] + "…") if len(best_answer) > 60 else (best_answer or "∅")
            log.info(
                "  %s → answer=%r score=%.4f (%dms)",
                fixture["id"],
                display,
                best.score if best else 0.0,
                elapsed_ms,
            )
    log.info("wrote %s (%d rows)", args.out, len(FIXTURES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
