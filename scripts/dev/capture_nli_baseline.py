#!/usr/bin/env python3
"""Capture NLI entailment scores for #382 regression guard.

Same design as ``capture_qa_baseline.py``: fixed model + fixed premise/
hypothesis pairs → frozen entailment probabilities. Phase E's NLI backend
refactor drove this file; Phase F's HFSeq2SeqBackend + everything after
touch nothing here but the fixture is our safety net.

Deterministic: fixed model_id, device=cpu, ``local_files_only=True``,
fixed 12 pairs designed to exercise entailment / neutral / contradiction
across three-class NLI models. Emits JSONL:
``data/eval/references/nli_baseline_v5.jsonl``.

Usage::

    python scripts/dev/capture_nli_baseline.py \\
        --out data/eval/references/nli_baseline_v5.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Fixed 12-pair fixture. Deliberately spans:
# - 3 clear entailment pairs (paraphrase / hypernym / partial paraphrase)
# - 3 clear contradiction pairs (negation / opposite / mutually exclusive)
# - 3 neutral pairs (topic overlap without entailment)
# - 3 "extractive" pairs like the GI grounding call sites hit
FIXTURES: list[dict[str, str]] = [
    # Entailment
    {
        "id": "ent_01_paraphrase",
        "premise": "The cat sat on the mat.",
        "hypothesis": "A cat is on the mat.",
    },
    {
        "id": "ent_02_hypernym",
        "premise": "The Golden Retriever ran across the yard.",
        "hypothesis": "A dog moved across the yard.",
    },
    {
        "id": "ent_03_partial",
        "premise": "She published two novels last year.",
        "hypothesis": "She published novels last year.",
    },
    # Contradiction
    {
        "id": "con_01_negation",
        "premise": "The team won the championship.",
        "hypothesis": "The team lost the championship.",
    },
    {
        "id": "con_02_opposite",
        "premise": "The room was completely empty.",
        "hypothesis": "The room was full of people.",
    },
    {
        "id": "con_03_exclusive",
        "premise": "He is currently in Paris.",
        "hypothesis": "He is currently in Tokyo.",
    },
    # Neutral
    {
        "id": "neu_01_related",
        "premise": "The bike had a carbon frame.",
        "hypothesis": "The bike was expensive.",
    },
    {
        "id": "neu_02_topic",
        "premise": "Machine learning models require data.",
        "hypothesis": "Deep learning is a subfield of machine learning.",
    },
    {
        "id": "neu_03_context",
        "premise": "She lives in Boston.",
        "hypothesis": "She commutes to work by train.",
    },
    # GI-grounding-style (quote → claim entailment)
    {
        "id": "gi_01_direct",
        "premise": "The host said trail maintenance requires drainage grading every spring.",
        "hypothesis": "Trail maintenance involves regular drainage work.",
    },
    {
        "id": "gi_02_indirect",
        "premise": (
            "Index funds have consistently outperformed active " "management over 20-year periods."
        ),
        "hypothesis": "Passive investing beats active investing long-term.",
    },
    {
        "id": "gi_03_unsupported",
        "premise": "The guest recommended shooting at f/8 for landscape work.",
        "hypothesis": "The guest recommended shooting at f/22 for portraits.",
    },
]

DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-base"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("capture_nli_baseline")

    import transformers

    from podcast_scraper.providers.ml import nli_loader

    log.info(
        "Capturing NLI baseline: model=%s device=%s transformers=%s pairs=%d",
        args.model,
        args.device,
        transformers.__version__,
        len(FIXTURES),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for fx in FIXTURES:
            t0 = time.perf_counter()
            score = nli_loader.entailment_score(
                premise=fx["premise"],
                hypothesis=fx["hypothesis"],
                model_id=args.model,
                device=args.device,
            )
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            row = {
                "id": fx["id"],
                "premise": fx["premise"],
                "hypothesis": fx["hypothesis"],
                "entailment_score": score,
                "model": args.model,
                "device": args.device,
                "transformers_version": transformers.__version__,
                "elapsed_ms": elapsed_ms,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.info("  %-22s → score=%.4f (%dms)", fx["id"], score, elapsed_ms)
    log.info("wrote %s (%d rows)", args.out, len(FIXTURES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
