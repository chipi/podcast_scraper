#!/usr/bin/env python3
"""Calibrate a local entailment gate against a trusted annotator's judgements (#1179).

The DGX pilot grounded 13.3% of its insights where the cloud grounded 91.3%. The funnel showed the
collapse is one stage: of the candidates it found, gemini's entailment gate passes ~66% and qwen's
passes ~6%. So the gate is miscalibrated — but "tune it until more quotes pass" is not a target, it
is a way to break the ADR-053 trust contract by letting unsupported quotes certify insights.

The fix is a **silver reference**. Gemini annotated the *same episodes* and grounded 91.3% of them,
so the (insight -> quote) pairs it ACCEPTED are pairs a trusted annotator judged to be real
evidence. That gives labelled data for free:

    POSITIVES  insight -> quote pairs gemini linked with a SUPPORTED_BY edge
    NEGATIVES  the same quotes paired with insights from OTHER episodes — same style, same domain,
               same distribution, but genuinely unrelated

A candidate gate is then judged on both, because either alone is trivially gameable:

    recall     % of gemini's positives it accepts   (too low -> evidence is discarded; 6% today)
    rejection  % of the negatives it rejects        (too low -> hallucinations get certified)

Maximise both. A gate that accepts everything scores 100% recall and 0% rejection, and is worthless.

Usage:
    python scripts/eval/score/entailment_calibration_v1.py \
        --silver .test_outputs/manual/prod-v2/corpus --feed simplecast \
        --model qwen3.5:35b --prompt src/podcast_scraper/prompts/ollama/evidence/entailment/v1.j2

Exit codes: 0 ok · 2 input error
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import httpx


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _silver_pairs(corpus: Path, feed_filter: str) -> List[Tuple[str, str, str]]:
    """(episode, insight, quote) for every SUPPORTED_BY edge the trusted annotator created."""
    pairs: List[Tuple[str, str, str]] = []
    for feed in sorted(corpus.glob("feeds/*")):
        if feed_filter and feed_filter not in feed.name:
            continue
        runs = sorted(feed.glob("run_*"))
        if not runs:
            continue
        for gi_path in sorted((runs[-1] / "metadata").glob("*.gi.json")):
            gi = _load(gi_path)
            if not isinstance(gi, dict):
                continue
            nodes = {n.get("id"): n for n in gi.get("nodes") or []}
            for edge in gi.get("edges") or []:
                a, b = nodes.get(edge.get("from")) or {}, nodes.get(edge.get("to")) or {}
                insight = (
                    a if a.get("type") == "Insight" else (b if b.get("type") == "Insight" else None)
                )
                quote = a if a.get("type") == "Quote" else (b if b.get("type") == "Quote" else None)
                if not insight or not quote:
                    continue
                itext = str((insight.get("properties") or {}).get("text") or "").strip()
                qtext = str((quote.get("properties") or {}).get("text") or "").strip()
                if itext and qtext:
                    pairs.append((gi_path.name, itext, qtext))
    return pairs


def _negatives(pairs: List[Tuple[str, str, str]], seed: int) -> List[Tuple[str, str]]:
    """Quotes paired with insights from a DIFFERENT episode.

    Same domain, same style, same generator — but genuinely unrelated. A gate that cannot reject
    these is not a gate, and would certify hallucinations (the exact failure ADR-053 exists to
    prevent).
    """
    rng = random.Random(seed)
    out: List[Tuple[str, str]] = []
    for episode, _insight, quote in pairs:
        others = [p for p in pairs if p[0] != episode]
        if others:
            out.append((rng.choice(others)[1], quote))
    return out


def _ask(host: str, model: str, prompt: str, num_ctx: int) -> float:
    try:
        response = httpx.post(
            f"{host}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_ctx": num_ctx},
            },
            timeout=300.0,
        )
        response.raise_for_status()
        raw = str(response.json().get("response", "")).strip()
    except Exception as exc:  # noqa: BLE001
        print(f"    call failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 0.0
    # Reasoning models bury the answer after a thinking block whose digits a naive regex would grab.
    raw = re.sub(r"(?is).*?</think\s*>", "", raw, count=1)
    match = re.search(r"\d*\.?\d+", raw)
    return float(match.group(0)) if match else 0.0


def _render(template: str, insight: str, quote: str) -> str:
    text = template.replace("{{ hypothesis }}", insight).replace("{{ premise }}", quote)
    return re.sub(r"\{\{[^}]+\}\}", "", text)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--silver", type=Path, required=True, help="Corpus the trusted annotator built")
    ap.add_argument("--feed", default="", help="Restrict to feeds whose dir name contains this")
    ap.add_argument("--model", default="qwen3.5:35b")
    ap.add_argument(
        "--prompt",
        type=Path,
        default=Path("src/podcast_scraper/prompts/ollama/evidence/entailment/v1.j2"),
    )
    ap.add_argument("--threshold", type=float, default=0.5, help="gi_nli_entailment_min")
    ap.add_argument("--host", default="http://dgx-llm-1:11434")
    ap.add_argument("--num-ctx", type=int, default=32768)
    ap.add_argument("--limit", type=int, default=40, help="Pairs per class")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    if not args.silver.is_dir() or not args.prompt.is_file():
        print(
            "error: --silver must be a corpus dir and --prompt an existing template",
            file=sys.stderr,
        )
        return 2

    pairs = _silver_pairs(args.silver, args.feed)
    if not pairs:
        print("error: no SUPPORTED_BY edges found in the silver corpus", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    positives = pairs[: args.limit]
    negatives = _negatives(pairs, args.seed)[: args.limit]
    template = args.prompt.read_text()

    print(
        f"silver: {len(pairs)} accepted pairs | testing {len(positives)} pos / {len(negatives)} neg"
    )
    print(f"model: {args.model}  prompt: {args.prompt.name}  threshold: {args.threshold}\n")

    accepted_pos = 0
    for _episode, insight, quote in positives:
        score = _ask(args.host, args.model, _render(template, insight, quote), args.num_ctx)
        accepted_pos += score >= args.threshold

    rejected_neg = 0
    for insight, quote in negatives:
        score = _ask(args.host, args.model, _render(template, insight, quote), args.num_ctx)
        rejected_neg += score < args.threshold

    recall = 100.0 * accepted_pos / len(positives) if positives else 0.0
    rejection = 100.0 * rejected_neg / len(negatives) if negatives else 0.0

    print(f"  recall    {recall:5.1f}%  of the annotator's evidence it accepts  (gemini: ~66%)")
    print(f"  rejection {rejection:5.1f}%  of unrelated pairs it correctly rejects")
    print()
    print(
        "  Maximise BOTH. High recall with low rejection is a gate that certifies hallucinations;"
    )
    print("  high rejection with low recall is today's failure (6% -> 13.3% grounded).")

    payload = {
        "model": args.model,
        "prompt": str(args.prompt),
        "threshold": args.threshold,
        "positives": len(positives),
        "negatives": len(negatives),
        "recall_pct": round(recall, 1),
        "rejection_pct": round(rejection, 1),
    }
    print()
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
