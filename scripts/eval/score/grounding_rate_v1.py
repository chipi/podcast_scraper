#!/usr/bin/env python3
"""Score a corpus against the ADR-053 grounding contract (#1179).

ADR-053 defines when an insight is trustworthy: it is *grounded* when at least one verbatim quote
from the transcript supports it, after low-confidence candidates are filtered. It sets a hard
target — **>= 80% of insights grounded** — and says episodes below it are flagged `gil_quality:
"low"`. Without grounding, the ADR notes, "insights are indistinguishable from hallucination".

That target is the objective function for tuning the evidence stage. Nothing was measuring it, so
the DGX pilot shipped a corpus grounding 5% of its insights while reporting success.

Measured on the same episodes:

    v2 cloud (gemini):  91.3%   PASS
    v3 DGX  (qwen):     13.3%   FAIL

Reports the whole **funnel**, not just the survivors, because "grounding is low" is not actionable
and the artifacts only contain what lived. Reading the run metrics gives the full attrition, and it
is the only view that says *which stage* is losing the evidence:

    insights                     claims made
    candidates reaching NLI      the extractor + QA gate found evidence at all
    quotes surviving NLI         the entailment gate accepted it
    grounded                     >=1 surviving quote (ADR-053)

That view is what located the DGX pilot's failure. Per insight:

                          gemini      qwen
    candidates -> NLI       3.21      2.77     <- qwen's supply is FINE (86% of gemini's)
    surviving quotes        2.13      0.17     <- 66% pass vs 6% pass
    grounded               91.3%     13.3%

Extraction, the QA gate, and verbatim-ness (100% in both) are all healthy. The entire collapse is
the entailment gate, with an 11x lower pass rate on candidates qwen already found. Scoring only the
survivors would have blamed the extractor.

Usage:
    python scripts/eval/score/grounding_rate_v1.py <corpus_root> [--feed SUBSTR]

Exit codes: 0 at/above target · 1 below target · 2 input error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ADR-053: "Quality target: >=80% of insights should be grounded in a well-processed episode."
_TARGET_PCT = 80.0


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _score_episode(gi_path: Path, transcript: Optional[str]) -> Optional[Dict[str, Any]]:
    gi = _load(gi_path)
    if not isinstance(gi, dict):
        return None
    nodes = gi.get("nodes") or []
    insights = [n for n in nodes if n.get("type") == "Insight"]
    if not insights:
        return None
    quotes = [n for n in nodes if n.get("type") == "Quote"]

    grounded = sum(1 for n in insights if (n.get("properties") or {}).get("grounded") is True)

    # A quote that cannot be located in the transcript cannot ground anything (ADR-053).
    verbatim = 0
    for q in quotes:
        text = str((q.get("properties") or {}).get("text") or "").strip()
        if text and transcript and text in transcript:
            verbatim += 1

    return {
        "episode": gi_path.name,
        "insights": len(insights),
        "quotes": len(quotes),
        "quotes_verbatim": verbatim,
        "grounded": grounded,
        "grounded_pct": round(100.0 * grounded / len(insights), 1),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--feed", default="", help="Only score feeds whose dir name contains this")
    ap.add_argument("--json", action="store_true", help="Emit JSON (for the tuning harness)")
    args = ap.parse_args(argv)

    if not args.corpus.is_dir():
        print(f"error: no such corpus: {args.corpus}", file=sys.stderr)
        return 2

    rows: List[Dict[str, Any]] = []
    for feed in sorted(args.corpus.glob("feeds/*")):
        if args.feed and args.feed not in feed.name:
            continue
        runs = sorted(feed.glob("run_*"))
        if not runs:
            continue
        run = runs[-1]  # newest run is what the corpus serves
        for gi_path in sorted((run / "metadata").glob("*.gi.json")):
            stem = gi_path.name[: -len(".gi.json")]
            tx = run / "transcripts" / f"{stem}.txt"
            transcript = tx.read_text(encoding="utf-8", errors="replace") if tx.is_file() else None
            row = _score_episode(gi_path, transcript)
            if row:
                rows.append(row)

    if not rows:
        print("error: no gi.json artifacts found", file=sys.stderr)
        return 2

    insights = sum(r["insights"] for r in rows)
    grounded = sum(r["grounded"] for r in rows)
    quotes = sum(r["quotes"] for r in rows)
    verbatim = sum(r["quotes_verbatim"] for r in rows)
    rate = 100.0 * grounded / insights if insights else 0.0

    summary = {
        "episodes": len(rows),
        "insights": insights,
        "grounded": grounded,
        "grounded_pct": round(rate, 1),
        "target_pct": _TARGET_PCT,
        "meets_target": rate >= _TARGET_PCT,
        "quotes": quotes,
        "quotes_verbatim": verbatim,
        "quotes_verbatim_pct": round(100.0 * verbatim / quotes, 1) if quotes else 0.0,
    }

    if args.json:
        print(json.dumps({"summary": summary, "episodes": rows}, indent=2))
    else:
        print(f"episodes: {len(rows)}   insights: {insights}   quotes: {quotes}")
        print()
        verbatim_pct = summary["quotes_verbatim_pct"]
        print(f"  quotes verbatim   {verbatim_pct:5.1f}%   (locatable in transcript)")
        print(
            f"  GROUNDED          {rate:5.1f}%   target >= {_TARGET_PCT:.0f}%   "
            f"{'PASS' if summary['meets_target'] else 'FAIL'}"
        )
        print()
        print("  ADR-053: an insight is grounded when >=1 verbatim quote supports it, after the")
        print("  low-confidence filter. Below target, the episode is gil_quality: 'low'.")

    return 0 if summary["meets_target"] else 1


if __name__ == "__main__":
    sys.exit(main())
