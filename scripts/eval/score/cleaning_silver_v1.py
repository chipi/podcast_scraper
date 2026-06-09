"""Generate Sonnet 4.6 silver cleaned transcripts for #594.

For each input transcript, call Anthropic Sonnet 4.6 with the canonical cleaning
system prompt (mirrors `providers/anthropic/anthropic_provider.py::clean_transcript`)
and save the cleaned output to disk. Each provider variant in the sweep is later
scored against this silver.

Sonnet 4.6 was the v1↔v2 silver winner across all four cells (PR #918 / #903 eval
report) — using it as the gold for cleaning quality is consistent with how the
summarization silver was picked.

Usage:
    python scripts/eval/score/cleaning_silver_v1.py \\
        --sources data/eval/sources/curated_5feeds_raw_v2 \\
        --episodes p01_e01 p02_e01 p03_e01 p04_e01 p05_e01 \\
        --output data/eval/references/silver/cleaning_v1
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

CLEANING_SYSTEM = (
    "You are a transcript cleaning assistant. "
    "Remove sponsors, ads, intros, outros, and meta-commentary. "
    "Preserve all substantive content and speaker information. "
    "Return only the cleaned text, no explanations."
)

CLEANING_USER_TMPL = (
    "Clean the following podcast transcript. Apply the rules in the system "
    "prompt. Return ONLY the cleaned transcript text (no preamble, no markdown "
    "fences, no commentary).\n\n"
    "Transcript:\n```\n{transcript}\n```"
)


def clean_one(client, transcript: str) -> str:
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8000,
        temperature=0.0,
        system=CLEANING_SYSTEM,
        messages=[{"role": "user", "content": CLEANING_USER_TMPL.format(transcript=transcript)}],
    )
    return resp.content[0].text.strip()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--episodes", nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 1
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    args.output.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for ep in args.episodes:
        # Find transcript file (search under feed-* subdirs)
        candidates = list(args.sources.rglob(f"{ep}.txt"))
        if not candidates:
            print(f"  SKIP {ep}: no transcript", file=sys.stderr)
            continue
        transcript = candidates[0].read_text(encoding="utf-8")
        cleaned = clean_one(client, transcript)
        out = args.output / f"{ep}.silver.txt"
        out.write_text(cleaned, encoding="utf-8")
        elapsed = time.time() - t0
        print(
            f"  {ep}: raw={len(transcript)} → silver={len(cleaned)} "
            f"({100*len(cleaned)/max(len(transcript),1):.0f}%) ({elapsed:.1f}s)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
