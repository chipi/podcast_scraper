"""Broader cleaning_v3 vs cleaning_v4 pairwise judge — #989 acceptance gate.

#905 Tier 2 found cleaning_v3 wins 10W-0L-5T vs cleaning_v4 on 5 v2 episodes
under a single Sonnet 4.6 judge. #989 requires a broader 15-20 episode
sample before flipping the production default — to confirm the lift survives
sample-size noise. This script is that gate.

## Method

For each of 15 v2 episodes (p[1-5]_e[1-3]) the script:

1. Reads the raw transcript source from
   ``data/eval/sources/curated_5feeds_raw_v2/feed-<feed>/<episode>.txt``.
2. Applies ``cleaning_v3`` and ``cleaning_v4`` from the registered
   ``preprocessing.profiles``.
3. Asks Sonnet 4.6 to pairwise-judge A vs B (anonymised) against the same
   rubric #905 used (remove sponsor/ad/intro/outro/meta; preserve
   substantive content; no invention/paraphrase).
4. Aggregates wins / losses / ties.

Each episode is judged twice with A/B order swapped to neutralise
position bias. The headline "v3 wins" number is the avg over both
positionings.

Acceptance gate per #989: cleaning_v3 wins ≥ 60% (9/15 episodes).

## Cost / time

15 episodes × 2 orderings × 1 Sonnet 4.6 call ≈ 30 calls × ~$0.02 = ~$0.60.
Wall-clock ~3 minutes sequential.

## Usage

    export $(grep -E '^ANTHROPIC_API_KEY=' .env)
    PYTHONPATH=. .venv/bin/python \\
        scripts/eval/score/cleaning_v3_vs_v4_broader_judge_v1.py \\
        --sources data/eval/sources/curated_5feeds_raw_v2 \\
        --output  data/eval/runs/cleaning_v3_vs_v4_broader_judge_v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from podcast_scraper.preprocessing import profiles as profile_mod  # noqa: E402

_EPISODES = [f"p{feed:02d}_e{ep:02d}" for feed in range(1, 6) for ep in range(1, 4)]

_JUDGE_SYSTEM = (
    "You are evaluating two candidate cleanings (A and B) of the same podcast "
    "transcript. Better cleaning means: removed all sponsor / ad / intro / "
    "outro / meta-commentary, preserved all substantive content, kept speaker "
    "labels intact, did NOT invent or paraphrase content. "
    "Pick the better one. Reply STRICT JSON: "
    '{"winner": "A" | "B" | "TIE", "reason": "<one short sentence>"}'
)


def _judge_pair(client: Any, transcript: str, cleaning_a: str, cleaning_b: str) -> Dict[str, Any]:
    msg = (
        f"Original transcript (first 4000 chars):\n```\n{transcript[:4000]}\n```\n\n"
        f"Cleaning A (first 4000 chars):\n```\n{cleaning_a[:4000]}\n```\n\n"
        f"Cleaning B (first 4000 chars):\n```\n{cleaning_b[:4000]}\n```\n\n"
        "Which cleaning is better? Reply STRICT JSON only."
    )
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        temperature=0.0,
        system=_JUDGE_SYSTEM,
        messages=[{"role": "user", "content": msg}],
    )
    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.strip("`").lstrip("json").strip()
    try:
        return dict(json.loads(text))
    except json.JSONDecodeError:
        return {"winner": "PARSE_ERROR", "reason": text[:120]}


def _resolve_source(sources_root: Path, episode: str) -> Path:
    feed = episode.split("_")[0]
    return sources_root / f"feed-{feed}" / f"{episode}.txt"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--episodes", nargs="*", default=_EPISODES)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY env var required", file=sys.stderr)
        return 2

    try:
        from anthropic import Anthropic
    except ImportError:
        print("ERROR: anthropic SDK required (.venv/bin/pip install anthropic)", file=sys.stderr)
        return 2
    client = Anthropic(api_key=api_key)

    args.output.mkdir(parents=True, exist_ok=True)
    per_episode: List[Dict[str, Any]] = []

    for ep in args.episodes:
        source_path = _resolve_source(args.sources, ep)
        if not source_path.is_file():
            print(f"  SKIP {ep}: source not found at {source_path}", file=sys.stderr)
            continue
        transcript = source_path.read_text(encoding="utf-8")
        cleaned_v3 = profile_mod.apply_profile(transcript, "cleaning_v3")
        cleaned_v4 = profile_mod.apply_profile(transcript, "cleaning_v4")

        # Order 1: A=v3, B=v4
        v1 = _judge_pair(client, transcript, cleaned_v3, cleaned_v4)
        time.sleep(0.5)
        # Order 2: A=v4, B=v3 (swap to neutralize positional bias)
        v2 = _judge_pair(client, transcript, cleaned_v4, cleaned_v3)
        time.sleep(0.5)

        # Translate verdicts to v3/v4/TIE.
        def _decode(verdict: Dict[str, Any], a_is_v3: bool) -> str:
            w = verdict.get("winner", "PARSE_ERROR")
            if w == "TIE":
                return "TIE"
            if w == "A":
                return "v3" if a_is_v3 else "v4"
            if w == "B":
                return "v4" if a_is_v3 else "v3"
            return "PARSE_ERROR"

        verdicts = {"order1": _decode(v1, a_is_v3=True), "order2": _decode(v2, a_is_v3=False)}
        # Consensus: if both orderings agree, that's the verdict. If they disagree, TIE.
        consensus = (
            verdicts["order1"] if verdicts["order1"] == verdicts["order2"] else "TIE_POSITIONAL"
        )
        per_episode.append(
            {
                "episode_id": ep,
                "v3_chars": len(cleaned_v3),
                "v4_chars": len(cleaned_v4),
                "v3_chars_removed": len(transcript) - len(cleaned_v3),
                "v4_chars_removed": len(transcript) - len(cleaned_v4),
                "verdicts": verdicts,
                "v1_reason": v1.get("reason"),
                "v2_reason": v2.get("reason"),
                "consensus": consensus,
            }
        )
        print(
            f"  {ep}: order1={verdicts['order1']:>3s} order2={verdicts['order2']:>3s} "
            f"consensus={consensus}"
        )

    counter = Counter(row["consensus"] for row in per_episode)
    total = len(per_episode) or 1
    v3_rate = counter.get("v3", 0) / total
    v4_rate = counter.get("v4", 0) / total
    tie_rate = (counter.get("TIE", 0) + counter.get("TIE_POSITIONAL", 0)) / total
    decision_v3_wins = counter.get("v3", 0) >= max(1, int(0.6 * total))

    summary = {
        "episodes_judged": total,
        "v3_wins": counter.get("v3", 0),
        "v4_wins": counter.get("v4", 0),
        "ties": counter.get("TIE", 0),
        "tie_positional": counter.get("TIE_POSITIONAL", 0),
        "parse_errors": counter.get("PARSE_ERROR", 0),
        "v3_win_rate": v3_rate,
        "v4_win_rate": v4_rate,
        "tie_rate": tie_rate,
        "passes_60pct_gate": decision_v3_wins,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(
            {
                "schema": "cleaning_v3_vs_v4_broader_judge_v1",
                "summary": summary,
                "rows": per_episode,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    n_ties = counter.get("TIE", 0) + counter.get("TIE_POSITIONAL", 0)
    gate = "PASS" if decision_v3_wins else "FAIL"
    flip = "is" if decision_v3_wins else "is NOT"
    print()
    print(f"v3 wins:  {counter.get('v3', 0)}/{total} ({v3_rate:.1%})")
    print(f"v4 wins:  {counter.get('v4', 0)}/{total} ({v4_rate:.1%})")
    print(f"ties:     {n_ties}/{total} ({tie_rate:.1%})")
    print(f"60% gate: {gate} — production flip to v3 {flip} indicated")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
