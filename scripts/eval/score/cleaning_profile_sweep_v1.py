"""Cleaning-profile selection sweep — #905 Phase A.

Runs each of the 6 registered `preprocessing.profiles` over the v2 smoke
sources + 3 real-prod episodes; computes pattern-recall metrics (cheap) and
LLM-judge pairwise comparison against the #594 Sonnet silver (expensive but
decisive).

Output: per-profile per-episode metrics + a head-to-head ranking, so the
default profile choice has evidence under it.

Usage:
    python scripts/eval/score/cleaning_profile_sweep_v1.py \\
        --sources data/eval/sources/curated_5feeds_raw_v2 \\
        --silver  data/eval/references/silver/cleaning_v1 \\
        --episodes p01_e01 p02_e01 p03_e01 p04_e01 p05_e01 \\
        --output  data/eval/runs/baseline_cleaning_profile_sweep_v1
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from podcast_scraper.cleaning.commercial.patterns import SPONSOR_PATTERNS
from podcast_scraper.preprocessing import profiles as profile_mod

PROFILES = [
    "cleaning_none",
    "cleaning_v1",
    "cleaning_v2",
    "cleaning_v3",
    "cleaning_v4",
    "cleaning_hybrid_after_pattern",
]

# Block-end patterns are transition markers, not content. Don't count them.
_CONTENT_PATTERNS = [p for p in SPONSOR_PATTERNS if p.boundary_hint != "block_end"]


def _content_hits(text: str) -> int:
    return sum(sum(1 for _ in pat.pattern.finditer(text)) for pat in _CONTENT_PATTERNS)


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def judge_pair(client: Any, transcript: str, a: str, b: str) -> dict[str, Any]:
    """Sonnet pairwise: which of two cleaned outputs is better."""
    sys_prompt = (
        "You are evaluating two cleaned versions (A and B) of the same podcast "
        "transcript. Better cleaning means: removed sponsor / ad / intro / outro / "
        "meta-commentary, preserved substantive content, kept speaker labels "
        "intact, did not invent or paraphrase. Pick the better one. "
        'Reply STRICT JSON: {"winner": "A"|"B"|"TIE", "reason": "<one short sentence>"}'
    )
    user_prompt = (
        f"Original transcript (truncated to 4000 chars):\n```\n{transcript[:4000]}\n```\n\n"
        f"Cleaning A (truncated):\n```\n{a[:4000]}\n```\n\n"
        f"Cleaning B (truncated):\n```\n{b[:4000]}\n```\n\n"
        "Which is better? Reply STRICT JSON only."
    )
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        temperature=0.0,
        system=sys_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = resp.content[0].text.strip()
    if text.startswith("```"):
        text = text.strip("`").lstrip("json").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"winner": "PARSE_ERROR", "reason": text}


def main() -> int:  # noqa: C901 — orchestrator, simple control flow
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--silver", type=Path, required=True)
    p.add_argument("--episodes", nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip the LLM-judge pass (cheap-metrics-only run)",
    )
    args = p.parse_args()

    transcripts: dict[str, str] = {}
    silvers: dict[str, str] = {}
    for ep in args.episodes:
        srcs = list(args.sources.rglob(f"{ep}.txt"))
        if not srcs:
            print(f"  SKIP {ep}: no source", file=sys.stderr)
            continue
        transcripts[ep] = srcs[0].read_text(encoding="utf-8")
        sil = args.silver / f"{ep}.silver.txt"
        if sil.exists():
            silvers[ep] = sil.read_text(encoding="utf-8")

    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    cleaned_outputs: dict[tuple[str, str], str] = {}

    for profile_id in PROFILES:
        try:
            fn = profile_mod.get_profile(profile_id)
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIP profile {profile_id}: {exc}", file=sys.stderr)
            continue
        if fn is None:
            continue
        for ep, raw in transcripts.items():
            try:
                cleaned = fn(raw)
            except Exception as exc:  # noqa: BLE001
                rows.append({"profile": profile_id, "episode_id": ep, "error": str(exc)})
                continue
            cleaned_outputs[(profile_id, ep)] = cleaned
            silver = silvers.get(ep, "")
            rows.append(
                {
                    "profile": profile_id,
                    "episode_id": ep,
                    "raw_chars": len(raw),
                    "cleaned_chars": len(cleaned),
                    "chars_removed_pct": round(
                        100 * (len(raw) - len(cleaned)) / max(len(raw), 1), 2
                    ),
                    "content_hits_raw": _content_hits(raw),
                    "content_hits_cleaned": _content_hits(cleaned),
                    "similarity_to_silver": (
                        round(_similarity(cleaned, silver), 4) if silver else None
                    ),
                }
            )

    # Cheap-metric summary per profile
    by_profile: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if "error" in r:
            continue
        by_profile[r["profile"]].append(r)

    print(
        f"{'profile':<32} {'chars%':>7} {'content_hits':>12} " f"{'recall%':>8} {'sim_silver':>10}"
    )
    summary: list[dict[str, Any]] = []
    for prof in PROFILES:
        rs = by_profile.get(prof, [])
        if not rs:
            continue
        mean_pct = sum(r["chars_removed_pct"] for r in rs) / len(rs)
        mean_resid = sum(r["content_hits_cleaned"] for r in rs) / len(rs)
        raw_total = sum(r["content_hits_raw"] for r in rs)
        residual_total = sum(r["content_hits_cleaned"] for r in rs)
        recall = round(100 * (raw_total - residual_total) / raw_total, 1) if raw_total else 0.0
        sims = [r["similarity_to_silver"] for r in rs if r["similarity_to_silver"] is not None]
        mean_sim = round(sum(sims) / len(sims), 4) if sims else None
        summary.append(
            {
                "profile": prof,
                "mean_chars_removed_pct": round(mean_pct, 2),
                "mean_content_residual": round(mean_resid, 2),
                "content_recall_pct": recall,
                "mean_sim_to_silver": mean_sim,
            }
        )
        print(
            f"{prof:<32} {mean_pct:>7.2f} {mean_resid:>12.2f} "
            f"{recall:>8.1f} {str(mean_sim):>10}"
        )

    # Pairwise judge — only the profiles with non-zero recall against silver
    judge_results: list[dict[str, Any]] = []
    judge_ranking: dict[str, dict[str, int]] = {}
    if not args.skip_judge and silvers:
        if "ANTHROPIC_API_KEY" not in os.environ:
            print(
                "ANTHROPIC_API_KEY not set; skipping judge pass",
                file=sys.stderr,
            )
        else:
            from anthropic import Anthropic

            client = Anthropic()
            # Restrict tournament to profiles that actually clean something
            candidate_profiles = [
                s["profile"]
                for s in summary
                if s["mean_chars_removed_pct"] > 1.0  # ignore cleaning_none
            ]
            judge_ranking = {p: {"wins": 0, "losses": 0, "ties": 0} for p in candidate_profiles}
            t0 = time.time()
            for ep in args.episodes:
                if ep not in transcripts:
                    continue
                transcript = transcripts[ep]
                for i, a in enumerate(candidate_profiles):
                    for b in candidate_profiles[i + 1 :]:
                        if (a, ep) not in cleaned_outputs or (b, ep) not in cleaned_outputs:
                            continue
                        verdict = judge_pair(
                            client,
                            transcript,
                            cleaned_outputs[(a, ep)],
                            cleaned_outputs[(b, ep)],
                        )
                        w = verdict.get("winner")
                        judge_results.append({"episode": ep, "a": a, "b": b, **verdict})
                        if w == "A":
                            judge_ranking[a]["wins"] += 1
                            judge_ranking[b]["losses"] += 1
                        elif w == "B":
                            judge_ranking[b]["wins"] += 1
                            judge_ranking[a]["losses"] += 1
                        elif w == "TIE":
                            judge_ranking[a]["ties"] += 1
                            judge_ranking[b]["ties"] += 1
                        print(f"  [{time.time()-t0:6.1f}s] {ep} {a:32s} vs {b:32s} -> {w}")

    payload = {
        "schema": "metrics_cleaning_profile_sweep_v1",
        "summary": summary,
        "rows": rows,
        "judge_results": judge_results,
        "judge_ranking": judge_ranking,
    }
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if judge_ranking:
        ranked = sorted(
            judge_ranking.items(),
            key=lambda x: (-(x[1]["wins"] - x[1]["losses"]), -x[1]["wins"]),
        )
        print("\nPairwise ranking (net = wins - losses):")
        for prof, r in ranked:
            net = r["wins"] - r["losses"]
            print(
                f"  {prof:32s}  W={r['wins']:2d}  L={r['losses']:2d}  "
                f"T={r['ties']:2d}  net={net:+d}"
            )

    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
