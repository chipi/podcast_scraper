"""Pairwise judge for #594 cleaning autoresearch.

For each episode, picks the best (temperature) cell per provider, then runs a
Sonnet-judged pairwise tournament between provider winners. Output:
- per-provider best temperature
- per-episode pairwise verdicts
- overall provider ranking (Bradley-Terry-ish: wins minus losses)

Judge is given both cleaned outputs (anonymised as A/B) plus the original
transcript and the canonical cleaning rubric. Asked to pick the better
output or declare a tie.

Usage:
    python scripts/eval/score/cleaning_judge_v1.py \\
        --sweep-output data/eval/runs/baseline_cleaning_autoresearch_v1 \\
        --sources data/eval/sources/curated_5feeds_raw_v2 \\
        --episodes p01_e01 p02_e01 p03_e01 p04_e01 p05_e01 \\
        --output data/eval/runs/baseline_cleaning_autoresearch_v1/judge.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

JUDGE_SYSTEM = (
    "You are evaluating two candidate cleanings (A and B) of the same podcast "
    "transcript. Better cleaning means: removed all sponsor / ad / intro / "
    "outro / meta-commentary, preserved all substantive content, kept speaker "
    "labels intact, did NOT invent or paraphrase content. "
    "Pick the better one. Reply STRICT JSON: "
    '{"winner": "A" | "B" | "TIE", "reason": "<one short sentence>"}'
)


def judge_pair(client: Any, transcript: str, a: str, b: str) -> dict[str, Any]:
    msg = (
        f"Original transcript (truncated to 4000 chars):\n```\n{transcript[:4000]}\n```\n\n"
        f"Cleaning A (truncated):\n```\n{a[:4000]}\n```\n\n"
        f"Cleaning B (truncated):\n```\n{b[:4000]}\n```\n\n"
        "Which cleaning is better? Reply STRICT JSON only."
    )
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        temperature=0.0,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": msg}],
    )
    text = resp.content[0].text.strip()
    if text.startswith("```"):
        text = text.strip("`").lstrip("json").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"winner": "PARSE_ERROR", "reason": text}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep-output", type=Path, required=True)
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--episodes", nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    # Load per-cell metrics
    rows = [
        json.loads(line)
        for line in (args.sweep_output / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    rows = [r for r in rows if "error" not in r]

    # Per-provider best temperature: maximise similarity_to_silver
    # (tie-break by min sponsor residual)
    by_pmodel: dict[tuple[str, str], dict[float, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        by_pmodel[(r["provider"], r["model"])][r["temperature"]].append(r["similarity_to_silver"])
    best_temp_per_pmodel: dict[tuple[str, str], float] = {}
    for k, by_t in by_pmodel.items():
        scored = sorted(
            ((t, sum(v) / len(v)) for t, v in by_t.items()),
            key=lambda x: -x[1],
        )
        best_temp_per_pmodel[k] = scored[0][0]

    # Per-provider best model: maximise mean similarity at its best temp
    best_per_provider: dict[str, tuple[str, float, float]] = {}
    for (prov, model), best_t in best_temp_per_pmodel.items():
        mean_sim = sum(by_pmodel[(prov, model)][best_t]) / len(by_pmodel[(prov, model)][best_t])
        if prov not in best_per_provider or mean_sim > best_per_provider[prov][2]:
            best_per_provider[prov] = (model, best_t, mean_sim)

    print("Per-provider best (model, temp, similarity_to_silver):")
    for prov, (model, temp, sim) in sorted(best_per_provider.items()):
        print(f"  {prov:10s}  {model:30s}  t={temp}  sim={sim:.4f}")

    # Tournament: for each episode, pairwise compare all provider winners
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 1
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)

    wins_losses: dict[str, dict[str, int]] = defaultdict(
        lambda: {"wins": 0, "losses": 0, "ties": 0}
    )
    verdicts: list[dict[str, Any]] = []

    t0 = time.time()
    for ep in args.episodes:
        # Load source transcript
        srcs = list(args.sources.rglob(f"{ep}.txt"))
        if not srcs:
            continue
        transcript = srcs[0].read_text(encoding="utf-8")

        # Load each provider's best cleaning for this episode
        cleanings: dict[str, str] = {}
        for prov, (model, temp, _) in best_per_provider.items():
            slug = f"{prov}__{model.replace('/', '-').replace(':', '-')}__t{temp}"
            cleaned_path = args.sweep_output / slug / f"{ep}.cleaned.txt"
            if cleaned_path.exists():
                cleanings[prov] = cleaned_path.read_text(encoding="utf-8")

        for a_prov, b_prov in itertools.combinations(sorted(cleanings), 2):
            verdict = judge_pair(client, transcript, cleanings[a_prov], cleanings[b_prov])
            verdicts.append(
                {
                    "episode_id": ep,
                    "a_provider": a_prov,
                    "b_provider": b_prov,
                    "winner": verdict.get("winner"),
                    "reason": verdict.get("reason"),
                }
            )
            w = verdict.get("winner")
            if w == "A":
                wins_losses[a_prov]["wins"] += 1
                wins_losses[b_prov]["losses"] += 1
            elif w == "B":
                wins_losses[b_prov]["wins"] += 1
                wins_losses[a_prov]["losses"] += 1
            elif w == "TIE":
                wins_losses[a_prov]["ties"] += 1
                wins_losses[b_prov]["ties"] += 1
            print(f"  [{time.time()-t0:6.1f}s] {ep} {a_prov:10s} vs {b_prov:10s} -> {w}")

    ranking = sorted(
        ((p, c) for p, c in wins_losses.items()),
        key=lambda x: (-(x[1]["wins"] - x[1]["losses"]), -x[1]["wins"]),
    )

    payload = {
        "schema": "metrics_cleaning_autoresearch_judge_v1",
        "best_per_provider": {
            p: {"model": m, "temperature": t, "mean_sim_to_silver": round(s, 4)}
            for p, (m, t, s) in best_per_provider.items()
        },
        "ranking": [{"provider": p, **c, "net_score": c["wins"] - c["losses"]} for p, c in ranking],
        "verdicts": verdicts,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nProvider ranking (wins - losses):")
    for p, c in ranking:
        print(f"  {p:10s}  W={c['wins']:2d}  L={c['losses']:2d}  T={c['ties']:2d}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
