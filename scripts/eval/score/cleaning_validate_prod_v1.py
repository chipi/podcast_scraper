"""Validate #594 picks against real-prod transcripts.

Picks 3 manual-run-10 episodes, cleans each with both the OLD default
(temp=0.2) and NEW default (temp=0.4) for each provider's recommended model,
then has Sonnet pairwise-judge: does the new temp produce as-good-or-better
cleaning on real prod? If t=0.4 wins or ties on real podcast content, the
temperature bump is safe.

Usage:
    python scripts/eval/score/cleaning_validate_prod_v1.py \\
        --prod-transcripts-dir <path> \\
        --output data/eval/runs/baseline_cleaning_validate_prod_v1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Re-use callers + prompts from the sweep script
import importlib.util

_SWEEP = PROJECT_ROOT / "scripts" / "eval" / "score" / "cleaning_sweep_v1.py"
_spec = importlib.util.spec_from_file_location("cleaning_sweep_v1", _SWEEP)
assert _spec and _spec.loader
sweep = importlib.util.module_from_spec(_spec)
sys.modules["cleaning_sweep_v1"] = sweep
_spec.loader.exec_module(sweep)

JUDGE_SYSTEM = (
    "You are evaluating two candidate cleanings (A and B) of the same real-prod "
    "podcast transcript. Better cleaning means: removed sponsors / ads / intros / "
    "outros / meta-commentary, preserved substantive content, kept speaker labels "
    "intact, did NOT invent or paraphrase content, did NOT over-trim real material. "
    'Pick the better one. Reply STRICT JSON: {"winner": "A"|"B"|"TIE", "reason": "..."}'
)


# Per-provider: (model, old_temp, new_temp)
RECOMMENDATIONS = [
    ("openai", "gpt-4o-mini", 0.2, 0.4),
    ("anthropic", "claude-haiku-4-5", 0.2, 0.4),
    ("gemini", "gemini-2.5-flash-lite", 0.2, 0.4),
]


def judge(client: Any, transcript: str, a: str, b: str) -> dict[str, Any]:
    msg = (
        f"Real-prod transcript (truncated to 4000 chars):\n```\n{transcript[:4000]}\n```\n\n"
        f"Cleaning A (truncated):\n```\n{a[:4000]}\n```\n\n"
        f"Cleaning B (truncated):\n```\n{b[:4000]}\n```\n\n"
        "Which is better? Reply STRICT JSON only."
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
    p.add_argument("--prod-transcripts-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--episodes",
        nargs="+",
        default=[
            "0001 - Boing_ Springtime for the market",
            "0023 - The incredible shrinking dollar",
            "0024 - How does a gold rush end_",
        ],
    )
    args = p.parse_args()

    clients = sweep.build_clients()
    if "anthropic" not in clients:
        print("ANTHROPIC_API_KEY required for judge", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    t0 = time.time()
    for ep_prefix in args.episodes:
        matches = list(args.prod_transcripts_dir.glob(f"{ep_prefix}*.txt"))
        if not matches:
            print(f"  SKIP {ep_prefix!r}: no matching file", file=sys.stderr)
            continue
        transcript = matches[0].read_text(encoding="utf-8")
        # Cap at 12k chars to keep judge prompt size bounded
        if len(transcript) > 12000:
            transcript = transcript[:12000]
        ep_short = ep_prefix.split(" - ")[0].strip()

        for provider, model, old_temp, new_temp in RECOMMENDATIONS:
            if provider not in clients:
                continue
            client = clients[provider]
            caller = sweep.CALLERS[provider]
            try:
                a_text = caller(client, model, transcript, old_temp)
            except Exception as exc:  # noqa: BLE001
                print(f"  ERR {provider} t={old_temp} on {ep_short}: {exc}", file=sys.stderr)
                continue
            try:
                b_text = caller(client, model, transcript, new_temp)
            except Exception as exc:  # noqa: BLE001
                print(f"  ERR {provider} t={new_temp} on {ep_short}: {exc}", file=sys.stderr)
                continue

            verdict = judge(clients["anthropic"], transcript, a_text, b_text)
            row = {
                "episode": ep_short,
                "provider": provider,
                "model": model,
                "old_temp": old_temp,
                "new_temp": new_temp,
                "raw_chars": len(transcript),
                "old_cleaned_chars": len(a_text),
                "new_cleaned_chars": len(b_text),
                "winner": verdict.get("winner"),
                "reason": verdict.get("reason"),
            }
            rows.append(row)
            print(
                f"  [{time.time()-t0:6.1f}s] {ep_short:5s} {provider:10s} "
                f"OLD(t={old_temp}, {len(a_text)}c) vs NEW(t={new_temp}, {len(b_text)}c) "
                f"-> winner={verdict.get('winner')}"
            )

    # Aggregate
    summary: dict[str, dict[str, int]] = defaultdict(lambda: {"old": 0, "new": 0, "tie": 0})
    for r in rows:
        s = summary[r["provider"]]
        w = r["winner"]
        if w == "A":
            s["old"] += 1
        elif w == "B":
            s["new"] += 1
        elif w == "TIE":
            s["tie"] += 1
    payload = {
        "schema": "metrics_cleaning_validate_prod_v1",
        "rows": rows,
        "summary": dict(summary),
        "verdict": (
            "NEW temp safe to adopt (won or tied every prod episode)"
            if all(s["old"] == 0 for s in summary.values())
            else "MIXED — NEW temp lost on at least one prod episode; review per-row"
        ),
    }
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nPer-provider OLD(t=0.2) vs NEW(t=0.4):")
    for prov, s in sorted(summary.items()):
        print(f"  {prov:12s}  OLD wins={s['old']}  NEW wins={s['new']}  ties={s['tie']}")
    print(f"\nverdict: {payload['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
