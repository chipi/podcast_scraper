# flake8: noqa: E501  -- prompt string literals are long by intent
"""Prompt v1↔v2 validation for #906 Phase C (focused, not full RFC-057 sweep).

Question: does the current production paragraph-summary prompt (tuned on v1)
still win on v2 inputs, or does a v2-aware variant (mentioning recurring
guests + position arcs explicitly) produce judge-preferred summaries?

Approach:
- For each v2 smoke episode, generate two Sonnet 4.6 summaries:
  A = current production prompt (`anthropic/summarization/long_v1.j2`)
  B = v2-aware variant — same prompt + two extra bullets that explicitly call
      out v2's recurring-guests + position-arcs patterns
- Sonnet pairwise judge picks the better summary per episode
- Aggregate: if B wins or ties, recommend shipping the v2-aware variant; if
  A wins, the v1 prompt is robust to v2 inputs.

The full RFC-057 prompt sweep (many template tweaks × ratchet loop) is
deferred — this is the narrowest test to settle "v1 prompt vs v2-aware
variant" before deciding whether broader re-tuning is worth the spend.

Usage:
    python scripts/eval/score/prompt_v2_validation_v1.py \\
        --sources data/eval/sources/curated_5feeds_raw_v2 \\
        --episodes p01_e01 p02_e01 p03_e01 p04_e01 p05_e01 \\
        --output  data/eval/runs/baseline_prompt_v2_validation_v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = (
    "You are an expert at creating concise, informative summaries of podcast "
    "episodes. Focus on key insights, decisions, and lessons learned."
)

PROMPT_V1 = """Summarize the following podcast episode transcript.
- Write a detailed summary with 4-6 paragraphs
- Begin the first paragraph with a single sentence naming the episode's domain and its central arg or premise
- Cover ALL major discussion segments in the order they appear in the transcript
- Preserve key technical terms, concept names, product names, and specific vocabulary from the transcript verbatim — do not paraphrase or substitute synonyms for named concepts
- Anchor each paragraph in specific claims, data points, or named entities from the transcript
- Focus on key decisions, arguments, and lessons learned
- Ignore sponsorships, ads, and housekeeping
- Do not use quotes or speaker names
- Do not invent info not implied by the transcript

Transcript:
```
{transcript}
```
"""

PROMPT_V2_AWARE = """Summarize the following podcast episode transcript.
- Write a detailed summary with 4-6 paragraphs
- Begin the first paragraph with a single sentence naming the episode's domain and its central arg or premise
- Cover ALL major discussion segments in the order they appear in the transcript
- Preserve key technical terms, concept names, product names, and specific vocabulary from the transcript verbatim — do not paraphrase or substitute synonyms for named concepts
- Anchor each paragraph in specific claims, data points, or named entities from the transcript
- Focus on key decisions, arguments, and lessons learned
- **Surface any explicit position changes** ("I used to think X — after Y, I now think Z") as a distinct beat; these are the highest-signal moments in the conversation
- **Name recurring guests** (anyone who's appeared on the show before, indicated by host callbacks like "as Marco said last week") so the summary preserves the show's social graph
- Ignore sponsorships, ads, and housekeeping
- Do not use quotes or speaker names (except to name recurring guests)
- Do not invent info not implied by the transcript

Transcript:
```
{transcript}
```
"""

JUDGE_SYSTEM = (
    "You are evaluating two candidate summaries (A and B) of the same podcast "
    "episode transcript. Better = covers main claims faithfully, preserves "
    "named entities, captures position changes if present, mentions recurring "
    "guests if cited by the host, and does NOT invent or paraphrase. "
    'Reply STRICT JSON: {"winner": "A" | "B" | "TIE", "reason": "<short>"}'
)


def summarize(client: Any, prompt: str, transcript: str) -> str:
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        temperature=0.0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt.format(transcript=transcript)}],
    )
    return resp.content[0].text.strip()


def judge(client: Any, transcript: str, a: str, b: str) -> dict[str, Any]:
    msg = (
        f"Transcript (truncated to 4000 chars):\n```\n{transcript[:4000]}\n```\n\n"
        f"Summary A:\n```\n{a}\n```\n\n"
        f"Summary B:\n```\n{b}\n```\n\n"
        "Which is better? STRICT JSON only."
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
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--episodes", nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    if "ANTHROPIC_API_KEY" not in os.environ:
        print("ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 1
    from anthropic import Anthropic

    client = Anthropic()
    args.output.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    t0 = time.time()

    for ep in args.episodes:
        srcs = list(args.sources.rglob(f"{ep}.txt"))
        if not srcs:
            print(f"  SKIP {ep}: no source", file=sys.stderr)
            continue
        transcript = srcs[0].read_text(encoding="utf-8")
        try:
            summ_a = summarize(client, PROMPT_V1, transcript)
            summ_b = summarize(client, PROMPT_V2_AWARE, transcript)
        except Exception as exc:  # noqa: BLE001
            print(f"  ERR {ep}: {exc}", file=sys.stderr)
            continue
        verdict = judge(client, transcript, summ_a, summ_b)
        results.append(
            {
                "episode": ep,
                "summary_v1_chars": len(summ_a),
                "summary_v2_aware_chars": len(summ_b),
                "winner": verdict.get("winner"),
                "reason": verdict.get("reason"),
            }
        )
        print(
            f"  [{time.time()-t0:6.1f}s] {ep}  "
            f"v1={len(summ_a)}c  v2_aware={len(summ_b)}c  -> {verdict.get('winner')}"
        )

    wins = {"A_v1": 0, "B_v2_aware": 0, "TIE": 0, "OTHER": 0}
    for r in results:
        w = r.get("winner")
        if w == "A":
            wins["A_v1"] += 1
        elif w == "B":
            wins["B_v2_aware"] += 1
        elif w == "TIE":
            wins["TIE"] += 1
        else:
            wins["OTHER"] += 1

    payload = {
        "schema": "metrics_prompt_v2_validation_v1",
        "wins": wins,
        "results": results,
        "verdict": (
            "v2-aware variant wins or ties; consider shipping v2 prompt variant"
            if wins["B_v2_aware"] > wins["A_v1"]
            else (
                "v1 prompt robust to v2 inputs; no prompt change indicated"
                if wins["A_v1"] >= wins["B_v2_aware"]
                else "mixed"
            )
        ),
    }
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nv1 prompt (A) wins: {wins['A_v1']}")
    print(f"v2-aware prompt (B) wins: {wins['B_v2_aware']}")
    print(f"TIE: {wins['TIE']}")
    print(f"verdict: {payload['verdict']}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
