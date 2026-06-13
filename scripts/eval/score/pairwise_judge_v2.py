"""Pairwise judge v2 — multi-judge + position-swap + configurable rubric.

Replaces the four ad-hoc pairwise judges accumulated across cleaning /
summary / cil / GI evals (cleaning_judge_v1.py, cleaning_v3_vs_v4_broader_
judge_v1.py, …) with one well-tested harness. Built off lessons from #989's
position-bias finding (every #905 "tie" was a v3 win once A/B order was
swapped — see EVAL_PAIRWISE_JUDGING_LESSONS_2026_06_13.md for the full
methodology rationale).

## What this harness does

For each item (candidate_a vs candidate_b) and each requested judge model,
the harness:

1. **Anonymises** the candidates as A / B before sending to the judge.
2. **Swaps positions** when ``--orderings swap``: judges the same pair twice
   with A=v3,B=v4 and then A=v4,B=v3. Consensus only when both orderings
   agree. Disagreement is the signal that the verdict is positional, not
   substantive — flag, don't average.
3. **Aggregates across judges** when more than one is requested. The final
   item-level consensus requires a strict majority of the per-judge
   verdicts (e.g. 2-of-3 with the third disagreeing → 2-judge consensus;
   1-1-1 → DISAGREEMENT).
4. **Logs every raw judge call** (prompt + response + cost) to an audit
   JSONL so a future rerun can verify reproducibility.

## Tier-1 gate (default flips / customer-facing behaviour changes)

    --orderings swap
    --judges anthropic:claude-sonnet-4-6 gemini:gemini-2.5-flash openai:gpt-4o-mini

Three judges × two orderings = six calls per item. ~$0.02–0.05/item on
typical cleaning-comparison shapes. Use this gate for any change that
flips a production default — see EVAL_PAIRWISE_JUDGING_LESSONS for the
"why this tier".

## Tier-3 monitoring (continuous quality observation)

    --orderings single
    --judges anthropic:claude-sonnet-4-6

One call per item. Cheap; use for trend monitoring where false-positive
position bias is acceptable.

## Items JSONL schema

Each line is a JSON object:

    {
      "item_id": "p01_e01",
      "candidate_a_label": "cleaning_v3",
      "candidate_a_text": "...",
      "candidate_b_label": "cleaning_v4",
      "candidate_b_text": "...",
      "context": "...optional original transcript for the judge..."
    }

## Usage

    PYTHONPATH=. .venv/bin/python scripts/eval/score/pairwise_judge_v2.py \\
        --items data/eval/runs/.../items.jsonl \\
        --judges anthropic:claude-sonnet-4-6 \\
        --judges gemini:gemini-2.5-flash \\
        --judges openai:gpt-4o-mini \\
        --orderings swap \\
        --rubric scripts/eval/score/rubrics/cleaning_default.md \\
        --output data/eval/runs/cleaning_tier1_gate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Default rubric used when --rubric is not passed. Generic enough to drive
# any pairwise-cleaning comparison; specific rubrics belong under
# scripts/eval/score/rubrics/<name>.md.
_DEFAULT_RUBRIC = """You are evaluating two candidate cleaned outputs (A and B) of \
the same source content. Better cleaning means:

- removed all sponsor / ad / intro / outro / meta-commentary,
- preserved all substantive content,
- kept speaker labels intact,
- did NOT invent or paraphrase content.

Pick the better one."""

# Pricing snapshot 2026-06 ($/M tokens). Used only for cost ESTIMATES in
# the output — keep accurate to the nearest order-of-magnitude. Override
# via PAIRWISE_JUDGE_PRICING_JSON env var if a different pricing applies.
_DEFAULT_PRICING: Dict[str, Tuple[float, float]] = {
    # provider:model -> (input_per_M, output_per_M)
    "anthropic:claude-sonnet-4-6": (3.0, 15.0),
    "anthropic:claude-haiku-4-5": (0.8, 4.0),
    "openai:gpt-4o-mini": (0.15, 0.60),
    "openai:gpt-4o": (2.50, 10.0),
    "gemini:gemini-2.5-flash": (0.075, 0.30),
    "gemini:gemini-2.5-flash-lite": (0.04, 0.16),
}


# ---------------------------------------------------------------- Judge clients


@dataclass
class JudgeVerdict:
    """One single judge call's verdict."""

    winner: str  # "A" | "B" | "TIE" | "PARSE_ERROR"
    reason: str
    raw: str
    input_tokens: int = 0
    output_tokens: int = 0
    error: str = ""


JudgeFn = Callable[[str, str, str, str], JudgeVerdict]


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return text


def _parse_winner(raw: str) -> Tuple[str, str]:
    """Return (winner, reason) parsed from a judge's JSON response."""
    cleaned = _strip_code_fences(raw)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return ("PARSE_ERROR", cleaned[:200])
    winner = str(payload.get("winner", "")).strip().upper()
    if winner not in {"A", "B", "TIE"}:
        return ("PARSE_ERROR", f"unexpected winner {winner!r}: {cleaned[:120]}")
    reason = str(payload.get("reason", "")).strip()
    return (winner, reason[:300])


def _build_user_message(
    rubric: str, context: Optional[str], candidate_a: str, candidate_b: str
) -> str:
    ctx_block = (
        f"Original source (first 4000 chars):\n```\n{context[:4000]}\n```\n\n" if context else ""
    )
    return (
        f"{rubric}\n\n"
        f"{ctx_block}"
        f"Candidate A (first 4000 chars):\n```\n{candidate_a[:4000]}\n```\n\n"
        f"Candidate B (first 4000 chars):\n```\n{candidate_b[:4000]}\n```\n\n"
        "Which is better? Reply STRICT JSON only: "
        '{"winner": "A"|"B"|"TIE", "reason": "<one short sentence>"}'
    )


def _make_anthropic_judge(model: str) -> JudgeFn:
    from anthropic import Anthropic

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def judge(
        rubric: str, context: Optional[str], candidate_a: str, candidate_b: str
    ) -> JudgeVerdict:
        msg = _build_user_message(rubric, context, candidate_a, candidate_b)
        try:
            response = client.messages.create(
                model=model,
                max_tokens=400,
                temperature=0.0,
                system="Return ONLY the requested JSON. No prose, no markdown.",
                messages=[{"role": "user", "content": msg}],
            )
        except Exception as exc:  # pragma: no cover - external API
            return JudgeVerdict("PARSE_ERROR", "", "", error=str(exc)[:300])
        raw = response.content[0].text if response.content else ""
        winner, reason = _parse_winner(raw)
        usage = getattr(response, "usage", None)
        return JudgeVerdict(
            winner=winner,
            reason=reason,
            raw=raw,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0) if usage else 0,
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0) if usage else 0,
        )

    return judge


def _make_openai_judge(model: str) -> JudgeFn:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def judge(
        rubric: str, context: Optional[str], candidate_a: str, candidate_b: str
    ) -> JudgeVerdict:
        msg = _build_user_message(rubric, context, candidate_a, candidate_b)
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=400,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "Return ONLY the requested JSON. No prose."},
                    {"role": "user", "content": msg},
                ],
            )
        except Exception as exc:  # pragma: no cover - external API
            return JudgeVerdict("PARSE_ERROR", "", "", error=str(exc)[:300])
        choice = response.choices[0] if response.choices else None
        raw = choice.message.content if choice and choice.message else ""
        winner, reason = _parse_winner(raw or "")
        usage = getattr(response, "usage", None)
        return JudgeVerdict(
            winner=winner,
            reason=reason,
            raw=raw or "",
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0,
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0,
        )

    return judge


def _make_gemini_judge(model: str) -> JudgeFn:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def judge(
        rubric: str, context: Optional[str], candidate_a: str, candidate_b: str
    ) -> JudgeVerdict:
        msg = _build_user_message(rubric, context, candidate_a, candidate_b)
        try:
            response = client.models.generate_content(
                model=model,
                contents=msg,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    system_instruction="Return ONLY the requested JSON. No prose, no markdown.",
                ),
            )
        except Exception as exc:  # pragma: no cover - external API
            return JudgeVerdict("PARSE_ERROR", "", "", error=str(exc)[:300])
        raw = response.text or ""
        winner, reason = _parse_winner(raw)
        usage = getattr(response, "usage_metadata", None)
        return JudgeVerdict(
            winner=winner,
            reason=reason,
            raw=raw,
            input_tokens=int(getattr(usage, "prompt_token_count", 0) or 0) if usage else 0,
            output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0) if usage else 0,
        )

    return judge


_JUDGE_FACTORIES: Dict[str, Callable[[str], JudgeFn]] = {
    "anthropic": _make_anthropic_judge,
    "openai": _make_openai_judge,
    "gemini": _make_gemini_judge,
}


def _parse_judge_spec(spec: str) -> Tuple[str, str]:
    if ":" not in spec:
        raise argparse.ArgumentTypeError(
            "--judges value must be 'provider:model' "
            f"(e.g. 'anthropic:claude-sonnet-4-6'); got {spec!r}"
        )
    provider, _, model = spec.partition(":")
    if provider not in _JUDGE_FACTORIES:
        raise argparse.ArgumentTypeError(
            f"unknown provider {provider!r}; choose from {list(_JUDGE_FACTORIES)}"
        )
    return provider, model


# ---------------------------------------------------------------- consensus rules


def _per_judge_consensus(o1: str, o2: Optional[str]) -> str:
    """Return the per-judge consensus across the two A/B orderings.

    o1 is the verdict's mapped winner (already de-anonymised — "v3" / "v4" /
    "TIE" / "PARSE_ERROR"). o2 is the swapped-ordering equivalent, or None
    if --orderings single.
    """
    if o2 is None:
        return o1
    if o1 == o2:
        return o1
    if o1 == "PARSE_ERROR" or o2 == "PARSE_ERROR":
        return "PARSE_ERROR"
    return "TIE_POSITIONAL"


def _multi_judge_consensus(per_judge: Dict[str, str]) -> str:
    """Strict majority rule across judges. Returns DISAGREEMENT on no majority."""
    from collections import Counter

    votes = Counter(v for v in per_judge.values() if v not in {"PARSE_ERROR"})
    if not votes:
        return "ALL_PARSE_ERROR"
    top, top_n = votes.most_common(1)[0]
    needed = (len(per_judge) // 2) + 1
    return top if top_n >= needed else "DISAGREEMENT"


# ---------------------------------------------------------------- pricing


def _load_pricing() -> Dict[str, Tuple[float, float]]:
    raw = os.environ.get("PAIRWISE_JUDGE_PRICING_JSON")
    if not raw:
        return dict(_DEFAULT_PRICING)
    try:
        override = json.loads(raw)
        return {k: (float(v[0]), float(v[1])) for k, v in override.items()}
    except (ValueError, KeyError, IndexError, TypeError):
        print(
            "WARNING: PAIRWISE_JUDGE_PRICING_JSON malformed — using defaults",
            file=sys.stderr,
        )
        return dict(_DEFAULT_PRICING)


def _cost_for(
    verdict: JudgeVerdict, judge_key: str, pricing: Dict[str, Tuple[float, float]]
) -> float:
    pin, pout = pricing.get(judge_key, (0.0, 0.0))
    return (verdict.input_tokens / 1_000_000) * pin + (verdict.output_tokens / 1_000_000) * pout


# ---------------------------------------------------------------- main


def _load_items(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        items.append(json.loads(line))
    return items


def _decode_winner(raw_winner: str, a_label: str, b_label: str) -> str:
    if raw_winner == "A":
        return a_label
    if raw_winner == "B":
        return b_label
    return raw_winner  # "TIE" / "PARSE_ERROR"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument(
        "--judges",
        action="append",
        required=True,
        type=_parse_judge_spec,
        help="Repeatable: provider:model (e.g. anthropic:claude-sonnet-4-6).",
    )
    parser.add_argument(
        "--orderings",
        choices=("swap", "single"),
        default="swap",
        help="Tier-1 defaults to 'swap'. 'single' is for cheap monitoring.",
    )
    parser.add_argument(
        "--rubric",
        type=Path,
        default=None,
        help="Optional rubric file (default: generic cleaning rubric).",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sleep-s", type=float, default=0.3, help="Inter-call sleep (rate politeness)."
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    raw_log_path = args.output / "raw_log.jsonl"
    raw_log_path.unlink(missing_ok=True)

    rubric = _DEFAULT_RUBRIC if args.rubric is None else args.rubric.read_text(encoding="utf-8")
    items = _load_items(args.items)
    if not items:
        print(f"ERROR: no items in {args.items}", file=sys.stderr)
        return 2

    pricing = _load_pricing()
    judge_fns: List[Tuple[str, JudgeFn]] = []
    for provider, model in args.judges:
        key = f"{provider}:{model}"
        judge_fns.append((key, _JUDGE_FACTORIES[provider](model)))

    per_item_rows: List[Dict[str, Any]] = []
    total_cost = 0.0
    raw_lines: List[str] = []

    for item in items:
        item_id = item.get("item_id", "?")
        a_label = item["candidate_a_label"]
        b_label = item["candidate_b_label"]
        a_text = item["candidate_a_text"]
        b_text = item["candidate_b_text"]
        context = item.get("context")

        per_judge_consensus_map: Dict[str, str] = {}
        per_judge_detail: Dict[str, Dict[str, Any]] = {}
        for judge_key, judge in judge_fns:
            v1 = judge(rubric, context, a_text, b_text)
            time.sleep(args.sleep_s)
            v1_label = _decode_winner(v1.winner, a_label, b_label)
            v1_cost = _cost_for(v1, judge_key, pricing)
            total_cost += v1_cost
            raw_lines.append(
                json.dumps(
                    {
                        "item_id": item_id,
                        "judge": judge_key,
                        "ordering": 1,
                        "a_was": a_label,
                        "b_was": b_label,
                        "raw_winner": v1.winner,
                        "decoded": v1_label,
                        "reason": v1.reason,
                        "input_tokens": v1.input_tokens,
                        "output_tokens": v1.output_tokens,
                        "cost_usd": v1_cost,
                        "error": v1.error,
                    },
                    ensure_ascii=False,
                )
            )

            v2_label: Optional[str] = None
            v2_detail: Optional[Dict[str, Any]] = None
            if args.orderings == "swap":
                # Swap: A becomes the old B; B becomes the old A.
                v2 = judge(rubric, context, b_text, a_text)
                time.sleep(args.sleep_s)
                # Decode using swapped labels (a_position_label=b_label, b_position_label=a_label).
                v2_label = _decode_winner(v2.winner, b_label, a_label)
                v2_cost = _cost_for(v2, judge_key, pricing)
                total_cost += v2_cost
                raw_lines.append(
                    json.dumps(
                        {
                            "item_id": item_id,
                            "judge": judge_key,
                            "ordering": 2,
                            "a_was": b_label,
                            "b_was": a_label,
                            "raw_winner": v2.winner,
                            "decoded": v2_label,
                            "reason": v2.reason,
                            "input_tokens": v2.input_tokens,
                            "output_tokens": v2.output_tokens,
                            "cost_usd": v2_cost,
                            "error": v2.error,
                        },
                        ensure_ascii=False,
                    )
                )
                v2_detail = {"raw_winner": v2.winner, "decoded": v2_label, "reason": v2.reason}

            consensus_for_judge = _per_judge_consensus(v1_label, v2_label)
            per_judge_consensus_map[judge_key] = consensus_for_judge
            per_judge_detail[judge_key] = {
                "ordering1": {"raw_winner": v1.winner, "decoded": v1_label, "reason": v1.reason},
                "ordering2": v2_detail,
                "consensus": consensus_for_judge,
            }

        final = _multi_judge_consensus(per_judge_consensus_map)
        per_item_rows.append(
            {
                "item_id": item_id,
                "candidate_a_label": a_label,
                "candidate_b_label": b_label,
                "judges": per_judge_detail,
                "per_judge_consensus": per_judge_consensus_map,
                "final_consensus": final,
            }
        )
        judge_summary = " ".join(
            f"{k.split(':',1)[1][:18]}={v}" for k, v in per_judge_consensus_map.items()
        )
        print(f"  {item_id}: {judge_summary} → {final}", file=sys.stderr)

    raw_log_path.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")

    # ---- Aggregate ----
    from collections import Counter

    consensus_counter = Counter(row["final_consensus"] for row in per_item_rows)
    per_judge_winner_counters: Dict[str, Counter] = {key: Counter() for key, _ in judge_fns}
    for row in per_item_rows:
        for judge_key, judge_verdict in row["per_judge_consensus"].items():
            per_judge_winner_counters[judge_key][judge_verdict] += 1

    summary = {
        "items": len(per_item_rows),
        "orderings": args.orderings,
        "judges": [k for k, _ in judge_fns],
        "rubric_first_120_chars": rubric[:120],
        "cost_usd_estimate": round(total_cost, 4),
        "final_consensus_counts": dict(consensus_counter),
        "per_judge_consensus_counts": {k: dict(c) for k, c in per_judge_winner_counters.items()},
    }

    (args.output / "metrics.json").write_text(
        json.dumps(
            {
                "schema": "pairwise_judge_v2",
                "summary": summary,
                "rows": per_item_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=== pairwise_judge_v2 summary ===")
    print(f"  items: {summary['items']}")
    print(f"  orderings: {summary['orderings']}")
    print(f"  judges: {summary['judges']}")
    print(f"  cost (est): ${summary['cost_usd_estimate']}")
    print("  final_consensus_counts:")
    for k, n in consensus_counter.most_common():
        print(f"    {k:>20s}: {n}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
