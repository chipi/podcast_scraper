"""Silver-label harvested Insight pairs for the NLI-contradiction eval via Opus 4.8 (#1106).

Reads a harvest JSONL (see ``enrichment_nli_harvest_v1.py``), sends each cross-Person
Insight pair to Anthropic Claude Opus 4.8 with a two-lens rubric, and writes a
self-contained silver JSONL with each pair annotated:

    label            ∈ {contradiction, entailment, neutral}
    contradiction_type ∈ {logical, competing_claim, null}
    confidence        ∈ [0, 1]
    rationale         one sentence

Two-lens (per #1106 / the epic-1101 audit): ``contradiction_type`` lets us score the
current DeBERTa enricher two ways from ONE labeling pass —
  * against the *logical* subset (its fair, strict-NLI bar), and
  * against the *full* set incl. ``competing_claim`` (the Option-B product bar, which a
    strict-NLI model structurally can't reach — quantifies the recall gap).

Opus is vendor-disjoint from the candidate (DeBERTa) → no silver/judge vendor bias
([[feedback_silver_judge_vendor_bias]]). Runs OFFLINE; the committed silver JSONL is the
fixture — CI never calls this ([[feedback_no_llm_in_ci]]).

Usage:
    export ANTHROPIC_API_KEY=...            # or: set -a; source .env; set +a
    python scripts/eval/score/enrichment_nli_silver_v1.py \\
        --input  data/eval/enrichment/nli_contradiction/harvest_prodv2_v1.jsonl \\
        --output data/eval/enrichment/nli_contradiction/gold/silver_prodv2_v1.jsonl

Exit codes:
    0 — silver written
    1 — missing API key / input
    2 — invocation error
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

MODEL = "claude-opus-4-8"

SYSTEM = (
    "You label pairs of INSIGHTS pulled from two DIFFERENT speakers on a podcast "
    "knowledge graph, tagged to the same topic. You are building a reference set to "
    "evaluate an automated contradiction detector, so be precise and conservative.\n\n"
    "Decide the relationship between claim A and claim B:\n\n"
    '- "contradiction" — a rational person cannot fully endorse BOTH claims at once. '
    "Two sub-types:\n"
    '    - "logical": mutually-exclusive facts or predictions about the SAME proposition '
    '("X will happen" vs "X will not happen"; "the cause is A" vs "the cause is not A").\n'
    '    - "competing_claim": mutually-exclusive normative/evaluative positions on the same '
    'question ("carbon tax is the BEST lever" vs "cap-and-trade is the BEST lever"), or '
    'opposing net-effect assessments of the same thing ("remote work boosts productivity" vs '
    '"remote work hurts output").\n'
    '- "entailment" — one claim restates, supports, or is directly implied by the other.\n'
    '- "neutral" — everything else: the claims are compatible, about DIFFERENT '
    "propositions/aspects, or merely share a topic keyword without opposing each other.\n\n"
    "CRITICAL guardrails (this is exactly where the automated detector fails):\n"
    '- Sharing a topic or named entity is NOT a contradiction. "Spirit Airlines ceased '
    'operations" and "Spirit Airlines pioneered the low-cost model" are BOTH true, about '
    "different aspects → neutral.\n"
    "- The two claims must target the SAME underlying proposition to contradict. Different "
    "subjects, time periods, or scopes → neutral.\n"
    "- Different emphasis or focus is not contradiction unless the positions are mutually "
    "exclusive.\n"
    '- When unsure, prefer "neutral": for this reference set a false contradiction is worse '
    "than a missed one.\n\n"
    "Return ONLY a JSON object, no prose, no code fence:\n"
    '{"label": "contradiction|entailment|neutral", '
    '"contradiction_type": "logical|competing_claim|null", '
    '"confidence": 0.0-1.0, "rationale": "one sentence"}\n'
    'contradiction_type is null unless label is "contradiction".'
)


def _build_user_prompt(pair: dict[str, Any]) -> str:
    return (
        f"Topic: {pair.get('topic_id', '')}\n"
        f"A ({pair.get('person_a_name', 'A')}): {pair.get('insight_a_text', '')}\n"
        f"B ({pair.get('person_b_name', 'B')}): {pair.get('insight_b_text', '')}\n\n"
        "Label the relationship between claims A and B."
    )


# Fields we persist from the harvest row (keep the silver file self-contained but lean).
_KEEP = (
    "pair_id",
    "stratum",
    "topic_id",
    "person_a_name",
    "person_b_name",
    "insight_a_id",
    "insight_b_id",
    "insight_a_text",
    "insight_b_text",
    "deberta_flagged",
    "deberta_score",
)


def label_one(client: Any, pair: dict[str, Any]) -> dict[str, Any]:
    """Label one pair; returns the kept harvest fields merged with the silver verdict."""
    resp = client.messages.create(
        model=MODEL,
        max_tokens=300,
        system=SYSTEM,
        messages=[{"role": "user", "content": _build_user_prompt(pair)}],
    )
    text = resp.content[0].text.strip()
    if text.startswith("```"):
        text = text.strip("`").lstrip("json").strip()
    base = {k: pair.get(k) for k in _KEEP}
    try:
        verdict = json.loads(text)
    except json.JSONDecodeError as exc:
        return {**base, "label": "PARSE_ERROR", "raw_response": text, "error": str(exc)}
    return {**base, **verdict}


def main() -> int:
    """Label every harvested pair via Opus; stream results to the output JSONL."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0, help="0 = process all")
    args = p.parse_args()

    if not args.input.is_file():
        print(f"input not found: {args.input}", file=sys.stderr)
        return 1
    lines = args.input.read_text(encoding="utf-8").splitlines()
    pairs = [json.loads(x) for x in lines if x.strip()]
    if args.limit:
        pairs = pairs[: args.limit]

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set (try: set -a; source .env; set +a)", file=sys.stderr)
        return 1
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    labelled: list[dict[str, Any]] = []
    with args.output.open("w", encoding="utf-8") as f:
        for i, pair in enumerate(pairs, 1):
            try:
                out = label_one(client, pair)
            except Exception as exc:  # noqa: BLE001 — one bad call shouldn't drop the batch
                out = {k: pair.get(k) for k in _KEEP}
                out.update(label="ERROR", error=str(exc))
            labelled.append(out)
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()
            if i % 20 == 0 or i == len(pairs):
                print(f"  {i}/{len(pairs)}  elapsed={time.time() - t0:.1f}s", file=sys.stderr)

    by_label: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for o in labelled:
        by_label[o.get("label", "?")] = by_label.get(o.get("label", "?"), 0) + 1
        if o.get("label") == "contradiction":
            ct = o.get("contradiction_type") or "unspecified"
            by_type[ct] = by_type.get(ct, 0) + 1
    print(f"\nlabel distribution: {by_label}", file=sys.stderr)
    print(f"contradiction sub-types: {by_type}", file=sys.stderr)
    print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
