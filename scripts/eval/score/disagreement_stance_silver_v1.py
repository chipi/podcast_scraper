"""Judge stance-level disagreement for the #1144 feasibility spike via Opus 4.8.

Reads the stance harvest (``disagreement_stance_harvest_v1.py``) and asks Opus, per
speaker-pair, whether their overall positions on the topic DISAGREE — stance-vs-stance,
with an explicit ``no_shared_question`` category for the common "same topic, different
facet" case #1106 taught us to expect.

Purpose: verify the "where thinkers disagree" signal actually EXISTS at the stance level
before investing in an RFC + an offline-LLM enrichment tier. If most viable pairs come
back ``no_shared_question`` / ``agree``, the premise is weak and we reconsider before
building.

Opus is the reference labeler (offline, one-time). CI never calls this
([[feedback_no_llm_in_ci]]).

Usage:
    export ANTHROPIC_API_KEY=...            # or: set -a; source .env; set +a
    python scripts/eval/score/disagreement_stance_silver_v1.py \\
        --input  data/eval/enrichment/disagreement/harvest_prodv2_v1.jsonl \\
        --output data/eval/enrichment/disagreement/silver_prodv2_v1.jsonl
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
    "You compare two speakers' STANCES on a shared topic, from a podcast knowledge graph, "
    "to test whether an automated 'where thinkers disagree' detector is worth building. "
    "You get the topic, speaker A's insights on it, and speaker B's insights on it. Judge "
    "whether their overall positions DISAGREE.\n\n"
    '- "disagree" — A and B take opposing positions on a SHARED question within the topic: '
    "one asserts X, the other asserts not-X or a mutually-exclusive alternative. Include "
    'opposing net-effect / normative positions ("regulation helps" vs "regulation hurts"; '
    '"X is the best approach" vs "Y is the best approach").\n'
    '- "agree" — their positions align or reinforce each other on a shared question.\n'
    '- "no_shared_question" — they discuss DIFFERENT aspects of the topic and never engage '
    "a common proposition, so there is nothing to agree or disagree about. THIS IS COMMON: "
    "sharing a broad topic (e.g. 'ai-development') is NOT disagreement if one talks model "
    "architecture and the other talks a specific company's market position.\n\n"
    'CRITICAL: only "disagree" when they genuinely oppose each other on the SAME question. '
    "Different facets, different companies, different sub-topics, different time frames → "
    '"no_shared_question". When unsure, prefer "no_shared_question".\n\n'
    "Return ONLY a JSON object, no prose, no code fence:\n"
    '{"label": "disagree|agree|no_shared_question", '
    '"shared_question": "the common question they both address, or null", '
    '"a_position": "one clause or null", "b_position": "one clause or null", '
    '"strength": 0.0-1.0, "confidence": 0.0-1.0, "rationale": "one sentence"}'
)


def _bullets(insights: list[str]) -> str:
    return "\n".join(f"  - {t}" for t in insights)


def _build_user_prompt(pair: dict[str, Any]) -> str:
    return (
        f"Topic: {pair.get('topic_id', '')}\n\n"
        f"Speaker A = {pair.get('speaker_a_name', 'A')}\n"
        f"{_bullets(pair.get('speaker_a_insights', []))}\n\n"
        f"Speaker B = {pair.get('speaker_b_name', 'B')}\n"
        f"{_bullets(pair.get('speaker_b_insights', []))}\n\n"
        "Do A and B disagree on this topic?"
    )


_KEEP = (
    "pair_id",
    "topic_id",
    "speaker_a_name",
    "speaker_b_name",
    "speaker_a_insights",
    "speaker_b_insights",
)


def label_one(client: Any, pair: dict[str, Any]) -> dict[str, Any]:
    """Judge one speaker-pair; merge the kept fields with the verdict."""
    resp = client.messages.create(
        model=MODEL,
        max_tokens=400,
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
    """Judge every stance pair via Opus; stream results to the output JSONL."""
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
            if i % 10 == 0 or i == len(pairs):
                print(f"  {i}/{len(pairs)}  elapsed={time.time() - t0:.1f}s", file=sys.stderr)

    by_label: dict[str, int] = {}
    for o in labelled:
        by_label[o.get("label", "?")] = by_label.get(o.get("label", "?"), 0) + 1
    print(f"\nlabel distribution: {by_label}", file=sys.stderr)
    print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
