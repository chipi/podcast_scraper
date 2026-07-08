"""Silver-label topic-similarity candidate pools via Opus 4.8 (#1105).

Reads the harvest (``enrichment_topic_similarity_harvest_v1.py``) and, per topic, asks Opus
which candidates are genuinely RELATED — producing the gold ``expected_neighbours`` the
recall@K scorer (``enrichment_topic_similarity.py``) consumes.

Output rows are written in the scorer's gold schema:
    {"topic_id": "...", "expected_neighbours": ["topic:...", ...], ...}

Opus is the reference labeler (vendor-disjoint from the sentence-transformers candidate);
runs OFFLINE, the committed JSONL is the fixture ([[feedback_no_llm_in_ci]]).

Usage:
    export ANTHROPIC_API_KEY=...            # or: set -a; source .env; set +a
    python scripts/eval/score/enrichment_topic_similarity_silver_v1.py \\
        --input  data/eval/enrichment/topic_similarity/harvest_prodv2_v1.jsonl \\
        --output data/eval/enrichment/topic_similarity/gold/silver_prodv2_v1.jsonl
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
    "You judge which candidate topics are genuinely RELATED to a target topic, to build a "
    "reference set for evaluating an automated 'related topics' feature on a podcast "
    "knowledge graph.\n\n"
    "RELATED = a user exploring the target topic would find the candidate a natural, on-theme "
    "neighbour: the same subject area, a sub-topic, or a closely-linked theme. NOT merely "
    "sharing a word, and NOT a different subject that just happens to co-occur.\n\n"
    "Examples: target 'organizational culture' — RELATED: 'company culture', 'engineering "
    "culture', 'corporate practices'; NOT related: 'nvidia', 'banking history', 'scientific "
    "communication'. Be strict; when unsure, EXCLUDE.\n\n"
    "You get the target topic and a numbered candidate list (id + label). Return ONLY a JSON "
    "object, no prose, no code fence:\n"
    '{"related_ids": ["topic:...", ...], "rationale": "one sentence"}\n'
    "Include only candidate ids that are genuinely related; [] if none."
)


def _build_user_prompt(row: dict[str, Any]) -> str:
    lines = [f"Target topic: {row.get('topic_label')} ({row.get('topic_id')})", "", "Candidates:"]
    for c in row.get("candidates", []):
        lines.append(f"  - {c['id']}  ({c.get('label', '')})")
    lines.append("")
    lines.append("Which candidate ids are genuinely related to the target topic?")
    return "\n".join(lines)


_KEEP = ("pair_id", "topic_id", "topic_label", "predicted_top_k", "candidates")


def label_one(client: Any, row: dict[str, Any]) -> dict[str, Any]:
    """Judge one topic's candidate pool; return the scorer gold row + provenance."""
    resp = client.messages.create(
        model=MODEL,
        max_tokens=400,
        system=SYSTEM,
        messages=[{"role": "user", "content": _build_user_prompt(row)}],
    )
    text = resp.content[0].text.strip()
    if text.startswith("```"):
        text = text.strip("`").lstrip("json").strip()
    base = {k: row.get(k) for k in _KEEP}
    try:
        verdict = json.loads(text)
    except json.JSONDecodeError as exc:
        return {**base, "expected_neighbours": [], "error": str(exc), "raw_response": text}
    valid = {c["id"] for c in row.get("candidates", [])}
    related = [i for i in (verdict.get("related_ids") or []) if i in valid]
    return {**base, "expected_neighbours": related, "rationale": verdict.get("rationale")}


def main() -> int:
    """Label every harvested topic via Opus; stream gold rows to the output JSONL."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    if not args.input.is_file():
        print(f"input not found: {args.input}", file=sys.stderr)
        return 1
    lines = args.input.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(x) for x in lines if x.strip()]
    if args.limit:
        rows = rows[: args.limit]

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set (try: set -a; source .env; set +a)", file=sys.stderr)
        return 1
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    out: list[dict[str, Any]] = []
    with args.output.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows, 1):
            try:
                rec = label_one(client, row)
            except Exception as exc:  # noqa: BLE001
                rec = {k: row.get(k) for k in _KEEP}
                rec.update(expected_neighbours=[], error=str(exc))
            out.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if i % 8 == 0 or i == len(rows):
                print(f"  {i}/{len(rows)}  elapsed={time.time() - t0:.1f}s", file=sys.stderr)

    total_expected = sum(len(r.get("expected_neighbours") or []) for r in out)
    print(
        f"\nwrote {args.output}: {len(out)} topics, {total_expected} expected-neighbour labels",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
