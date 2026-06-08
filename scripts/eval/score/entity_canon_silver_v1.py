"""Auto-label candidate entity-canonicalization pairs via Sonnet 4.6 (#853).

Reads `data/eval/sources/entity_canon_v1/candidate_pairs.jsonl`, sends each
pair to Anthropic Claude Sonnet 4.6 with a strict rubric, and writes
`data/eval/references/silver/entity_canon_v1/labels.jsonl` with each pair
annotated `label ∈ {SAME, DIFFERENT, BORDERLINE}` plus rationale + confidence.

The judge is told that the input pair comes from a financial-podcast corpus
where ASR (Whisper) is the common source of variants — so the priors are:
- Same first/last name + 1-2 letter swap on the surname → likely SAME (Whisper garble)
- Different first names → DIFFERENT
- Nickname-vs-full-name (Rich/Richard, Rob/Robert) → SAME
- Otherwise familiar variant pattern → SAME unless prominent name collision

Output also includes the input pair so the labels file is self-contained.

Usage:
    python scripts/eval/score/entity_canon_silver_v1.py \
        --input data/eval/sources/entity_canon_v1/candidate_pairs.jsonl \
        --output data/eval/references/silver/entity_canon_v1/labels.jsonl

Requires ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


SYSTEM = (
    "You decide whether two person identifiers extracted from a financial-podcast "
    "knowledge graph refer to the SAME real person or DIFFERENT people. "
    "The corpus is dominated by Bloomberg / Odd Lots / Pushkin / Financial Times shows. "
    "Whisper transcription is the upstream source — many variants are ASR garbles "
    "(e.g. `Tracy Allaway` vs `Tracy Alloway`, `Byrne Hobart` vs `Burne Hobart`, "
    "`Joe Wassenthal` vs `Joe Weisenthal`, `Henry Blodgett` vs `Henry Blodget`). "
    "Nickname-and-full-name variants (`Rich Clarida` vs `Richard Clarida`, "
    "`Rob` vs `Robert`) are SAME person. "
    "Same first name but unrelated surnames "
    "(`Jacob Goldstein` vs `Rob Goldstein`) are DIFFERENT. "
    "Single-letter surname differences are USUALLY SAME (ASR garble) unless "
    "both names are well-known and distinct people "
    "(e.g. `Powell` vs `Powers`). When uncertain answer BORDERLINE.\n\n"
    "Reply with STRICT JSON only:\n"
    '{"label": "SAME"|"DIFFERENT"|"BORDERLINE", "confidence": 0.0-1.0,'
    ' "reason": "<one short sentence>"}'
)


def _build_user_prompt(pair: dict[str, Any]) -> str:
    return (
        f"Candidate pair:\n"
        f"  A: id={pair['a_id']!r}  label={pair['a_label']!r}  episodes={pair['a_episode_count']}\n"
        f"  B: id={pair['b_id']!r}  label={pair['b_label']!r}  episodes={pair['b_episode_count']}\n"
        f"  shared_episode: {pair['shared_episode']}\n"
        f"  slug_similarity: {pair['slug_ratio']}\n\n"
        f"Are A and B the SAME real person or DIFFERENT people?"
    )


def label_one(client: Any, pair: dict[str, Any]) -> dict[str, Any]:
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        temperature=0.0,
        system=SYSTEM,
        messages=[{"role": "user", "content": _build_user_prompt(pair)}],
    )
    text = resp.content[0].text.strip()
    # Tolerate code fences
    if text.startswith("```"):
        text = text.strip("`").lstrip("json").strip()
    try:
        verdict = json.loads(text)
    except json.JSONDecodeError as exc:
        return {**pair, "label": "PARSE_ERROR", "raw_response": text, "error": str(exc)}
    out = {**pair, **verdict}
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0, help="0 = process all")
    args = p.parse_args()

    pairs = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if args.limit:
        pairs = pairs[: args.limit]

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 1
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    labelled: list[dict[str, Any]] = []
    with args.output.open("w") as f:
        for i, pair in enumerate(pairs, 1):
            try:
                out = label_one(client, pair)
            except Exception as exc:  # noqa: BLE001
                out = {**pair, "label": "ERROR", "error": str(exc)}
            labelled.append(out)
            f.write(json.dumps(out) + "\n")
            f.flush()
            if i % 20 == 0 or i == len(pairs):
                elapsed = time.time() - t0
                print(f"  {i}/{len(pairs)}  elapsed={elapsed:.1f}s")

    by_label: dict[str, int] = {}
    for o in labelled:
        by_label[o.get("label", "?")] = by_label.get(o.get("label", "?"), 0) + 1
    print(f"\nLabel distribution: {by_label}")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
