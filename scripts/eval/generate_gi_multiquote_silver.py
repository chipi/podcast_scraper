"""Generate multi-quote silver GI references using Sonnet 4.6.

For each episode, prompts the model to produce 8 insights with 2-3
verbatim supporting quotes each. Verifies quote char offsets.

Usage:
    python scripts/eval/generate_gi_multiquote_silver.py \
        --dataset curated_5feeds_benchmark_v2 \
        --output-id silver_sonnet46_gi_multiquote_benchmark_v2
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from podcast_scraper.evaluation.autoresearch_track_a import (
    load_local_dotenv_files,
)

load_local_dotenv_files(Path.cwd())
os.environ.setdefault("AUTORESEARCH_ALLOW_PRODUCTION_KEYS", "1")

PROMPT = """\
You are generating reference-quality grounded insights for a podcast \
transcript evaluation benchmark.

Read the following podcast transcript carefully. Produce 8 key insights \
with 2-3 VERBATIM supporting quotes each.

For EACH insight:
1. Write a clear, single-sentence insight statement
2. Find 2-3 EXACT VERBATIM quotes from the transcript that support it
3. Each quote must be an EXACT substring — copy character-for-character
4. Quotes must be from DIFFERENT parts of the transcript (not overlapping)
5. Each quote should be 1-3 sentences long

Return valid JSON:
{{
  "insights": [
    {{
      "text": "The insight statement",
      "insight_type": "claim",
      "supporting_quotes": [
        {{"text": "exact verbatim quote 1"}},
        {{"text": "exact verbatim quote 2"}},
        {{"text": "exact verbatim quote 3"}}
      ]
    }}
  ]
}}

Transcript:

{transcript}"""


def find_quote_offset(transcript, quote_text):
    """Find exact char offset of a quote in the transcript."""
    idx = transcript.find(quote_text)
    if idx >= 0:
        return (idx, idx + len(quote_text))
    normalized_quote = re.sub(r"\s+", " ", quote_text.strip())
    normalized_transcript = re.sub(r"\s+", " ", transcript)
    idx = normalized_transcript.find(normalized_quote)
    if idx >= 0:
        return (idx, idx + len(normalized_quote))
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-id", required=True)
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    args = parser.parse_args()

    import anthropic

    client = anthropic.Anthropic()

    materialized = Path(f"data/eval/materialized/{args.dataset}")
    meta = json.load(open(materialized / "meta.json"))
    episodes = meta["episodes"]
    print(f"Dataset: {args.dataset} — {len(episodes)} episodes")

    output_dir = Path(f"data/eval/references/silver/{args.output_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_artifacts = []
    total_quotes = 0
    verified_quotes = 0

    for ep in episodes:
        ep_id = ep["episode_id"]
        transcript = (materialized / f"{ep_id}.txt").read_text()
        print(f"\n  {ep_id}: {len(transcript)} chars")

        t0 = time.time()
        response = client.messages.create(
            model=args.model,
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": PROMPT.format(transcript=transcript[:80000]),
                }
            ],
            temperature=0.0,
        )
        elapsed = time.time() - t0

        text = response.content[0].text if response.content else "{}"
        text = re.sub(
            r"^```(?:json)?\s*\n?",
            "",
            text.strip(),
            flags=re.MULTILINE,
        )
        text = re.sub(r"\n?```\s*$", "", text.strip(), flags=re.MULTILINE)
        result = json.loads(text.strip(), strict=False)

        insights = result.get("insights", [])
        verified_insights = []

        for insight in insights:
            verified_sq = []
            for sq in insight.get("supporting_quotes", []):
                total_quotes += 1
                quote_text = sq["text"]
                offsets = find_quote_offset(transcript, quote_text)
                if offsets:
                    verified_quotes += 1
                    verified_sq.append(
                        {
                            "text": quote_text,
                            "char_start": offsets[0],
                            "char_end": offsets[1],
                        }
                    )
                else:
                    print(f"    ⚠ Quote not found: " f"'{quote_text[:60]}...'")

            verified_insights.append(
                {
                    "text": insight["text"],
                    "insight_type": insight.get("insight_type", "unknown"),
                    "grounded": len(verified_sq) > 0,
                    "supporting_quotes": verified_sq,
                }
            )

        # Check quote overlap within each insight
        overlap_count = 0
        for ins in verified_insights:
            sqs = ins["supporting_quotes"]
            for i_q in range(len(sqs)):
                for j_q in range(i_q + 1, len(sqs)):
                    a_start = sqs[i_q]["char_start"]
                    a_end = sqs[i_q]["char_end"]
                    b_start = sqs[j_q]["char_start"]
                    b_end = sqs[j_q]["char_end"]
                    if a_start < b_end and b_start < a_end:
                        overlap_count += 1

        avg_quotes = sum(len(ins["supporting_quotes"]) for ins in verified_insights) / max(
            len(verified_insights), 1
        )

        print(
            f"    {len(insights)} insights, "
            f"avg {avg_quotes:.1f} quotes/insight, "
            f"{overlap_count} overlaps, "
            f"{elapsed:.1f}s"
        )

        artifact = {
            "episode_id": ep_id,
            "dataset_id": args.dataset,
            "output": {"insights": verified_insights},
            "metadata": {
                "model": args.model,
                "total_insights": len(verified_insights),
                "avg_quotes_per_insight": round(avg_quotes, 2),
                "overlap_count": overlap_count,
                "generation_time_seconds": elapsed,
            },
        }
        all_artifacts.append(artifact)

    preds_path = output_dir / "predictions.jsonl"
    with preds_path.open("w") as f:
        for art in all_artifacts:
            f.write(json.dumps(art) + "\n")

    baseline = {
        "run_id": args.output_id,
        "dataset_id": args.dataset,
        "task": "grounded_insights_multiquote",
        "backend": {"type": "anthropic_silver", "model": args.model},
        "stats": {
            "num_episodes": len(episodes),
            "total_quotes": total_quotes,
            "verified_quotes": verified_quotes,
            "verification_rate": (round(verified_quotes / max(total_quotes, 1), 3)),
        },
    }
    (output_dir / "baseline.json").write_text(json.dumps(baseline, indent=2))

    print(f"\n{'='*60}")
    print(f"Silver: {output_dir}")
    print(f"Quotes: {verified_quotes}/{total_quotes} verified")
    print(f"Rate: " f"{verified_quotes / max(total_quotes, 1) * 100:.0f}%")


if __name__ == "__main__":
    main()
