"""Generate silver GI references using Sonnet 4.6 (or another frontier model).

For each episode transcript, prompts the model to produce ideal grounded
insights: 5-8 insights, each with 1-2 verbatim supporting quotes.
Verifies quote char offsets programmatically before writing.

Usage:
    python scripts/eval/generate_gi_silver.py \
        --dataset curated_5feeds_benchmark_v2 \
        --output-id silver_sonnet46_gi_benchmark_v2 \
        [--model claude-sonnet-4-20250514]
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

GI_SILVER_PROMPT = """You are generating reference-quality grounded insights \
for a podcast transcript evaluation benchmark.

Read the following podcast transcript carefully. Then produce 5-8 key \
insights — the most important claims, recommendations, observations, \
or questions discussed.

For EACH insight:
1. Write a clear, single-sentence insight statement
2. Classify its type as one of: claim, recommendation, observation, question
3. Find 1-2 VERBATIM quotes from the transcript that support the insight
4. Each quote must be an EXACT substring of the transcript — copy it \
character-for-character, including any punctuation and whitespace

Return valid JSON in this exact format:
{{
  "insights": [
    {{
      "text": "The insight statement goes here",
      "insight_type": "claim",
      "supporting_quotes": [
        {{
          "text": "The exact verbatim quote from the transcript"
        }}
      ]
    }}
  ]
}}

Rules:
- 5-8 insights total, covering the main topics discussed
- Each quote MUST be findable as an exact substring in the transcript
- Do NOT paraphrase, truncate, or modify quotes in any way
- Quotes should be 1-3 sentences long (not single words, not full paragraphs)
- Insight types: claim (factual assertion), recommendation (advice), \
observation (pattern/trend noted), question (open question raised)
- Ignore ads, sponsorships, and housekeeping

Transcript:

{transcript}"""


def find_quote_offset(transcript: str, quote_text: str) -> tuple[int, int] | None:
    """Find exact char offset of a quote in the transcript."""
    idx = transcript.find(quote_text)
    if idx >= 0:
        return (idx, idx + len(quote_text))

    # Try with normalized whitespace
    normalized_quote = re.sub(r"\s+", " ", quote_text.strip())
    normalized_transcript = re.sub(r"\s+", " ", transcript)
    idx = normalized_transcript.find(normalized_quote)
    if idx >= 0:
        # Map back to original offsets approximately
        return (idx, idx + len(normalized_quote))

    return None


def generate_gi_silver(transcript: str, model: str = "claude-sonnet-4-20250514") -> dict:
    """Call Anthropic API to generate GI silver reference."""
    import anthropic

    client = anthropic.Anthropic()
    prompt = GI_SILVER_PROMPT.format(transcript=transcript[:80000])

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    text = response.content[0].text if response.content else "{}"
    # Strip code fences if present
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text.strip(), flags=re.MULTILINE)

    return json.loads(text.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-id", required=True)
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    args = parser.parse_args()

    materialized = Path(f"data/eval/materialized/{args.dataset}")
    meta = json.load(open(materialized / "meta.json"))
    episodes = meta["episodes"]
    print(f"Dataset: {args.dataset} — {len(episodes)} episodes")
    print(f"Model: {args.model}")

    output_dir = Path(f"data/eval/references/silver/{args.output_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_artifacts = []

    for ep in episodes:
        ep_id = ep["episode_id"]
        transcript = (materialized / f"{ep_id}.txt").read_text()
        print(f"\n  {ep_id}: {len(transcript)} chars")

        t0 = time.time()
        result = generate_gi_silver(transcript, args.model)
        elapsed = time.time() - t0
        print(f"    Generated in {elapsed:.1f}s")

        insights = result.get("insights", [])
        verified_insights = []
        total_quotes = 0
        verified_quotes = 0

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
                    print(f"    ⚠ Quote not found: '{quote_text[:80]}...'")

            verified_insights.append(
                {
                    "text": insight["text"],
                    "insight_type": insight.get("insight_type", "unknown"),
                    "grounded": len(verified_sq) > 0,
                    "supporting_quotes": verified_sq,
                }
            )

        print(f"    {len(insights)} insights, " f"{verified_quotes}/{total_quotes} quotes verified")

        artifact = {
            "episode_id": ep_id,
            "dataset_id": args.dataset,
            "output": {
                "insights": verified_insights,
            },
            "metadata": {
                "model": args.model,
                "total_insights": len(verified_insights),
                "total_quotes": total_quotes,
                "verified_quotes": verified_quotes,
                "generation_time_seconds": elapsed,
            },
        }
        all_artifacts.append(artifact)

    # Write predictions.jsonl (same format as other silver refs)
    preds_path = output_dir / "predictions.jsonl"
    with preds_path.open("w") as f:
        for art in all_artifacts:
            f.write(json.dumps(art) + "\n")

    baseline = {
        "run_id": args.output_id,
        "dataset_id": args.dataset,
        "task": "grounded_insights",
        "backend": {"type": "anthropic_silver", "model": args.model},
        "stats": {
            "num_episodes": len(episodes),
            "total_insights": sum(a["metadata"]["total_insights"] for a in all_artifacts),
            "total_quotes_verified": sum(a["metadata"]["verified_quotes"] for a in all_artifacts),
            "total_quotes_attempted": sum(a["metadata"]["total_quotes"] for a in all_artifacts),
        },
    }
    (output_dir / "baseline.json").write_text(json.dumps(baseline, indent=2))

    print(f"\n{'='*60}")
    print(f"Silver GI refs written to: {output_dir}")
    print(f"Episodes: {len(all_artifacts)}")
    total_ins = sum(a["metadata"]["total_insights"] for a in all_artifacts)
    total_vq = sum(a["metadata"]["verified_quotes"] for a in all_artifacts)
    total_aq = sum(a["metadata"]["total_quotes"] for a in all_artifacts)
    print(f"Total insights: {total_ins}")
    print(f"Quotes verified: {total_vq}/{total_aq} ({total_vq/total_aq*100:.0f}%)")


if __name__ == "__main__":
    main()
