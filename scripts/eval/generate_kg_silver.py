"""Generate silver KG references using Sonnet 4.6.

For each episode transcript, prompts the model to produce ideal KG extraction:
topics (2-8 word noun phrases) and entities (person/org with roles).

Usage:
    python scripts/eval/generate_kg_silver.py \
        --dataset curated_5feeds_benchmark_v2 \
        --output-id silver_sonnet46_kg_benchmark_v2
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

KG_SILVER_PROMPT = """\
You are generating reference-quality knowledge graph extraction for a \
podcast transcript evaluation benchmark.

Read the following podcast transcript carefully. Extract:

1. **Topics** (8-12): the main themes, subjects, and discussion areas. \
Each topic should be a 2-8 word noun phrase (e.g., "mountain bike \
suspension tuning", "on-call rotation design"). Cover the major themes \
discussed, ordered by importance.

2. **Entities** (all mentioned): people and organizations referenced in \
the episode. For each, indicate:
   - name: full name as it appears in the transcript
   - kind: "person" or "org"
   - role: "host" (if they run the show), "guest" (if interviewed), \
or "mentioned" (if just referenced)

Return valid JSON:
{{
  "topics": [
    {{
      "label": "2-8 word noun phrase",
      "description": "1-2 sentence context from this episode"
    }}
  ],
  "entities": [
    {{
      "name": "Full Name",
      "kind": "person",
      "role": "host",
      "description": "1 sentence context"
    }}
  ]
}}

Rules:
- Topics should be noun phrases, NOT full sentences
- Entity names must appear verbatim in the transcript
- Include ALL people mentioned by name (hosts, guests, referenced experts)
- Include show/podcast name as an org entity
- Order topics by importance (most important first)

Transcript:

{transcript}"""


def generate_kg_silver(transcript: str, model: str = "claude-sonnet-4-20250514") -> dict:
    """Call Anthropic API to generate KG silver reference."""
    import anthropic

    client = anthropic.Anthropic()
    prompt = KG_SILVER_PROMPT.format(transcript=transcript[:80000])

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    text = response.content[0].text if response.content else "{}"
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text.strip(), flags=re.MULTILINE)

    return json.loads(text.strip(), strict=False)


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
        result = generate_kg_silver(transcript, args.model)
        elapsed = time.time() - t0

        topics = result.get("topics", [])
        entities = result.get("entities", [])

        # Verify entity names appear in transcript
        verified_entities = []
        for ent in entities:
            name = ent.get("name", "")
            if name and name in transcript:
                verified_entities.append(ent)
            else:
                print(f"    ⚠ Entity not in transcript: '{name}'")

        print(
            f"    {len(topics)} topics, "
            f"{len(verified_entities)}/{len(entities)} entities verified, "
            f"{elapsed:.1f}s"
        )

        artifact = {
            "episode_id": ep_id,
            "dataset_id": args.dataset,
            "output": {
                "topics": topics,
                "entities": verified_entities,
            },
            "metadata": {
                "model": args.model,
                "total_topics": len(topics),
                "total_entities": len(entities),
                "verified_entities": len(verified_entities),
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
        "task": "knowledge_graph",
        "backend": {"type": "anthropic_silver", "model": args.model},
        "stats": {
            "num_episodes": len(episodes),
            "total_topics": sum(a["metadata"]["total_topics"] for a in all_artifacts),
            "total_entities_verified": sum(
                a["metadata"]["verified_entities"] for a in all_artifacts
            ),
        },
    }
    (output_dir / "baseline.json").write_text(json.dumps(baseline, indent=2))

    total_t = sum(a["metadata"]["total_topics"] for a in all_artifacts)
    total_ev = sum(a["metadata"]["verified_entities"] for a in all_artifacts)
    total_ea = sum(a["metadata"]["total_entities"] for a in all_artifacts)
    print(f"\n{'='*60}")
    print(f"Silver KG refs: {output_dir}")
    print(f"Episodes: {len(all_artifacts)}")
    print(f"Topics: {total_t}")
    print(f"Entities verified: {total_ev}/{total_ea}")


if __name__ == "__main__":
    main()
