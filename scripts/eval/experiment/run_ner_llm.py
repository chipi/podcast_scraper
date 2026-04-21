"""LLM-based NER runner for eval comparison against spaCy baselines.

Calls LLM providers to extract PERSON + ORG entities from transcripts,
outputs predictions.jsonl compatible with the NER scorer (entity_set mode).

Usage:
    python scripts/eval/run_ner_llm.py \
        --provider openai --model gpt-4o-mini \
        --dataset curated_5feeds_smoke_v1 \
        --run-id ner_openai_gpt4omini_smoke_v1
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from podcast_scraper.evaluation.autoresearch_track_a import (
    load_local_dotenv_files,
)

load_local_dotenv_files(Path.cwd())
os.environ.setdefault("AUTORESEARCH_ALLOW_PRODUCTION_KEYS", "1")

NER_PROMPT = """Extract all person names and organization names from the following text.

Rules:
- PERSON: real people mentioned by name (hosts, guests, experts, public figures)
- ORG: organizations, companies, show/podcast names, institutions
- Return ONLY names that actually appear in the text
- Deduplicate: return each unique name only once
- Do NOT include generic terms like "host", "guest", "speaker"

Return JSON:
{
  "persons": ["Name One", "Name Two"],
  "organizations": ["Org One", "Org Two"]
}

Text:
"""


def call_llm_ner(provider: str, model: str, transcript: str) -> dict:
    """Call LLM provider to extract entities."""
    if provider == "openai":
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You extract named entities from text. Return valid JSON only.",
                },
                {"role": "user", "content": NER_PROMPT + transcript[:40000]},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)

    elif provider == "gemini":
        import google.genai as genai

        client = genai.Client()
        response = client.models.generate_content(
            model=model,
            contents=NER_PROMPT + transcript[:40000],
            config=genai.types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )
        return json.loads(response.text)

    elif provider == "ollama":
        from openai import OpenAI

        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You extract named entities from text. Return valid JSON only.",
                },
                {"role": "user", "content": NER_PROMPT + transcript[:40000]},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)

    else:
        raise ValueError(f"Unsupported provider: {provider}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True, choices=["openai", "gemini", "ollama"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    materialized = Path(f"data/eval/materialized/{args.dataset}")
    meta = json.load(open(materialized / "meta.json"))
    episodes = meta["episodes"]
    print(f"Dataset: {args.dataset} — {len(episodes)} episodes")
    print(f"Provider: {args.provider}/{args.model}")

    run_dir = Path(f"data/eval/runs/{args.run_id}")
    run_dir.mkdir(parents=True, exist_ok=True)

    preds_path = run_dir / "predictions.jsonl"
    durations = []

    with preds_path.open("w") as f:
        for ep in episodes:
            ep_id = ep["episode_id"]
            transcript = (materialized / f"{ep_id}.txt").read_text()
            print(f"  {ep_id}: {len(transcript)} chars -> ", end="", flush=True)

            t0 = time.time()
            try:
                result = call_llm_ner(args.provider, args.model, transcript)
            except Exception as e:
                print(f"ERROR: {e}")
                result = {"persons": [], "organizations": []}
            elapsed = time.time() - t0
            durations.append(elapsed)

            persons = result.get("persons", result.get("PERSON", []))
            orgs = result.get("organizations", result.get("ORG", []))

            entities = []
            for name in persons:
                entities.append({"start": 0, "end": len(name), "text": name, "label": "PERSON"})
            for name in orgs:
                entities.append({"start": 0, "end": len(name), "text": name, "label": "ORG"})

            print(f"{len(persons)} persons, {len(orgs)} orgs in {elapsed:.1f}s")

            pred = {
                "episode_id": ep_id,
                "dataset_id": args.dataset,
                "output": {"entities": entities},
            }
            f.write(json.dumps(pred) + "\n")

    baseline = {
        "run_id": args.run_id,
        "dataset_id": args.dataset,
        "task": "ner_entities",
        "backend": {"type": f"{args.provider}_llm", "model": args.model},
        "params": {"model": args.model, "temperature": 0.0},
        "stats": {
            "num_episodes": len(episodes),
            "total_time_seconds": sum(durations),
            "avg_time_seconds": sum(durations) / len(durations) if durations else 0,
        },
    }
    (run_dir / "baseline.json").write_text(json.dumps(baseline, indent=2))
    print(f"\nDone. Predictions: {preds_path}")


if __name__ == "__main__":
    main()
