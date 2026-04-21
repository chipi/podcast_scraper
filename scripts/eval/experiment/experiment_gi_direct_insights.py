"""Experiment: direct LLM insight generation vs summary-derived bullets.

Bypasses the summary step entirely — asks the LLM to extract insights
directly from the transcript. Tests whether this closes the 30% coverage
gap observed when using summary bullets.

Also tests varying the number of requested insights (5, 8, 12).

Usage:
    python scripts/eval/experiment_gi_direct_insights.py \
        --dataset curated_5feeds_benchmark_v2 \
        --silver silver_sonnet46_gi_benchmark_v2
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from podcast_scraper.evaluation.autoresearch_track_a import (
    load_local_dotenv_files,
)

load_local_dotenv_files(Path.cwd())
os.environ.setdefault("AUTORESEARCH_ALLOW_PRODUCTION_KEYS", "1")

DIRECT_INSIGHT_PROMPT = """Read the following podcast transcript and extract \
exactly {n} key insights — the most important claims, recommendations, \
observations, or questions discussed.

Each insight should be a clear, single-sentence statement that captures \
a distinct idea from the transcript. Cover the main topics in order.

Return valid JSON:
{{"insights": ["Insight one.", "Insight two.", ...]}}

Transcript:

{transcript}"""


def extract_direct_insights(
    transcript: str,
    n: int,
    provider: str,
    model: str,
) -> list[str]:
    """Extract insights directly from transcript via LLM."""
    prompt = DIRECT_INSIGHT_PROMPT.format(n=n, transcript=transcript[:60000])

    if provider == "gemini":
        import google.genai as genai

        client = genai.Client()
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )
        raw = str(response.text or "{}").strip()
        try:
            parsed = json.loads(raw, strict=False)
        except json.JSONDecodeError:
            parsed = json.loads(raw.split("\n")[0], strict=False)

    elif provider == "openai":
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Extract key insights from text. Return JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(response.choices[0].message.content or "{}")

    elif provider == "ollama":
        from openai import OpenAI

        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Extract key insights from text. Return JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(response.choices[0].message.content or "{}")

    else:
        raise ValueError(f"Unsupported provider: {provider}")

    insights = parsed.get("insights", [])
    return [s.strip() for s in insights if isinstance(s, str) and s.strip()]


def compute_coverage(
    bullets: list[str],
    silver_insights: list[dict],
    embedder: object,
    threshold: float = 0.65,
) -> dict:
    """Compute insight coverage."""
    if not bullets or not silver_insights:
        return {
            "covered": 0,
            "total": len(silver_insights),
            "rate": 0.0,
            "avg_sim": 0.0,
        }

    insight_texts = [ins["text"] for ins in silver_insights]
    b_embs = embedder.encode(bullets, normalize_embeddings=True)  # type: ignore[union-attr]
    i_embs = embedder.encode(insight_texts, normalize_embeddings=True)  # type: ignore[union-attr]
    sim = np.dot(i_embs, b_embs.T)

    covered = 0
    max_sims = []
    for i in range(len(silver_insights)):
        ms = float(np.max(sim[i]))
        max_sims.append(ms)
        if ms >= threshold:
            covered += 1

    return {
        "covered": covered,
        "total": len(silver_insights),
        "rate": covered / len(silver_insights),
        "avg_sim": float(np.mean(max_sims)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--silver", required=True)
    parser.add_argument(
        "--providers",
        default="gemini:gemini-2.5-flash-lite",
        help="comma-sep provider:model pairs",
    )
    parser.add_argument(
        "--counts",
        default="5,8,12",
        help="comma-sep insight counts to test",
    )
    parser.add_argument("--threshold", type=float, default=0.65)
    args = parser.parse_args()

    materialized = Path(f"data/eval/materialized/{args.dataset}")
    meta = json.load(open(materialized / "meta.json"))
    episodes = meta["episodes"]

    silver_path = Path(f"data/eval/references/silver/{args.silver}/predictions.jsonl")
    silver_by_ep = {}
    for line in silver_path.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            silver_by_ep[d["episode_id"]] = d["output"]["insights"]

    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    counts = [int(c) for c in args.counts.split(",")]
    providers = [p.split(":") for p in args.providers.split(",")]

    print(f"Dataset: {args.dataset} ({len(episodes)} episodes)")
    print(f"Silver: {args.silver}")
    print(f"Providers: {providers}")
    print(f"Insight counts: {counts}")
    print(f"Threshold: {args.threshold}")
    print()

    # Results matrix
    results = []

    for prov_name, model_name in providers:
        for n in counts:
            total_covered = 0
            total_insights = 0
            all_sims = []
            t0 = time.time()

            for ep in episodes:
                ep_id = ep["episode_id"]
                if ep_id not in silver_by_ep:
                    continue
                transcript = (materialized / f"{ep_id}.txt").read_text()
                insights_direct = extract_direct_insights(transcript, n, prov_name, model_name)
                cov = compute_coverage(
                    insights_direct,
                    silver_by_ep[ep_id],
                    embedder,
                    args.threshold,
                )
                total_covered += cov["covered"]
                total_insights += cov["total"]
                all_sims.append(cov["avg_sim"])

            elapsed = time.time() - t0
            rate = total_covered / total_insights if total_insights else 0
            avg_sim = float(np.mean(all_sims)) if all_sims else 0

            results.append(
                {
                    "provider": prov_name,
                    "model": model_name,
                    "n_insights": n,
                    "covered": total_covered,
                    "total": total_insights,
                    "rate": rate,
                    "avg_sim": avg_sim,
                    "time_s": elapsed,
                }
            )
            print(
                f"  {prov_name}/{model_name} n={n:2d}: "
                f"{total_covered}/{total_insights} covered "
                f"({rate:.0%}), sim={avg_sim:.3f}, {elapsed:.1f}s"
            )

    print()
    print("=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Provider':<30s} {'N':>3s} " f"{'Covered':>8s} {'Rate':>6s} {'Sim':>6s}")
    print("-" * 60)

    # Add summary-derived baselines for comparison
    print("--- Summary-derived (from v2 runs, for comparison) ---")
    print(f"{'bart-led (summary)':<30s} {'~8':>3s} " f"{'3/40':>8s} {'8%':>6s} {'0.393':>6s}")
    print(
        f"{'qwen3.5:9b bundled (summary)':<30s} {'~9':>3s} "
        f"{'28/40':>8s} {'70%':>6s} {'0.748':>6s}"
    )
    print(
        f"{'gemini flash-lite (summary)':<30s} {'~7':>3s} "
        f"{'29/40':>8s} {'72%':>6s} {'0.762':>6s}"
    )
    print("--- Direct from transcript (this experiment) ---")
    for r in results:
        print(
            f"{r['provider']+'/'+r['model']:<30s} {r['n_insights']:>3d} "
            f"{r['covered']}/{r['total']:>3d}"
            f"{'':>2s} {r['rate']:.0%}"
            f"{'':>2s} {r['avg_sim']:.3f}"
        )


if __name__ == "__main__":
    main()
