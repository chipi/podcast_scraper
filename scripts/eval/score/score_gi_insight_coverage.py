"""Score summary bullet coverage against GI silver insights.

For each episode, compares the summary's bullet points against the silver
GI reference insights using embedding cosine similarity. Measures: do
better summaries produce bullets that capture the right insights?

Usage:
    python scripts/eval/score_gi_insight_coverage.py \
        --run-id <predictions_run> \
        --silver silver_sonnet46_gi_benchmark_v2 \
        --dataset curated_5feeds_benchmark_v2
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def extract_bullets_from_prediction(pred: dict) -> list[str]:
    """Extract bullet points from a prediction's summary_final field."""
    output = pred.get("output", {})
    summary = output.get("summary_final", "")

    # Try parsing as JSON (bundled/bullets runs store JSON in summary_final)
    # Strip markdown code fences first (Gemini wraps JSON in ```json ... ```)
    clean = re.sub(r"^```(?:json)?\s*\n?", "", summary.strip(), flags=re.MULTILINE)
    clean = re.sub(r"\n?```\s*$", "", clean.strip(), flags=re.MULTILINE)
    try:
        parsed = json.loads(clean.strip(), strict=False)
        if isinstance(parsed, dict):
            bullets = parsed.get("bullets", [])
            if bullets:
                return [b.strip() for b in bullets if isinstance(b, str) and b.strip()]
    except (json.JSONDecodeError, TypeError):
        pass

    # Fall back to extracting bullet lines from plain text
    bullets = []
    for line in summary.split("\n"):
        line = line.strip()
        if line.startswith(("- ", "• ", "* ", "– ")):
            bullets.append(line.lstrip("-•*– ").strip())
        elif re.match(r"^\d+[\.\)]\s", line):
            bullets.append(re.sub(r"^\d+[\.\)]\s*", "", line).strip())

    # If no bullets found, split paragraph into sentences as fallback
    if not bullets and summary:
        sentences = re.split(r"(?<=[.!?])\s+", summary.strip())
        bullets = [s.strip() for s in sentences if len(s.strip()) > 20]

    return bullets


def compute_coverage(
    bullets: list[str],
    silver_insights: list[dict],
    embedder: object,
    threshold: float = 0.65,
) -> dict:
    """Compute how many silver insights are covered by the bullets.

    Returns:
        Dict with coverage metrics: covered_count, total_insights,
        coverage_rate, avg_max_similarity, per_insight details.
    """
    if not bullets or not silver_insights:
        return {
            "covered_count": 0,
            "total_insights": len(silver_insights),
            "coverage_rate": 0.0,
            "avg_max_similarity": 0.0,
            "per_insight": [],
        }

    insight_texts = [ins["text"] for ins in silver_insights]

    # Encode all texts
    bullet_embs = embedder.encode(bullets, normalize_embeddings=True)  # type: ignore[union-attr]
    insight_embs = embedder.encode(  # type: ignore[union-attr]
        insight_texts, normalize_embeddings=True
    )

    # Compute similarity matrix: insights × bullets
    sim_matrix = np.dot(insight_embs, bullet_embs.T)

    per_insight = []
    covered = 0
    max_sims = []

    for i, insight in enumerate(silver_insights):
        max_sim = float(np.max(sim_matrix[i]))
        best_bullet_idx = int(np.argmax(sim_matrix[i]))
        is_covered = max_sim >= threshold

        if is_covered:
            covered += 1
        max_sims.append(max_sim)

        per_insight.append(
            {
                "insight": insight["text"],
                "insight_type": insight.get("insight_type", "unknown"),
                "best_bullet": bullets[best_bullet_idx] if bullets else "",
                "max_similarity": round(max_sim, 3),
                "covered": is_covered,
            }
        )

    return {
        "covered_count": covered,
        "total_insights": len(silver_insights),
        "coverage_rate": covered / len(silver_insights) if silver_insights else 0.0,
        "avg_max_similarity": float(np.mean(max_sims)) if max_sims else 0.0,
        "per_insight": per_insight,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, help="Predictions run to score")
    parser.add_argument("--silver", required=True, help="Silver GI reference ID")
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--threshold", type=float, default=0.65, help="Cosine sim threshold for coverage"
    )
    args = parser.parse_args()

    # Load silver GI refs
    silver_path = Path(f"data/eval/references/silver/{args.silver}/predictions.jsonl")
    silver_by_ep = {}
    for line in silver_path.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            silver_by_ep[d["episode_id"]] = d["output"]["insights"]

    # Load predictions
    preds_path = Path(f"data/eval/runs/{args.run_id}/predictions.jsonl")
    preds_by_ep = {}
    for line in preds_path.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            preds_by_ep[d["episode_id"]] = d

    # Load embedding model
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Run: {args.run_id}")
    print(f"Silver: {args.silver}")
    print(f"Threshold: {args.threshold}")
    print(f"Episodes: {len(silver_by_ep)}")
    print()

    total_covered = 0
    total_insights = 0
    all_sims = []

    for ep_id in sorted(silver_by_ep.keys()):
        insights = silver_by_ep[ep_id]
        pred = preds_by_ep.get(ep_id)

        if not pred:
            print(f"  {ep_id}: MISSING prediction")
            total_insights += len(insights)
            continue

        bullets = extract_bullets_from_prediction(pred)
        result = compute_coverage(bullets, insights, embedder, args.threshold)

        total_covered += result["covered_count"]
        total_insights += result["total_insights"]
        all_sims.append(result["avg_max_similarity"])

        print(
            f"  {ep_id}: {result['covered_count']}/{result['total_insights']} covered "
            f"({result['coverage_rate']:.0%}), "
            f"avg_sim={result['avg_max_similarity']:.3f}, "
            f"bullets={len(bullets)}"
        )

    print()
    rate = total_covered / total_insights if total_insights else 0
    avg_sim = float(np.mean(all_sims)) if all_sims else 0
    print(f"OVERALL: {total_covered}/{total_insights} insights covered ({rate:.0%})")
    print(f"AVG MAX SIMILARITY: {avg_sim:.3f}")


if __name__ == "__main__":
    main()
