"""Score candidate GI predictions vs silver GI insights — insight-to-insight (#1016 Phase 2c).

Companion to ``score_gi_insight_coverage.py``. The original scorer measures
whether a SUMMARY's bullets cover silver GI insights (summary-side eval).
This scorer compares candidate-EXTRACTED insights vs silver insights
(GI-side eval) — required when the candidate is being evaluated as the
GI extractor, not as the summarizer.

For each episode, embeds both the candidate's extracted insights and the
silver insights with all-MiniLM-L6-v2, then computes for each silver
insight: max cosine similarity vs any candidate insight. Coverage =
fraction of silver insights with at least one candidate insight above
threshold.

Output shape mirrors ``score_gi_insight_coverage.py`` so dashboards and
reports can read both with the same loader.

Usage:
    python scripts/eval/score/score_gi_insight_to_insight.py \\
        --run-id autoresearch_prompt_vllm_qwen3_5_35b_a3b_dev_grounded_insights_v1 \\
        --silver silver_opus47_gi_dev_v1 \\
        --dataset curated_5feeds_dev_v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def extract_insights_from_prediction(pred: dict[str, Any]) -> list[str]:
    """Pull insight texts from a candidate GI prediction.

    Handles three known prediction shapes:
    - ``output.insights`` (direct insight list, matches silver shape)
    - ``output.gil.nodes`` (Grounded Insights Layer with mixed node types —
      filter to nodes whose type is Insight / Claim / Recommendation)
    - ``output.gil.insights`` (some pipelines nest insights under gil)
    """
    output = pred.get("output", {})

    direct = output.get("insights")
    if isinstance(direct, list):
        return [ins.get("text", "") for ins in direct if isinstance(ins, dict) and ins.get("text")]

    gil = output.get("gil")
    if isinstance(gil, dict):
        gil_insights = gil.get("insights")
        if isinstance(gil_insights, list):
            return [
                ins.get("text", "")
                for ins in gil_insights
                if isinstance(ins, dict) and ins.get("text")
            ]
        nodes = gil.get("nodes")
        if isinstance(nodes, list):
            texts: list[str] = []
            for n in nodes:
                if not isinstance(n, dict):
                    continue
                ntype = (n.get("type") or "").lower()
                if ntype in ("insight", "claim", "recommendation", "observation"):
                    props = n.get("properties", {}) or {}
                    txt = (
                        n.get("text")
                        or props.get("text")
                        or props.get("statement")
                        or props.get("label")
                        or ""
                    )
                    if txt:
                        texts.append(txt)
            if texts:
                return texts

    return []


def compute_insight_to_insight_coverage(
    cand_insights: list[str],
    silver_insights: list[dict[str, Any]],
    embedder: Any,
    threshold: float,
) -> dict[str, Any]:
    """Return coverage metrics: how many silver insights have a candidate match."""
    if not cand_insights or not silver_insights:
        return {
            "covered_count": 0,
            "total_insights": len(silver_insights),
            "coverage_rate": 0.0,
            "avg_max_similarity": 0.0,
            "candidate_insight_count": len(cand_insights),
            "per_insight": [],
        }

    silver_texts = [ins["text"] for ins in silver_insights]

    cand_embs = embedder.encode(cand_insights, normalize_embeddings=True)
    silver_embs = embedder.encode(silver_texts, normalize_embeddings=True)

    sim_matrix = np.dot(silver_embs, cand_embs.T)

    per_insight = []
    covered = 0
    max_sims = []
    for i, ins in enumerate(silver_insights):
        max_sim = float(np.max(sim_matrix[i]))
        best_cand_idx = int(np.argmax(sim_matrix[i]))
        is_covered = max_sim >= threshold
        if is_covered:
            covered += 1
        max_sims.append(max_sim)
        per_insight.append(
            {
                "silver_insight": ins["text"],
                "silver_insight_type": ins.get("insight_type", "unknown"),
                "best_candidate_insight": cand_insights[best_cand_idx],
                "max_similarity": round(max_sim, 3),
                "covered": is_covered,
            }
        )

    return {
        "covered_count": covered,
        "total_insights": len(silver_insights),
        "coverage_rate": covered / len(silver_insights),
        "avg_max_similarity": float(np.mean(max_sims)),
        "candidate_insight_count": len(cand_insights),
        "per_insight": per_insight,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--silver", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument(
        "--output-name",
        default=None,
        help="Output JSON filename inside the run dir. Default: " "metrics_vs_<silver>.json",
    )
    args = parser.parse_args()

    silver_path = Path(f"data/eval/references/silver/{args.silver}/predictions.jsonl")
    silver_by_ep: dict[str, list[dict[str, Any]]] = {}
    for line in silver_path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        silver_by_ep[d["episode_id"]] = d["output"]["insights"]

    run_dir = Path(f"data/eval/runs/{args.run_id}")
    preds_path = run_dir / "predictions.jsonl"
    preds_by_ep: dict[str, dict[str, Any]] = {}
    for line in preds_path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        preds_by_ep[d["episode_id"]] = d

    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Run: {args.run_id}")
    print(f"Silver: {args.silver}")
    print(f"Threshold: {args.threshold}")
    print(f"Episodes: {len(silver_by_ep)}")
    print()

    per_episode_results = []
    total_covered = 0
    total_silver = 0
    total_cand = 0
    all_sims = []
    missing = 0

    for ep_id in sorted(silver_by_ep.keys()):
        silver_ins = silver_by_ep[ep_id]
        pred = preds_by_ep.get(ep_id)
        if not pred:
            missing += 1
            total_silver += len(silver_ins)
            print(f"  {ep_id}: MISSING prediction")
            per_episode_results.append({"episode_id": ep_id, "status": "missing"})
            continue

        cand_ins = extract_insights_from_prediction(pred)
        result = compute_insight_to_insight_coverage(cand_ins, silver_ins, embedder, args.threshold)
        per_episode_results.append({"episode_id": ep_id, **result})

        total_covered += result["covered_count"]
        total_silver += result["total_insights"]
        total_cand += result["candidate_insight_count"]
        if result["candidate_insight_count"] > 0:
            all_sims.append(result["avg_max_similarity"])

        print(
            f"  {ep_id}: silver={result['total_insights']} "
            f"cand={result['candidate_insight_count']} "
            f"covered={result['covered_count']} ({result['coverage_rate']:.0%}) "
            f"avg_sim={result['avg_max_similarity']:.3f}"
        )

    overall_rate = total_covered / total_silver if total_silver else 0.0
    avg_sim = float(np.mean(all_sims)) if all_sims else 0.0

    print()
    print(
        f"OVERALL: silver={total_silver} cand={total_cand} "
        f"covered={total_covered} ({overall_rate:.0%})"
    )
    print(f"AVG MAX SIMILARITY: {avg_sim:.3f}")
    print(f"MISSING: {missing} / {len(silver_by_ep)}")

    output_name = args.output_name or f"metrics_vs_{args.silver}.json"
    output_path = run_dir / output_name
    payload = {
        "reference_id": args.silver,
        "reference_quality": "silver",
        "dataset_id": args.dataset,
        "run_id": args.run_id,
        "metric_type": "gi_insight_to_insight",
        "threshold": args.threshold,
        "vs_reference": {
            "covered_count": total_covered,
            "total_silver_insights": total_silver,
            "total_candidate_insights": total_cand,
            "coverage_rate": overall_rate,
            "avg_max_similarity": avg_sim,
            "missing_episodes": missing,
            "n_episodes": len(silver_by_ep),
        },
        "per_episode": per_episode_results,
    }
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
