"""Score candidate KG predictions vs silver KG nodes — node-to-node (#1016 Phase 2c).

Companion to ``score_gi_insight_to_insight.py``. Compares the candidate's
extracted KG nodes (Topic, Entity, Insight, Claim) against the silver KG's
nodes using embedding cosine similarity. Coverage = fraction of silver
nodes that have a candidate node above the similarity threshold.

Three node-class buckets are reported separately and as a weighted mean,
because topic-vs-entity coverage may diverge for some models:
- ``topics``: silver Topic nodes vs candidate Topic-class nodes
- ``entities``: silver Entity nodes vs candidate Entity-class nodes
- ``claims``: silver Insight/Claim nodes vs candidate Insight/Claim-class

Usage:
    python scripts/eval/score/score_kg_node_to_node.py \\
        --run-id autoresearch_prompt_vllm_qwen3_5_35b_a3b_dev_knowledge_graph_v1 \\
        --silver silver_opus47_kg_dev_v1 \\
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


_TOPIC_TYPES = {"topic", "theme", "subject"}
_ENTITY_TYPES = {"entity", "person", "organization", "place", "product", "concept"}
_CLAIM_TYPES = {"insight", "claim", "recommendation", "observation", "argument"}


def _classify(ntype: str) -> str:
    t = (ntype or "").lower()
    if t in _TOPIC_TYPES:
        return "topics"
    if t in _ENTITY_TYPES:
        return "entities"
    if t in _CLAIM_TYPES:
        return "claims"
    return "other"


def _node_text(node: dict[str, Any]) -> str:
    props = node.get("properties") or {}
    return (
        node.get("text")
        or node.get("label")
        or node.get("name")
        or props.get("label")
        or props.get("name")
        or props.get("description")
        or props.get("text")
        or props.get("statement")
        or ""
    )


def extract_buckets(nodes: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Split a list of KG nodes into (topics, entities, claims) text buckets."""
    out: dict[str, list[str]] = {"topics": [], "entities": [], "claims": []}
    for n in nodes:
        if not isinstance(n, dict):
            continue
        bucket = _classify(n.get("type", ""))
        if bucket == "other":
            continue
        text = _node_text(n)
        if text:
            out[bucket].append(text)
    return out


def extract_silver_nodes(silver_entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull node list from a silver KG entry. Supports three shapes:
    1. ``output.kg.nodes``
    2. ``output.nodes``
    3. ``output.topics`` + ``output.entities`` (flat) — synthesises typed
       node dicts so downstream bucket-extraction works uniformly.
    """
    output = silver_entry.get("output", {})
    kg = output.get("kg")
    if isinstance(kg, dict):
        nodes = kg.get("nodes")
        if isinstance(nodes, list):
            return nodes
    nodes = output.get("nodes")
    if isinstance(nodes, list):
        return nodes
    synth: list[dict[str, Any]] = []
    topics = output.get("topics")
    if isinstance(topics, list):
        for t in topics:
            if isinstance(t, dict):
                synth.append({"type": "Topic", **t})
            elif isinstance(t, str):
                synth.append({"type": "Topic", "label": t})
    entities = output.get("entities")
    if isinstance(entities, list):
        for e in entities:
            if isinstance(e, dict):
                synth.append({"type": "Entity", **e})
            elif isinstance(e, str):
                synth.append({"type": "Entity", "label": e})
    return synth


def extract_candidate_nodes(pred: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull node list from a candidate KG prediction."""
    output = pred.get("output", {})
    # Try nested: output.kg.nodes
    kg = output.get("kg")
    if isinstance(kg, dict):
        nodes = kg.get("nodes")
        if isinstance(nodes, list):
            return nodes
    # Try flat: output.nodes
    nodes = output.get("nodes")
    if isinstance(nodes, list):
        return nodes
    # Try gil: some pipelines may put KG under gil
    gil = output.get("gil")
    if isinstance(gil, dict):
        nodes = gil.get("nodes")
        if isinstance(nodes, list):
            return nodes
    return []


def compute_bucket_coverage(
    cand_texts: list[str],
    silver_texts: list[str],
    embedder: Any,
    threshold: float,
) -> dict[str, Any]:
    if not silver_texts:
        return {
            "covered_count": 0,
            "total_silver": 0,
            "coverage_rate": 0.0,
            "avg_max_similarity": 0.0,
            "candidate_count": len(cand_texts),
        }
    if not cand_texts:
        return {
            "covered_count": 0,
            "total_silver": len(silver_texts),
            "coverage_rate": 0.0,
            "avg_max_similarity": 0.0,
            "candidate_count": 0,
        }
    cand_embs = embedder.encode(cand_texts, normalize_embeddings=True)
    silver_embs = embedder.encode(silver_texts, normalize_embeddings=True)
    sim_matrix = np.dot(silver_embs, cand_embs.T)
    max_sims = sim_matrix.max(axis=1)
    covered = int((max_sims >= threshold).sum())
    return {
        "covered_count": covered,
        "total_silver": len(silver_texts),
        "coverage_rate": covered / len(silver_texts),
        "avg_max_similarity": float(max_sims.mean()),
        "candidate_count": len(cand_texts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--silver", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()

    silver_path = Path(f"data/eval/references/silver/{args.silver}/predictions.jsonl")
    silver_by_ep: dict[str, list[dict[str, Any]]] = {}
    for line in silver_path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        silver_by_ep[d["episode_id"]] = extract_silver_nodes(d)

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
    totals = {
        b: {"covered": 0, "silver": 0, "candidate": 0, "sims": []}
        for b in ("topics", "entities", "claims")
    }
    missing = 0

    for ep_id in sorted(silver_by_ep.keys()):
        silver_nodes = silver_by_ep[ep_id]
        pred = preds_by_ep.get(ep_id)
        if not pred:
            missing += 1
            per_episode_results.append({"episode_id": ep_id, "status": "missing"})
            print(f"  {ep_id}: MISSING prediction")
            continue
        silver_buckets = extract_buckets(silver_nodes)
        cand_buckets = extract_buckets(extract_candidate_nodes(pred))

        ep_result: dict[str, Any] = {"episode_id": ep_id, "buckets": {}}
        for bucket in ("topics", "entities", "claims"):
            r = compute_bucket_coverage(
                cand_buckets[bucket], silver_buckets[bucket], embedder, args.threshold
            )
            ep_result["buckets"][bucket] = r
            totals[bucket]["covered"] += r["covered_count"]
            totals[bucket]["silver"] += r["total_silver"]
            totals[bucket]["candidate"] += r["candidate_count"]
            if r["total_silver"] > 0:
                totals[bucket]["sims"].append(r["avg_max_similarity"])
        per_episode_results.append(ep_result)
        topic_rate = ep_result["buckets"]["topics"]["coverage_rate"]
        entity_rate = ep_result["buckets"]["entities"]["coverage_rate"]
        claim_rate = ep_result["buckets"]["claims"]["coverage_rate"]
        print(
            f"  {ep_id}: topics={topic_rate:.0%} entities={entity_rate:.0%} claims={claim_rate:.0%}"
        )

    print()
    overall_payload: dict[str, Any] = {"per_bucket": {}}
    weighted_rate = 0.0
    weight_sum = 0.0
    for bucket in ("topics", "entities", "claims"):
        t = totals[bucket]
        rate = t["covered"] / t["silver"] if t["silver"] else 0.0
        avg_sim = float(np.mean(t["sims"])) if t["sims"] else 0.0
        overall_payload["per_bucket"][bucket] = {
            "covered_count": t["covered"],
            "total_silver": t["silver"],
            "coverage_rate": rate,
            "avg_max_similarity": avg_sim,
            "total_candidate": t["candidate"],
        }
        if t["silver"] > 0:
            weighted_rate += rate * t["silver"]
            weight_sum += t["silver"]
        print(
            f"  {bucket}: silver={t['silver']} cand={t['candidate']} "
            f"covered={t['covered']} ({rate:.0%}) sim={avg_sim:.3f}"
        )
    overall_rate = weighted_rate / weight_sum if weight_sum else 0.0
    overall_payload["overall_weighted_coverage_rate"] = overall_rate
    print()
    print(f"OVERALL WEIGHTED COVERAGE: {overall_rate:.0%}")
    print(f"MISSING: {missing} / {len(silver_by_ep)}")

    output_name = args.output_name or f"metrics_vs_{args.silver}.json"
    output_path = run_dir / output_name
    payload = {
        "reference_id": args.silver,
        "reference_quality": "silver",
        "dataset_id": args.dataset,
        "run_id": args.run_id,
        "metric_type": "kg_node_to_node",
        "threshold": args.threshold,
        "vs_reference": overall_payload,
        "per_episode": per_episode_results,
    }
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
