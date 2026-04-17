"""Score KG topic coverage against silver KG references.

Compares pipeline-extracted KG topics against silver topics using
embedding cosine similarity. Same methodology as GI insight coverage.

Usage:
    python scripts/eval/score_kg_topic_coverage.py \
        --run-id <predictions_run> \
        --silver silver_sonnet46_kg_benchmark_v2 \
        --dataset curated_5feeds_benchmark_v2
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def extract_topics_from_prediction(pred: dict) -> list[str]:
    """Extract topic labels from a KG prediction."""
    output = pred.get("output", {})

    # Direct KG format: output.topics is a list of dicts with "label"
    topics = output.get("topics", [])
    if topics and isinstance(topics[0], dict):
        return [t["label"].strip() for t in topics if t.get("label")]

    # Summarization format: extract from summary_final JSON (bundled)
    summary = output.get("summary_final", "")
    if summary:
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

    return []


def extract_entities_from_prediction(pred: dict) -> list[dict]:
    """Extract entity dicts from a KG prediction."""
    output = pred.get("output", {})
    entities = output.get("entities", [])
    return [e for e in entities if isinstance(e, dict) and e.get("name")]


def compute_topic_coverage(
    pred_topics: list[str],
    silver_topics: list[dict],
    embedder: object,
    threshold: float = 0.65,
) -> dict:
    """Compute topic coverage using embedding similarity."""
    if not pred_topics or not silver_topics:
        return {
            "covered": 0,
            "total": len(silver_topics),
            "rate": 0.0,
            "avg_sim": 0.0,
        }

    silver_labels = [t["label"] for t in silver_topics]
    p_embs = embedder.encode(pred_topics, normalize_embeddings=True)  # type: ignore[union-attr]
    s_embs = embedder.encode(silver_labels, normalize_embeddings=True)  # type: ignore[union-attr]
    sim = np.dot(s_embs, p_embs.T)

    covered = 0
    max_sims = []
    for i in range(len(silver_topics)):
        ms = float(np.max(sim[i]))
        max_sims.append(ms)
        if ms >= threshold:
            covered += 1

    return {
        "covered": covered,
        "total": len(silver_topics),
        "rate": covered / len(silver_topics),
        "avg_sim": float(np.mean(max_sims)),
    }


def compute_entity_coverage(
    pred_entities: list[dict],
    silver_entities: list[dict],
) -> dict:
    """Compute entity coverage using name matching."""
    if not silver_entities:
        return {"covered": 0, "total": 0, "rate": 0.0, "extra": 0}

    silver_names = {e["name"].lower().strip() for e in silver_entities}
    pred_names = {e["name"].lower().strip() for e in pred_entities}

    covered = len(silver_names & pred_names)
    extra = len(pred_names - silver_names)

    return {
        "covered": covered,
        "total": len(silver_names),
        "rate": covered / len(silver_names) if silver_names else 0.0,
        "extra": extra,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--silver", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--threshold", type=float, default=0.65)
    args = parser.parse_args()

    silver_path = Path(f"data/eval/references/silver/{args.silver}/predictions.jsonl")
    silver_by_ep = {}
    for line in silver_path.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            silver_by_ep[d["episode_id"]] = d["output"]

    preds_path = Path(f"data/eval/runs/{args.run_id}/predictions.jsonl")
    preds_by_ep = {}
    for line in preds_path.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            preds_by_ep[d["episode_id"]] = d

    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Run: {args.run_id}")
    print(f"Silver: {args.silver}")
    print()

    tc = 0
    tt = 0
    ec = 0
    et = 0
    all_sims = []

    for ep_id in sorted(silver_by_ep.keys()):
        silver = silver_by_ep[ep_id]
        pred = preds_by_ep.get(ep_id)

        if not pred:
            print(f"  {ep_id}: MISSING")
            tt += len(silver.get("topics", []))
            et += len(silver.get("entities", []))
            continue

        pred_topics = extract_topics_from_prediction(pred)
        t_cov = compute_topic_coverage(
            pred_topics,
            silver.get("topics", []),
            embedder,
            args.threshold,
        )

        pred_entities = extract_entities_from_prediction(pred)
        e_cov = compute_entity_coverage(pred_entities, silver.get("entities", []))

        tc += t_cov["covered"]
        tt += t_cov["total"]
        ec += e_cov["covered"]
        et += e_cov["total"]
        all_sims.append(t_cov["avg_sim"])

        print(
            f"  {ep_id}: topics {t_cov['covered']}/{t_cov['total']} "
            f"({t_cov['rate']:.0%}) sim={t_cov['avg_sim']:.3f}, "
            f"entities {e_cov['covered']}/{e_cov['total']} "
            f"({e_cov['rate']:.0%}) +{e_cov['extra']} extra, "
            f"pred_topics={len(pred_topics)}"
        )

    print()
    t_rate = tc / tt if tt else 0
    e_rate = ec / et if et else 0
    avg_sim = float(np.mean(all_sims)) if all_sims else 0
    print(f"TOPICS:   {tc}/{tt} covered ({t_rate:.0%}), avg_sim={avg_sim:.3f}")
    print(f"ENTITIES: {ec}/{et} covered ({e_rate:.0%})")


if __name__ == "__main__":
    main()
