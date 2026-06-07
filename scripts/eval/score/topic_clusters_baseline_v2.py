"""Topic-clusters baseline over v2 KG outputs (issue #903 AC).

Embeds KG topic labels with the same MiniLM model the pipeline uses, builds
cosine clusters with the pipeline's default threshold, and emits a baseline
measuring `tc:` parents per podcast plus cross-feed `tc:` count.

Inputs:
- KG predictions.jsonl from a `task: knowledge_graph` run on the v2 dataset
- The episode->feed map from the v2 dataset JSON

Usage:
    python scripts/eval/score/topic_clusters_baseline_v2.py \\
        --kg-run kg_gemini_curated_5feeds_kg_v2_provider \\
        --dataset curated_5feeds_kg_v2 \\
        --output data/eval/runs/baseline_topic_clusters_curated_5feeds_v2/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from podcast_scraper.graph_id_utils import slugify_label as slugify
from podcast_scraper.search.topic_clusters import (
    cluster_indices_by_threshold,
    cosine_similarity_matrix,
    pick_centroid_closest_label,
)

DEFAULT_THRESHOLD = 0.75
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _collect_kg_topics(kg_predictions: Path, ep_to_feed: dict[str, str]) -> list[dict[str, Any]]:
    """One row per (episode, topic_id) with label + feed."""
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for line in kg_predictions.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        ep_id = rec["episode_id"]
        feed = ep_to_feed.get(ep_id, "unknown")
        kg = rec["output"]["kg"]
        for node in kg.get("nodes", []):
            if node.get("type") != "Topic":
                continue
            label = (node.get("properties") or {}).get("label")
            if not label:
                continue
            topic_id = node.get("id") or f"topic:{slugify(label)}"
            key = (ep_id, topic_id)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "episode_id": ep_id,
                    "feed": feed,
                    "topic_id": topic_id,
                    "label": label,
                }
            )
    return rows


def _embed_labels(labels: list[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    vectors = model.encode(labels, normalize_embeddings=True, convert_to_numpy=True)
    return np.asarray(vectors, dtype=np.float32)


def _episode_to_feed(dataset_path: Path) -> dict[str, str]:
    data = json.loads(dataset_path.read_text())
    feed_map: dict[str, str] = {}
    for ep in data["episodes"]:
        eid = ep["episode_id"]
        path = ep.get("transcript_path", "")
        parts = Path(path).parts
        feed = next((p for p in parts if p.startswith("feed-")), "unknown")
        feed_map[eid] = feed
    return feed_map


def _podcast_of(feed: str) -> str:
    return feed.replace("feed-", "")


def build_clusters(rows: list[dict[str, Any]], threshold: float, model: str) -> dict[str, Any]:
    if not rows:
        return {"clusters": [], "row_count": 0}
    labels = [r["label"] for r in rows]
    vectors = _embed_labels(labels, model)
    sim = cosine_similarity_matrix(vectors)
    cluster_ids = cluster_indices_by_threshold(sim, threshold)

    # Group rows by cluster
    by_cluster: dict[int, list[int]] = defaultdict(list)
    for idx, cid in enumerate(cluster_ids.tolist()):
        by_cluster[cid].append(idx)

    out: list[dict[str, Any]] = []
    for cid, member_idxs in sorted(by_cluster.items()):
        members = [rows[i] for i in member_idxs]
        # Cluster only contributes a `tc:` parent if it actually links multiple topic_ids
        unique_topic_ids = {m["topic_id"] for m in members}
        if len(unique_topic_ids) < 2:
            continue
        centroid_idx = pick_centroid_closest_label(member_idxs, vectors)
        parent_label = rows[centroid_idx]["label"]
        parent_id = f"tc:{slugify(parent_label)}"
        feeds = sorted({m["feed"] for m in members})
        podcasts = sorted({_podcast_of(f) for f in feeds})
        out.append(
            {
                "tc_id": parent_id,
                "label": parent_label,
                "member_count": len(members),
                "unique_topic_count": len(unique_topic_ids),
                "feeds": feeds,
                "feed_count": len(feeds),
                "podcasts": podcasts,
                "podcast_count": len(podcasts),
                "topic_ids": sorted(unique_topic_ids),
                "labels": sorted({m["label"] for m in members}),
            }
        )
    return {"clusters": out, "row_count": len(rows)}


FRAME_ROOT = "frame"
FRAME_AMBIGUITY_FEED = "feed-p04"


def _frame_negative_test(clusters: list[dict[str, Any]]) -> dict[str, Any]:
    """v2 spec (#900) deliberately introduces `topic:frame` ambiguity in p04 to
    verify clustering doesn't bundle unrelated uses of 'frame' across feeds.

    A cluster fails the negative test if it contains at least one p04 label
    with `frame` in it AND at least one label from a non-p04 feed where
    `frame` is also present. Such a cluster would indicate the embedder
    merged p04's photographic 'frame' with a non-photo 'frame' from a
    different domain.
    """
    violations: list[dict[str, Any]] = []
    for c in clusters:
        labels = c.get("labels") or []
        if not any(FRAME_ROOT in lbl.lower() for lbl in labels):
            continue
        if c["feed_count"] < 2:
            continue
        if FRAME_AMBIGUITY_FEED not in c["feeds"]:
            continue
        # Cluster has frame-rooted labels, spans >=2 feeds, and includes p04 →
        # bundling photographic frame with something from another domain.
        violations.append(
            {
                "tc_id": c["tc_id"],
                "feeds": c["feeds"],
                "labels": labels,
            }
        )
    return {
        "violations": violations,
        "pass": len(violations) == 0,
        "note": (
            "Pass means no tc:* cluster bundles p04 frame-rooted labels with "
            "non-p04 frame-rooted labels (deliberate ambiguity left isolated)."
        ),
    }


def aggregate(clusters_payload: dict[str, Any]) -> dict[str, Any]:
    clusters = clusters_payload["clusters"]
    cross_feed = [c for c in clusters if c["feed_count"] >= 2]

    # tc: parents per podcast
    parents_per_podcast: dict[str, int] = defaultdict(int)
    for c in clusters:
        for podcast in c["podcasts"]:
            parents_per_podcast[podcast] += 1

    frame_test = _frame_negative_test(clusters)

    return {
        "tc_parent_count": len(clusters),
        "tc_cross_feed_count": len(cross_feed),
        "tc_parents_per_podcast": dict(sorted(parents_per_podcast.items())),
        "topic_row_count": clusters_payload["row_count"],
        "frame_negative_test": frame_test,
        "ac_targets": {
            "tc_parent_count_gt": 0,
            "tc_cross_feed_count_gt": 0,
            "frame_negative_test_pass": True,
        },
        "ac_pass": {
            "tc_parent_count": len(clusters) > 0,
            "tc_cross_feed_count": len(cross_feed) > 0,
            "frame_negative_test": frame_test["pass"],
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kg-run", required=True, help="KG run_id under data/eval/runs/")
    p.add_argument("--dataset", required=True, help="Dataset id (under data/eval/datasets/)")
    p.add_argument("--output", type=Path, required=True, help="Output directory for baseline")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--baseline-id", default="baseline_topic_clusters_curated_5feeds_v2")
    args = p.parse_args()

    kg_path = PROJECT_ROOT / "data" / "eval" / "runs" / args.kg_run / "predictions.jsonl"
    dataset_path = PROJECT_ROOT / "data" / "eval" / "datasets" / f"{args.dataset}.json"
    for p_ in (kg_path, dataset_path):
        if not p_.exists():
            print(f"Missing input: {p_}", file=sys.stderr)
            return 1

    ep_to_feed = _episode_to_feed(dataset_path)
    rows = _collect_kg_topics(kg_path, ep_to_feed)
    if not rows:
        print("No KG topic nodes found in predictions", file=sys.stderr)
        return 1

    clusters_payload = build_clusters(rows, args.threshold, args.model)
    agg = aggregate(clusters_payload)

    args.output.mkdir(parents=True, exist_ok=True)
    metrics = {
        "baseline_id": args.baseline_id,
        "task": "topic_clustering",
        "kg_run_id": args.kg_run,
        "dataset_id": args.dataset,
        "threshold": args.threshold,
        "embedding_model": args.model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "aggregate": agg,
        "clusters": clusters_payload["clusters"],
        "schema": "metrics_topic_clusters_baseline_v1",
    }
    metrics_path = args.output / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    cf_v = "PASS" if agg["ac_pass"]["tc_cross_feed_count"] else "FAIL"
    p_v = "PASS" if agg["ac_pass"]["tc_parent_count"] else "FAIL"
    fr_v = "PASS" if agg["ac_pass"]["frame_negative_test"] else "FAIL"
    frame_violations = agg["frame_negative_test"]["violations"]
    report = [
        f"# Topic-clusters Baseline — {args.baseline_id}",
        "",
        f"**KG run:** `{args.kg_run}`  ",
        f"**Dataset:** `{args.dataset}`  ",
        f"**Threshold:** {args.threshold}  ",
        f"**Embedding model:** `{args.model}`  ",
        f"**KG topic rows:** {agg['topic_row_count']}",
        "",
        "## Aggregate",
        "",
        "| Metric | Value | AC target |",
        "| --- | ---: | --- |",
        f"| tc:* parent clusters | {agg['tc_parent_count']} | >0 ({p_v}) |",
        f"| tc:* parents spanning >=2 feeds | {agg['tc_cross_feed_count']} | >0 ({cf_v}) |",
        f"| frame negative test | {len(frame_violations)} violations | 0 ({fr_v}) |",
        "",
        "### tc:* parents per podcast",
        "",
        "| Podcast | tc: count |",
        "| --- | ---: |",
    ]
    for podcast, n in agg["tc_parents_per_podcast"].items():
        report.append(f"| {podcast} | {n} |")
    report += [
        "",
        "## Top cross-feed tc:* clusters",
        "",
        "| tc:id | Feeds | Member labels |",
        "| --- | ---: | --- |",
    ]
    for c in sorted(clusters_payload["clusters"], key=lambda x: -x["feed_count"]):
        if c["feed_count"] < 2:
            continue
        labels = ", ".join(c["labels"][:4])
        report.append(f"| `{c['tc_id']}` | {c['feed_count']} | {labels} |")
    (args.output / "metrics_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote {metrics_path}")
    print(f"  tc_parents={agg['tc_parent_count']} " f"cross_feed={agg['tc_cross_feed_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
