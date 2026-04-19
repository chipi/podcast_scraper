"""Cross-episode insight clustering.

Groups semantically similar insights from different episodes using the same
average-linkage algorithm as topic clustering (RFC-075). Each cluster
aggregates supporting quotes from multiple episodes/speakers.

Usage:
    from podcast_scraper.search.insight_clusters import build_insight_clusters_for_corpus
    payload = build_insight_clusters_for_corpus(output_dir, threshold=0.75)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .topic_clusters import (
    cluster_indices_by_threshold,
    cosine_similarity_matrix,
    pick_centroid_closest_label,
)

logger = logging.getLogger(__name__)

INSIGHT_CLUSTERS_FILENAME = "insight_clusters.json"
INSIGHT_CLUSTERS_SCHEMA_VERSION = "1"


def collect_insight_rows_from_corpus(
    output_dir: Path,
) -> List[Dict[str, Any]]:
    """Scan all gi.json artifacts and collect insight texts + metadata."""
    rows: List[Dict[str, Any]] = []

    for gi_path in sorted(output_dir.rglob("*.gi.json")):
        try:
            gi = json.loads(gi_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("Skipping %s: %s", gi_path, exc)
            continue

        episode_id = gi.get("episode_id", gi_path.stem)
        nodes = gi.get("nodes", [])
        edges = gi.get("edges", [])

        # Build SUPPORTED_BY map: insight_id -> [quote_ids]
        supported_by: Dict[str, List[str]] = {}
        for edge in edges:
            if edge.get("type") == "SUPPORTED_BY":
                supported_by.setdefault(edge["from"], []).append(edge["to"])

        # Collect quote nodes by id
        quote_nodes: Dict[str, Dict[str, Any]] = {}
        for node in nodes:
            if node.get("type") == "Quote":
                quote_nodes[node["id"]] = node

        for node in nodes:
            if node.get("type") != "Insight":
                continue
            props = node.get("properties", {})
            text = props.get("text", "")
            if not text:
                continue

            # Collect supporting quotes for this insight
            quote_ids = supported_by.get(node["id"], [])
            quotes = []
            for qid in quote_ids:
                qnode = quote_nodes.get(qid)
                if qnode:
                    qprops = qnode.get("properties", {})
                    quotes.append(
                        {
                            "quote_id": qid,
                            "text": qprops.get("text", ""),
                            "speaker_id": qprops.get("speaker_id"),
                            "char_start": qprops.get("char_start"),
                            "char_end": qprops.get("char_end"),
                        }
                    )

            rows.append(
                {
                    "insight_id": node["id"],
                    "text": text,
                    "insight_type": props.get("insight_type", "unknown"),
                    "episode_id": episode_id,
                    "grounded": props.get("grounded", False),
                    "supporting_quotes": quotes,
                }
            )

    return rows


def build_insight_clusters_payload(
    rows: List[Dict[str, Any]],
    threshold: float = 0.75,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Dict[str, Any]:
    """Cluster insight rows by semantic similarity."""
    if not rows:
        return {
            "schema_version": INSIGHT_CLUSTERS_SCHEMA_VERSION,
            "model": embedding_model,
            "threshold": threshold,
            "insight_count": 0,
            "cluster_count": 0,
            "singletons": 0,
            "clusters": [],
        }

    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(embedding_model)
    texts = [r["text"] for r in rows]
    embs = embedder.encode(texts, normalize_embeddings=True)
    sim = cosine_similarity_matrix(embs)
    labels = cluster_indices_by_threshold(sim, threshold)

    # Group by cluster
    from collections import defaultdict

    cluster_members: Dict[int, List[int]] = defaultdict(list)
    for i, lbl in enumerate(labels):
        cluster_members[int(lbl)].append(i)

    clusters: List[Dict[str, Any]] = []
    singletons = 0

    for _lbl, members in sorted(cluster_members.items(), key=lambda x: -len(x[1])):
        if len(members) < 2:
            singletons += 1
            continue

        # Check if cross-episode
        episode_ids = sorted(set(rows[m]["episode_id"] for m in members))

        # Pick canonical insight
        member_vecs = embs[np.array(members)]
        centroid_idx = pick_centroid_closest_label(list(range(len(members))), member_vecs)
        canonical = rows[members[centroid_idx]]

        # Compute centroid for similarity scores
        centroid = np.mean(member_vecs, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 1e-12:
            centroid = centroid / norm

        cluster_members_out = []
        for m in members:
            sim_to_centroid = float(np.dot(embs[m], centroid))
            cluster_members_out.append(
                {
                    "insight_id": rows[m]["insight_id"],
                    "text": rows[m]["text"],
                    "insight_type": rows[m]["insight_type"],
                    "episode_id": rows[m]["episode_id"],
                    "similarity_to_centroid": round(sim_to_centroid, 4),
                    "supporting_quotes": rows[m]["supporting_quotes"],
                }
            )

        from podcast_scraper.graph_id_utils import slugify_label

        slug = slugify_label(canonical["text"][:80])
        clusters.append(
            {
                "cluster_id": f"ic:{slug}",
                "canonical_insight": canonical["text"],
                "member_count": len(members),
                "episode_count": len(episode_ids),
                "cross_episode": len(episode_ids) > 1,
                "episode_ids": episode_ids,
                "members": cluster_members_out,
            }
        )

    return {
        "schema_version": INSIGHT_CLUSTERS_SCHEMA_VERSION,
        "model": embedding_model,
        "threshold": threshold,
        "insight_count": len(rows),
        "cluster_count": len(clusters),
        "cross_episode_clusters": sum(1 for c in clusters if c["cross_episode"]),
        "singletons": singletons,
        "clusters": clusters,
    }


def build_insight_clusters_for_corpus(
    output_dir: str | Path,
    *,
    threshold: float = 0.75,
    out_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build insight clusters for an entire corpus."""
    output_dir = Path(output_dir)
    rows = collect_insight_rows_from_corpus(output_dir)
    logger.info("Collected %d insights from %s", len(rows), output_dir)

    payload = build_insight_clusters_payload(rows, threshold=threshold)

    if out_path is None:
        search_dir = output_dir / "search"
        search_dir.mkdir(parents=True, exist_ok=True)
        out_path = search_dir / INSIGHT_CLUSTERS_FILENAME

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(
        "Wrote %s: %d clusters (%d cross-episode), %d singletons",
        out_path,
        payload["cluster_count"],
        payload["cross_episode_clusters"],
        payload["singletons"],
    )
    return payload
