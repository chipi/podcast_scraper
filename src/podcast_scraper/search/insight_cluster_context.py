"""Cluster context expansion for gi explore results.

Given explore results (insights matched by topic/speaker/semantic search),
expands each insight with cross-episode cluster context — additional quotes
from other episodes that support the same claim.

Usage:
    from podcast_scraper.search.insight_cluster_context import (
        expand_with_cluster_context,
    )
    expanded = expand_with_cluster_context(insights, clusters_path)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def load_insight_clusters(
    clusters_path: Path,
) -> Dict[str, Dict[str, Any]]:
    """Load insight_clusters.json and index by member insight_id."""
    if not clusters_path.exists():
        return {}

    payload = json.loads(clusters_path.read_text(encoding="utf-8"))
    index: Dict[str, Dict[str, Any]] = {}
    for cluster in payload.get("clusters", []):
        for member in cluster.get("members", []):
            index[member["insight_id"]] = {
                "cluster_id": cluster["cluster_id"],
                "canonical_insight": cluster["canonical_insight"],
                "member_count": cluster["member_count"],
                "episode_count": cluster["episode_count"],
                "cross_episode": cluster["cross_episode"],
                "all_members": cluster["members"],
            }
    return index


def expand_with_cluster_context(
    insights: List[Dict[str, Any]],
    clusters_path: Optional[Path] = None,
    cluster_index: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Add cluster information to explore results.

    For each insight that belongs to a cluster, adds:
    - cluster_id
    - canonical_insight (centroid)
    - cross_episode_quotes: quotes from OTHER episodes in the cluster
    - cluster_size: total insights in cluster
    - cluster_episode_count: episodes represented

    Args:
        insights: List of insight dicts from gi explore.
        clusters_path: Path to insight_clusters.json.
        cluster_index: Pre-loaded index (if already loaded).

    Returns:
        Same insights list with cluster context added where applicable.
    """
    if cluster_index is None:
        if clusters_path is None:
            return insights
        cluster_index = load_insight_clusters(clusters_path)

    if not cluster_index:
        return insights

    expanded = []
    for ins in insights:
        ins_id = ins.get("insight_id", ins.get("id", ""))
        cluster_info = cluster_index.get(ins_id)

        if cluster_info and cluster_info["cross_episode"]:
            # Collect quotes from OTHER episodes in the cluster
            this_episode = ins.get("episode_id", "")
            cross_quotes = []
            for member in cluster_info["all_members"]:
                if member["episode_id"] != this_episode:
                    for sq in member.get("supporting_quotes", []):
                        cross_quotes.append(
                            {
                                "text": sq.get("text", ""),
                                "episode_id": member["episode_id"],
                                "speaker_id": sq.get("speaker_id"),
                                "from_insight": member["text"][:100],
                            }
                        )

            ins_expanded = dict(ins)
            ins_expanded["cluster"] = {
                "cluster_id": cluster_info["cluster_id"],
                "canonical_insight": cluster_info["canonical_insight"],
                "cluster_size": cluster_info["member_count"],
                "cluster_episodes": cluster_info["episode_count"],
                "cross_episode_quotes": cross_quotes,
            }
            expanded.append(ins_expanded)
        else:
            expanded.append(ins)

    return expanded


def format_cluster_context(insight: Dict[str, Any]) -> str:
    """Format cluster context for CLI display."""
    cluster = insight.get("cluster")
    if not cluster:
        return ""

    lines = [
        f"\n  [Cluster: {cluster['cluster_id']}]",
        f"  Canonical: {cluster['canonical_insight'][:70]}...",
        f"  {cluster['cluster_size']} insights across " f"{cluster['cluster_episodes']} episodes",
    ]

    cross_quotes = cluster.get("cross_episode_quotes", [])
    if cross_quotes:
        lines.append(f"  Cross-episode evidence ({len(cross_quotes)} quotes):")
        for cq in cross_quotes[:3]:
            ep = cq.get("episode_id", "?")[:20]
            speaker = cq.get("speaker_id", "")
            speaker_str = f" ({speaker})" if speaker else ""
            lines.append(f"    [{ep}{speaker_str}] " f"\"{cq['text'][:60]}...\"")
        if len(cross_quotes) > 3:
            lines.append(f"    ... +{len(cross_quotes) - 3} more")

    return "\n".join(lines)
