"""Resolve KG artifact paths by episode id (scan metadata/*.kg.json)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast, Dict, Optional


def find_kg_artifact_by_episode_id(
    output_dir: Path,
    episode_id: str,
    *,
    feed_id: Optional[str] = None,
) -> Optional[Path]:
    """Resolve ``.kg.json`` for ``episode_id`` (flat or multi-feed corpus root).

    When several feeds share the same ``episode_id``, pass ``feed_id`` (metadata
    ``feed.feed_id``) to disambiguate.
    """
    from podcast_scraper.utils.corpus_episode_paths import (
        list_artifact_paths_for_episode,
        pick_single_artifact_path,
    )

    paths = list_artifact_paths_for_episode(
        output_dir,
        episode_id,
        feed_id=feed_id,
        kind="kg",
    )
    return pick_single_artifact_path(paths)


def episode_node(artifact: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the Episode node from a KG artifact, if any."""
    for n in artifact.get("nodes", []):
        if n.get("type") == "Episode":
            return cast(Dict[str, Any], n)
    return None
