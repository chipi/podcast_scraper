"""Shared artifact-loading helpers for deterministic enrichers.

The chunk-2 enrichers all read some combination of GI, KG, and bridge
JSON. This module isolates the read/parse logic so enricher bodies stay
focused on the algorithm.

All helpers tolerate missing files (return ``{}`` / empty list) — the
``BadInputError`` non-retryable path is reserved for the case where the
required input is *expected* but malformed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from podcast_scraper.enrichment.protocol import EpisodeArtifactBundle


def _read_json(path: Path | None) -> dict[str, Any]:
    """Read a JSON file; return ``{}`` when absent or unparseable."""
    if path is None or not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def load_kg(bundle: EpisodeArtifactBundle) -> dict[str, Any]:
    """Load the episode KG (or ``{}`` when missing)."""
    return _read_json(bundle.kg_path)


def load_gi(bundle: EpisodeArtifactBundle) -> dict[str, Any]:
    """Load the episode GI (or ``{}`` when missing)."""
    return _read_json(bundle.gi_path)


def load_bridge(bundle: EpisodeArtifactBundle) -> dict[str, Any]:
    """Load the episode bridge (or ``{}`` when missing)."""
    return _read_json(bundle.bridge_path)


def load_metadata(bundle: EpisodeArtifactBundle) -> dict[str, Any]:
    """Load the episode metadata JSON (or ``{}`` when missing)."""
    return _read_json(bundle.metadata_path)


def nodes_of_type(art: dict[str, Any], node_type: str) -> list[dict[str, Any]]:
    """Filter ``art["nodes"]`` to the given node type."""
    nodes = art.get("nodes") or []
    if not isinstance(nodes, list):
        return []
    return [n for n in nodes if isinstance(n, dict) and n.get("type") == node_type]


def edges_of_type(art: dict[str, Any], edge_type: str) -> list[dict[str, Any]]:
    """Filter ``art["edges"]`` to the given edge type."""
    edges = art.get("edges") or []
    if not isinstance(edges, list):
        return []
    return [e for e in edges if isinstance(e, dict) and e.get("type") == edge_type]


def node_label(node: dict[str, Any]) -> str:
    """Best-effort label for a node — `properties.label` or `properties.name` or `id`."""
    props = node.get("properties") or {}
    label = props.get("label") or props.get("name") or node.get("id")
    return str(label) if label else ""


def publish_date(art: dict[str, Any]) -> str | None:
    """Extract the episode publish_date from a KG/GI artifact.

    Looks at the Episode node's ``properties.publish_date`` (an ISO-8601
    timestamp). Returns ``None`` when unavailable.
    """
    for node in nodes_of_type(art, "Episode"):
        date = (node.get("properties") or {}).get("publish_date")
        if isinstance(date, str) and date:
            return date
    return None


__all__ = [
    "edges_of_type",
    "load_bridge",
    "load_gi",
    "load_kg",
    "load_metadata",
    "node_label",
    "nodes_of_type",
    "publish_date",
]
