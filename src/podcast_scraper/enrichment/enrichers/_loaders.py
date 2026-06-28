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
import re
from pathlib import Path
from typing import Any

from podcast_scraper.enrichment.protocol import EpisodeArtifactBundle

_SPEAKER_PLACEHOLDER_PATTERN = re.compile(
    r"^(?:person:)?speaker[_\-]?\d+$",
    re.IGNORECASE,
)


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


def is_unresolved_speaker_placeholder(person_id: str, name: str | None = None) -> bool:
    """True when *person_id* / *name* looks like a per-episode diarization label.

    Upstream NER / speaker-link sometimes leaves diarization output
    (``SPEAKER_00``, ``SPEAKER_18``) un-resolved to a real Person. The
    pipeline still slugs them as ``person:speaker-NN`` and puts them in
    the Person table. Each episode's ``SPEAKER_NN`` is independent —
    ``SPEAKER_00`` in episode A has nothing to do with ``SPEAKER_00`` in
    episode B — so corpus-scope enrichers that aggregate across episodes
    must drop them or they over-count the same label's cross-episode
    coincidence as a real co-occurrence / co-grounding.
    """
    if person_id and _SPEAKER_PLACEHOLDER_PATTERN.match(person_id):
        return True
    if name and _SPEAKER_PLACEHOLDER_PATTERN.match(name):
        return True
    return False


def episode_duration_seconds(meta: dict[str, Any]) -> float:
    """Pull duration from metadata.json — top-level or ``episode.``-nested.

    The metadata writer puts ``duration_seconds`` under ``episode.``;
    some legacy fixtures + the chunk-1 enricher contract assume it's
    top-level. Both shapes accepted, ``0.0`` when neither carries it.
    """
    for source in (meta, meta.get("episode") if isinstance(meta.get("episode"), dict) else {}):
        if not isinstance(source, dict):
            continue
        raw = source.get("duration_seconds")
        if isinstance(raw, (int, float)) and raw > 0:
            return float(raw)
    return 0.0


__all__ = [
    "edges_of_type",
    "episode_duration_seconds",
    "is_unresolved_speaker_placeholder",
    "load_bridge",
    "load_gi",
    "load_kg",
    "load_metadata",
    "node_label",
    "nodes_of_type",
    "publish_date",
]
