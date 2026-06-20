"""Derive meaning-bearing relational edges for the GIL semantic model (#874 family).

Two derivable, no-ML, no-diarization edges that the search-powered surfaces (PRD-033)
need from the foundation:

- ``Insight ─MENTIONS→ Entity`` — an insight concerns a person/org. Cross-layer: the
  target is a KG ``person:``/``org:`` id (unified in the cross-layer graph). Matched by
  the entity's surface name appearing verbatim in the insight text. Grounds FR4 entity
  views / Topic Entity View.
- ``Podcast ─HAS_EPISODE→ Episode`` — ties an episode to its canonical show node so
  show-name navigation (FR3.4) and show-scoped views have a real edge, not hashed ids.

Both are persisted into the gi.json artifact (``MENTIONS`` was added to the GIL edge
schema; ``Podcast``/``HAS_EPISODE`` already exist). Conservative + idempotent.
"""

from __future__ import annotations

import re
from typing import Dict, Mapping

from ..identity.slugify import slugify


def add_insight_entity_edges(artifact: Dict, entity_names: Mapping[str, str]) -> int:
    """Add ``Insight ─MENTIONS→ Entity`` edges to *artifact* in place.

    *entity_names* maps ``entity_id`` (``person:``/``org:`` slug) → surface name. An
    edge is added when the name appears as a whole-word phrase in an insight's text.
    Idempotent. Returns the number of edges added.
    """
    nodes = artifact.setdefault("nodes", [])
    edges = artifact.setdefault("edges", [])
    insights = [
        (n["id"], (n.get("properties") or {}).get("text") or "")
        for n in nodes
        if n.get("type") == "Insight" and isinstance(n.get("id"), str)
    ]
    patterns = [
        (eid, re.compile(r"\b" + re.escape(name) + r"\b"))
        for eid, name in entity_names.items()
        if name and len(name) >= 2
    ]
    existing = {(e.get("from"), e.get("to")) for e in edges if e.get("type") == "MENTIONS"}
    added = 0
    for insight_id, text in insights:
        for entity_id, pattern in patterns:
            if (insight_id, entity_id) not in existing and pattern.search(text):
                edges.append({"type": "MENTIONS", "from": insight_id, "to": entity_id})
                existing.add((insight_id, entity_id))
                added += 1
    return added


def add_episode_show_edges(artifact: Dict, show_title: str) -> int:
    """Add a ``Podcast`` node + ``Podcast ─HAS_EPISODE→ Episode`` edges in place.

    The canonical show node is ``podcast:{slug(show_title)}`` so it unifies across
    episodes of the same show (cross-show navigation). Idempotent. Returns edges added.
    """
    if not show_title:
        return 0
    nodes = artifact.setdefault("nodes", [])
    edges = artifact.setdefault("edges", [])
    podcast_id = "podcast:" + slugify(show_title)
    episode_ids = [
        n["id"] for n in nodes if n.get("type") == "Episode" and isinstance(n.get("id"), str)
    ]
    node_ids = {n.get("id") for n in nodes}
    existing = {(e.get("from"), e.get("to")) for e in edges if e.get("type") == "HAS_EPISODE"}
    if podcast_id not in node_ids:
        nodes.append({"id": podcast_id, "type": "Podcast", "properties": {"name": show_title}})
    added = 0
    for episode_id in episode_ids:
        if (podcast_id, episode_id) not in existing:
            edges.append({"type": "HAS_EPISODE", "from": podcast_id, "to": episode_id})
            existing.add((podcast_id, episode_id))
            added += 1
    return added


def kg_entity_names(kg_artifact: Dict) -> Dict[str, str]:
    """Extract ``{entity_id: surface_name}`` from a KG artifact (the match source).

    Permissive over v1.x ``Entity`` and v2.0 ``Person`` / ``Organization``
    node types (RFC-097).
    """
    from ..graph_id_utils import is_person_or_org_node

    out: Dict[str, str] = {}
    for node in kg_artifact.get("nodes") or []:
        if not is_person_or_org_node(node.get("type")):
            continue
        eid = node.get("id")
        name = (node.get("properties") or {}).get("name")
        if isinstance(eid, str) and isinstance(name, str) and name.strip():
            out[eid] = name.strip()
    return out
