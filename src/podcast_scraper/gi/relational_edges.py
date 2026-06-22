"""Derive meaning-bearing relational edges for the GIL semantic model (#874 family).

Two derivable, no-ML, no-diarization edges that the search-powered surfaces (PRD-033)
need from the foundation:

- ``Insight ─MENTIONS_PERSON→ Person`` / ``Insight ─MENTIONS_ORG→ Organization``
  — an insight concerns a person/org. Cross-layer: the target is a KG
  ``person:``/``org:`` id (unified in the cross-layer graph). Matched by the
  entity's surface name appearing verbatim in the insight text. Grounds FR4
  entity views / Topic Entity View.
- ``Podcast ─HAS_EPISODE→ Episode`` — ties an episode to its canonical show node so
  show-name navigation (FR3.4) and show-scoped views have a real edge, not hashed ids.

RFC-097 v3.0 split the generic ``MENTIONS`` into the typed ``MENTIONS_PERSON`` /
``MENTIONS_ORG`` so the viewer + relational query layer (RFC-094) can style/filter
descriptive edges without re-reading node types. Schema-permissive: artifacts with
the legacy generic ``MENTIONS`` still read; the post-pass here emits typed edges
and bumps ``schema_version`` to ``3.0``.

Both are persisted into the gi.json artifact. Conservative + idempotent.
"""

from __future__ import annotations

import re
from typing import Dict, Mapping, Tuple

from ..identity.slugify import slugify


def add_insight_entity_edges(artifact: Dict, entity_index: Mapping[str, Tuple[str, str]]) -> int:
    """Add typed ``Insight ─MENTIONS_PERSON/ORG→`` edges to *artifact* in place.

    *entity_index* maps ``entity_id`` (``person:``/``org:`` slug) → ``(name, kind)``
    where ``kind`` is ``"person"`` or ``"organization"``. An edge is added when
    the name appears as a whole-word phrase in an insight's text.

    Edge types: ``MENTIONS_PERSON`` (when kind=person) or ``MENTIONS_ORG`` (when
    kind=organization). Permissive over the legacy generic ``MENTIONS`` edges —
    deduplication considers all three types so repeated calls don't double up.

    RFC-097 chunk-4 retroactive sweep: when an edge points to a Person /
    Organization id that does NOT already exist as a node in the artifact,
    the function adds a minimal ``{name, label}`` Person / Organization
    node so the edge resolves locally and the viewer doesn't have to
    join across kg.json to render the relationship.

    When at least one typed edge is added, the artifact's ``schema_version``
    is bumped to ``3.0`` (RFC-097); earlier versions otherwise pass through.

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
        (eid, kind, name, re.compile(r"\b" + re.escape(name) + r"\b"))
        for eid, (name, kind) in entity_index.items()
        if name and len(name) >= 2
    ]
    existing = {
        (e.get("from"), e.get("to"), e.get("type"))
        for e in edges
        if e.get("type") in ("MENTIONS", "MENTIONS_PERSON", "MENTIONS_ORG")
    }
    existing_node_ids = {n.get("id") for n in nodes if isinstance(n.get("id"), str)}
    added = 0
    for insight_id, text in insights:
        for entity_id, kind, surface_name, pattern in patterns:
            edge_type = "MENTIONS_ORG" if kind == "organization" else "MENTIONS_PERSON"
            # Skip if EITHER the legacy generic MENTIONS or the typed edge is already present.
            keys = {
                (insight_id, entity_id, "MENTIONS"),
                (insight_id, entity_id, "MENTIONS_PERSON"),
                (insight_id, entity_id, "MENTIONS_ORG"),
            }
            if keys & existing:
                continue
            if pattern.search(text):
                # Add the missing Person / Organization node so the edge target
                # resolves within gi.json (no cross-layer join required).
                if entity_id not in existing_node_ids:
                    node_type = "Organization" if kind == "organization" else "Person"
                    nodes.append(
                        {
                            "id": entity_id,
                            "type": node_type,
                            "properties": {"name": surface_name},
                        }
                    )
                    existing_node_ids.add(entity_id)
                edges.append({"type": edge_type, "from": insight_id, "to": entity_id})
                existing.add((insight_id, entity_id, edge_type))
                added += 1
    if added and isinstance(artifact.get("schema_version"), str):
        # Bump to v3.0 (RFC-097) on first typed-edge addition; preserves existing higher versions.
        if artifact["schema_version"] in ("1.0", "2.0"):
            artifact["schema_version"] = "3.0"
    return added


def add_episode_show_edges(artifact: Dict, show_title: str) -> int:
    """Add a ``Podcast`` node + ``Podcast ─HAS_EPISODE→ Episode`` edges in place.

    The canonical show node is ``podcast:{slug(show_title)}`` so it unifies across
    episodes of the same show (cross-show navigation). Idempotent. Returns edges added.

    RFC-097 chunk-4 retroactive sweep: Podcast node carries ``title`` (not ``name``)
    to match ``gi.schema.json`` ``podcast_node.properties`` which requires ``title``.
    ``rss_url`` is unknowable from ``show_title`` alone and stays absent — see the
    schema's optional ``rss_url`` annotation.
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
        nodes.append({"id": podcast_id, "type": "Podcast", "properties": {"title": show_title}})
    added = 0
    for episode_id in episode_ids:
        if (podcast_id, episode_id) not in existing:
            edges.append({"type": "HAS_EPISODE", "from": podcast_id, "to": episode_id})
            existing.add((podcast_id, episode_id))
            added += 1
    return added


def kg_entity_names(kg_artifact: Dict) -> Dict[str, str]:
    """Extract ``{entity_id: surface_name}`` from a KG artifact.

    Permissive over v1.x ``Entity`` and v2.0 ``Person`` / ``Organization``
    node types (RFC-097). Kept for backward compatibility with consumers
    that only need names; for typed-MENTIONS edges use :func:`kg_entity_index`.
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


def kg_entity_index(kg_artifact: Dict) -> Dict[str, Tuple[str, str]]:
    """Extract ``{entity_id: (surface_name, kind)}`` from a KG artifact.

    ``kind`` is ``"person"`` or ``"organization"``, derived from the node type
    (v2.0 ``Person`` / ``Organization``) or properties (v1.x ``Entity`` with
    ``kind`` / ``entity_kind``). Required input for typed-MENTIONS emission
    via :func:`add_insight_entity_edges` (RFC-097 v3.0).
    """
    from ..graph_id_utils import is_person_or_org_node, normalized_entity_kind_from_node

    out: Dict[str, Tuple[str, str]] = {}
    for node in kg_artifact.get("nodes") or []:
        if not is_person_or_org_node(node.get("type")):
            continue
        eid = node.get("id")
        name = (node.get("properties") or {}).get("name")
        if isinstance(eid, str) and isinstance(name, str) and name.strip():
            kind = normalized_entity_kind_from_node(node)
            out[eid] = (name.strip(), kind)
    return out


def apply_typed_mentions_to_gi_artifact(
    gi_payload: Dict,
    kg_payload: Dict,
) -> int:
    """Apply the RFC-097 v3.0 typed-MENTIONS post-pass to a GI artifact.

    The live GI emit path (``gi/pipeline.py::build_artifact``) does not
    materialize typed ``MENTIONS_PERSON`` / ``MENTIONS_ORG`` cross-layer
    edges — they need a KG entity index that isn't available inside the
    GI build alone. This helper plugs the gap: caller passes in the
    finished KG payload, helper extracts the entity index and calls the
    in-place mutator.

    Mutates *gi_payload* in place. Caller owns any I/O (e.g. re-writing
    the updated artifact back to disk). Returns the number of typed edges
    added. Idempotent — calling repeatedly with the same payloads is a
    no-op after the first call.

    Args:
        gi_payload: GI artifact dict (mutated in place when edges are added).
        kg_payload: KG artifact dict (read-only).

    Returns:
        Count of typed edges added (0 if KG had no Person/Org entities, or
        if every match was already typed).
    """
    entity_index = kg_entity_index(kg_payload)
    if not entity_index:
        return 0
    return add_insight_entity_edges(gi_payload, entity_index)


def apply_typed_mentions_and_rewrite_gi(
    gi_payload: Dict,
    kg_payload: Dict,
    gi_path: str,
) -> int:
    """Apply the typed-MENTIONS post-pass and re-write the GI artifact on disk.

    Composes :func:`apply_typed_mentions_to_gi_artifact` with a guarded
    disk re-write — re-writes only when edges are actually added so the
    on-disk artifact's mtime stays stable for the zero-op case (no KG
    entities matched, or every match was already typed).

    Used by the production orchestrator
    (:mod:`podcast_scraper.workflow.metadata_generation`). Caller must
    have validated that both payloads exist and the path is non-empty.

    Args:
        gi_payload: GI artifact dict (mutated in place when edges added).
        kg_payload: KG artifact dict (read-only).
        gi_path: Filesystem path to re-write when edges are added.

    Returns:
        Count of typed edges added. ``0`` means the on-disk artifact was
        left untouched.
    """
    from pathlib import Path

    from .io import write_artifact

    added = apply_typed_mentions_to_gi_artifact(gi_payload, kg_payload)
    if added > 0:
        write_artifact(Path(gi_path), gi_payload, validate=True)
    return added
