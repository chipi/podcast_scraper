"""A news item's own KG 3.0 nodes: its publisher and its byline (E05 "One graph", ADR-180).

The publisher is the registry's organisation, ``PUBLISHED_BY`` from the item's source. Each byline
name is a person with the role ``author`` on the item (``PARTICIPATES_IN``); the person id is the
one every other path mints (``graph_id_utils.person_node_id``), and joining it to anyone else is
the joining rules' work (ADR-180), never this module's.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..content_types import ItemContext
from ..graph_id_utils import person_node_id
from ..publishers import Registry
from ..utils.canonical_url import canonical_url
from . import byline as byline_parser, document

AUTHOR = "author"


def extend(
    payload: Dict[str, Any], item: ItemContext, *, registry: Optional[Registry] = None
) -> Dict[str, Any]:
    """``payload`` with the item's publisher and byline people added (no node twice)."""
    nodes: List[Dict[str, Any]] = list(payload.get("nodes") or [])
    edges: List[Dict[str, Any]] = list(payload.get("edges") or [])
    have = {n.get("id") for n in nodes}
    f = item.fields
    url = canonical_url(str(f.get("url") or "")) if f.get("url") else ""
    reg = registry if registry is not None else document.load_registry()
    pub = document.publisher(reg, item.source_id, url) if url else None
    if pub is not None and item.source_id:
        if pub.id not in have:
            nodes.append({"id": pub.id, "type": "Organization", "properties": {"name": pub.name}})
            have.add(pub.id)
        edges.append({"type": "PUBLISHED_BY", "from": item.source_id, "to": pub.id})
    organisations = [pub.name] if pub is not None and not f.get("personal_publisher") else []
    for name in byline_parser.parse(f.get("byline_raw"), organisations=organisations):
        pid = person_node_id(name)
        if pid == "person:unknown":
            continue
        if pid not in have:
            nodes.append({"id": pid, "type": "Person", "properties": {"name": name}})
            have.add(pid)
        edges.append(
            {
                "type": "PARTICIPATES_IN",
                "from": pid,
                "to": item.item_id,
                "properties": {"role": AUTHOR},
            }
        )
    return {**payload, "nodes": nodes, "edges": edges}
