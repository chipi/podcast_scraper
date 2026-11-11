"""KG 3.0 (ontology v3) for an item of a non-podcast content type.

The extraction is the one every item gets: :func:`kg.pipeline.build_artifact` produces the KG 2.x
graph from the item's body. :func:`item_artifact` then writes it in v3's shape for that item
(docs/architecture/corpus/ontology-v3.md):

- the ``Podcast`` and ``Episode`` anchors become the item's ``Source`` and ``ContentItem``, with
  ``HAS_ITEM`` between them;
- ``HOSTS`` and ``GUESTS_ON`` become ``CONTRIBUTES_TO`` with that role;
- ``MENTIONS`` and ``RELATED_TO``, and the ``Person``, ``Organization``, ``Object`` and ``Topic``
  nodes, are kept, each with the properties v3 defines for it.

Podcasts keep writing KG 2.x until R3b (ADR-178); nothing here runs for them.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

SCHEMA_VERSION = "3.0"

#: The item times v3 records on ``ContentItem``, each optional and never guessed.
ITEM_TIMES = (
    "published_at",
    "filed_at",
    "period_start",
    "period_end",
    "observed_at",
    "ingested_at",
)

_NAMED = ("name", "label", "aliases", "role", "description")
_KEPT_PROPERTIES: Dict[str, tuple] = {
    "Person": _NAMED,
    "Organization": _NAMED + ("identifiers",),
    "Object": _NAMED,
    "Topic": ("label", "slug", "description"),
}
_ROLE_BY_V2_EDGE = {"HOSTS": "host", "GUESTS_ON": "guest"}
_KEPT_EDGES = ("MENTIONS", "RELATED_TO")


class V3MappingError(ValueError):
    """A KG 2.x node or edge that v3 has no place for; refused rather than dropped."""


def item_artifact(
    v2: Mapping[str, Any],
    *,
    item_id: str,
    content_type: str,
    item_title: str,
    source_id: str,
    source_title: str,
    source_kind: str,
    transport: str,
    document_ref: str,
    variant: str = "raw",
    item_url: Optional[str] = None,
    source_url: Optional[str] = None,
    times: Optional[Mapping[str, Optional[str]]] = None,
) -> Dict[str, Any]:
    """The item's KG 3.0 artifact, from the KG 2.x graph its extraction built."""
    anchors = {
        n["id"]: n["type"] for n in v2.get("nodes") or [] if n.get("type") in ("Podcast", "Episode")
    }
    remap = {old: (item_id if kind == "Episode" else source_id) for old, kind in anchors.items()}

    item_props: Dict[str, Any] = {
        "title": item_title,
        "content_type": content_type,
        "source_id": source_id,
    }
    if item_url:
        item_props["url"] = item_url
    for name in ITEM_TIMES:
        value = (times or {}).get(name)
        if value:
            item_props[name] = value
    source_props: Dict[str, Any] = {
        "title": source_title,
        "source_kind": source_kind,
        "transport": transport,
    }
    if source_url:
        source_props["url"] = source_url

    nodes: List[Dict[str, Any]] = [
        {"id": source_id, "type": "Source", "properties": source_props},
        {"id": item_id, "type": "ContentItem", "properties": item_props},
    ]
    for node in v2.get("nodes") or []:
        if node["id"] in anchors:
            continue
        kept = _KEPT_PROPERTIES.get(node.get("type", ""))
        if kept is None:
            raise V3MappingError(f"KG 2.x node type {node.get('type')!r} has no v3 form")
        props = {k: v for k, v in (node.get("properties") or {}).items() if k in kept}
        out = {"id": node["id"], "type": node["type"], "properties": props}
        if "confidence" in node:
            out["confidence"] = node["confidence"]
        nodes.append(out)

    edges: List[Dict[str, Any]] = [{"type": "HAS_ITEM", "from": source_id, "to": item_id}]
    for edge in v2.get("edges") or []:
        kind = edge.get("type")
        ends = {
            "from": remap.get(edge["from"], edge["from"]),
            "to": remap.get(edge["to"], edge["to"]),
        }
        if kind == "HAS_EPISODE":
            continue
        if kind in _ROLE_BY_V2_EDGE:
            props = dict(edge.get("properties") or {})
            props["role"] = _ROLE_BY_V2_EDGE[kind]
            edges.append({"type": "CONTRIBUTES_TO", **ends, "properties": props})
        elif kind in _KEPT_EDGES:
            out_edge: Dict[str, Any] = {"type": kind, **ends}
            if edge.get("properties"):
                out_edge["properties"] = dict(edge["properties"])
            edges.append(out_edge)
        else:
            raise V3MappingError(f"KG 2.x edge type {kind!r} has no v3 form")

    extraction = v2.get("extraction") or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "item_id": item_id,
        "content_type": content_type,
        "extraction": {
            "model_version": extraction.get("model_version") or "metadata_only",
            "extracted_at": extraction.get("extracted_at"),
            "document_ref": document_ref,
            "variant": variant,
        },
        "nodes": nodes,
        "edges": edges,
    }
