"""KG document format transforms (KG v1.2, v2.0), idempotent; corpus migration 0006 applies v2.

Kept public with the KG (E04 split line, section 5). The GI transforms that lived beside them in
``gil_kg_identity_migrations`` left with GI.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


def migrate_kg_document(data: Dict[str, Any]) -> Dict[str, Any]:
    """Rewrite ``entity:person:`` / ``entity:organization:`` ids; ``entity_kind`` -> ``kind``."""
    out = deepcopy(data)
    nodes = out.get("nodes")
    if not isinstance(nodes, list):
        return out

    id_map: Dict[str, str] = {}
    for n in nodes:
        if not isinstance(n, dict):
            continue
        oid = n.get("id")
        if not isinstance(oid, str):
            continue
        if oid.startswith("entity:person:"):
            id_map[oid] = "person:" + oid[len("entity:person:") :]
        elif oid.startswith("entity:organization:"):
            id_map[oid] = "org:" + oid[len("entity:organization:") :]

    for n in nodes:
        if not isinstance(n, dict):
            continue
        oid = n.get("id")
        if isinstance(oid, str) and oid in id_map:
            n["id"] = id_map[oid]
        if n.get("type") != "Entity":
            continue
        props = n.get("properties")
        if not isinstance(props, dict):
            continue
        if "kind" not in props and "entity_kind" in props:
            ek = props.get("entity_kind")
            if ek == "organization":
                props["kind"] = "org"
            else:
                props["kind"] = "person"
            del props["entity_kind"]
        n["properties"] = props

    for e in out.get("edges") or []:
        if not isinstance(e, dict):
            continue
        for k in ("from", "to"):
            v = e.get(k)
            if isinstance(v, str) and v in id_map:
                e[k] = id_map[v]

    sv = out.get("schema_version")
    if sv in ("1.0", "1.1"):
        out["schema_version"] = "1.2"
    return out


def migrate_kg_document_v2(data: Dict[str, Any]) -> Dict[str, Any]:
    """RFC-097 v2.0: rewrite Entity nodes to typed Person / Organization.

    Order: first applies ``migrate_kg_document`` (v1.x → v1.2 normalisation) so
    every Entity node has a stable ``kind`` property; then rewrites each Entity
    to either a ``Person`` or ``Organization`` node depending on ``kind``,
    dropping ``kind`` from properties (the node type carries the discriminator
    now). Bumps ``schema_version`` to ``2.0``.

    HAS_EPISODE edges + Podcast nodes are NOT synthesised here — those require
    feed metadata not present in the kg.json artifact. Use the GI workflow
    (``add_episode_show_edges``) for that.

    Idempotent: a v2.0 artifact passes through unchanged.
    """
    out = migrate_kg_document(data)
    nodes = out.get("nodes")
    if not isinstance(nodes, list):
        return out
    sv = out.get("schema_version")
    if sv == "2.0":
        # Already v2.0; check for stray Entity nodes anyway (defensive).
        pass
    for n in nodes:
        if not isinstance(n, dict) or n.get("type") != "Entity":
            continue
        raw_props = n.get("properties")
        props: Dict[str, Any] = raw_props if isinstance(raw_props, dict) else {}
        kind = props.get("kind")
        if kind == "org":
            n["type"] = "Organization"
        else:
            n["type"] = "Person"
        # Drop the legacy `kind` property — node type encodes it.
        if "kind" in props:
            props.pop("kind", None)
            n["properties"] = props
    if out.get("schema_version") in ("1.0", "1.1", "1.2"):
        out["schema_version"] = "2.0"
    return out
