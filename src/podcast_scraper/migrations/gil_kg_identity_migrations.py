"""Migrate legacy GI/KG ids and property names in JSON documents (idempotent transforms).

Callers should back up corpora before rewriting files.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


def migrate_gil_document(data: Dict[str, Any]) -> Dict[str, Any]:
    """Rewrite Speaker nodes and ``speaker:`` ids to Person / ``person:``; bump schema to 2.0."""
    out = deepcopy(data)
    nodes = out.get("nodes")
    if not isinstance(nodes, list):
        return out

    id_map: Dict[str, str] = {}
    for n in nodes:
        if not isinstance(n, dict):
            continue
        oid = n.get("id")
        if n.get("type") == "Speaker":
            n["type"] = "Person"
            if isinstance(oid, str) and oid.startswith("speaker:"):
                nid = "person:" + oid.split(":", 1)[1]
                id_map[oid] = nid
                n["id"] = nid

    for e in out.get("edges") or []:
        if not isinstance(e, dict):
            continue
        for k in ("from", "to"):
            v = e.get(k)
            if isinstance(v, str) and v in id_map:
                e[k] = id_map[v]

    for n in nodes:
        if not isinstance(n, dict) or n.get("type") != "Quote":
            continue
        props = n.get("properties")
        if not isinstance(props, dict):
            continue
        sid = props.get("speaker_id")
        if isinstance(sid, str) and sid.startswith("speaker:"):
            props["speaker_id"] = "person:" + sid.split(":", 1)[1]

    sv = out.get("schema_version")
    if sv == "1.0":
        out["schema_version"] = "2.0"
    return out


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
