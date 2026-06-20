"""Migrate legacy GI/KG ids and property names in JSON documents (idempotent transforms).

Callers should back up corpora before rewriting files.

RFC-097 (2026-06-20) introduces v2.0 (KG) + v3.0 (GI) shapes that
are migrated by ``migrate_kg_document_v2`` and ``migrate_gi_document_v3``.
The pre-existing ``migrate_kg_document`` and ``migrate_gil_document``
land the v1.2 / v2.0 transitions and remain stable.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

#: RFC-097 v3.0 vocab — must stay in sync with gi.schema.json + gi/pipeline.py.
_V3_INSIGHT_TYPES = frozenset({"claim", "recommendation", "observation", "question", "unknown"})
#: Legacy synonyms from the pre-RFC-097 megabundle vocab.
_V3_LEGACY_SYNONYMS = {"fact": "claim", "opinion": "observation"}


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


def migrate_kg_document_v2(data: Dict[str, Any]) -> Dict[str, Any]:
    """RFC-097 v2.0: rewrite Entity nodes to typed Person / Organization.

    Order: first applies ``migrate_kg_document`` (v1.x → v1.2 normalisation) so
    every Entity node has a stable ``kind`` property; then rewrites each Entity
    to either a ``Person`` or ``Organization`` node depending on ``kind``,
    dropping ``kind`` from properties (the node type carries the discriminator
    now). Bumps ``schema_version`` to ``2.0``.

    HAS_EPISODE edges + Podcast nodes are NOT synthesised here — those require
    feed metadata not present in the kg.json artifact. Use the GI workflow
    (``add_episode_show_edges`` in ``gi/relational_edges.py``) for that.

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


def migrate_gi_document_v3(data: Dict[str, Any]) -> Dict[str, Any]:
    """RFC-097 v3.0: rewrite legacy ``MENTIONS`` (Insight → Entity) into
    typed ``MENTIONS_PERSON`` / ``MENTIONS_ORG`` based on each target's id
    prefix (``person:`` / ``org:``). Also normalises ``insight_type`` to the
    v3 vocab (legacy synonyms ``fact``/``opinion`` mapped to ``claim`` /
    ``observation``; out-of-vocab → ``unknown``). Bumps ``schema_version`` to
    ``3.0``. Order: first applies ``migrate_gil_document`` (Speaker → Person).

    Idempotent: a v3.0 artifact with already-typed MENTIONS_PERSON /
    MENTIONS_ORG edges + normalised insight types passes through unchanged.
    """
    out = migrate_gil_document(data)
    _raw_nodes = out.get("nodes")
    nodes: List[Any] = _raw_nodes if isinstance(_raw_nodes, list) else []
    _raw_edges = out.get("edges")
    edges: List[Any] = _raw_edges if isinstance(_raw_edges, list) else []

    # Build id → kind index from cross-layer canonical ids (person:/org:).
    id_to_kind: Dict[str, str] = {}
    for n in nodes:
        if not isinstance(n, dict):
            continue
        nid = n.get("id")
        if not isinstance(nid, str):
            continue
        if nid.startswith("person:"):
            id_to_kind[nid] = "person"
        elif nid.startswith("org:"):
            id_to_kind[nid] = "organization"

    # Rewrite Insight→Entity MENTIONS to typed forms (best-effort by id prefix).
    insight_ids = {
        n["id"]
        for n in nodes
        if isinstance(n, dict) and n.get("type") == "Insight" and isinstance(n.get("id"), str)
    }
    for e in edges:
        if not isinstance(e, dict):
            continue
        if e.get("type") != "MENTIONS":
            continue
        src = e.get("from")
        dst = e.get("to")
        if not isinstance(src, str) or not isinstance(dst, str):
            continue
        # Only rewrite Insight→Person|Org MENTIONS (descriptive). KG-side
        # MENTIONS (e.g. Topic→Episode) stay as discovery edges — those
        # don't live in gi.json so this filter rarely matters but is safe.
        if src not in insight_ids:
            continue
        if dst.startswith("person:"):
            e["type"] = "MENTIONS_PERSON"
        elif dst.startswith("org:"):
            e["type"] = "MENTIONS_ORG"
        # Fall through if neither prefix matches (rare; preserve legacy).

    # Normalise insight_type vocab on every Insight node.
    for n in nodes:
        if not isinstance(n, dict) or n.get("type") != "Insight":
            continue
        props = n.get("properties")
        if not isinstance(props, dict):
            continue
        raw = props.get("insight_type")
        if raw is None:
            continue
        if isinstance(raw, str):
            k = raw.strip().lower()
            if k in _V3_INSIGHT_TYPES:
                if k != raw:
                    props["insight_type"] = k
            elif k in _V3_LEGACY_SYNONYMS:
                props["insight_type"] = _V3_LEGACY_SYNONYMS[k]
            else:
                props["insight_type"] = "unknown"

    if out.get("schema_version") in ("1.0", "2.0"):
        out["schema_version"] = "3.0"
    return out


def compute_position_hints_for_document(
    data: Dict[str, Any],
    *,
    transcript_segments: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """RFC-097: backfill ``Insight.position_hint`` on a GI document using the
    4-step waterfall (RSS duration → segments end → max Quote ts → skip).

    Computes per-Insight ``position_hint = mean(Quote.timestamp_start_ms) /
    duration_ms`` for every Insight that has ≥1 SUPPORTED_BY Quote. Uses
    ``Episode.duration_ms`` from the artifact (step 1), the caller-supplied
    ``transcript_segments`` (step 2), or ``max(Quote.timestamp_end_ms)``
    across each Insight's supporting Quotes (step 3). Skips emission when no
    duration is recoverable (step 4 — field remains absent).

    Idempotent (existing position_hint values are overwritten with the
    recomputed value so the function is also a *recompute* not just a
    *backfill*).
    """
    from ..gi.position_hint import compute_position_hint

    out = deepcopy(data)
    _raw_nodes = out.get("nodes")
    nodes: List[Any] = _raw_nodes if isinstance(_raw_nodes, list) else []
    _raw_edges = out.get("edges")
    edges: List[Any] = _raw_edges if isinstance(_raw_edges, list) else []

    quote_by_id: Dict[str, Dict[str, Any]] = {}
    for n in nodes:
        if isinstance(n, dict) and n.get("type") == "Quote" and isinstance(n.get("id"), str):
            quote_by_id[n["id"]] = n

    ep_duration_ms: Optional[int] = None
    for n in nodes:
        if isinstance(n, dict) and n.get("type") == "Episode":
            dur = (n.get("properties") or {}).get("duration_ms")
            if isinstance(dur, int) and dur > 0:
                ep_duration_ms = dur
            break

    quotes_by_insight: Dict[str, List[Dict[str, Any]]] = {}
    for e in edges:
        if not isinstance(e, dict) or e.get("type") != "SUPPORTED_BY":
            continue
        src = e.get("from")
        dst = e.get("to")
        if not isinstance(src, str) or not isinstance(dst, str):
            continue
        q = quote_by_id.get(dst)
        if q is None:
            continue
        quotes_by_insight.setdefault(src, []).append(q)

    for n in nodes:
        if not isinstance(n, dict) or n.get("type") != "Insight":
            continue
        props = n.get("properties")
        if not isinstance(props, dict):
            continue
        iid = n.get("id")
        if not isinstance(iid, str):
            continue
        qs = quotes_by_insight.get(iid, [])
        starts: List[int] = []
        ends: List[int] = []
        for q in qs:
            qp = q.get("properties") or {}
            s = qp.get("timestamp_start_ms")
            e_ts = qp.get("timestamp_end_ms")
            if isinstance(s, int) and s >= 0:
                starts.append(s)
            if isinstance(e_ts, int) and e_ts > 0:
                ends.append(e_ts)
        ph_value, _step = compute_position_hint(
            starts,
            ep_duration_ms,
            transcript_segments=transcript_segments,
            quote_end_fallback_ms=max(ends) if ends else None,
        )
        if ph_value is not None:
            props["position_hint"] = ph_value
        # Otherwise leave whatever was there alone (step 4: skip).
    return out
