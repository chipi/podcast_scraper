"""RFC-072 per-episode ``bridge.json`` — canonical CIL identities across GI and KG."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, MutableMapping, Set

_CIL_PREFIXES = ("person:", "org:", "topic:")


def _strip_layer_prefixes(raw_id: str) -> str:
    s = str(raw_id).strip()
    if s.startswith("g:"):
        s = s[2:]
    if s.startswith("k:") and not s.startswith("kg:"):
        s = s[2:]
    if s.startswith("kg:"):
        s = s[3:]
    return s


def _node_aliases(props: Mapping[str, Any]) -> List[str]:
    raw = props.get("aliases")
    if raw is None:
        return []
    if isinstance(raw, str) and raw.strip():
        return [raw.strip()]
    if isinstance(raw, list):
        out: List[str] = []
        for x in raw:
            if isinstance(x, str) and x.strip():
                out.append(x.strip())
        return out
    return []


def _merge_display_name(existing: str, incoming: str) -> str:
    a = (existing or "").strip()
    b = (incoming or "").strip()
    if not a:
        return b
    if not b:
        return a
    if a == b:
        return a
    return a if len(a) >= len(b) else b


def _collect_identity(
    identities: MutableMapping[str, Dict[str, Any]],
    node: Mapping[str, Any],
    *,
    source: str,
) -> None:
    nid_raw = node.get("id")
    if nid_raw is None:
        return
    nid = _strip_layer_prefixes(str(nid_raw))
    if not nid.startswith(_CIL_PREFIXES):
        return
    kind = nid.split(":", 1)[0]
    props_any = node.get("properties")
    props: Dict[str, Any] = props_any if isinstance(props_any, dict) else {}
    node_aliases = _node_aliases(props)
    name_prop = props.get("name")
    label_prop = props.get("label")
    display = ""
    if isinstance(name_prop, str) and name_prop.strip():
        display = name_prop.strip()
    elif isinstance(label_prop, str) and label_prop.strip():
        display = label_prop.strip()

    if nid in identities:
        ex = identities[nid]
        ex["sources"][source] = True
        merged_aliases: Set[str] = set(ex.get("aliases") or [])
        merged_aliases.update(node_aliases)
        ex["aliases"] = sorted(merged_aliases)
        ex["display_name"] = _merge_display_name(str(ex.get("display_name", "")), display)
    else:
        identities[nid] = {
            "id": nid,
            "type": kind,
            "display_name": display,
            "aliases": list(node_aliases),
            "sources": {"gi": source == "gi", "kg": source == "kg"},
        }


def build_bridge(
    episode_id: str,
    gi_artifact: Mapping[str, Any] | None,
    kg_artifact: Mapping[str, Any] | None,
    *,
    emitted_at: datetime | None = None,
) -> Dict[str, Any]:
    """Build a ``bridge.json`` payload (RFC-072 §4)."""
    identities: Dict[str, Dict[str, Any]] = {}
    gi = gi_artifact or {}
    kg = kg_artifact or {}
    for node in gi.get("nodes") or []:
        if isinstance(node, dict):
            _collect_identity(identities, node, source="gi")
    for node in kg.get("nodes") or []:
        if isinstance(node, dict):
            _collect_identity(identities, node, source="kg")
    when = emitted_at or datetime.now(timezone.utc)
    ts = when.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    id_list = sorted(identities.values(), key=lambda x: str(x["id"]))
    return {
        "schema_version": "1.0",
        "episode_id": str(episode_id),
        "emitted_at": ts,
        "identities": id_list,
    }
