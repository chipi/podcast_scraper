"""Read a KG 3.0 file with the KG 2.x names this repository's readers use (R3b).

R3b rewrites every served KG file to 3.0 (``kg/v3_file.to_v3_file``: the :func:`kg_as_v3` view
stored), and news items are KG 3.0 from their first one. The readers here were written against
2.x, so each KG file goes through :func:`kg_read_view` where it is loaded. It undoes
:func:`podcast_scraper.kg.as_v3.kg_as_v3` additively: every 3.0 name stays, and its 2.x name is
filled in only where the file does not carry it (``ContentItem`` and ``Source`` nodes are renamed,
as a reader tests a node's type). A 2.x file comes back unchanged (the same object). The input is
never mutated, and a view is never written back.
"""

from __future__ import annotations

import copy
from typing import Any, cast, Dict, Mapping, TypeVar

_T = TypeVar("_T")

_NODE_TYPE_2 = {"ContentItem": "Episode", "Source": "Podcast"}
_EDGE_BY_ROLE_2 = {"host": "HOSTS", "guest": "GUESTS_ON"}


def is_kg_v3(artifact: Mapping[str, Any]) -> bool:
    """Whether ``artifact`` is KG 3.x (its ``schema_version``, or a 3.0 shape without one)."""
    version = str(artifact.get("schema_version") or "")
    if version.startswith("3."):
        return True
    if "item_id" in artifact and "episode_id" not in artifact:
        return True
    return any(
        isinstance(n, Mapping) and n.get("type") in _NODE_TYPE_2
        for n in artifact.get("nodes") or []
    )


def kg_read_view(artifact: _T) -> _T:
    """``artifact`` with the KG 2.x names filled in from its 3.0 names (this module's docstring)."""
    if not isinstance(artifact, Mapping) or not is_kg_v3(artifact):
        return artifact
    out: Dict[str, Any] = copy.deepcopy(dict(artifact))
    if "episode_id" not in out and out.get("item_id"):
        out["episode_id"] = out["item_id"]
    for node in out.get("nodes") or []:
        if not isinstance(node, dict) or node.get("type") not in _NODE_TYPE_2:
            continue
        node["type"] = _NODE_TYPE_2[node["type"]]
        props = node.get("properties")
        if node["type"] == "Episode" and isinstance(props, dict):
            if "podcast_id" not in props and props.get("feed_id"):
                props["podcast_id"] = props["feed_id"]
            if "publish_date" not in props and props.get("published_at"):
                props["publish_date"] = props["published_at"]
    for edge in out.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        if edge.get("type") == "HAS_ITEM":
            edge["type"] = "HAS_EPISODE"
        elif edge.get("type") == "CONTRIBUTES_TO":
            role = str((edge.get("properties") or {}).get("role") or "")
            if role in _EDGE_BY_ROLE_2:
                edge["type"] = _EDGE_BY_ROLE_2[role]
    extraction = out.get("extraction")
    if isinstance(extraction, dict) and "transcript_ref" not in extraction:
        if extraction.get("document_ref"):
            extraction["transcript_ref"] = extraction["document_ref"]
    return cast(_T, out)
