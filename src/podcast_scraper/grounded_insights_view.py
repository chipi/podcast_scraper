"""Read a GI 4.0 file with the GI 3.1 names this repository's readers use.

GI 4.0 (R3b) rewrites every served GI file: ``Episode`` becomes ``ContentItem``, ``Podcast``
becomes ``Source`` (with KG's source id), ``HAS_EPISODE`` becomes ``HAS_ITEM``, the evidence
fields are renamed (``item_id``, ``document_ref``, ``start_ms``/``end_ms``, ``published_at``) and
the podcast-only fields move under ``properties.podcast_episode``. The readers here were written
against 3.1, so each GI file goes through :func:`gi_read_view` where it is loaded: every 4.0 name
is read first, its 3.1 name is filled in only when the file does not carry it, and nothing a file
already holds is changed. A 3.1 file comes back unchanged.
"""

from __future__ import annotations

import copy
from typing import Any, cast, Dict, Mapping, TypeVar

_T = TypeVar("_T")

_NODE_TYPE_3_1 = {"ContentItem": "Episode", "Source": "Podcast"}
_EDGE_TYPE_3_1 = {"HAS_ITEM": "HAS_EPISODE"}

#: 3.1 name <- 4.0 name, per node type, for fields renamed at the top of ``properties``.
_RENAMED: Dict[str, tuple[tuple[str, str], ...]] = {
    "Episode": (("publish_date", "published_at"),),
    "Insight": (("episode_id", "item_id"),),
    "Quote": (
        ("episode_id", "item_id"),
        ("transcript_ref", "document_ref"),
        ("timestamp_start_ms", "start_ms"),
        ("timestamp_end_ms", "end_ms"),
    ),
}

#: Fields GI 4.0 nests under ``properties.podcast_episode`` (ADR-181), per node type.
_NESTED: Dict[str, tuple[str, ...]] = {
    "Episode": ("podcast_id",),
    "Insight": ("speaker", "speaker_voice_type"),
    "Quote": ("speaker_id", "speaker_name", "speaker_voice_type"),
}


def is_gi_v4(artifact: Mapping[str, Any]) -> bool:
    """Whether ``artifact`` carries any GI 4.0 name this view maps."""
    if "item_id" in artifact:
        return True
    for node in artifact.get("nodes") or []:
        if isinstance(node, Mapping) and node.get("type") in _NODE_TYPE_3_1:
            return True
    for edge in artifact.get("edges") or []:
        if isinstance(edge, Mapping) and edge.get("type") in _EDGE_TYPE_3_1:
            return True
    return False


def _fill_props(props: Dict[str, Any], kind: str) -> None:
    for old, new in _RENAMED.get(kind, ()):
        if old not in props and props.get(new) is not None:
            props[old] = props[new]
    nested = props.get("podcast_episode")
    if isinstance(nested, Mapping):
        for name in _NESTED.get(kind, ()):
            if name not in props and name in nested:
                props[name] = nested[name]


def gi_read_view(artifact: _T) -> _T:
    """``artifact`` with the 3.1 names filled in from its 4.0 names; a 3.1 file is returned as
    is (the same object). The input is never mutated."""
    if not isinstance(artifact, Mapping) or not is_gi_v4(artifact):
        return artifact
    out: Dict[str, Any] = copy.deepcopy(dict(artifact))
    if "episode_id" not in out and out.get("item_id"):
        out["episode_id"] = out["item_id"]
    for node in out.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        kind = _NODE_TYPE_3_1.get(str(node.get("type")), node.get("type"))
        node["type"] = kind
        props = node.get("properties")
        if isinstance(props, dict):
            _fill_props(props, str(kind))
    for edge in out.get("edges") or []:
        if isinstance(edge, dict) and edge.get("type") in _EDGE_TYPE_3_1:
            edge["type"] = _EDGE_TYPE_3_1[edge["type"]]
    return cast(_T, out)
