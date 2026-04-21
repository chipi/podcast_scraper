"""Lift FAISS transcript chunks to GIL Insights + bridge display names."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional

from podcast_scraper.builders.bridge_artifact_paths import bridge_path_next_to_gi_json
from podcast_scraper.gi.edge_normalization import normalize_gil_edge_type
from podcast_scraper.search.cil_lift_overrides import (
    CilLiftOverrides,
    resolve_id_alias,
)
from podcast_scraper.utils.path_validation import normpath_if_under_root, safe_resolve_directory

logger = logging.getLogger(__name__)

# Public alias (same name as historical API); canonical implementation in builders.
bridge_path_next_to_gi = bridge_path_next_to_gi_json


def _bridge_json_str_next_to_gi_json(safe_gi_path: str) -> str:
    """Sibling ``*.bridge.json`` path for a sanitized ``*.gi.json`` (string ops only)."""
    parent = os.path.dirname(safe_gi_path)
    name = os.path.basename(safe_gi_path)
    if name.endswith(".gi.json"):
        return os.path.normpath(os.path.join(parent, name[: -len(".gi.json")] + ".bridge.json"))
    stem, dot, _ext = name.rpartition(".")
    if not dot:
        return os.path.normpath(os.path.join(parent, f"{name}.bridge.json"))
    if stem.endswith(".gi"):
        return os.path.normpath(os.path.join(parent, stem[: -len(".gi")] + ".bridge.json"))
    return os.path.normpath(os.path.join(parent, f"{stem}.bridge.json"))


def _nodes_by_id(artifact: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for n in artifact.get("nodes") or []:
        if isinstance(n, dict):
            nid = n.get("id")
            if isinstance(nid, str) and nid:
                out[nid] = n
    return out


def _display_name_from_bridge(bridge: Mapping[str, Any], canonical_id: str) -> str:
    for row in bridge.get("identities") or []:
        if not isinstance(row, dict):
            continue
        rid = row.get("id")
        if rid == canonical_id and isinstance(rid, str):
            dn = row.get("display_name")
            return str(dn).strip() if isinstance(dn, str) else ""
    return ""


def _best_overlapping_quote(
    gi: Mapping[str, Any],
    chunk_start: int,
    chunk_end: int,
) -> Optional[str]:
    """Pick Quote id with largest half-open overlap with ``[chunk_start, chunk_end)``."""
    best_id: Optional[str] = None
    best_w = -1
    for node in gi.get("nodes") or []:
        if not isinstance(node, dict) or node.get("type") != "Quote":
            continue
        qid = node.get("id")
        props = node.get("properties")
        if not isinstance(qid, str) or not isinstance(props, dict):
            continue
        try:
            q0 = int(props["char_start"])
            q1 = int(props["char_end"])
        except (KeyError, TypeError, ValueError):
            continue
        if q1 <= q0:
            continue
        lo = max(chunk_start, q0)
        hi = min(chunk_end, q1)
        w = max(0, hi - lo)
        if w > best_w:
            best_w = w
            best_id = qid
    if best_w <= 0:
        return None
    return best_id


def _insight_for_quote(gi: Mapping[str, Any], quote_id: str) -> Optional[str]:
    for e in gi.get("edges") or []:
        if not isinstance(e, dict):
            continue
        if normalize_gil_edge_type(e.get("type")) != "SUPPORTED_BY":
            continue
        if str(e.get("to")) != quote_id:
            continue
        iid = e.get("from")
        if isinstance(iid, str) and iid:
            return iid
    return None


def _topic_ids_for_insight(gi: Mapping[str, Any], insight_id: str) -> List[str]:
    topics: List[str] = []
    for e in gi.get("edges") or []:
        if not isinstance(e, dict):
            continue
        if normalize_gil_edge_type(e.get("type")) != "ABOUT":
            continue
        if str(e.get("from")) != insight_id:
            continue
        tid = e.get("to")
        if isinstance(tid, str) and tid.startswith("topic:"):
            topics.append(tid)
    return topics


def _person_id_for_quote(gi: Mapping[str, Any], quote_id: str) -> Optional[str]:
    for e in gi.get("edges") or []:
        if not isinstance(e, dict):
            continue
        if normalize_gil_edge_type(e.get("type")) != "SPOKEN_BY":
            continue
        if str(e.get("from")) != quote_id:
            continue
        pid = e.get("to")
        if isinstance(pid, str) and pid.startswith("person:"):
            return pid
    return None


def _insight_payload(nodes: Mapping[str, Dict[str, Any]], insight_id: str) -> Dict[str, Any]:
    raw_node = nodes.get(insight_id)
    node: Dict[str, Any] = raw_node if isinstance(raw_node, dict) else {}
    props_raw = node.get("properties")
    props = props_raw if isinstance(props_raw, dict) else {}
    text = props.get("text")
    grounded = props.get("grounded")
    itype = props.get("insight_type")
    ph = props.get("position_hint")
    ph_out: float | None
    if isinstance(ph, (int, float)):
        ph_out = float(ph)
    elif isinstance(ph, str):
        try:
            ph_out = float(ph.strip())
        except ValueError:
            ph_out = None
    else:
        ph_out = None
    return {
        "id": insight_id,
        "text": str(text) if isinstance(text, str) else "",
        "grounded": bool(grounded) if isinstance(grounded, bool) else False,
        "insight_type": str(itype) if isinstance(itype, str) else None,
        "position_hint": ph_out,
    }


def try_lift_transcript_chunk_from_gi(
    gi: Mapping[str, Any],
    corpus_root: Path,
    gi_path: Path,
    *,
    char_start: int,
    char_end: int,
    overrides: Optional[CilLiftOverrides] = None,
) -> Optional[Dict[str, Any]]:
    """If a Quote overlaps the chunk span, return a ``lifted`` dict, else ``None``."""
    quote_id = _best_overlapping_quote(gi, char_start, char_end)
    if quote_id is None:
        return None

    insight_id = _insight_for_quote(gi, quote_id)
    if not insight_id:
        return None

    nodes = _nodes_by_id(gi)
    ins_node = nodes.get(insight_id)
    if not ins_node or ins_node.get("type") != "Insight":
        return None

    topic_ids = _topic_ids_for_insight(gi, insight_id)
    topic_id = topic_ids[0] if topic_ids else None

    raw_q = nodes.get(quote_id)
    qnode: Dict[str, Any] = raw_q if isinstance(raw_q, dict) else {}
    qprops_raw = qnode.get("properties")
    qprops = qprops_raw if isinstance(qprops_raw, dict) else {}
    ts0 = qprops.get("timestamp_start_ms")
    ts1 = qprops.get("timestamp_end_ms")
    try:
        ts0_i = int(ts0) if ts0 is not None else 0
        ts1_i = int(ts1) if ts1 is not None else 0
    except (TypeError, ValueError):
        ts0_i, ts1_i = 0, 0

    person_id = _person_id_for_quote(gi, quote_id)
    if overrides and person_id:
        person_id = resolve_id_alias(person_id, overrides.entity_id_aliases)
    if overrides and topic_id:
        topic_id = resolve_id_alias(topic_id, overrides.topic_id_aliases)

    speaker_name = ""
    topic_name = ""
    root_resolved = safe_resolve_directory(corpus_root)
    if root_resolved is not None:
        root_s = os.path.normpath(str(root_resolved))
        safe_prefix = root_s + os.sep
        safe_gi = normpath_if_under_root(os.path.normpath(str(gi_path)), root_s)
        if safe_gi and safe_gi.startswith(safe_prefix):
            bridge_cand = _bridge_json_str_next_to_gi_json(safe_gi)
            safe_bridge = normpath_if_under_root(os.path.normpath(bridge_cand), root_s)
            if safe_bridge and safe_bridge.startswith(safe_prefix) and os.path.isfile(safe_bridge):
                try:
                    with open(safe_bridge, encoding="utf-8") as fh:
                        bridge = json.loads(fh.read())
                except (OSError, json.JSONDecodeError):
                    bridge = {}
                if isinstance(bridge, dict):
                    if person_id:
                        speaker_name = _display_name_from_bridge(bridge, person_id)
                    if topic_id:
                        topic_name = _display_name_from_bridge(bridge, topic_id)

    lifted: Dict[str, Any] = {
        "insight": _insight_payload(nodes, insight_id),
        "quote": {
            "timestamp_start_ms": ts0_i,
            "timestamp_end_ms": ts1_i,
        },
    }
    if person_id:
        lifted["speaker"] = {"id": person_id, "display_name": speaker_name}
    if topic_id:
        lifted["topic"] = {"id": topic_id, "display_name": topic_name}
    return lifted


class TranscriptLiftGiCache:
    """In-memory GI JSON cache for a search request."""

    def __init__(self) -> None:
        self._gi: Dict[str, Optional[Dict[str, Any]]] = {}

    def get(self, gi_path: Path) -> Optional[Dict[str, Any]]:
        """Return parsed GI JSON for *gi_path*, loading and caching on first access."""
        key = str(gi_path.resolve())
        if key in self._gi:
            return self._gi[key]
        try:
            doc = json.loads(gi_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.debug("lift cache: gi load failed %s: %s", gi_path, exc)
            self._gi[key] = None
            return None
        if not isinstance(doc, dict):
            self._gi[key] = None
            return None
        self._gi[key] = doc
        return doc


def lift_row_if_transcript(
    row: MutableMapping[str, Any],
    corpus_root: Path,
    gi_path: Path,
    cache: TranscriptLiftGiCache,
    overrides: Optional[CilLiftOverrides] = None,
) -> None:
    """Mutate ``row`` in place: set ``lifted`` when doc_type is transcript and lift succeeds."""
    meta = row.get("metadata")
    if not isinstance(meta, dict):
        return
    if meta.get("doc_type") != "transcript":
        return
    ep = meta.get("episode_id")
    if not isinstance(ep, str) or not ep.strip():
        return
    try:
        cs = int(meta["char_start"])
        ce = int(meta["char_end"])
    except (KeyError, TypeError, ValueError):
        return
    if ce <= cs:
        return
    shift = int(overrides.transcript_char_shift) if overrides else 0
    cs_adj = cs + shift
    ce_adj = ce + shift
    if ce_adj <= cs_adj:
        return

    doc = cache.get(gi_path)
    if doc is None:
        return

    lifted = try_lift_transcript_chunk_from_gi(
        doc,
        corpus_root,
        gi_path,
        char_start=cs_adj,
        char_end=ce_adj,
        overrides=overrides,
    )
    if lifted is not None:
        row["lifted"] = lifted
