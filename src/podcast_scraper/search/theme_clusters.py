"""Consumer read helpers for THEME clusters (co-occurrence).

Sibling of ``topic_clusters.py`` (semantic / embedding clusters). Theme clusters
group topics *discussed together* and are produced by the ``topic_theme_clusters``
enricher under ``enrichments/topic_theme_clusters.json`` (NOT ``search/``). This
module only *reads* that artifact for the server; the clustering itself lives in
the enricher.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, cast, Dict, Mapping, Optional

from podcast_scraper.utils.path_validation import safe_resolve_directory

logger = logging.getLogger(__name__)

THEME_CLUSTERS_REL = os.path.join("enrichments", "topic_theme_clusters.json")


def _load_theme_clusters_payload(corpus_root: Path) -> Optional[Dict[str, Any]]:
    """Path-safe load of ``enrichments/topic_theme_clusters.json`` (None if missing/invalid)."""
    root_p = safe_resolve_directory(corpus_root)
    if root_p is None:
        return None
    root_s = os.path.normpath(str(root_p))
    safe_prefix = root_s + os.sep
    joined = os.path.normpath(os.path.join(root_s, THEME_CLUSTERS_REL))
    if joined != root_s and not joined.startswith(safe_prefix):
        return None
    # codeql[py/path-injection] -- joined under root_s (Type 1; CODEQL_DISMISSALS.md).
    if not os.path.isfile(joined):
        return None
    try:
        # codeql[py/path-injection] -- joined sanitized above.
        with open(joined, encoding="utf-8") as fh:
            payload = cast(Dict[str, Any], json.loads(fh.read()))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("theme clusters: skip %s: %s", joined, exc)
        return None
    if not isinstance(payload, dict):
        return None
    # The enrichment framework wraps enricher output in an envelope
    # ({derived, enricher_id, ..., data: {...}}). Unwrap to the payload so callers
    # read ``clusters`` at the top level (parity with the un-enveloped semantic
    # topic_clusters.json). Tolerates an already-unwrapped payload.
    inner = payload.get("data")
    return inner if isinstance(inner, dict) else payload


def consumer_theme_cluster_map(corpus_root: Path) -> Dict[str, Dict[str, Any]]:
    """Per-topic theme-cluster info for attaching to episode topics.

    ``topic_id`` → ``{theme_cluster_id, theme_cluster_label, theme_cluster_size}`` where
    ``theme_cluster_id`` is the cluster's ``graph_compound_parent_id`` (``thc:…``),
    ``theme_cluster_label`` its canonical label, and ``theme_cluster_size`` its member
    count. Topics not in any theme cluster are simply absent. Empty when the artifact
    is missing/invalid (→ no theme markers, today's behaviour).
    """
    payload = _load_theme_clusters_payload(corpus_root)
    if payload is None:
        return {}
    raw = payload.get("clusters")
    if not isinstance(raw, list):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for cl in raw:
        if not isinstance(cl, Mapping):
            continue
        gpid = cl.get("graph_compound_parent_id")
        if not isinstance(gpid, str) or not gpid.strip():
            continue
        label_raw = cl.get("canonical_label")
        label = (
            str(label_raw).strip()
            if isinstance(label_raw, str) and label_raw.strip()
            else gpid.strip()
        )
        members = cl.get("members")
        if not isinstance(members, list):
            continue
        size = len(members)
        for m in members:
            if not isinstance(m, Mapping):
                continue
            tid = m.get("topic_id")
            if isinstance(tid, str) and tid.strip():
                out[tid.strip()] = {
                    "theme_cluster_id": gpid.strip(),
                    "theme_cluster_label": label,
                    "theme_cluster_size": size,
                }
    return out


def _anchor_topic_id(members: list[Any]) -> Optional[str]:
    """Most-central member of a theme cluster — highest ``lift_to_cluster`` (tie: topic_id asc).

    The Home "Storylines" rail opens this topic's card on tap; every member's card shows the same
    "discussed together" set, so the anchor just picks the most representative entry. Falls back to
    the first valid ``topic_id`` when lifts are absent.
    """
    best_id: Optional[str] = None
    best_lift = float("-inf")
    for m in members:
        if not isinstance(m, Mapping):
            continue
        tid = m.get("topic_id")
        if not isinstance(tid, str) or not tid.strip():
            continue
        lift_raw = m.get("lift_to_cluster")
        lift = float(lift_raw) if isinstance(lift_raw, (int, float)) else 0.0
        if best_id is None or lift > best_lift or (lift == best_lift and tid.strip() < best_id):
            best_id, best_lift = tid.strip(), lift
    return best_id


def _theme_cluster_summary(cl: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Project a theme cluster to ``{id, label, size, anchor_topic_id}`` (``None`` if unusable)."""
    gpid = cl.get("graph_compound_parent_id")
    if not isinstance(gpid, str) or not gpid.strip():
        return None
    members = cl.get("members")
    members_list = members if isinstance(members, list) else []
    anchor = _anchor_topic_id(members_list)
    if anchor is None:
        return None  # a themeless / empty cluster can't be opened — skip it
    mc = cl.get("member_count")
    size = mc if isinstance(mc, int) else len(members_list)
    label_raw = cl.get("canonical_label")
    label = (
        str(label_raw).strip() if isinstance(label_raw, str) and label_raw.strip() else gpid.strip()
    )
    return {"id": gpid.strip(), "label": label, "size": size, "anchor_topic_id": anchor}


def top_theme_clusters_by_member_count(corpus_root: Path, top_n: int = 12) -> list[Dict[str, Any]]:
    """Top-N THEME clusters ("storylines") by member count (desc) — for the picker + Home rail.

    Returns ``[{"id", "label", "size", "anchor_topic_id"}, ...]``; empty when the artifact is
    missing/invalid. ``id`` is the cluster's ``graph_compound_parent_id`` (``thc:…``, the interest
    key stored per-user); ``size`` is ``member_count`` when present else ``len(members)``;
    ``anchor_topic_id`` is the most-central member (see :func:`_anchor_topic_id`). Sibling of the
    semantic ``top_clusters_by_member_count`` but over ``enrichments/topic_theme_clusters.json``.
    """
    payload = _load_theme_clusters_payload(corpus_root)
    if payload is None:
        return []
    raw = payload.get("clusters")
    if not isinstance(raw, list):
        return []
    out = [s for cl in raw if isinstance(cl, Mapping) and (s := _theme_cluster_summary(cl))]
    out.sort(key=lambda c: c["size"], reverse=True)
    return out[: max(top_n, 0)]


def consumer_theme_cluster_siblings(corpus_root: Path, topic_id: str) -> list[Dict[str, str]]:
    """Sibling topics sharing ``topic_id``'s THEME cluster, excluding itself.

    Returns ``[{"id", "label"}, ...]`` from the theme cluster's ``members``. Empty when the
    topic is in no theme cluster, or the artifact is missing/invalid. Mirrors the semantic
    ``consumer_cluster_siblings`` but over ``enrichments/topic_theme_clusters.json``.
    """
    tid = topic_id.strip()
    if not tid:
        return []
    payload = _load_theme_clusters_payload(corpus_root)
    if payload is None:
        return []
    raw = payload.get("clusters")
    if not isinstance(raw, list):
        return []
    for cl in raw:
        if not isinstance(cl, Mapping):
            continue
        members = cl.get("members")
        if not isinstance(members, list):
            continue
        member_ids = {
            str(m.get("topic_id")).strip()
            for m in members
            if isinstance(m, Mapping) and isinstance(m.get("topic_id"), str)
        }
        if tid not in member_ids:
            continue
        siblings: list[Dict[str, str]] = []
        for m in members:
            if not isinstance(m, Mapping):
                continue
            mid_raw = m.get("topic_id")
            if not isinstance(mid_raw, str) or not mid_raw.strip():
                continue
            mid = mid_raw.strip()
            if mid == tid:
                continue
            label_raw = m.get("label")
            label = (
                label_raw.strip()
                if isinstance(label_raw, str) and label_raw.strip()
                else mid.split(":", 1)[-1]
            )
            siblings.append({"id": mid, "label": label})
        return siblings
    return []


__all__ = [
    "consumer_theme_cluster_map",
    "consumer_theme_cluster_siblings",
    "top_theme_clusters_by_member_count",
]
