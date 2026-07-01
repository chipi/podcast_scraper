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
    return payload if isinstance(payload, dict) else None


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


__all__ = ["consumer_theme_cluster_map", "consumer_theme_cluster_siblings"]
