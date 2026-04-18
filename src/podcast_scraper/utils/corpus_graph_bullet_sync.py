"""Patch existing GI/KG artifacts from repaired summary bullets (no LLM re-run).

Used by ``scripts/tools/repair_fenced_summary_metadata.py`` when ``--patch-graphs``
is set: updates Insight text and Topic nodes/edges in ``*.gi.json``, replaces
bullet-derived Topic nodes in ``*.kg.json`` when safe, then callers may rebuild
``*.bridge.json`` via :func:`podcast_scraper.builders.bridge_builder.build_bridge`.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set, Tuple

from podcast_scraper import config_constants
from podcast_scraper.graph_id_utils import (
    episode_node_id,
    slugify_label,
    topic_node_id_from_slug,
)
from podcast_scraper.kg.llm_extract import strip_known_ml_bullet_prefixes


def bullet_labels_from_summary_bullets(
    bullets: Any, *, max_topics: Optional[int] = None
) -> List[str]:
    """Normalize metadata ``summary.bullets`` the same way as GI/KG ingestion."""
    cap = max_topics or int(config_constants.DEFAULT_SUMMARY_BULLETS_DOWNSTREAM_MAX)
    if not isinstance(bullets, list) or not bullets:
        return []
    out: List[str] = []
    for raw in bullets[:cap]:
        s = strip_known_ml_bullet_prefixes(str(raw))
        if s:
            out.append(s)
    return out


def _dedupe_topic_node_specs(topic_labels: List[str]) -> List[Tuple[str, str]]:
    """Mirror ``gi.pipeline._dedupe_topic_node_specs`` (topic id + display label)."""
    seen_slugs: Set[str] = set()
    out: List[Tuple[str, str]] = []
    for lab in topic_labels:
        raw = (lab or "").strip()
        if not raw:
            continue
        slug = slugify_label(raw)
        if slug in seen_slugs:
            continue
        seen_slugs.add(slug)
        out.append((topic_node_id_from_slug(slug), raw[:200]))
    return out


def _insight_ids_has_insight_order(gi: Dict[str, Any]) -> List[str]:
    """Insight node ids in the order of ``HAS_INSIGHT`` edges (episode → insight)."""
    episode_ids = {
        str(n["id"])
        for n in (gi.get("nodes") or [])
        if isinstance(n, dict) and n.get("type") == "Episode" and n.get("id") is not None
    }
    if len(episode_ids) != 1:
        return []
    ordered: List[str] = []
    seen: Set[str] = set()
    for e in gi.get("edges") or []:
        if not isinstance(e, dict):
            continue
        if e.get("type") != "HAS_INSIGHT":
            continue
        fr = e.get("from")
        to_id = e.get("to")
        if fr not in episode_ids or not isinstance(to_id, str) or not to_id.strip():
            continue
        tid = to_id.strip()
        if tid not in seen:
            seen.add(tid)
            ordered.append(tid)
    return ordered


def _topic_label_looks_corrupt(label: Any) -> bool:
    if not isinstance(label, str):
        return False
    t = label.strip()
    if "```" in label:
        return True
    if t.startswith("{") and ("bullets" in label or "key_quotes" in label):
        return True
    return False


def kg_should_replace_topics_from_bullets(kg: Dict[str, Any]) -> bool:
    """True when KG topic nodes were bullet-derived or look like fenced JSON."""
    ext = kg.get("extraction") or {}
    mv = str(ext.get("model_version") or "").strip()
    if mv in ("summary_bullets", "stub"):
        return True
    if mv.startswith("provider:summary_bullets:"):
        return True
    for n in kg.get("nodes") or []:
        if not isinstance(n, dict) or n.get("type") != "Topic":
            continue
        props = n.get("properties") or {}
        if _topic_label_looks_corrupt(props.get("label")):
            return True
    return False


def patch_gi_for_bullet_labels(
    gi: Dict[str, Any], bullet_labels: List[str]
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Return a deep-copied GI dict with Insight texts and Topic/ABOUT graph synced."""
    if not bullet_labels:
        return None, "skip-gi: no bullet labels"
    eid_raw = gi.get("episode_id")
    if not isinstance(eid_raw, str) or not eid_raw.strip():
        return None, "skip-gi: missing episode_id"
    episode_id = eid_raw.strip()

    out = json.loads(json.dumps(gi))
    nodes = out.get("nodes")
    edges = out.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        return None, "skip-gi: invalid nodes/edges"

    insight_order = _insight_ids_has_insight_order(out)
    if not insight_order:
        return None, "skip-gi: no HAS_INSIGHT order"

    id_to_node = {str(n["id"]): n for n in nodes if isinstance(n, dict) and n.get("id") is not None}
    for idx, iid in enumerate(insight_order):
        node = id_to_node.get(iid)
        if not isinstance(node, dict):
            continue
        props = node.get("properties")
        if not isinstance(props, dict):
            props = {}
            node["properties"] = props
        if idx < len(bullet_labels):
            props["text"] = bullet_labels[idx]

    old_topic_ids = {
        str(n["id"])
        for n in nodes
        if isinstance(n, dict) and n.get("type") == "Topic" and n.get("id") is not None
    }
    new_nodes = [n for n in nodes if isinstance(n, dict) and n.get("type") != "Topic"]
    new_edges = [
        e
        for e in edges
        if isinstance(e, dict)
        and not (
            e.get("type") == "ABOUT" and isinstance(e.get("to"), str) and e["to"] in old_topic_ids
        )
    ]

    topic_specs = _dedupe_topic_node_specs(bullet_labels)
    for tid, display_label in topic_specs:
        new_nodes.append(
            {"id": tid, "type": "Topic", "properties": {"label": display_label}},
        )
    ep_nid = episode_node_id(episode_id)
    for iid in insight_order:
        for tid, _ in topic_specs:
            new_edges.append({"type": "ABOUT", "from": iid, "to": tid})

    out["nodes"] = new_nodes
    out["edges"] = new_edges
    if ep_nid not in {str(n.get("id")) for n in new_nodes if isinstance(n, dict)}:
        return None, "skip-gi: episode node missing after patch"
    return out, "patched-gi"


def patch_kg_for_bullet_labels(
    kg: Dict[str, Any], bullet_labels: List[str]
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Replace bullet-derived Topic nodes + MENTIONS edges when policy allows."""
    if not bullet_labels:
        return None, "skip-kg: no bullet labels"
    if not kg_should_replace_topics_from_bullets(kg):
        return None, "skip-kg: provider-derived topics preserved"

    eid_raw = kg.get("episode_id")
    if not isinstance(eid_raw, str) or not eid_raw.strip():
        return None, "skip-kg: missing episode_id"
    episode_id = eid_raw.strip()
    ep_nid = episode_node_id(episode_id)

    out = json.loads(json.dumps(kg))
    nodes = out.get("nodes")
    edges = out.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        return None, "skip-kg: invalid nodes/edges"

    old_topic_ids = {
        str(n["id"])
        for n in nodes
        if isinstance(n, dict) and n.get("type") == "Topic" and n.get("id") is not None
    }
    new_nodes = [n for n in nodes if isinstance(n, dict) and n.get("type") != "Topic"]
    new_edges: List[Dict[str, Any]] = []
    for e in edges:
        if not isinstance(e, dict):
            continue
        fr = e.get("from")
        to_id = e.get("to")
        if (
            e.get("type") == "MENTIONS"
            and isinstance(fr, str)
            and isinstance(to_id, str)
            and (fr in old_topic_ids or to_id in old_topic_ids)
        ):
            continue
        new_edges.append(e)

    seen_slugs: Set[str] = set()
    for raw in bullet_labels:
        lab = raw.strip()
        if not lab:
            continue
        lab_store = lab[:500]
        slug = slugify_label(lab_store)
        if slug in seen_slugs:
            continue
        seen_slugs.add(slug)
        topic_id = topic_node_id_from_slug(slug)
        new_nodes.append(
            {
                "id": topic_id,
                "type": "Topic",
                "properties": {"label": lab_store[:200], "slug": slug},
            }
        )
        new_edges.append(
            {"from": topic_id, "to": ep_nid, "type": "MENTIONS", "properties": {}},
        )

    out["nodes"] = new_nodes
    out["edges"] = new_edges
    return out, "patched-kg"
