"""KG extraction pipeline: transcript + episode context -> kg.json-shaped dict."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, cast, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


def _safe_iso_date(s: Optional[str]) -> str:
    """Return ISO date-time string; use placeholder if missing."""
    if s and s.strip():
        return s.strip()
    return "1970-01-01T00:00:00Z"


def _slugify(label: str, max_len: int = 80) -> str:
    """Build a filesystem-safe slug from a human label."""
    s = label.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-") or "topic"
    return s[:max_len]


def _resolve_source(cfg: Optional[Any]) -> str:
    if cfg is not None:
        return str(getattr(cfg, "kg_extraction_source", "summary_bullets") or "summary_bullets")
    return "summary_bullets"


def _max_topics(cfg: Optional[Any]) -> int:
    if cfg is not None:
        v = int(getattr(cfg, "kg_max_topics", 5) or 5)
        return max(1, min(v, 20))
    return 5


def _max_entities(cfg: Optional[Any]) -> int:
    if cfg is not None:
        v = int(getattr(cfg, "kg_max_entities", 15) or 15)
        return max(1, min(v, 50))
    return 15


def _merge_pipeline_default(cfg: Optional[Any]) -> bool:
    if cfg is None:
        return True
    return bool(getattr(cfg, "kg_merge_pipeline_entities", True))


def _topic_labels_from_args(
    topic_labels: Optional[List[str]],
    topic_label: Optional[str],
    cfg: Optional[Any],
) -> List[str]:
    labels: List[str] = []
    if topic_labels:
        labels.extend(str(x).strip() for x in topic_labels if str(x).strip())
    elif topic_label and topic_label.strip():
        labels.append(topic_label.strip())
    max_t = _max_topics(cfg)
    return labels[:max_t]


def _try_provider_extraction(
    transcript_text: str,
    episode_title: str,
    cfg: Optional[Any],
    kg_extraction_provider: Any,
    pipeline_metrics: Optional[Any],
) -> Optional[Dict[str, Any]]:
    if kg_extraction_provider is None:
        return None
    extract_fn = getattr(kg_extraction_provider, "extract_kg_graph", None)
    if not callable(extract_fn):
        return None
    max_t = _max_topics(cfg)
    max_e = _max_entities(cfg)
    params: Dict[str, Any] = {}
    if cfg is not None and getattr(cfg, "kg_extraction_model", None):
        params["kg_extraction_model"] = cfg.kg_extraction_model
    try:
        partial = extract_fn(
            transcript_text,
            episode_title=episode_title,
            max_topics=max_t,
            max_entities=max_e,
            params=params or None,
        )
    except Exception as exc:
        logger.debug("KG provider extract_kg_graph failed: %s", exc, exc_info=True)
        return None
    if not partial or not isinstance(partial, dict):
        return None
    topics = partial.get("topics") or []
    entities = partial.get("entities") or []
    if not topics and not entities:
        return None
    if pipeline_metrics is not None:
        pipeline_metrics.kg_provider_extractions += 1
    return cast(Dict[str, Any], partial)


def build_artifact(
    episode_id: str,
    transcript_text: str,
    *,
    podcast_id: str,
    episode_title: str,
    publish_date: Optional[str] = None,
    transcript_ref: str = "transcript.txt",
    model_version: Optional[str] = None,
    topic_label: Optional[str] = None,
    topic_labels: Optional[List[str]] = None,
    detected_hosts: Optional[List[str]] = None,
    detected_guests: Optional[List[str]] = None,
    cfg: Optional[Any] = None,
    kg_extraction_provider: Optional[Any] = None,
    pipeline_metrics: Optional[Any] = None,
) -> Dict[str, Any]:
    """Build a KG artifact dict for one episode.

    ``kg_extraction_source`` (from ``cfg``): ``stub`` | ``summary_bullets`` | ``provider``.
    When ``provider`` is set, calls ``extract_kg_graph`` on the summarization provider when
    available; otherwise falls back to ``summary_bullets`` when bullet/topic hints exist.

    Args:
        episode_id: Stable episode id (same family as metadata / GIL).
        transcript_text: Full transcript (used for provider extraction).
        podcast_id: Feed / podcast id string.
        episode_title: Episode title for Episode node.
        publish_date: ISO publish date string or None.
        transcript_ref: Relative transcript path for provenance.
        model_version: Optional override for extraction.model_version.
        topic_label: Optional single summary bullet / topic hint (legacy).
        topic_labels: Optional list of topic hints (e.g. summary bullets); capped by cfg.
        detected_hosts: Optional host names from speaker pipeline.
        detected_guests: Optional guest names from speaker pipeline.
        cfg: Optional Config (KG extraction source, limits, merge flag).
        kg_extraction_provider: Summarization provider instance for ``provider`` source.
        pipeline_metrics: Optional metrics collector (increments kg_provider_extractions).

    Returns:
        Dict matching docs/kg/kg.schema.json (minimal validation via kg.schema).
    """
    date_str = _safe_iso_date(publish_date)
    ep_node_id = f"kg:episode:{episode_id}"
    source = _resolve_source(cfg)

    nodes: List[Dict[str, Any]] = [
        {
            "id": ep_node_id,
            "type": "Episode",
            "properties": {
                "podcast_id": podcast_id or "podcast:unknown",
                "title": (episode_title or "Episode").strip() or "Episode",
                "publish_date": date_str,
            },
        }
    ]
    edges: List[Dict[str, Any]] = []

    bullet_labels = _topic_labels_from_args(topic_labels, topic_label, cfg)
    used_provider = False
    llm_partial: Optional[Dict[str, Any]] = None
    resolved_model = model_version

    if source == "provider":
        llm_partial = _try_provider_extraction(
            transcript_text,
            episode_title or "",
            cfg,
            kg_extraction_provider,
            pipeline_metrics,
        )
        used_provider = llm_partial is not None

    if llm_partial:
        _append_topics_and_entities_from_partial(
            episode_id,
            ep_node_id,
            llm_partial,
            nodes,
            edges,
            _max_topics(cfg),
            _max_entities(cfg),
        )
        if resolved_model is None:
            mid = None
            if kg_extraction_provider is not None:
                mid = getattr(kg_extraction_provider, "summary_model", None)
            if cfg is not None and getattr(cfg, "kg_extraction_model", None):
                mid = cfg.kg_extraction_model
            resolved_model = f"provider:{mid or 'unknown'}"
    elif source != "stub" and bullet_labels:
        _append_topics_from_labels(episode_id, ep_node_id, bullet_labels, nodes, edges)
        if resolved_model is None:
            # Legacy: callers without cfg keep "stub" label (tests / old metadata).
            resolved_model = "summary_bullets" if cfg is not None else "stub"
    elif source == "summary_bullets" and not bullet_labels and not used_provider:
        if resolved_model is None and cfg is not None:
            resolved_model = "summary_bullets"

    if source == "stub" and resolved_model is None:
        resolved_model = "stub"
    if resolved_model is None:
        resolved_model = "stub"

    merge_pipe = _merge_pipeline_default(cfg)
    if llm_partial and not merge_pipe:
        pass
    else:
        _append_pipeline_entities(
            episode_id,
            ep_node_id,
            detected_hosts,
            detected_guests,
            nodes,
            edges,
            existing_entity_names=_entity_names_lower(nodes),
        )

    extracted_at = (
        datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )

    return {
        "schema_version": "1.0",
        "episode_id": episode_id,
        "extraction": {
            "model_version": resolved_model,
            "extracted_at": extracted_at,
            "transcript_ref": transcript_ref,
        },
        "nodes": nodes,
        "edges": edges,
    }


def _entity_names_lower(nodes: List[Dict[str, Any]]) -> Set[str]:
    out: Set[str] = set()
    for n in nodes:
        if n.get("type") != "Entity":
            continue
        props = n.get("properties") or {}
        name = props.get("name")
        if isinstance(name, str) and name.strip():
            out.add(name.strip().lower())
    return out


def _append_topics_from_labels(
    episode_id: str,
    ep_node_id: str,
    labels: List[str],
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
) -> None:
    seen_slugs: Set[str] = set()
    for raw in labels:
        if not raw.strip():
            continue
        lab = raw.strip()[:500]
        slug = _slugify(lab)
        if slug in seen_slugs:
            continue
        seen_slugs.add(slug)
        topic_id = f"kg:topic:{episode_id}:{slug}"
        nodes.append(
            {
                "id": topic_id,
                "type": "Topic",
                "properties": {"label": lab[:200], "slug": slug},
            }
        )
        edges.append({"from": topic_id, "to": ep_node_id, "type": "MENTIONS", "properties": {}})


def _append_topics_and_entities_from_partial(
    episode_id: str,
    ep_node_id: str,
    partial: Dict[str, Any],
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    max_topics: int,
    max_entities: int,
) -> None:
    topics = partial.get("topics") or []
    entities = partial.get("entities") or []
    seen_slugs: Set[str] = set()
    t_count = 0
    if isinstance(topics, list):
        for item in topics:
            if t_count >= max_topics:
                break
            if isinstance(item, dict):
                lab = item.get("label")
            else:
                lab = None
            if not isinstance(lab, str) or not lab.strip():
                continue
            lab_s = lab.strip()[:500]
            slug = _slugify(lab_s)
            if slug in seen_slugs:
                continue
            seen_slugs.add(slug)
            topic_id = f"kg:topic:{episode_id}:llm:{slug}"
            nodes.append(
                {
                    "id": topic_id,
                    "type": "Topic",
                    "properties": {"label": lab_s[:200], "slug": slug},
                }
            )
            edges.append({"from": topic_id, "to": ep_node_id, "type": "MENTIONS", "properties": {}})
            t_count += 1

    e_count = 0
    seen_names: Set[str] = set()
    if isinstance(entities, list):
        for item in entities:
            if e_count >= max_entities:
                break
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            kind = item.get("entity_kind") or "person"
            if not isinstance(name, str) or not name.strip():
                continue
            name_s = name.strip()[:500]
            key = name_s.lower()
            if key in seen_names:
                continue
            seen_names.add(key)
            ek = kind if kind in ("person", "organization") else "person"
            eid = f"kg:entity:{episode_id}:llm:{e_count}"
            nodes.append(
                {
                    "id": eid,
                    "type": "Entity",
                    "properties": {"name": name_s, "entity_kind": ek, "role": "mentioned"},
                }
            )
            edges.append({"from": eid, "to": ep_node_id, "type": "MENTIONS", "properties": {}})
            e_count += 1


def _append_pipeline_entities(
    episode_id: str,
    ep_node_id: str,
    detected_hosts: Optional[List[str]],
    detected_guests: Optional[List[str]],
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    existing_entity_names: Set[str],
) -> None:
    def _add(names: List[str], role: str, kind: str, start_idx: int) -> int:
        idx = start_idx
        for name in names:
            n = (name or "").strip()
            if not n:
                continue
            if n.lower() in existing_entity_names:
                continue
            existing_entity_names.add(n.lower())
            eid = f"kg:entity:{episode_id}:{role}:{idx}"
            idx += 1
            nodes.append(
                {
                    "id": eid,
                    "type": "Entity",
                    "properties": {
                        "name": n[:500],
                        "entity_kind": kind,
                        "role": role,
                    },
                }
            )
            edges.append(
                {
                    "from": eid,
                    "to": ep_node_id,
                    "type": "MENTIONS",
                    "properties": {},
                }
            )
        return idx

    hosts = [h for h in (detected_hosts or []) if isinstance(h, str)]
    guests = [g for g in (detected_guests or []) if isinstance(g, str)]
    nxt = _add(hosts[:20], "host", "person", 0)
    _add(guests[:20], "guest", "person", nxt)
