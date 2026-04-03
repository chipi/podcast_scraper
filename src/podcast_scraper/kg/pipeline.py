"""KG extraction pipeline: transcript + episode context -> kg.json-shaped dict."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, cast, Dict, List, Optional, Set

from .. import config_constants
from ..graph_id_utils import (
    entity_node_id,
    episode_node_id,
    slugify_label,
    topic_node_id_from_slug,
)
from .llm_extract import _normalize_entity_kind

logger = logging.getLogger(__name__)


def _safe_iso_date(s: Optional[str]) -> str:
    """Return ISO date-time string; use placeholder if missing."""
    if s and s.strip():
        return s.strip()
    return "1970-01-01T00:00:00Z"


def _resolve_source(cfg: Optional[Any]) -> str:
    if cfg is not None:
        return str(getattr(cfg, "kg_extraction_source", "summary_bullets") or "summary_bullets")
    return "summary_bullets"


def _max_topics(cfg: Optional[Any]) -> int:
    if cfg is not None:
        v = int(
            getattr(cfg, "kg_max_topics", config_constants.DEFAULT_SUMMARY_BULLETS_DOWNSTREAM_MAX)
            or config_constants.DEFAULT_SUMMARY_BULLETS_DOWNSTREAM_MAX
        )
        return max(1, min(v, 20))
    return config_constants.DEFAULT_SUMMARY_BULLETS_DOWNSTREAM_MAX


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
            pipeline_metrics=pipeline_metrics,
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


def _try_bullet_derived_extraction(
    bullet_labels: List[str],
    episode_title: str,
    cfg: Optional[Any],
    kg_extraction_provider: Any,
    pipeline_metrics: Optional[Any],
) -> Optional[Dict[str, Any]]:
    """Call ``extract_kg_from_summary_bullets`` when the provider implements it."""
    if not bullet_labels or kg_extraction_provider is None:
        return None
    extract_fn = getattr(kg_extraction_provider, "extract_kg_from_summary_bullets", None)
    if not callable(extract_fn):
        return None
    max_t = _max_topics(cfg)
    max_e = _max_entities(cfg)
    params: Dict[str, Any] = {}
    if cfg is not None and getattr(cfg, "kg_extraction_model", None):
        params["kg_extraction_model"] = cfg.kg_extraction_model
    try:
        partial = extract_fn(
            bullet_labels,
            episode_title=episode_title,
            max_topics=max_t,
            max_entities=max_e,
            params=params or None,
            pipeline_metrics=pipeline_metrics,
        )
    except Exception as exc:
        logger.debug("KG provider extract_kg_from_summary_bullets failed: %s", exc, exc_info=True)
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
    available. When ``summary_bullets`` and ``kg_extraction_provider`` implements
    ``extract_kg_from_summary_bullets``, topics/entities are derived from bullets via LLM;
    otherwise topic nodes use bullet text as labels (legacy). Stub skips topic hints.

    Args:
        episode_id: Stable episode key (RSS GUID family). Do not prefix with ``episode:``;
            the Episode node id becomes ``episode:{episode_id}``.
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
        kg_extraction_provider: Summarization provider for ``provider`` source or optional
            bullet-derived KG when ``summary_bullets`` (if caller passes an API provider).
        pipeline_metrics: Optional metrics collector (increments kg_provider_extractions).

    Returns:
        Dict matching docs/kg/kg.schema.json (minimal validation via kg.schema).
    """
    date_str = _safe_iso_date(publish_date)
    ep_node_id = episode_node_id(episode_id)
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
    llm_from_summary_bullets = False
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
    elif source == "summary_bullets" and bullet_labels:
        llm_partial = _try_bullet_derived_extraction(
            bullet_labels,
            episode_title or "",
            cfg,
            kg_extraction_provider,
            pipeline_metrics,
        )
        used_provider = llm_partial is not None
        llm_from_summary_bullets = llm_partial is not None

    if llm_partial:
        _append_topics_and_entities_from_partial(
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
            mid_s = mid or "unknown"
            if llm_from_summary_bullets:
                resolved_model = f"provider:summary_bullets:{mid_s}"
            else:
                resolved_model = f"provider:{mid_s}"
    elif source != "stub" and bullet_labels:
        _append_topics_from_labels(ep_node_id, bullet_labels, nodes, edges)
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
            ep_node_id,
            detected_hosts,
            detected_guests,
            nodes,
            edges,
            existing_entity_keys=_entity_identity_keys(nodes),
        )

    extracted_at = (
        datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )

    return {
        "schema_version": "1.1",
        "episode_id": episode_id,
        "extraction": {
            "model_version": resolved_model,
            "extracted_at": extracted_at,
            "transcript_ref": transcript_ref,
        },
        "nodes": nodes,
        "edges": edges,
    }


def _entity_dedup_key(*, name: str, entity_kind: Optional[str]) -> str:
    """Stable (entity_kind, name) key for intra-artifact deduplication."""
    ek = _normalize_entity_kind(entity_kind)
    return f"{ek}:{name.strip().lower()}"


def _entity_identity_keys(nodes: List[Dict[str, Any]]) -> Set[str]:
    """Keys for existing Entity nodes: kind + lowercased name (matches node id semantics)."""
    out: Set[str] = set()
    for n in nodes:
        if n.get("type") != "Entity":
            continue
        props = n.get("properties") or {}
        name = props.get("name")
        if isinstance(name, str) and name.strip():
            out.add(_entity_dedup_key(name=name, entity_kind=props.get("entity_kind")))
    return out


def _entity_properties(
    *,
    name: str,
    entity_kind: str,
    role: str,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """Entity node properties: name, label (for graphs / Topic parity), kind, role."""
    name_s = (name or "").strip()[:500]
    props: Dict[str, Any] = {
        "name": name_s,
        "label": name_s[:200],
        "entity_kind": entity_kind,
        "role": role,
    }
    if description and str(description).strip():
        props["description"] = str(description).strip()[:2000]
    return props


def _append_topics_from_labels(
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
        slug = slugify_label(lab)
        if slug in seen_slugs:
            continue
        seen_slugs.add(slug)
        topic_id = topic_node_id_from_slug(slug)
        nodes.append(
            {
                "id": topic_id,
                "type": "Topic",
                "properties": {"label": lab[:200], "slug": slug},
            }
        )
        edges.append({"from": topic_id, "to": ep_node_id, "type": "MENTIONS", "properties": {}})


def _append_topics_and_entities_from_partial(
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
            slug = slugify_label(lab_s)
            if slug in seen_slugs:
                continue
            seen_slugs.add(slug)
            topic_id = topic_node_id_from_slug(slug)
            tprops: Dict[str, Any] = {"label": lab_s[:200], "slug": slug}
            raw_desc = item.get("description") if isinstance(item, dict) else None
            if isinstance(raw_desc, str) and raw_desc.strip():
                tprops["description"] = raw_desc.strip()[:2000]
            nodes.append(
                {
                    "id": topic_id,
                    "type": "Topic",
                    "properties": tprops,
                }
            )
            edges.append({"from": topic_id, "to": ep_node_id, "type": "MENTIONS", "properties": {}})
            t_count += 1

    e_count = 0
    seen_entity_keys: Set[str] = set()
    if isinstance(entities, list):
        for item in entities:
            if e_count >= max_entities:
                break
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            name_s = name.strip()[:500]
            ek = _normalize_entity_kind(item.get("entity_kind"))
            key = _entity_dedup_key(name=name_s, entity_kind=ek)
            if key in seen_entity_keys:
                continue
            seen_entity_keys.add(key)
            eid = entity_node_id(ek, name_s)
            ent_desc = item.get("description") if isinstance(item.get("description"), str) else None
            nodes.append(
                {
                    "id": eid,
                    "type": "Entity",
                    "properties": _entity_properties(
                        name=name_s,
                        entity_kind=ek,
                        role="mentioned",
                        description=ent_desc,
                    ),
                }
            )
            edges.append({"from": eid, "to": ep_node_id, "type": "MENTIONS", "properties": {}})
            e_count += 1


def _append_pipeline_entities(
    ep_node_id: str,
    detected_hosts: Optional[List[str]],
    detected_guests: Optional[List[str]],
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    existing_entity_keys: Set[str],
) -> None:
    def _add(names: List[str], role: str, kind: str) -> None:
        for name in names:
            n = (name or "").strip()
            if not n:
                continue
            key = _entity_dedup_key(name=n, entity_kind=kind)
            if key in existing_entity_keys:
                continue
            existing_entity_keys.add(key)
            eid = entity_node_id(kind, n)
            nodes.append(
                {
                    "id": eid,
                    "type": "Entity",
                    "properties": _entity_properties(name=n[:500], entity_kind=kind, role=role),
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

    hosts = [h for h in (detected_hosts or []) if isinstance(h, str)]
    guests = [g for g in (detected_guests or []) if isinstance(g, str)]
    _add(hosts[:20], "host", "person")
    _add(guests[:20], "guest", "person")
