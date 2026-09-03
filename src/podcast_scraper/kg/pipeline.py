"""KG extraction pipeline: transcript + episode context -> kg.json-shaped dict."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, cast, Dict, List, Optional, Set

from .. import config_constants
from ..graph_id_utils import (
    entity_node_id,
    episode_node_id,
    is_person_or_org_node,
    normalized_entity_kind_from_node,
    slugify_label,
    topic_node_id_from_slug,
)
from .llm_extract import _enforce_noun_phrase_label, _normalize_entity_kind

logger = logging.getLogger(__name__)


def _safe_iso_date(s: Optional[str]) -> str:
    """Return ISO date-time string; use placeholder if missing."""
    if s and s.strip():
        return s.strip()
    return "1970-01-01T00:00:00Z"


def _resolve_source(cfg: Optional[Any]) -> str:
    if cfg is not None:
        return str(getattr(cfg, "kg_extraction_source", "provider") or "provider")
    return "provider"


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


def _apply_kg_filters(
    llm_partial: Dict[str, Any], pipeline_metrics: Optional[Any]
) -> Dict[str, Any]:
    """#652 Part B — run topic normalizer + entity-kind repair on a KG partial.

    Pure rewrite: returns a new dict when anything changed, else the input
    untouched. Wires into pipeline_metrics counters when available. Kept out
    of ``build_artifact`` to keep its cyclomatic complexity within budget.
    """
    from .filters import consolidate_entity_names, normalize_topic_labels, repair_entity_kind

    raw_topic_dicts = [t for t in (llm_partial.get("topics") or []) if isinstance(t, dict)]
    raw_topic_labels = [str(t.get("label") or "").strip() for t in raw_topic_dicts]
    norm_labels, topics_changed = normalize_topic_labels(raw_topic_labels)
    if topics_changed:
        # Preserve per-topic fields (e.g. description) when the normalized
        # label maps back to a source row. First-occurrence wins on ties.
        by_norm: Dict[str, Dict[str, Any]] = {}
        for raw_label, src in zip(raw_topic_labels, raw_topic_dicts):
            norm_key = normalize_topic_labels([raw_label])[0]
            if not norm_key:
                continue
            if norm_key[0] not in by_norm:
                by_norm[norm_key[0]] = src
        new_topics: List[Dict[str, Any]] = []
        for lab in norm_labels:
            src_fields = by_norm.get(lab, {})
            merged = {k: v for k, v in src_fields.items() if k != "label"}
            merged["label"] = lab
            new_topics.append(merged)
        llm_partial = dict(llm_partial)
        llm_partial["topics"] = new_topics
        if pipeline_metrics is not None and hasattr(pipeline_metrics, "record_topics_normalized"):
            pipeline_metrics.record_topics_normalized(topics_changed)

    orig_entities = [e for e in (llm_partial.get("entities") or []) if isinstance(e, dict)]
    entities_for_repair = [
        {"name": e.get("name"), "kind": e.get("entity_kind")} for e in orig_entities
    ]
    repaired_entities, ents_repaired = repair_entity_kind(entities_for_repair)
    if ents_repaired:
        llm_partial = dict(llm_partial)
        # Merge the repaired kind back into the FULL original dict (parallel order)
        # so ``description`` and any other fields survive — previously the whole
        # entity list was replaced with {name, entity_kind}, dropping descriptions
        # for the entire batch whenever any kind was repaired (review H6).
        llm_partial["entities"] = [
            {**orig, "entity_kind": r["kind"]} for orig, r in zip(orig_entities, repaired_entities)
        ]
        if pipeline_metrics is not None and hasattr(
            pipeline_metrics, "record_entity_kinds_repaired"
        ):
            pipeline_metrics.record_entity_kinds_repaired(ents_repaired)

    # #851 — consolidate within-episode duplicate-spelling entities (runs after
    # kind repair so the merge is kind-aware). Conservative + type-aware; the
    # extraction prompt owns the correct surviving spelling.
    ent_dicts = [e for e in (llm_partial.get("entities") or []) if isinstance(e, dict)]
    consolidated, ents_merged = consolidate_entity_names(ent_dicts)
    if ents_merged:
        llm_partial = dict(llm_partial)
        llm_partial["entities"] = consolidated
        if pipeline_metrics is not None and hasattr(
            pipeline_metrics, "record_entities_consolidated"
        ):
            pipeline_metrics.record_entities_consolidated(ents_merged)

    return llm_partial


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


def _resolve_ner_prepass(
    transcript_text: str,
    cfg: Optional[Any],
    kg_extraction_provider: Any,
    max_entities: int,
) -> tuple[List[Dict[str, str]], str]:
    """#1035 — NER pre-pass: returns (hints, prompt_version).

    When ``cfg.kg_extraction_use_ner_prepass`` is False (the Field default is
    ``True``; see config.py) or the NER pipeline isn't available, returns
    ``([], "v4")`` — caller renders the v4 prompt with no hints. When enabled
    and the spaCy model resolves, returns the deduped + capped PERSON+ORG
    candidate list and prompt ``"v5"``.

    NB (#1540): the shipped cloud image is ``[llm,search]`` — spaCy lives only
    in ``[ml]`` — so on cloud_balanced this pass load-fails and silently
    downgrades to v4. Whether that materially hurts cloud KG entity recall
    (the #1035 0%→100% win was measured on DGX models, not the litellm cloud
    model) is an open question tracked in the ML-opt-in discussion issue.

    Reuses ``kg_extraction_provider._spacy_nlp`` when cached (Issue #387)
    or falls back to a fresh ``get_ner_model(cfg)`` load. Any failure
    silently downgrades to v4 — entity recall stays at the pre-#1035
    baseline, never worse.
    """
    if cfg is None or not getattr(cfg, "kg_extraction_use_ner_prepass", False):
        return [], "v4"
    nlp = getattr(kg_extraction_provider, "_spacy_nlp", None)
    if nlp is None:
        try:
            from ..providers.ml.speaker_detection import get_ner_model

            nlp = get_ner_model(cfg)
        except Exception as exc:
            logger.warning(
                "#1035 NER pre-pass: could not load spaCy model (%s); falling back to v4 prompt",
                exc,
            )
            return [], "v4"
    if nlp is None:
        logger.debug("#1035 NER pre-pass: cfg.ner_model not set; falling back to v4 prompt")
        return [], "v4"

    from .ner_prepass import extract_kg_ner_hints

    cap = max(1, min(int(max_entities) * 3, 40))
    hints = extract_kg_ner_hints(transcript_text, nlp, max_candidates=cap)
    return hints, "v5"


def _usable_partial(partial: Optional[Dict[str, Any]]) -> bool:
    """A KG partial is usable only if it actually carries topics or entities."""
    if not partial or not isinstance(partial, dict):
        return False
    return bool(partial.get("topics") or partial.get("entities"))


def _retry_extraction_on_fallback_chain(
    provider: Any,
    transcript_text: str,
    episode_title: str,
    max_topics: int,
    max_entities: int,
    params: Dict[str, Any],
    pipeline_metrics: Optional[Any],
) -> Optional[Dict[str, Any]]:
    """Hand the extraction to the fallback chain when the primary returned nothing.

    No-ops (returns ``None``) when the provider is not fallback-aware, so a plain provider behaves
    exactly as before. Never raises: an exhausted chain is the same outcome as no chain, and the
    caller already handles "no partial" honestly.
    """
    call_via_fallback = getattr(provider, "call_via_fallback", None)
    if not callable(call_via_fallback):
        return None
    logger.warning(
        "kg: primary extraction returned no topics/entities — escalating to the fallback chain "
        "(a dead endpoint is indistinguishable from an empty result on this provider)"
    )
    try:
        result = call_via_fallback(
            "extract_kg_graph",
            transcript_text,
            episode_title=episode_title,
            max_topics=max_topics,
            max_entities=max_entities,
            params=params or None,
            pipeline_metrics=pipeline_metrics,
        )
    except Exception as exc:  # noqa: BLE001 — an exhausted chain is a normal outcome here
        logger.warning("kg: fallback chain could not extract either: %s", exc)
        return None
    return result if isinstance(result, dict) else None


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
    ner_hints, prompt_version = _resolve_ner_prepass(
        transcript_text, cfg, kg_extraction_provider, max_e
    )
    if ner_hints:
        params["ner_entity_hints"] = ner_hints
    if prompt_version != "v4":
        params["kg_prompt_version"] = prompt_version
    try:
        partial = extract_fn(
            transcript_text,
            episode_title=episode_title,
            max_topics=max_t,
            max_entities=max_e,
            params=params or None,
            pipeline_metrics=pipeline_metrics,
        )
        primary_raised = False
    except Exception as exc:
        logger.debug("KG provider extract_kg_graph failed: %s", exc, exc_info=True)
        partial = None
        # The chain has ALREADY been walked if this provider is fallback-aware:
        # ``_wrap_call`` catches, walks every tier, and only re-raises once they have all failed.
        # So an exception reaching here means the tiers are exhausted, and escalating would walk
        # the same dead endpoints a second time — paying the full retry/backoff window twice per
        # stage per episode (this branch's profiles allow 12 retries and a 120s cap).
        primary_raised = True
    if not _usable_partial(partial) and not primary_raised:
        # #1878 escalation: the primary returned NOTHING WITHOUT RAISING, and that is invisible to
        # the failover contract.
        #
        # ``OpenAICompatibleProvider.extract_kg_graph`` ends in ``except Exception: return None``,
        # so a dead endpoint is indistinguishable from "the model found no topics" — and
        # ``FallbackAwareSummarizationProvider._wrap_call`` only walks the chain on an EXCEPTION.
        # A returned None sails straight through it. Measured on a real run with the primary
        # vLLM unreachable: ``llm_kg_calls=0, kg_provider_extractions=0, kg_failures=0`` while a
        # healthy ollama tier sat unused and ``llm_summary_fallback_active_count=1`` proved the
        # chain was working for the stage that DOES raise. ~106s per episode was spent in retry
        # backoff and then silently discarded.
        #
        # ``call_via_fallback`` exists for exactly this class (its docstring: "failures the
        # exception contract cannot see"). Using it here rather than changing
        # ``extract_kg_graph``'s exception contract keeps every other caller's semantics intact.
        partial = _retry_extraction_on_fallback_chain(
            kg_extraction_provider,
            transcript_text,
            episode_title,
            max_t,
            max_e,
            params,
            pipeline_metrics,
        )
    if not _usable_partial(partial):
        return None
    assert partial is not None  # narrowed by _usable_partial
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
    prefilled_partial: Optional[Dict[str, Any]] = None,
    feed_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a KG artifact dict for one episode.

    ``kg_extraction_source`` (from ``cfg``): ``metadata_only`` | ``provider``.
    When ``provider`` is set, calls ``extract_kg_graph`` on the summarization
    provider on the cleaned transcript directly. ``metadata_only`` skips topic hints
    entirely. The legacy ``summary_bullets`` route was removed in #1034 per
    the #1033 audit (lossy: routed extraction through name-stripped bullets).

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
        kg_extraction_provider: Summarization provider for ``provider`` source extraction.
        pipeline_metrics: Optional metrics collector (increments kg_provider_extractions).

    Returns:
        Dict matching docs/architecture/kg/kg.schema.json (minimal validation via kg.schema).
    """
    date_str = _safe_iso_date(publish_date)
    ep_node_id = episode_node_id(episode_id)
    source = _resolve_source(cfg)

    ep_props: Dict[str, Any] = {
        "podcast_id": podcast_id or "podcast:unknown",
        "title": (episode_title or "Episode").strip() or "Episode",
        "publish_date": date_str,
    }
    # #658 — Episode nodes carry the stable feed identifier so the
    # graph Feed chip can filter by feed across the merged corpus.
    if feed_id is not None:
        ep_props["feed_id"] = str(feed_id).strip()
    nodes: List[Dict[str, Any]] = [
        {
            "id": ep_node_id,
            "type": "Episode",
            "properties": ep_props,
        }
    ]
    edges: List[Dict[str, Any]] = []
    # RFC-097 v2.0: emit Podcast node + HAS_EPISODE edge alongside Episode.
    _append_podcast_and_has_episode(ep_props["podcast_id"], ep_node_id, nodes, edges)

    bullet_labels = _topic_labels_from_args(topic_labels, topic_label, cfg)
    llm_partial: Optional[Dict[str, Any]] = None
    resolved_model = model_version

    # #643: prefilled from mega_bundled / extraction_bundled short-circuits
    # provider dispatch entirely. Normalize to the {label,...}/{name,entity_kind}
    # shape that _append_topics_and_entities_from_partial expects.
    if prefilled_partial and (prefilled_partial.get("topics") or prefilled_partial.get("entities")):
        norm_topics: List[Dict[str, Any]] = []
        for t in prefilled_partial.get("topics") or []:
            if isinstance(t, str) and t.strip():
                norm_topics.append({"label": t.strip()})
            elif isinstance(t, dict) and isinstance(t.get("label"), str) and t["label"].strip():
                norm_topics.append(dict(t))
        norm_entities: List[Dict[str, Any]] = []
        for e in prefilled_partial.get("entities") or []:
            if not isinstance(e, dict):
                continue
            name = e.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            kind = e.get("kind") or e.get("entity_kind") or "person"
            norm_entities.append({"name": name.strip(), "entity_kind": str(kind).strip().lower()})
        llm_partial = {"topics": norm_topics, "entities": norm_entities}
    elif source == "provider":
        llm_partial = _try_provider_extraction(
            transcript_text,
            episode_title or "",
            cfg,
            kg_extraction_provider,
            pipeline_metrics,
        )

    if llm_partial:
        # #652 Part B — deterministic topic + entity filters (extracted helper
        # to keep build_artifact within complexity budget).
        llm_partial = _apply_kg_filters(llm_partial, pipeline_metrics)

        _append_topics_and_entities_from_partial(
            ep_node_id,
            llm_partial,
            nodes,
            edges,
            _max_topics(cfg),
            _max_entities(cfg),
            episode_id=episode_id,
        )
        if resolved_model is None:
            mid = None
            if kg_extraction_provider is not None:
                mid = getattr(kg_extraction_provider, "summary_model", None)
            if cfg is not None and getattr(cfg, "kg_extraction_model", None):
                mid = cfg.kg_extraction_model
            mid_s = mid or "unknown"
            resolved_model = f"provider:{mid_s}"
    elif source == "provider" and kg_extraction_provider is not None:
        # Provider extraction was ATTEMPTED and produced nothing. Do NOT fall through to the
        # bullet path below.
        #
        # The ``kg_extraction_provider is not None`` half is load-bearing: "a provider was
        # configured and failed" is a different state from "no provider was ever wired up", and
        # only the first one makes bullet substitution a lie. Callers that pass a ``topic_label``
        # hint with no provider — tests, legacy callers — are untouched and still get their hint.
        #
        # That fallback exists for callers who pass a ``topic_label`` hint and no extraction
        # provider — tests and legacy callers, per its own docstring. It was also silently
        # catching production extraction FAILURES, and the result is not a degraded KG, it is a
        # fabricated one: summary bullets are sentences ("Product development in frontier AI
        # requires building for model capabilities two to three months ahead"), so every Topic
        # node became a truncated proposition unique to its episode, while Insight, Person and
        # Quote nodes were absent entirely because GI never ran. Observed on a real ingest —
        # 8 topics, 0 insights, 0 entities, and nothing anywhere saying extraction had failed.
        #
        # An empty topic set is the honest outcome: the artifact then reports its provenance as
        # a failed extraction, and every downstream consumer (clustering, co-occurrence,
        # trending) correctly sees nothing rather than being poisoned with sentences.
        logger.warning(
            "kg: provider extraction produced no topics/entities for episode_id=%s; NOT "
            "substituting %d summary bullet(s) as topics (they are sentences, not subjects). "
            "The KG will have no Topic nodes — check the extraction provider.",
            episode_id or "?",
            len(bullet_labels or []),
        )
        if resolved_model is None:
            resolved_model = "provider:extraction_failed"
    elif source != "metadata_only" and bullet_labels:
        # No extraction provider configured — the caller supplied topic labels as a hint. This is
        # the tests / legacy path the comment above refers to, and it is legitimate here because
        # nothing was attempted and failed.
        _append_topics_from_labels(ep_node_id, bullet_labels, nodes, edges)
        if resolved_model is None:
            resolved_model = "topic_labels" if cfg is not None else "metadata_only"

    # "metadata_only" is a real provenance value: this KG came from episode metadata and the
    # pipeline's own hosts/guests, with no LLM involved. It was labelled "stub" until #1657,
    # which made a legitimate reduced mode read like the fabricated GI placeholder.
    if resolved_model is None:
        resolved_model = "metadata_only"

    merge_pipe = _merge_pipeline_default(cfg)
    # Respect kg_merge_pipeline_entities unconditionally. The old guard
    # (`if llm_partial and not merge_pipe: pass`) still injected pipeline
    # hosts/guests whenever the LLM call FAILED (llm_partial falsy), silently
    # ignoring the flag (review 2026-07-17 M15).
    if merge_pipe:
        _append_pipeline_entities(
            ep_node_id,
            detected_hosts,
            detected_guests,
            nodes,
            edges,
            existing_entity_keys=_entity_identity_keys(nodes),
            episode_id=episode_id,
            podcast_id=ep_props["podcast_id"],
        )

    extracted_at = (
        datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )

    # Final id-level node dedup, across EVERY append site. The per-site checks key on the raw
    # NAME ("david clark" != "david clark)") while the node id keys on the SLUG (parens
    # stripped) — so near-identical names collide on id, and hosts vs mentioned-entities append
    # from different sites with no cross-site check at all. Two nodes sharing an id crashed the
    # whole corpus reindex on 2026-08-26 (LanceDB refuses ambiguous merges). Last-writer's
    # properties fill gaps; an explicit host/guest role always beats 'mentioned'.
    nodes = _dedupe_nodes_by_id(nodes)

    return {
        # RFC-097 v2.0: typed Person / Organization / Podcast nodes + HAS_EPISODE.
        "schema_version": "2.0",
        "episode_id": episode_id,
        "extraction": {
            "model_version": resolved_model,
            "extracted_at": extracted_at,
            "transcript_ref": transcript_ref,
        },
        "nodes": nodes,
        "edges": edges,
    }


def _dedupe_nodes_by_id(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge nodes sharing an id into one (first kept, later ones fill missing properties).

    Role precedence: host/guest (an explicit stated role) beats 'mentioned' (the default the
    entity path mints). Order is otherwise preserved; edges are untouched — they reference ids,
    which all still resolve to the surviving node.
    """
    seen: Dict[str, Dict[str, Any]] = {}
    out: List[Dict[str, Any]] = []
    for node in nodes:
        nid = str(node.get("id") or "")
        if nid and nid in seen:
            kept = seen[nid].setdefault("properties", {})
            extra = node.get("properties") or {}
            if extra.get("role") in ("host", "guest") and kept.get("role") == "mentioned":
                kept["role"] = extra["role"]
            for k, v in extra.items():
                kept.setdefault(k, v)
            logger.warning(
                "kg build_artifact: merged duplicate node id %r (near-identical labels from "
                "different append sites)",
                nid,
            )
            continue
        if nid:
            seen[nid] = node
        out.append(node)
    return out


def _entity_dedup_key(*, name: str, entity_kind: Optional[str]) -> str:
    """Stable (entity_kind, name) key for intra-artifact deduplication."""
    ek = _normalize_entity_kind(entity_kind)
    return f"{ek}:{name.strip().lower()}"


def _normalized_kind_from_props(props: Dict[str, Any]) -> str:
    """Map stored ``kind`` or legacy ``entity_kind`` to person/organization (legacy Entity)."""
    raw = props.get("kind")
    if raw == "org":
        return "organization"
    if raw == "person":
        return "person"
    return _normalize_entity_kind(props.get("entity_kind"))


def _entity_identity_keys(nodes: List[Dict[str, Any]]) -> Set[str]:
    """Keys for existing person/org nodes: kind + lowercased name.

    Scans both v2.0 (Person/Organization) and legacy (Entity) nodes so
    pipeline-side merges deduplicate across both shapes during the
    permissive bake window (RFC-097 chunk 3).
    """
    out: Set[str] = set()
    for n in nodes:
        nt = n.get("type")
        if not is_person_or_org_node(nt):
            continue
        props = n.get("properties") or {}
        name = props.get("name")
        if isinstance(name, str) and name.strip():
            out.add(
                _entity_dedup_key(
                    name=name,
                    entity_kind=normalized_entity_kind_from_node(n),
                )
            )
    return out


def _typed_person_org_node(
    *,
    name: str,
    entity_kind: str,
    role: str,
    description: Optional[str] = None,
    episode_id: Optional[str] = None,
) -> Dict[str, Any]:
    """RFC-097 v2.0 emission: typed ``Person`` / ``Organization`` node.

    Replaces legacy ``Entity(kind=...)``: node type encodes the discriminator,
    so ``kind`` is no longer a property. Id format ``person:{slug}`` /
    ``org:{slug}`` stays identical (preserves cross-layer canonical ids) — except a bare
    diarization label (``SPEAKER_03``) is episode-scoped when ``episode_id`` is given, so an
    unnamed voice never merges across episodes into a phantom person (#1b).
    """
    name_s = (name or "").strip()[:500]
    ek = _normalize_entity_kind(entity_kind)
    node_type = "Organization" if ek == "organization" else "Person"
    props: Dict[str, Any] = {
        "name": name_s,
        "label": name_s[:200],
        "role": role,
    }
    if description and str(description).strip():
        props["description"] = str(description).strip()[:2000]
    return {
        "id": entity_node_id(ek, name_s, episode_id),
        "type": node_type,
        "properties": props,
    }


def _podcast_node_id(raw_podcast_id: str) -> Optional[str]:
    """Normalised ``podcast:{slug}`` node id for a raw feed/podcast id, or ``None`` for the
    ``podcast:unknown`` placeholder / empty input. Single source of truth so the Podcast node
    and the per-show host/guest edges (#1169 Path A) always reference the same id."""
    raw = (raw_podcast_id or "").strip()
    if not raw or raw == "podcast:unknown":
        return None
    pid = raw if raw.startswith("podcast:") else f"podcast:{slugify_label(raw)}"
    if pid in ("podcast:", "podcast:unknown"):
        return None
    return pid


def _append_podcast_and_has_episode(
    raw_podcast_id: str,
    ep_node_id: str,
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
) -> None:
    """RFC-097 v2.0: emit a Podcast node + HAS_EPISODE edge.

    Skipped for the ``podcast:unknown`` placeholder (no real podcast metadata
    available, so a synthetic node would be noise). Otherwise the Podcast
    node id is normalised to ``podcast:{slug}`` and the title is derived
    from the slug as a readable default; downstream code can refine via
    feed metadata if/when needed.
    """
    pid = _podcast_node_id(raw_podcast_id)
    if pid is None:
        return
    slug = pid[len("podcast:") :]
    title = slug.replace("-", " ").replace("_", " ").strip().title() or pid
    nodes.append(
        {
            "id": pid,
            "type": "Podcast",
            "properties": {"title": title[:500]},
        }
    )
    edges.append(
        {
            "type": "HAS_EPISODE",
            "from": pid,
            "to": ep_node_id,
            "properties": {},
        }
    )


def _append_topics_from_labels(
    ep_node_id: str,
    labels: List[str],
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
) -> None:
    seen_slugs: Set[str] = set()
    dropped_propositions: List[str] = []
    for raw in labels:
        if not raw.strip():
            continue
        lab = raw.strip()[:500]
        slug = slugify_label(lab)
        if slug in seen_slugs:
            continue
        seen_slugs.add(slug)
        topic_id = topic_node_id_from_slug(slug)
        # #587: enforce the noun-phrase cap here too. This site used to slice at [:200], which
        # bypassed the enforcement in ``llm_extract`` and let sentence-shaped labels through.
        # Measured on the 1,066-episode corpus: 679 labels (7.3%) exceeded 50 chars, and NOT ONE
        # of them ever recurred in a second episode (vs 6.9% of short labels) — a truncated
        # sentence cannot be emitted identically twice, so those topics were structurally
        # incapable of clustering or forming a theme. Overflow goes to ``description`` so nothing
        # is lost; the embedder reads label + description either way.
        # Returns None for a PROPOSITION — a sentence, not a topic. Dropped rather than
        # truncated: cutting it produces a unique mid-sentence fragment that reads like a real
        # topic and can never cluster. This is the site the live provider path uses.
        t_enforced = _enforce_noun_phrase_label(lab.strip())
        if t_enforced is None:
            dropped_propositions.append(lab.strip())
            continue
        t_label, t_overflow = t_enforced
        t_props: Dict[str, Any] = {"label": t_label[:200], "slug": slug}
        if t_overflow:
            t_props["description"] = t_overflow[:2000]
        nodes.append(
            {
                "id": topic_id,
                "type": "Topic",
                "properties": t_props,
            }
        )
        edges.append({"from": topic_id, "to": ep_node_id, "type": "MENTIONS", "properties": {}})

    if dropped_propositions:
        # Loud: a provider emitting propositions instead of noun phrases is a signal about the
        # RUN (it is what a degraded fallback tier does), and an episode whose topics all vanish
        # must be attributable to that rather than reading as an episode about nothing.
        logger.warning(
            "kg: dropped %d topic label(s) that were propositions, not noun phrases; sample=%r",
            len(dropped_propositions),
            dropped_propositions[:3],
        )


def _append_topics_and_entities_from_partial(
    ep_node_id: str,
    partial: Dict[str, Any],
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    max_topics: int,
    max_entities: int,
    episode_id: Optional[str] = None,
) -> None:
    topics = partial.get("topics") or []
    entities = partial.get("entities") or []
    seen_slugs: Set[str] = set()
    dropped_propositions: List[str] = []
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
            # #587 (see the sibling site above): cap the label at a noun phrase rather than
            # slicing at 200. Any overflow is PREPENDED to the description so the detail survives
            # — appending would bury it after a 2000-char description on the exact topics whose
            # label was too long in the first place.
            enforced = _enforce_noun_phrase_label(lab_s)
            if enforced is None:
                dropped_propositions.append(lab_s)
                continue
            lab_capped, lab_overflow = enforced
            tprops: Dict[str, Any] = {"label": lab_capped[:200], "slug": slug}
            raw_desc = item.get("description") if isinstance(item, dict) else None
            desc_parts = [x for x in (lab_overflow, (raw_desc or "").strip()) if x]
            if desc_parts:
                tprops["description"] = " ".join(desc_parts)[:2000]
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
            ent_desc = item.get("description") if isinstance(item.get("description"), str) else None
            v2_node = _typed_person_org_node(
                name=name_s,
                entity_kind=ek,
                role="mentioned",
                description=ent_desc,
                episode_id=episode_id,
            )
            nodes.append(v2_node)
            edges.append(
                {
                    "from": v2_node["id"],
                    "to": ep_node_id,
                    "type": "MENTIONS",
                    "properties": {},
                }
            )
            e_count += 1

    if dropped_propositions:
        # Same signal as the sibling site above — see its comment.
        logger.warning(
            "kg: dropped %d topic label(s) that were propositions, not noun phrases; sample=%r",
            len(dropped_propositions),
            dropped_propositions[:3],
        )


def _append_pipeline_entities(
    ep_node_id: str,
    detected_hosts: Optional[List[str]],
    detected_guests: Optional[List[str]],
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    existing_entity_keys: Set[str],
    episode_id: Optional[str] = None,
    podcast_id: Optional[str] = None,
) -> None:
    podcast_nid = _podcast_node_id(podcast_id or "")
    seen_show_edges: Set[tuple[str, str]] = set()

    def _add(names: List[str], role: str, kind: str, show_edge: Optional[str]) -> None:
        for name in names:
            n = (name or "").strip()
            if not n:
                continue
            v2_node = _typed_person_org_node(
                name=n[:500],
                entity_kind=kind,
                role=role,
                episode_id=episode_id,
            )
            # Per-SHOW host/guest fact — person HOSTS / GUESTS_ON the podcast, not just a
            # per-episode MENTIONS. Durable across episodes so a host in one sampled episode
            # is known show-wide (#1169 epic Path A). Emitted even when the person node is
            # deduped, and de-duplicated itself.
            if podcast_nid and show_edge and (v2_node["id"], show_edge) not in seen_show_edges:
                seen_show_edges.add((v2_node["id"], show_edge))
                edges.append(
                    {
                        "from": v2_node["id"],
                        "to": podcast_nid,
                        "type": show_edge,
                        "properties": {},
                    }
                )
            key = _entity_dedup_key(name=n, entity_kind=kind)
            if key in existing_entity_keys:
                continue
            existing_entity_keys.add(key)
            nodes.append(v2_node)
            edges.append(
                {
                    "from": v2_node["id"],
                    "to": ep_node_id,
                    "type": "MENTIONS",
                    "properties": {},
                }
            )

    hosts = [h for h in (detected_hosts or []) if isinstance(h, str)]
    guests = [g for g in (detected_guests or []) if isinstance(g, str)]
    _add(hosts[:20], "host", "person", "HOSTS")
    _add(guests[:20], "guest", "person", "GUESTS_ON")
