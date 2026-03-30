"""Shared helpers for LLM-based KG graph extraction (topics + entities JSON)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Whisper / subword summarizer artifacts at bullet starts (non-exhaustive; conservative).
_KNOWN_ML_BULLET_PREFIX_RES = (re.compile(r"^(?:Unden|Exting|Distingu)-\s*", re.IGNORECASE),)


def strip_known_ml_bullet_prefixes(text: str) -> str:
    """Remove common broken subword prefixes from summary bullets (ML / ASR noise)."""
    s = (text or "").strip()
    for pat in _KNOWN_ML_BULLET_PREFIX_RES:
        s = pat.sub("", s)
    return s.strip()


def build_kg_transcript_system_prompt(max_topics: int, max_entities: int) -> str:
    """System message for transcript-based KG extraction with explicit array caps."""
    mt = max(1, min(int(max_topics), 20))
    me = max(1, min(int(max_entities), 50))
    return (
        "You extract a small knowledge-graph fragment from a podcast transcript. "
        "Return ONLY valid JSON (no markdown fences, no commentary) with this shape:\n"
        '{"topics":[{"label":"short topic phrase"}],"entities":[{"name":"Full Name",'
        '"entity_kind":"person"}]}\n'
        'entity_kind must be "person" or "organization" only. '
        "Use concise topic labels (under 200 characters each). "
        f"Hard limits: at most {mt} objects in topics and at most {me} in entities — "
        "never exceed these array lengths. Order by importance (strongest first); "
        "extras are truncated if you exceed the limit."
    )


def build_kg_from_bullets_system_prompt(max_topics: int, max_entities: int) -> str:
    """System message for summary-bullet-derived KG with explicit array caps."""
    mt = max(1, min(int(max_topics), 20))
    me = max(1, min(int(max_entities), 50))
    return (
        "You derive a small knowledge-graph fragment from episode summary bullets only "
        "(there is no full transcript). "
        "Return ONLY valid JSON (no markdown fences, no commentary) with this shape:\n"
        '{"topics":[{"label":"short topic phrase"}],"entities":[{"name":"Full Name",'
        '"entity_kind":"person"}]}\n'
        'entity_kind must be "person" or "organization" only. '
        "Topic labels must be short thematic phrases "
        "(not full bullet sentences pasted as one label). "
        f"Hard limits: at most {mt} topic objects and at most {me} entity objects — "
        "never exceed these array lengths. Order topics by importance (strongest first). "
        "Prefer fewer, broader themes that still cover the bullets over many "
        "overlapping micro-topics."
    )


def truncate_transcript_for_kg(text: str, limit: int = 120000) -> str:
    """Trim transcript for LLM context windows."""
    text_slice = (text or "").strip()
    if len(text_slice) > limit:
        return text_slice[:limit] + "\n\n[Transcript truncated.]"
    return text_slice


def build_kg_user_prompt(
    transcript: str,
    title: str,
    max_topics: int,
    max_entities: int,
) -> str:
    """Render shared Jinja prompt for KG extraction."""
    from ..prompts.store import render_prompt

    return render_prompt(
        "shared/kg_graph_extraction/v1",
        transcript=transcript,
        title=title or "",
        max_topics=max_topics,
        max_entities=max_entities,
    )


def normalize_bullet_labels_for_kg(labels: List[str], max_bullets: int = 30) -> List[str]:
    """Trim empty entries and cap count for LLM prompts."""
    out: List[str] = []
    for raw in labels or []:
        s = strip_known_ml_bullet_prefixes(str(raw))
        if s:
            out.append(s[:2000])
        if len(out) >= max_bullets:
            break
    return out


def build_kg_from_bullets_user_prompt(
    bullet_labels: List[str],
    title: str,
    max_topics: int,
    max_entities: int,
) -> str:
    """Render Jinja prompt for KG extraction from summary bullets only."""
    from ..prompts.store import render_prompt

    bullets = normalize_bullet_labels_for_kg(bullet_labels)
    return render_prompt(
        "shared/kg_graph_extraction/from_summary_bullets_v1",
        bullets=bullets,
        title=title or "",
        max_topics=max_topics,
        max_entities=max_entities,
    )


def _strip_json_fence(raw: str) -> str:
    content = (raw or "").strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return content


def _normalize_entity_kind(kind: Optional[str]) -> str:
    k = (kind or "person").strip().lower()
    if k in ("organization", "org", "company", "corporation", "institution"):
        return "organization"
    return "person"


def parse_kg_graph_response(
    raw: str,
    *,
    max_topics: Optional[int] = None,
    max_entities: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Parse model output into {\"topics\": [...], \"entities\": [...]} or None.

    When ``max_topics`` / ``max_entities`` are set, lists are truncated after parsing
    (defense in depth alongside pipeline caps).
    """
    content = _strip_json_fence(raw)
    if not content:
        return None
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}\s*$", content)
        if not m:
            logger.debug("KG JSON parse failed: not valid JSON")
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            logger.debug("KG JSON parse failed after brace extract")
            return None

    if not isinstance(obj, dict):
        return None

    raw_topics = obj.get("topics")
    raw_entities = obj.get("entities")
    topics_out: List[Dict[str, str]] = []
    if isinstance(raw_topics, list):
        for item in raw_topics:
            if isinstance(item, str) and item.strip():
                topics_out.append({"label": item.strip()[:500]})
            elif isinstance(item, dict):
                lab = item.get("label") or item.get("name") or item.get("topic")
                if isinstance(lab, str) and lab.strip():
                    topics_out.append({"label": lab.strip()[:500]})

    entities_out: List[Dict[str, str]] = []
    if isinstance(raw_entities, list):
        for item in raw_entities:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("label")
            if not isinstance(name, str) or not name.strip():
                continue
            ek_raw = item.get("entity_kind")
            ek_in = ek_raw if isinstance(ek_raw, str) else None
            entities_out.append(
                {
                    "name": name.strip()[:500],
                    "entity_kind": _normalize_entity_kind(ek_in),
                }
            )

    if max_topics is not None and max_topics >= 1:
        topics_out = topics_out[: int(max_topics)]
    if max_entities is not None and max_entities >= 1:
        entities_out = entities_out[: int(max_entities)]

    if not topics_out and not entities_out:
        return None
    return {"topics": topics_out, "entities": entities_out}


def resolve_kg_model_id(provider: Any, params: Optional[Dict[str, Any]]) -> str:
    """Pick model id for KG call: params[\"kg_extraction_model\"] or provider.summary_model."""
    if params and params.get("kg_extraction_model"):
        return str(params["kg_extraction_model"])
    return str(getattr(provider, "summary_model", "") or "unknown")
