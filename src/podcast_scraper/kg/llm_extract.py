"""Shared helpers for LLM-based KG graph extraction (topics + entities JSON)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

KG_GRAPH_JSON_SYSTEM = (
    "You extract a small knowledge-graph fragment from a podcast transcript. "
    "Return ONLY valid JSON (no markdown fences, no commentary) with this shape:\n"
    '{"topics":[{"label":"short topic phrase"}],"entities":[{"name":"Full Name",'
    '"entity_kind":"person"}]}\n'
    'entity_kind must be "person" or "organization" only. '
    "Use concise topic labels (under 200 characters each)."
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


def parse_kg_graph_response(raw: str) -> Optional[Dict[str, Any]]:
    """Parse model output into {\"topics\": [...], \"entities\": [...]} or None."""
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

    if not topics_out and not entities_out:
        return None
    return {"topics": topics_out, "entities": entities_out}


def resolve_kg_model_id(provider: Any, params: Optional[Dict[str, Any]]) -> str:
    """Pick model id for KG call: params[\"kg_extraction_model\"] or provider.summary_model."""
    if params and params.get("kg_extraction_model"):
        return str(params["kg_extraction_model"])
    return str(getattr(provider, "summary_model", "") or "unknown")
