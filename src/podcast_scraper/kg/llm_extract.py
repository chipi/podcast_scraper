"""Shared helpers for LLM-based KG graph extraction (topics + entities JSON)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

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
    """System message for transcript-based KG extraction with explicit array caps.

    #1033 — the entity/topic boundary is enforced via worked examples + a
    common-mistakes warning. Empirically (see #112 / EVAL_112) models given
    a "person | organization" rule alone happily emit conceptual nouns
    ("error budgets", "RFCs", "security practices") as orgs. The
    "ONLY proper-noun named entities" rule + the contrast table fix this.
    """
    mt = max(1, min(int(max_topics), 20))
    me = max(1, min(int(max_entities), 50))
    return (
        "You extract a small knowledge-graph fragment from a podcast transcript. "
        "Return ONLY valid JSON (no markdown fences, no commentary) with this shape:\n"
        '{"topics":[{"label":"short topic phrase","description":"optional 1-3 sentences '
        'why this topic matters in this episode"}],"entities":[{"name":"Full Name",'
        '"entity_kind":"person","description":"optional 1-3 sentences on their role '
        'or relevance here"}]}\n'
        "Omit description keys when not useful.\n\n"
        "ENTITY vs TOPIC — CRITICAL DISTINCTION:\n"
        'entity_kind must be "person" or "organization" only. An ENTITY is a '
        "specific, named, proper-noun referent — a real-world individual, "
        "company, brand, podcast, product, or institution that has a name. "
        "If you cannot capitalize it as a proper noun and point to one specific "
        "real-world thing, it is NOT an entity — put it in topics instead.\n\n"
        "Correct entity examples (extract these into entities):\n"
        '  - {"name":"Maya","entity_kind":"person"}\n'
        '  - {"name":"Cascadia Alliance","entity_kind":"organization"}\n'
        '  - {"name":"Strava","entity_kind":"organization"}\n'
        '  - {"name":"Singletrack Sessions","entity_kind":"organization"}\n\n'
        "Common mistakes — these are TOPICS not ENTITIES (do NOT put them in entities):\n"
        '  - "error budgets" → topic (concept), not entity\n'
        '  - "security practices" → topic (concept), not entity\n'
        '  - "RFCs" → topic (concept), not entity\n'
        '  - "trail building" → topic (concept), not entity\n'
        '  - "the host" → not an entity at all (no name)\n'
        '  - "a trail builder" → not an entity at all (no name)\n\n'
        "Quantity guidance: most episodes have 2-8 named entities. If you find "
        "yourself emitting 15+ entities, you are almost certainly labeling topics "
        "or concepts as entities — re-check the distinction above. It is BETTER "
        "to emit 0 entities than to fill the array with conceptual nouns.\n\n"
        "Use one canonical spelling per entity: if a name is spelled inconsistently "
        "in the transcript, pick the single most likely correct spelling and emit it "
        "once — do not also emit the variant as a separate entity. Keep genuinely "
        "different entities (e.g. UPS vs USPS) separate. "
        "Each topic label should be a compact heading: prefer about 2–8 words, "
        "noun-phrase style (hard cap 200 characters). Avoid long sentences, "
        'comma stacks, leading clauses ("How …", "Why …"), or pasting raw '
        "transcript lines as the label — put nuance in description instead. "
        "Short stable headings align better across episodes in later tooling. "
        f"Hard limits: at most {mt} objects in topics and at most {me} in entities — "
        "never exceed these array lengths. Order by importance (strongest first); "
        "extras are truncated if you exceed the limit."
    )


_MAX_TOPIC_LABEL_CHARS = 50


def _enforce_noun_phrase_label(label: str) -> Tuple[str, Optional[str]]:
    """Enforce noun-phrase length on a topic label.

    If *label* exceeds ``_MAX_TOPIC_LABEL_CHARS``, split at a word boundary:
    the short part becomes the label, the overflow is returned separately
    (caller should append it to ``description``).

    Returns ``(truncated_label, overflow_or_None)``.
    """
    if len(label) <= _MAX_TOPIC_LABEL_CHARS:
        return label, None
    # Cut at word boundary
    cut = label[:_MAX_TOPIC_LABEL_CHARS].rsplit(" ", 1)[0]
    if not cut:
        cut = label[:_MAX_TOPIC_LABEL_CHARS]
    overflow = label[len(cut) :].strip()
    return cut.rstrip(), overflow if overflow else None


def _truncate_kg_description(text: Optional[str], limit: int = 2000) -> Optional[str]:
    s = (text or "").strip()
    if not s:
        return None
    return s[:limit] if len(s) > limit else s


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
    prompt_version: str = "v4",  # v4: entity canonicalization (#851); v3 noun-phrase (#590)
) -> str:
    """Render shared Jinja prompt for KG extraction."""
    from ..prompts.store import render_prompt

    return render_prompt(
        f"shared/kg_graph_extraction/{prompt_version}",
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


def _parse_topic_items(raw_topics: Any) -> List[Dict[str, str]]:
    """Parse raw topic items with noun-phrase label enforcement."""
    out: List[Dict[str, str]] = []
    if not isinstance(raw_topics, list):
        return out
    for item in raw_topics:
        if isinstance(item, str) and item.strip():
            label, overflow = _enforce_noun_phrase_label(item.strip())
            row: Dict[str, str] = {"label": label}
            if overflow:
                row["description"] = overflow
            out.append(row)
        elif isinstance(item, dict):
            lab = item.get("label") or item.get("name") or item.get("topic")
            if not isinstance(lab, str) or not lab.strip():
                continue
            label, overflow = _enforce_noun_phrase_label(lab.strip())
            row = {"label": label}
            desc_parts: List[str] = []
            if overflow:
                desc_parts.append(overflow)
            desc = item.get("description")
            if isinstance(desc, str):
                td = _truncate_kg_description(desc)
                if td:
                    desc_parts.append(td)
            if desc_parts:
                row["description"] = ". ".join(desc_parts)
            out.append(row)
    return out


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
    topics_out = _parse_topic_items(raw_topics)

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
            erow: Dict[str, str] = {
                "name": name.strip()[:500],
                "entity_kind": _normalize_entity_kind(ek_in),
            }
            edesc = item.get("description")
            if isinstance(edesc, str):
                ed = _truncate_kg_description(edesc)
                if ed:
                    erow["description"] = ed
            entities_out.append(erow)

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
