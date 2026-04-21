"""Parser for mega-bundle / extraction-bundle JSON responses (#643).

Validates the structured JSON returned by a single mega-bundle LLM call and
splits the fields into shapes the existing pipeline consumers expect:

  - summary (title, summary, bullets) -> feeds the summarization-stage output
  - insights (text, type) -> feeds the GI stage
  - topics (label list) + entities (name, kind, role) -> feeds the KG stage

Designed to be tolerant of per-provider formatting quirks (code fences,
slight schema drift on optional fields) while failing fast on hard schema
violations (missing required fields, wrong types where it matters).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MegaBundleParseError(ValueError):
    """Raised when the provider response cannot be parsed into the expected shape."""


@dataclass
class MegaBundleResult:
    """Parsed + normalized result of a mega-bundle call.

    Fields are already in the shape downstream stages expect:
      - ``summary_artifact`` mirrors what a standalone ``summarize()`` returns.
      - ``insights`` is a list of dicts suitable for GI pipeline construction.
      - ``topics`` + ``entities`` feed the KG pipeline.
    """

    title: str
    summary: str
    bullets: List[str]
    insights: List[Dict[str, Any]]
    topics: List[str]
    entities: List[Dict[str, Any]]
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_summary_artifact(self) -> Dict[str, Any]:
        """Return a dict matching the standalone summarize() return shape."""
        return {
            "title": self.title,
            "summary": self.summary,
            "bullets": list(self.bullets),
        }

    def to_extraction_partial(self) -> Dict[str, Any]:
        """Return the insights/topics/entities payload for GIL+KG reuse.

        Shape is the canonical in-pipeline contract: GIL stage reads
        ``insights`` as ``[{"text": str, "insight_type": str}]``; KG stage reads
        ``topics`` as ``list[str]`` and ``entities`` as
        ``[{"name": str, "kind": str, "role": str}]``.
        """
        return {
            "insights": [dict(i) for i in self.insights],
            "topics": list(self.topics),
            "entities": [dict(e) for e in self.entities],
        }


_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def _strip_code_fences(text: str) -> str:
    return _CODE_FENCE_RE.sub("", text.strip()).strip()


def _as_list_of_strings(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if isinstance(v, (str, int, float)) and str(v).strip()]


def _normalize_insight(obj: Any) -> Optional[Dict[str, Any]]:
    """Accept {"text": "...", "insight_type": "..."} or a bare string."""
    if isinstance(obj, str):
        t = obj.strip()
        if not t:
            return None
        return {"text": t, "insight_type": "claim"}
    if isinstance(obj, dict):
        text = str(obj.get("text") or obj.get("insight") or "").strip()
        if not text:
            return None
        itype = str(obj.get("insight_type") or obj.get("type") or "claim").strip().lower()
        if itype not in {"claim", "fact", "opinion"}:
            itype = "claim"
        out = {"text": text, "insight_type": itype}
        # Preserve optional fields like grounded/quotes if present.
        for k in ("grounded", "supporting_quotes", "quotes"):
            if k in obj:
                out[k] = obj[k]
        return out
    return None


def _normalize_entity(obj: Any) -> Optional[Dict[str, Any]]:
    """Accept {"name": "...", "kind": "...", "role": "..."} or bare string."""
    if isinstance(obj, str):
        n = obj.strip()
        if not n:
            return None
        return {"name": n, "kind": "person", "role": "mentioned"}
    if isinstance(obj, dict):
        name = str(obj.get("name") or "").strip()
        if not name:
            return None
        kind = str(obj.get("kind") or obj.get("type") or "person").strip().lower()
        if kind not in {"person", "org", "place"}:
            kind = "person"
        role = str(obj.get("role") or "mentioned").strip().lower()
        if role not in {"host", "guest", "mentioned"}:
            role = "mentioned"
        return {"name": name, "kind": kind, "role": role}
    return None


def _normalize_topic(obj: Any) -> Optional[str]:
    """Accept a bare string or {"label": "..."} (KG v3 schema compatibility)."""
    if isinstance(obj, str):
        t = obj.strip()
        return t or None
    if isinstance(obj, dict):
        t = str(obj.get("label") or obj.get("text") or obj.get("name") or "").strip()
        return t or None
    return None


def parse_megabundle_response(
    text: str,
    *,
    require_summary: bool = True,
    require_insights: bool = True,
    require_topics: bool = True,
) -> MegaBundleResult:
    """Parse a mega-bundle JSON response into a normalized :class:`MegaBundleResult`.

    Args:
        text: Raw response text (may contain code fences; we strip them).
        require_summary: Raise if the parsed object has no non-empty summary.
        require_insights: Raise if the parsed object has no non-empty insights list.
        require_topics: Raise if the parsed object has no non-empty topics list.

    Raises:
        MegaBundleParseError: if the response is not valid JSON, not an object,
        or missing a required field.
    """
    if not isinstance(text, str) or not text.strip():
        raise MegaBundleParseError("Empty response text")

    cleaned = _strip_code_fences(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise MegaBundleParseError(f"Response is not valid JSON: {e}") from e

    if not isinstance(data, dict):
        raise MegaBundleParseError(
            f"Expected a JSON object at the top level, got {type(data).__name__}"
        )

    title = str(data.get("title") or "").strip()
    summary = str(data.get("summary") or "").strip()
    bullets = _as_list_of_strings(data.get("bullets"))
    insights = [i for i in (_normalize_insight(x) for x in (data.get("insights") or [])) if i]
    topics = [t for t in (_normalize_topic(x) for x in (data.get("topics") or [])) if t]
    entities = [e for e in (_normalize_entity(x) for x in (data.get("entities") or [])) if e]

    if require_summary and not summary:
        raise MegaBundleParseError("Missing 'summary' in mega-bundle response")
    if require_insights and not insights:
        raise MegaBundleParseError("Missing 'insights' in mega-bundle response")
    if require_topics and not topics:
        raise MegaBundleParseError("Missing 'topics' in mega-bundle response")

    return MegaBundleResult(
        title=title,
        summary=summary,
        bullets=bullets,
        insights=insights,
        topics=topics,
        entities=entities,
        raw=data,
    )


def parse_extraction_bundle_response(text: str) -> MegaBundleResult:
    """Parse an extraction-only JSON response (no title/summary/bullets required).

    Used by the 2-call ``extraction_bundled`` pipeline mode where the summary
    stage ran separately in the first call. Returns a :class:`MegaBundleResult`
    with empty title/summary/bullets and populated insights/topics/entities.
    """
    result = parse_megabundle_response(
        text,
        require_summary=False,
        require_insights=True,
        require_topics=True,
    )
    # Extraction-bundle has no title/summary/bullets by design.
    result.title = ""
    result.summary = ""
    result.bullets = []
    return result
