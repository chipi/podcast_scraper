"""Normalized summary output schema and parsing.

This module provides a stable, normalized schema for episode summaries with
strict parsing, validation, and repair capabilities.

The schema supports:
- Structured output from JSON mode providers (strict parsing)
- Best-effort parsing from text-only providers (heuristics, regex)
- Degraded status tracking when parsing fails
- Raw text preservation for failed parses
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from podcast_scraper import config_constants

logger = logging.getLogger(__name__)

# Sentence-split fallback for prose: merge overflow so we do not emit dozens of fragments.
# Align with downstream budget (GI/KG default max bullets), not a fixed “summary length”.
_HEURISTIC_BULLET_CAP = config_constants.DEFAULT_SUMMARY_BULLETS_DOWNSTREAM_MAX
# Multi-paragraph heuristic: allow a bit more than sentence cap for explicit paragraph breaks.
_HEURISTIC_PARAGRAPH_CAP = max(24, _HEURISTIC_BULLET_CAP + 4)


class SummarySchema(BaseModel):
    """Normalized summary schema with validation.

    This schema defines the standard structure for episode summaries,
    ensuring consistency across all providers.

    Attributes:
        title: Episode summary title (optional, can be derived from episode title)
        bullets: List of key takeaways/bullet points (required)
        key_quotes: Optional list of notable quotes from the episode
        named_entities: Optional list of important entities mentioned
        timestamps: Optional list of timestamp references with descriptions
        status: Parsing status (valid, degraded, invalid). JSON with title + bullets
            is valid even if optional key_quotes / named_entities are omitted.
        raw_text: Original raw text if parsing was degraded/invalid
    """

    title: Optional[str] = Field(default=None, description="Summary title")
    bullets: List[str] = Field(default_factory=list, description="Key takeaways/bullet points")
    key_quotes: Optional[List[str]] = Field(
        default=None, description="Notable quotes from the episode"
    )
    named_entities: Optional[List[str]] = Field(
        default=None, description="Important entities mentioned"
    )
    timestamps: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Timestamp references with descriptions"
    )
    status: Literal["valid", "degraded", "invalid"] = Field(
        default="valid", description="Parsing status"
    )
    raw_text: Optional[str] = Field(default=None, description="Original raw text if parsing failed")

    @field_validator("bullets")
    @classmethod
    def validate_bullets(cls, v: List[str]) -> List[str]:
        """Validate bullets are non-empty strings."""
        if not v:
            raise ValueError("bullets list cannot be empty")
        return [bullet.strip() for bullet in v if bullet.strip()]

    @field_validator("key_quotes")
    @classmethod
    def validate_key_quotes(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate key_quotes are non-empty strings."""
        if v is None:
            return None
        return [quote.strip() for quote in v if quote.strip()] or None

    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary."""
        result: Dict[str, Any] = {
            "bullets": self.bullets,
            "status": self.status,
        }
        if self.title:
            result["title"] = self.title
        if self.key_quotes:
            result["key_quotes"] = self.key_quotes
        if self.named_entities:
            result["named_entities"] = self.named_entities
        if self.timestamps:
            result["timestamps"] = self.timestamps
        if self.raw_text:
            result["raw_text"] = self.raw_text
        return result


@dataclass
class ParseResult:
    """Result of summary parsing attempt."""

    schema: Optional[SummarySchema]
    success: bool
    error: Optional[str] = None
    repair_attempted: bool = False


def _structured_summary_json_parse_failed(unfenced: str) -> bool:
    """True when *unfenced* is not valid JSON (strict parse).

    Used to refuse prose heuristics on truncated Gemini-style blobs: if the model
    clearly intended a ``title`` + ``bullets`` object but ``json.loads`` fails,
    treating the tail as "sentences" produces a single toxic bullet and hides the
    failure from callers that only check ``ParseResult.success``.
    """
    try:
        json.loads(unfenced)
    except json.JSONDecodeError:
        return True
    return False


def _looks_like_structured_summary_json_contract(text: str) -> bool:
    """Heuristic: model tried to return the normalized JSON summary object.

    Matches prompts that ask for ``title`` + ``bullets`` / ``key_points`` / ``takeaways``.
    Scoped to the opening of the payload to avoid scanning huge truncated blobs.
    """
    s = text.strip()
    if not s.startswith("{"):
        return False
    head = s[:8000].lower()
    return (
        '"bullets"' in head
        or '"key_points"' in head
        or '"takeaways"' in head
        or "'bullets'" in head
    )


def _strip_markdown_json_fence(text: str) -> str:
    """Remove leading `` ```json `` / `` ``` `` and trailing `` ``` `` from LLM output.

    Models often emit `` ```json { ... } ``` `` on **one line** (no newline after the
    fence). Older repair logic only stripped `` ```json `` when followed by ``\\n``,
    which left the fence in place and broke ``json.loads``.
    """
    t = text.strip()
    if not t:
        return t
    t = re.sub(r"^```(?:json)?\s*", "", t, count=1, flags=re.IGNORECASE)
    t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def parse_summary_output(
    text: str, provider: Any, episode_title: Optional[str] = None
) -> ParseResult:
    """Parse summary output from provider response.

    This function attempts multiple parsing strategies:
    1. Strict JSON parsing (for JSON mode providers)
    2. Best-effort JSON parsing (repair malformed JSON)
    3. Heuristic text parsing (extract bullets, quotes, etc.)

    If (1) and (2) do not yield a schema but the text still looks like the
    structured JSON summary contract (e.g. ``{"title":...,"bullets":[...``) and
    strict JSON parse of the fence-stripped text still fails, (3) is skipped
    and ``success=False`` is returned so callers can fail the episode instead
    of storing a single degraded "bullet" of raw JSON.

    Args:
        text: Raw summary text from provider
        provider: Provider instance (for capability detection)
        episode_title: Optional episode title for context

    Returns:
        ParseResult with parsed schema or error information
    """
    if not text or not text.strip():
        return ParseResult(
            schema=None,
            success=False,
            error="Empty summary text",
        )

    text_stripped = text.strip()
    unfenced = _strip_markdown_json_fence(text_stripped)

    # Strategy 1: Try strict JSON parsing
    try:
        data = json.loads(unfenced)
        schema = _validate_and_create_schema(data, text, episode_title)
        if schema:
            return ParseResult(schema=schema, success=True)
    except json.JSONDecodeError:
        pass  # Not JSON, try other strategies

    # Strategy 2: Try to repair malformed JSON
    repaired_json = _repair_json(text_stripped)
    if repaired_json:
        try:
            data = json.loads(repaired_json)
            schema = _validate_and_create_schema(data, text, episode_title)
            if schema:
                return ParseResult(schema=schema, success=True, repair_attempted=True)
        except json.JSONDecodeError:
            pass  # Repair failed

    if _looks_like_structured_summary_json_contract(
        unfenced
    ) and _structured_summary_json_parse_failed(unfenced):
        return ParseResult(
            schema=None,
            success=False,
            error=(
                "Structured summary JSON was incomplete or invalid (e.g. truncated). "
                "Skipping prose fallback; retry summarization for this episode."
            ),
        )

    # Strategy 3: Best-effort text parsing (use fence-stripped text)
    schema = _parse_text_heuristics(unfenced, episode_title)
    if schema:
        return ParseResult(
            schema=schema,
            success=True,
            repair_attempted=True,
        )

    # All strategies failed
    return ParseResult(
        schema=SummarySchema(
            bullets=[text[:200] + "..." if len(text) > 200 else text],
            status="invalid",
            raw_text=text,
        ),
        success=False,
        error="All parsing strategies failed",
    )


def _validate_and_create_schema(
    data: Dict[str, Any], raw_text: str, episode_title: Optional[str]
) -> Optional[SummarySchema]:
    """Validate parsed data and create SummarySchema.

    Args:
        data: Parsed JSON data
        raw_text: Original raw text
        episode_title: Optional episode title

    Returns:
        SummarySchema if valid, None otherwise
    """
    try:
        # Extract fields with defaults
        title = data.get("title") or episode_title
        bullets = data.get("bullets") or data.get("key_points") or data.get("takeaways") or []
        key_quotes = data.get("key_quotes") or data.get("quotes") or None
        named_entities = data.get("named_entities") or data.get("entities") or None
        timestamps = data.get("timestamps") or None

        # Ensure bullets is a list of strings
        if isinstance(bullets, str):
            bullets = [bullets]
        elif not isinstance(bullets, list):
            bullets = []

        # Normalize bullets
        bullets = [str(b).strip() for b in bullets if str(b).strip()]

        if not bullets:
            # Try to extract from summary field
            summary_text = data.get("summary") or data.get("text") or ""
            if summary_text:
                bullets = _extract_bullets_from_text(str(summary_text))

        if not bullets:
            return None  # Cannot create schema without bullets

        # JSON bullet summaries often omit key_quotes / named_entities; that is still a
        # complete contract when title (or episode_title fallback) and bullets exist.
        title_clean = title.strip() if isinstance(title, str) else ""
        if not title_clean:
            status: Literal["valid", "degraded", "invalid"] = "degraded"
        else:
            status = "valid"

        return SummarySchema(
            title=title,
            bullets=bullets,
            key_quotes=key_quotes,
            named_entities=named_entities,
            timestamps=timestamps,
            status=status,
            raw_text=None if status == "valid" else raw_text,
        )
    except Exception as e:
        logger.debug(f"Failed to create schema from data: {e}")
        return None


def _repair_json(text: str) -> Optional[str]:
    """Attempt to repair malformed JSON.

    Common repairs:
    - Remove trailing commas
    - Fix unclosed brackets/braces
    - Remove markdown code fences

    Args:
        text: Potentially malformed JSON text

    Returns:
        Repaired JSON string if possible, None otherwise
    """
    text = _strip_markdown_json_fence(text)

    # Remove trailing commas (simple heuristic)
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)

    return text.strip() if text.strip() else None


def _parse_text_heuristics(text: str, episode_title: Optional[str]) -> Optional[SummarySchema]:
    """Parse summary from plain text using heuristics.

    Extracts:
    - Bullets from bullet points (•, -, *, etc.)
    - Quotes from quoted text
    - Entities from capitalized phrases

    Args:
        text: Plain text summary
        episode_title: Optional episode title

    Returns:
        SummarySchema if parsing succeeds, None otherwise
    """
    bullets = _extract_bullets_from_text(text)
    if not bullets:
        return None

    key_quotes = _extract_quotes_from_text(text)
    named_entities = _extract_entities_from_text(text)

    return SummarySchema(
        title=episode_title,
        bullets=bullets,
        key_quotes=key_quotes if key_quotes else None,
        named_entities=named_entities if named_entities else None,
        status="degraded",  # Heuristic parsing is always degraded
        raw_text=text,
    )


def _extract_bullets_from_text(text: str) -> List[str]:
    """Extract bullet points from text.

    Supports:
    - • bullet points
    - - dashes
    - * asterisks
    - Numbered lists
    """
    bullets = []

    # Pattern for bullet points
    bullet_patterns = [
        r"^[\s]*[•\-\*]\s+(.+)$",  # •, -, or *
        r"^[\s]*\d+[\.\)]\s+(.+)$",  # Numbered lists
    ]

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        for pattern in bullet_patterns:
            match = re.match(pattern, line)
            if match:
                bullets.append(match.group(1).strip())
                break

    # If no bullets found, try to split by paragraphs
    if not bullets:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(paragraphs) == 1 and len(paragraphs[0]) > 200:
            # Single prose block (common for LLMs asked for "paragraphs"): split sentences
            raw = paragraphs[0].strip()
            sentences = re.split(r"(?<=[.!?])\s+", raw)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 12]
            cap = _HEURISTIC_BULLET_CAP
            if len(sentences) <= cap:
                bullets = sentences
            else:
                # Merge tail so we do not return dozens of tiny bullets
                head = sentences[: cap - 1]
                tail = " ".join(sentences[cap - 1 :])
                bullets = head + ([tail] if tail else [])
        else:
            bullets = paragraphs[:_HEURISTIC_PARAGRAPH_CAP]

    return bullets


def _extract_quotes_from_text(text: str) -> List[str]:
    """Extract quoted text from summary."""
    # Pattern for quoted text
    quote_pattern = r'["""]([^"""]+)["""]'
    quotes = re.findall(quote_pattern, text)
    return [q.strip() for q in quotes if q.strip()][:5]  # Limit to 5 quotes


def _extract_entities_from_text(text: str) -> List[str]:
    """Extract named entities (capitalized phrases) from text."""
    # Simple heuristic: extract capitalized phrases
    # This is a basic implementation; could be enhanced with NER
    entity_pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
    entities = re.findall(entity_pattern, text)
    # Filter out common words
    common_words = {"The", "This", "That", "These", "Those", "A", "An"}
    entities = [e for e in entities if e not in common_words]
    return list(set(entities))[:10]  # Limit to 10 unique entities


def validate_summary_schema(data: Dict[str, Any]) -> bool:
    """Validate that data conforms to SummarySchema.

    Args:
        data: Dictionary to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        SummarySchema(**data)
        return True
    except Exception:
        return False
