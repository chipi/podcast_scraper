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

logger = logging.getLogger(__name__)


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
        status: Parsing status (valid, degraded, invalid)
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


def parse_summary_output(
    text: str, provider: Any, episode_title: Optional[str] = None
) -> ParseResult:
    """Parse summary output from provider response.

    This function attempts multiple parsing strategies:
    1. Strict JSON parsing (for JSON mode providers)
    2. Best-effort JSON parsing (repair malformed JSON)
    3. Heuristic text parsing (extract bullets, quotes, etc.)

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

    # Strategy 1: Try strict JSON parsing
    try:
        data = json.loads(text.strip())
        schema = _validate_and_create_schema(data, text, episode_title)
        if schema:
            return ParseResult(schema=schema, success=True)
    except json.JSONDecodeError:
        pass  # Not JSON, try other strategies

    # Strategy 2: Try to repair malformed JSON
    repaired_json = _repair_json(text)
    if repaired_json:
        try:
            data = json.loads(repaired_json)
            schema = _validate_and_create_schema(data, text, episode_title)
            if schema:
                return ParseResult(schema=schema, success=True, repair_attempted=True)
        except json.JSONDecodeError:
            pass  # Repair failed

    # Strategy 3: Best-effort text parsing
    schema = _parse_text_heuristics(text, episode_title)
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

        # Determine status
        status: Literal["valid", "degraded", "invalid"] = "valid"
        if not all([title, key_quotes, named_entities]):
            status = "degraded"  # Missing optional fields

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
    # Remove markdown code fences
    text = re.sub(r"^```json\s*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"^```\s*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n```\s*$", "", text, flags=re.MULTILINE)

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
        bullets = paragraphs[:10]  # Limit to 10 paragraphs

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
