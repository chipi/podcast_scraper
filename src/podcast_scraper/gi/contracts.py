"""RFC-050 output contracts for GIL inspection and explore.

Dataclasses/Pydantic shapes: insight with supporting_quotes, evidence spans.
JSON-serializable for CLI --format json.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EvidenceSpan(BaseModel):
    """Evidence: reference to transcript and character span."""

    transcript_ref: str = Field(description="Reference to transcript artifact (e.g. filename)")
    char_start: int = Field(description="Character start in transcript (0-based)")
    char_end: int = Field(description="Character end in transcript (exclusive)")
    excerpt: Optional[str] = Field(
        default=None, description="Verbatim text from transcript if loaded"
    )


class SupportingQuote(BaseModel):
    """One supporting quote for an insight (RFC-050)."""

    quote_id: str = Field(description="Quote node ID")
    text: str = Field(description="Verbatim quote text")
    speaker_id: Optional[str] = Field(default=None, description="Speaker ID if available")
    timestamp_start_ms: Optional[int] = Field(default=None, description="Start time in ms")
    timestamp_end_ms: Optional[int] = Field(default=None, description="End time in ms")
    evidence: EvidenceSpan = Field(description="Transcript reference and span")


class InsightSummary(BaseModel):
    """One insight with optional supporting quotes (RFC-050)."""

    insight_id: str = Field(description="Insight node ID")
    text: str = Field(description="Insight statement")
    grounded: bool = Field(description="True if has at least one supporting quote")
    confidence: Optional[float] = Field(default=None, description="Extraction confidence 0-1")
    episode_id: str = Field(description="Episode this insight belongs to")
    supporting_quotes: List[SupportingQuote] = Field(
        default_factory=list,
        description="Quotes that support this insight",
    )


class InspectOutput(BaseModel):
    """Output shape for gi inspect (single episode)."""

    episode_id: str = Field(description="Episode identifier")
    schema_version: str = Field(description="Artifact schema version")
    model_version: str = Field(description="Model used for extraction")
    insights: List[InsightSummary] = Field(default_factory=list, description="Insights with quotes")
    stats: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary stats: grounded_count, ungrounded_count, quote_count, etc.",
    )


class ExploreOutput(BaseModel):
    """Output shape for gi explore (cross-episode topic query)."""

    topic: Optional[str] = Field(default=None, description="Topic filter used (if any)")
    insights: List[InsightSummary] = Field(
        default_factory=list, description="Insights with supporting quotes"
    )
    summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Counts: insight_count, grounded_insight_count, quote_count, episode_count",
    )
    episodes_searched: int = Field(description="Number of episode artifacts scanned")
