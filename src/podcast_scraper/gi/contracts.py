"""RFC-050 output contracts for GIL inspection and explore.

Dataclasses/Pydantic shapes: insight with supporting_quotes, evidence spans.
JSON-serializable for CLI --format json.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


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
    speaker_name: Optional[str] = Field(
        default=None,
        description="Speaker display name from SPOKEN_BY -> Speaker node when present",
    )
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
    episode_title: Optional[str] = Field(
        default=None,
        description="Episode title from GIL Episode node when present",
    )
    publish_date: Optional[str] = Field(
        default=None,
        description="Episode publish_date ISO string from Episode node when present",
    )
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


class TopSpeakerEntry(BaseModel):
    """Aggregated speaker stats for explore (RFC-050 top_speakers)."""

    speaker_id: str = Field(description="Speaker identifier from quotes")
    name: Optional[str] = Field(default=None, description="Display name when known")
    quote_count: int = Field(ge=0, description="Supporting quotes attributed to speaker")
    insight_count: int = Field(ge=0, description="Distinct insights with such quotes")


class TopicEntry(BaseModel):
    """Topic node aggregated across an explore result set (ABOUT edges; GitHub #487)."""

    topic_id: str = Field(description="Topic node id from gi.json (e.g. topic:slug)")
    label: str = Field(description="Topic label from Topic.properties.label")
    insight_count: int = Field(
        ge=0,
        description="Count of insights in the result set linked to this topic",
    )


class ExploreOutput(BaseModel):
    """Output shape for gi explore (cross-episode topic query)."""

    topic: Optional[str] = Field(default=None, description="Topic filter used (if any)")
    speaker_filter: Optional[str] = Field(
        default=None,
        description="Speaker substring filter used for explore (if any)",
    )
    insights: List[InsightSummary] = Field(
        default_factory=list, description="Insights with supporting quotes"
    )
    summary: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Counts: insight_count, grounded_insight_count, quote_count, episode_count, "
            "speaker_count (distinct speakers in result quotes)"
        ),
    )
    top_speakers: List[TopSpeakerEntry] = Field(
        default_factory=list,
        description="Speakers ranked by quote_count for this result set (RFC-050)",
    )
    topics: List[TopicEntry] = Field(
        default_factory=list,
        description="Topic nodes linked via ABOUT to insights in this result (#487)",
    )
    episodes_searched: int = Field(description="Number of episode artifacts scanned")


class GiArtifact(BaseModel):
    """Typed top-level shape for one gi.json payload (GitHub #487 / GIL-2).

    Node and edge payloads remain dicts until full discriminated unions exist.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    model_version: str
    prompt_version: str
    episode_id: str
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    edges: List[Dict[str, Any]] = Field(default_factory=list)


class GiCorpusArtifactEntry(GiArtifact):
    """One episode artifact inside a merged corpus bundle (allows ``_artifact_path``)."""

    model_config = ConfigDict(extra="allow")


class GiCorpusBundleOutput(BaseModel):
    """Validated shape for ``gi export --format merged``."""

    export_kind: Literal["gi_corpus_bundle"] = "gi_corpus_bundle"
    schema_version: str
    artifact_count: int
    insight_count_total: int
    quote_count_total: int
    artifacts: List[GiCorpusArtifactEntry] = Field(
        default_factory=list,
        description="Per-episode gi.json payloads plus optional _artifact_path",
    )


def build_gi_corpus_bundle_output(bundle: Dict[str, Any]) -> GiCorpusBundleOutput:
    """Validate merged GIL corpus export dict."""
    return GiCorpusBundleOutput.model_validate(bundle)
