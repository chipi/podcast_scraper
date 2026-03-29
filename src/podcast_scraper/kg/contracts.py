"""RFC-056 output contracts for KG inspect (machine-readable CLI JSON)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class KgTopicRow(BaseModel):
    """One Topic node summary for inspect output."""

    id: str = Field(description="Topic node id")
    label: str = Field(description="Topic label")
    slug: str = Field(description="Topic slug")


class KgEntityRow(BaseModel):
    """One Entity node summary for inspect output."""

    id: str = Field(description="Entity node id")
    name: str = Field(description="Display name")
    entity_kind: str = Field(description="person or organization")
    role: Optional[str] = Field(default=None, description="host, guest, mentioned, etc.")


class KgInspectOutput(BaseModel):
    """Output shape for kg inspect --format json (RFC-056)."""

    episode_id: Optional[str] = Field(default=None, description="Episode id from artifact")
    schema_version: Optional[str] = Field(default=None, description="Artifact schema version")
    extraction: Dict[str, Any] = Field(
        default_factory=dict,
        description="extraction block: model_version, extracted_at, transcript_ref",
    )
    node_count: int = Field(default=0, description="Total nodes")
    edge_count: int = Field(default=0, description="Total edges")
    nodes_by_type: Dict[str, int] = Field(
        default_factory=dict, description="Count of nodes per type label"
    )
    topics: List[KgTopicRow] = Field(default_factory=list, description="Topic nodes")
    entities: List[KgEntityRow] = Field(default_factory=list, description="Entity nodes")
    episode_title: Optional[str] = Field(default=None, description="Episode title if present")
    artifact_path: Optional[str] = Field(
        default=None, description="Path to .kg.json when resolved from filesystem"
    )


def build_kg_inspect_output(
    artifact: Dict[str, Any],
    *,
    artifact_path: Optional[Path] = None,
) -> KgInspectOutput:
    """Build validated inspect output from a kg.json dict (see corpus.inspect_summary)."""
    from .corpus import inspect_summary

    raw = inspect_summary(artifact, artifact_path=artifact_path)
    return KgInspectOutput.model_validate(raw)


class KgEntityEpisodeAppearance(BaseModel):
    """One episode row inside entity roll-up CLI output."""

    episode_id: str = Field(description="Episode id")
    title: str = Field(default="", description="Episode title if known")
    artifact_path: str = Field(description="Relative path to .kg.json")


class KgEntityRollupRow(BaseModel):
    """One aggregated entity from kg entities --format json."""

    entity_kind: str
    name: str
    episode_count: int
    mention_count: int
    episodes: List[KgEntityEpisodeAppearance]


class KgEntityRollupOutput(BaseModel):
    """Wrapper for kg entities --format json."""

    entities: List[KgEntityRollupRow]


class KgTopicPairRow(BaseModel):
    """One topic–topic pair from kg topics --format json."""

    topic_a_id: str
    topic_b_id: str
    topic_a_label: str
    topic_b_label: str
    episode_count: int


class KgTopicPairsOutput(BaseModel):
    """Wrapper for kg topics --format json."""

    topic_pairs: List[KgTopicPairRow]


class KgCorpusBundleOutput(BaseModel):
    """Validated shape for kg export --format merged."""

    export_kind: Literal["kg_corpus_bundle"] = "kg_corpus_bundle"
    schema_version: str
    artifact_count: int
    node_count_total: int
    edge_count_total: int
    artifacts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-episode kg.json payloads plus _artifact_path",
    )


def build_kg_entity_rollup_output(rows: List[Dict[str, Any]]) -> KgEntityRollupOutput:
    """Validate entity roll-up rows for machine-readable CLI output."""
    return KgEntityRollupOutput(entities=[KgEntityRollupRow.model_validate(r) for r in rows])


def build_kg_topic_pairs_output(rows: List[Dict[str, Any]]) -> KgTopicPairsOutput:
    """Validate topic pair rows for machine-readable CLI output."""
    return KgTopicPairsOutput(topic_pairs=[KgTopicPairRow.model_validate(r) for r in rows])


def build_kg_corpus_bundle_output(bundle: Dict[str, Any]) -> KgCorpusBundleOutput:
    """Validate merged corpus export dict."""
    return KgCorpusBundleOutput.model_validate(bundle)
