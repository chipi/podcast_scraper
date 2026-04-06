"""Pydantic models for viewer API responses (RFC-062)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ArtifactItem(BaseModel):
    """One GI or KG artifact file under a corpus directory."""

    name: str = Field(description="File name (basename).")
    relative_path: str = Field(description="Path relative to the listed corpus root (POSIX).")
    kind: Literal["gi", "kg"] = Field(description="Artifact kind.")
    size_bytes: int = Field(ge=0, description="File size in bytes.")


class ArtifactListResponse(BaseModel):
    """Response for GET /api/artifacts."""

    path: str = Field(description="Resolved absolute corpus root path.")
    artifacts: list[ArtifactItem] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Response for GET /api/health."""

    status: Literal["ok"] = "ok"


class IndexStatsBody(BaseModel):
    """Vector index aggregate stats (mirrors ``IndexStats`` dataclass)."""

    total_vectors: int = Field(ge=0)
    doc_type_counts: dict[str, int] = Field(default_factory=dict)
    feeds_indexed: list[str] = Field(default_factory=list)
    embedding_model: str = ""
    embedding_dim: int = Field(ge=0)
    last_updated: str = ""
    index_size_bytes: int = Field(ge=0)


class IndexStatsEnvelope(BaseModel):
    """Response for GET /api/index/stats — always 200 when the request is valid."""

    available: bool
    reason: str | None = Field(
        default=None,
        description="When ``available`` is false: no_corpus_path, no_index, load_failed, …",
    )
    index_path: str | None = None
    stats: IndexStatsBody | None = None


class SearchHitModel(BaseModel):
    """One semantic search hit (enriched metadata + optional quote stack)."""

    doc_id: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    text: str = ""
    supporting_quotes: list[dict[str, Any]] | None = None


class CorpusSearchApiResponse(BaseModel):
    """Response for GET /api/search."""

    query: str
    results: list[SearchHitModel] = Field(default_factory=list)
    error: str | None = None
    detail: str | None = None


class ExploreApiResponse(BaseModel):
    """Response for GET /api/explore (filters) or natural-language ``gi query`` (UC4)."""

    kind: Literal["explore", "natural_language"]
    error: str | None = None
    detail: str | None = None
    data: dict[str, Any] | None = Field(
        default=None,
        description="RFC-050 explore JSON when ``kind`` is ``explore``.",
    )
    question: str | None = None
    answer: dict[str, Any] | None = Field(
        default=None,
        description="UC4 answer (explore-shaped or topic leaderboard).",
    )
    explanation: str | None = None
