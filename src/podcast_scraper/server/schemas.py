"""Pydantic models for viewer API responses."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ArtifactItem(BaseModel):
    """One GI, KG, or bridge artifact file under a corpus directory."""

    name: str = Field(description="File name (basename).")
    relative_path: str = Field(description="Path relative to the listed corpus root (POSIX).")
    kind: Literal["gi", "kg", "bridge"] = Field(description="Artifact kind.")
    size_bytes: int = Field(ge=0, description="File size in bytes.")
    mtime_utc: str = Field(
        description="Last modification time (UTC ISO-8601 with Z suffix).",
    )
    publish_date: str = Field(
        description="Episode calendar date (YYYY-MM-DD) from metadata when present; "
        "otherwise UTC calendar date derived from this file's mtime (ingested surrogate).",
    )


class ArtifactListResponse(BaseModel):
    """Response for GET /api/artifacts."""

    path: str = Field(description="Resolved absolute corpus root path.")
    artifacts: list[ArtifactItem] = Field(default_factory=list)
    hints: list[str] = Field(
        default_factory=list,
        description="UX hints (e.g. multi-feed corpus root vs feed subtree for search).",
    )


class HealthResponse(BaseModel):
    """Response for GET /api/health."""

    status: Literal["ok"] = "ok"
    artifacts_api: bool = Field(
        default=True,
        description="True when GET /api/artifacts (GI/KG list + load for graph) is mounted.",
    )
    search_api: bool = Field(
        default=True,
        description="True when GET /api/search (semantic search) is mounted.",
    )
    explore_api: bool = Field(
        default=True,
        description="True when GET /api/explore (graph neighborhood) is mounted.",
    )
    index_routes_api: bool = Field(
        default=True,
        description="True when /api/index/stats and /api/index/rebuild are mounted.",
    )
    corpus_metrics_api: bool = Field(
        default=True,
        description=(
            "True when GET /api/corpus/stats, manifest, run-summary, " "runs/summary are mounted."
        ),
    )
    corpus_coverage_api: bool = Field(
        default=True,
        description="True when GET /api/corpus/coverage (GI/KG presence by month/feed) is mounted.",
    )
    corpus_library_api: bool = Field(
        default=True,
        description="True when GET /api/corpus/* catalog routes are mounted.",
    )
    corpus_digest_api: bool = Field(
        default=True,
        description=(
            "True when GET /api/corpus/digest is mounted. "
            "Omit or false on older server builds without digest."
        ),
    )
    corpus_binary_api: bool = Field(
        default=True,
        description="True when GET /api/corpus/binary is mounted (local artwork, corpus library).",
    )
    cil_queries_api: bool = Field(
        default=True,
        description="True when cross-layer CIL query routes are mounted (GitHub #527).",
    )


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
    reindex_recommended: bool = Field(
        default=False,
        description="Heuristic: artifacts newer than index, or search missing but metadata exists.",
    )
    reindex_reasons: list[str] = Field(
        default_factory=list,
        description="Stable reason codes (GitHub #507), e.g. artifacts_newer_than_index.",
    )
    artifact_newest_mtime: str | None = Field(
        default=None,
        description="Newest UTC ISO timestamp among index-relevant corpus files, if any.",
    )
    search_root_hints: list[str] = Field(
        default_factory=list,
        description="Same hints as GET /api/artifacts for multi-feed search root (may be empty).",
    )
    rebuild_in_progress: bool = Field(
        default=False,
        description="True while POST /api/index/rebuild is running for this corpus.",
    )
    rebuild_last_error: str | None = Field(
        default=None,
        description="Last background rebuild error message, if any.",
    )


class IndexRebuildAccepted(BaseModel):
    """Response for POST /api/index/rebuild (202 Accepted)."""

    accepted: bool = True
    corpus_path: str = Field(description="Resolved corpus root that was queued.")
    rebuild: bool = Field(description="Whether a full rebuild was requested.")


class SearchHitModel(BaseModel):
    """One semantic search hit (enriched metadata + optional quote stack)."""

    doc_id: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    text: str = ""
    supporting_quotes: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Insight hits only: optional supporting quote rows from indexer enrichment. "
            "Each entry may include speaker_id / speaker_name mirroring .gi.json; both are "
            "often null or absent when transcript segments lack diarization labels (GitHub #541)."
        ),
    )
    lifted: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Transcript hits only: chunk-to-Insight lift when GI offsets and bridge "
            "align (#528). Shape is loosely typed; typical keys include insight, speaker, "
            "topic, and quote (e.g. timestamp_start_ms / timestamp_end_ms). Speaker display "
            "and quote speaker fields follow the same .gi.json / segment rules as #541."
        ),
    )


class CorpusSearchLiftStatsModel(BaseModel):
    """Lift counters for the returned ``results`` page (after ``top_k`` slice)."""

    transcript_hits_returned: int = Field(
        default=0,
        ge=0,
        description="Rows in this response with metadata.doc_type == transcript.",
    )
    lift_applied: int = Field(
        default=0,
        ge=0,
        description="Rows in this response with a non-null ``lifted`` object.",
    )


class CorpusSearchApiResponse(BaseModel):
    """Response for GET /api/search."""

    query: str
    results: list[SearchHitModel] = Field(default_factory=list)
    error: str | None = None
    detail: str | None = None
    lift_stats: CorpusSearchLiftStatsModel | None = Field(
        default=None,
        description="Transcript lift coverage for this response page (#528).",
    )


class ExploreApiResponse(BaseModel):
    """Response for GET /api/explore (filters) or natural-language ``gi query`` (UC4)."""

    kind: Literal["explore", "natural_language"]
    error: str | None = None
    detail: str | None = None
    data: dict[str, Any] | None = Field(
        default=None,
        description="Explore-shaped GI JSON when ``kind`` is ``explore``.",
    )
    question: str | None = None
    answer: dict[str, Any] | None = Field(
        default=None,
        description="UC4 answer (explore-shaped or topic leaderboard).",
    )
    explanation: str | None = None


class CorpusFeedItem(BaseModel):
    """One feed row for GET /api/corpus/feeds."""

    feed_id: str = Field(description="Normalized feed id; empty string if missing in metadata.")
    display_title: str | None = Field(
        default=None,
        description="Feed title from metadata when present.",
    )
    episode_count: int = Field(ge=0, description="Episodes under this feed id in the catalog scan.")
    image_url: str | None = Field(
        default=None,
        description="Feed artwork URL from metadata when present (first non-empty seen).",
    )
    image_local_relpath: str | None = Field(
        default=None,
        description=("Corpus-relative path to downloaded feed art when file exists."),
    )
    rss_url: str | None = Field(
        default=None,
        description="RSS / feed URL from ``feed.url`` in metadata when present.",
    )
    description: str | None = Field(
        default=None,
        description="Feed description from ``feed.description`` in metadata when present.",
    )


class CorpusFeedsResponse(BaseModel):
    """Response for GET /api/corpus/feeds."""

    path: str = Field(description="Resolved corpus root.")
    feeds: list[CorpusFeedItem] = Field(default_factory=list)


class CilDigestTopicPill(BaseModel):
    """One CIL topic chip for digest / library (bridge identity + optional topic cluster)."""

    topic_id: str = Field(description="Canonical ``topic:…`` id from bridge.json.")
    label: str = Field(description="Human label (bridge display_name or derived from id).")
    in_topic_cluster: bool = Field(
        default=False,
        description="True when this topic_id is a member of a multi-topic cluster artifact.",
    )
    topic_cluster_compound_id: str | None = Field(
        default=None,
        description="Viewer compound parent ``tc:…`` when ``in_topic_cluster``.",
    )


class CorpusEpisodeListItem(BaseModel):
    """Summary row for GET /api/corpus/episodes."""

    metadata_relative_path: str
    feed_id: str
    feed_display_title: str | None = Field(
        default=None,
        description="Feed title from episode metadata when present.",
    )
    feed_rss_url: str | None = Field(
        default=None,
        description="RSS URL from ``feed.url`` (this row or any episode in the feed).",
    )
    feed_description: str | None = Field(
        default=None,
        description="From ``feed.description`` (this row or any episode in the feed).",
    )
    topics: list[str] = Field(
        default_factory=list,
        description="Summary-derived strings (capped); not used as list-row chips in the viewer.",
    )
    summary_title: str | None = Field(
        default=None,
        description="Summary headline from metadata (same shape as digest rows for list UI).",
    )
    summary_bullets_preview: list[str] = Field(
        default_factory=list,
        description="Up to four summary bullets (same cap as digest rows).",
    )
    summary_preview: str | None = Field(
        default=None,
        description="Compact recap for list rows (summary title, bullets, or truncated prose).",
    )
    episode_id: str | None = None
    episode_title: str
    publish_date: str | None = Field(
        default=None,
        description="YYYY-MM-DD when parseable from metadata.",
    )
    feed_image_url: str | None = Field(
        default=None,
        description="From ``feed.image_url`` in metadata when present.",
    )
    episode_image_url: str | None = Field(
        default=None,
        description="From ``episode.image_url`` in metadata when present.",
    )
    duration_seconds: int | None = Field(
        default=None,
        ge=0,
        description="From ``episode.duration_seconds`` when present.",
    )
    episode_number: int | None = Field(
        default=None,
        ge=0,
        description="From ``episode.episode_number`` when present.",
    )
    feed_image_local_relpath: str | None = Field(
        default=None,
        description="Verified path under ``.podcast_scraper/corpus-art/`` when present.",
    )
    episode_image_local_relpath: str | None = Field(
        default=None,
        description="Verified path under ``.podcast_scraper/corpus-art/`` when present.",
    )
    cil_digest_topics: list[CilDigestTopicPill] = Field(
        default_factory=list,
        description=(
            "Reserved on list responses (always empty); "
            "CIL pills are on digest rows and episode detail."
        ),
    )
    gi_relative_path: str = Field(
        default="",
        description="Corpus-relative GI artifact path (for graph loads from list rows).",
    )
    kg_relative_path: str = Field(
        default="",
        description="Corpus-relative KG artifact path (for graph loads from list rows).",
    )
    has_gi: bool = Field(default=False, description="True when GI artifact exists on disk.")
    has_kg: bool = Field(default=False, description="True when KG artifact exists on disk.")


class CorpusEpisodesResponse(BaseModel):
    """Response for GET /api/corpus/episodes."""

    path: str
    feed_id: str | None = Field(
        default=None,
        description="Echo of filter when set; null when listing all feeds.",
    )
    items: list[CorpusEpisodeListItem] = Field(default_factory=list)
    next_cursor: str | None = None


class CorpusEpisodeDetailResponse(BaseModel):
    """Response for GET /api/corpus/episodes/detail."""

    path: str
    metadata_relative_path: str
    feed_id: str
    feed_rss_url: str | None = Field(
        default=None,
        description="RSS URL from ``feed.url`` (this row or any episode in the feed).",
    )
    feed_description: str | None = Field(
        default=None,
        description="From ``feed.description`` (this row or any episode in the feed).",
    )
    episode_id: str | None = None
    episode_title: str
    publish_date: str | None = None
    summary_title: str | None = None
    summary_bullets: list[str] = Field(default_factory=list)
    summary_text: str | None = Field(
        default=None,
        description="Long-form summary from metadata (raw_text or short_summary).",
    )
    gi_relative_path: str
    kg_relative_path: str
    bridge_relative_path: str = Field(
        description="Sibling ``.bridge.json`` path from metadata stem.",
    )
    has_gi: bool = False
    has_kg: bool = False
    has_bridge: bool = Field(
        description="True when ``bridge_relative_path`` exists on disk.",
    )
    feed_image_url: str | None = Field(
        default=None,
        description="From ``feed.image_url`` in metadata when present.",
    )
    episode_image_url: str | None = Field(
        default=None,
        description="From ``episode.image_url`` in metadata when present.",
    )
    duration_seconds: int | None = Field(
        default=None,
        ge=0,
        description="From ``episode.duration_seconds`` when present.",
    )
    episode_number: int | None = Field(
        default=None,
        ge=0,
        description="From ``episode.episode_number`` when present.",
    )
    feed_image_local_relpath: str | None = Field(
        default=None,
        description="Verified local artwork path when file exists on disk.",
    )
    episode_image_local_relpath: str | None = Field(
        default=None,
        description="Verified local artwork path when file exists on disk.",
    )
    cil_digest_topics: list[CilDigestTopicPill] = Field(
        default_factory=list,
        description="CIL topic pills from bridge (cluster-first order).",
    )


class CorpusSimilarEpisodeItem(BaseModel):
    """One deduped peer episode from GET /api/corpus/episodes/similar."""

    score: float
    feed_id: str = Field(description="Normalized feed id (empty when unknown).")
    episode_id: str | None = None
    episode_title: str = Field(
        default="",
        description="From catalog when the episode is listed; otherwise empty.",
    )
    metadata_relative_path: str | None = Field(
        default=None,
        description="Catalog metadata path when known; use with episodes/detail.",
    )
    publish_date: str | None = None
    doc_type: str | None = Field(
        default=None,
        description="Vector doc_type of the highest-scoring chunk for this episode.",
    )
    snippet: str = Field(default="", description="Chunk text from the index hit.")
    feed_image_url: str | None = Field(
        default=None,
        description="From catalog when the peer episode is listed.",
    )
    episode_image_url: str | None = Field(
        default=None,
        description="From catalog when the peer episode is listed.",
    )
    duration_seconds: int | None = Field(default=None, ge=0)
    episode_number: int | None = Field(default=None, ge=0)
    feed_image_local_relpath: str | None = None
    episode_image_local_relpath: str | None = None


class CorpusSimilarEpisodesResponse(BaseModel):
    """Response for GET /api/corpus/episodes/similar."""

    path: str
    source_metadata_relative_path: str
    query_used: str = ""
    items: list[CorpusSimilarEpisodeItem] = Field(default_factory=list)
    error: str | None = Field(
        default=None,
        description="Machine-readable: insufficient_text, no_index, embed_failed, …",
    )
    detail: str | None = None


class CorpusDigestRow(BaseModel):
    """One ranked digest row."""

    metadata_relative_path: str
    feed_id: str
    feed_display_title: str | None = Field(
        default=None,
        description="Feed title from episode metadata when present.",
    )
    feed_rss_url: str | None = Field(
        default=None,
        description="RSS URL from ``feed.url`` (this row or any episode in the feed).",
    )
    feed_description: str | None = Field(
        default=None,
        description="From ``feed.description`` (this row or any episode in the feed).",
    )
    episode_id: str | None = None
    episode_title: str
    publish_date: str | None = None
    summary_title: str | None = None
    summary_bullets_preview: list[str] = Field(default_factory=list)
    summary_bullet_graph_topic_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Parallel to summary_bullets_preview: topic:{slug} hints from each bullet text "
            "(graph_id_utils.slugify_label), for Digest → Graph topic focus."
        ),
    )
    summary_preview: str | None = Field(
        default=None,
        description="Compact recap for list cards (aligned with Library episode list).",
    )
    gi_relative_path: str
    kg_relative_path: str
    has_gi: bool = False
    has_kg: bool = False
    feed_image_url: str | None = Field(default=None, description="From metadata when present.")
    episode_image_url: str | None = Field(default=None, description="From metadata when present.")
    duration_seconds: int | None = Field(default=None, ge=0)
    episode_number: int | None = Field(default=None, ge=0)
    feed_image_local_relpath: str | None = Field(default=None, description="Verified local path.")
    episode_image_local_relpath: str | None = Field(
        default=None, description="Verified local path."
    )
    cil_digest_topics: list[CilDigestTopicPill] = Field(
        default_factory=list,
        description="CIL topic pills from bridge (cluster-first order); empty when no bridge.",
    )


class CorpusDigestTopicHit(BaseModel):
    """Semantic topic hit scoped to digest window."""

    metadata_relative_path: str | None = None
    episode_title: str = ""
    feed_id: str = ""
    feed_display_title: str | None = Field(
        default=None,
        description="Feed title from catalog when present.",
    )
    feed_rss_url: str | None = Field(
        default=None,
        description="RSS URL from catalog row or sibling episode in the feed.",
    )
    feed_description: str | None = Field(
        default=None,
        description="Feed description from catalog row or sibling episode in the feed.",
    )
    score: float | None = None
    summary_preview: str | None = Field(
        default=None,
        description="Compact recap from catalog row when joined.",
    )
    episode_id: str | None = None
    publish_date: str | None = Field(
        default=None,
        description="Episode publish day from catalog (YYYY-MM-DD) when joined.",
    )
    gi_relative_path: str = ""
    kg_relative_path: str = ""
    has_gi: bool = False
    has_kg: bool = False
    feed_image_url: str | None = Field(default=None, description="From catalog row when joined.")
    episode_image_url: str | None = Field(default=None, description="From catalog row when joined.")
    duration_seconds: int | None = Field(default=None, ge=0)
    episode_number: int | None = Field(default=None, ge=0)
    feed_image_local_relpath: str | None = Field(default=None, description="Verified local path.")
    episode_image_local_relpath: str | None = Field(
        default=None, description="Verified local path."
    )


class CorpusDigestTopicBand(BaseModel):
    """Topic heading + hits for Digest tab."""

    topic_id: str
    label: str
    query: str
    graph_topic_id: str = Field(
        description=(
            "Suggested GI/KG topic node id (topic:{slug}) from the band label; "
            "may be absent in a given episode graph."
        ),
    )
    hits: list[CorpusDigestTopicHit] = Field(default_factory=list)


class CorpusDigestResponse(BaseModel):
    """Response for GET /api/corpus/digest."""

    path: str
    window: Literal["all", "24h", "7d", "1mo", "since"]
    window_start_utc: str
    window_end_utc: str
    compact: bool
    rows: list[CorpusDigestRow] = Field(default_factory=list)
    topics: list[CorpusDigestTopicBand] = Field(default_factory=list)
    topics_unavailable_reason: str | None = None


class CorpusStatsResponse(BaseModel):
    """Response for GET /api/corpus/stats (dashboard histograms)."""

    path: str
    publish_month_histogram: dict[str, int] = Field(
        default_factory=dict,
        description="YYYY-MM keys → episode count from catalog publish_date.",
    )
    catalog_episode_count: int = Field(
        0,
        description="Total catalog rows (metadata files); may exceed histogram sum.",
    )
    catalog_feed_count: int = Field(
        0,
        description="Distinct non-empty feed_id values in the catalog (normalized).",
    )
    digest_topics_configured: int = Field(
        0,
        description="Digest topic bands from server config.",
    )


class CoverageByMonthItem(BaseModel):
    """One publish-month bucket for GI/KG coverage (dashboard)."""

    month: str = Field(description="YYYY-MM from episode publish_date.")
    total: int = Field(ge=0)
    with_gi: int = Field(ge=0)
    with_kg: int = Field(ge=0)
    with_both: int = Field(ge=0)


class CoverageFeedItem(BaseModel):
    """Per-feed GI/KG coverage counts (dashboard)."""

    feed_id: str
    display_title: str
    total: int = Field(ge=0)
    with_gi: int = Field(ge=0)
    with_kg: int = Field(ge=0)


class CorpusCoverageResponse(BaseModel):
    """Response for GET /api/corpus/coverage."""

    path: str
    total_episodes: int = Field(ge=0)
    with_gi: int = Field(ge=0)
    with_kg: int = Field(ge=0)
    with_both: int = Field(ge=0)
    with_neither: int = Field(ge=0)
    by_month: list[CoverageByMonthItem] = Field(default_factory=list)
    by_feed: list[CoverageFeedItem] = Field(default_factory=list)


class TopPersonItem(BaseModel):
    """One row for GET /api/corpus/persons/top."""

    person_id: str = Field(description='Canonical id, e.g. "person:slug".')
    display_name: str
    episode_count: int = Field(ge=0)
    insight_count: int = Field(ge=0)
    top_topics: list[str] = Field(
        default_factory=list,
        description="Up to three canonical topic ids (ABOUT targets of grounded insights).",
    )


class CorpusTopPersonsResponse(BaseModel):
    """Response for GET /api/corpus/persons/top."""

    path: str
    persons: list[TopPersonItem] = Field(default_factory=list)
    total_persons: int = Field(
        ge=0,
        description="Distinct Person node ids seen across GI artifacts.",
    )


class CorpusRunSummaryItem(BaseModel):
    """Compact row parsed from a ``run.json`` under the corpus tree."""

    relative_path: str
    run_id: str = ""
    created_at: str | None = None
    run_duration_seconds: float | None = None
    episodes_scraped_total: int | None = None
    errors_total: int | None = None
    gi_artifacts_generated: int | None = None
    kg_artifacts_generated: int | None = None
    time_scraping_seconds: float | None = None
    time_parsing_seconds: float | None = None
    time_normalizing_seconds: float | None = None
    time_io_and_waiting_seconds: float | None = None
    episode_outcomes: dict[str, int] = Field(
        default_factory=dict,
        description="Counts for ok / failed / skipped from ``metrics.episode_statuses``.",
    )


class CorpusRunsSummaryResponse(BaseModel):
    """Response for GET /api/corpus/runs/summary."""

    path: str
    runs: list[CorpusRunSummaryItem] = Field(default_factory=list)


class CorpusResolveEpisodesRequest(BaseModel):
    """Body for POST /api/corpus/resolve-episode-artifacts."""

    episode_ids: list[str] = Field(
        min_length=1,
        description="Logical episode ids from metadata (same as catalog episode_id).",
    )
    path: str | None = Field(
        default=None,
        description="Corpus root. Omit when server default output_dir is set.",
    )


class CorpusResolvedEpisodeArtifact(BaseModel):
    """Resolved GI/KG/bridge relative paths for one episode."""

    episode_id: str
    publish_date: str | None = None
    gi_relative_path: str | None = None
    kg_relative_path: str | None = None
    bridge_relative_path: str | None = None


class CorpusResolveEpisodesResponse(BaseModel):
    """Response for POST /api/corpus/resolve-episode-artifacts."""

    path: str
    resolved: list[CorpusResolvedEpisodeArtifact] = Field(default_factory=list)
    missing_episode_ids: list[str] = Field(default_factory=list)


class CorpusNodeEpisodesRequest(BaseModel):
    """Body for POST /api/corpus/node-episodes (cross-episode graph expand)."""

    node_id: str = Field(min_length=1, description="Viewer or canonical CIL node id.")
    path: str | None = Field(
        default=None,
        description="Corpus root. Omit when server default output_dir is set.",
    )
    max_episodes: int | None = Field(
        default=None,
        ge=1,
        description="Optional cap after stable sort by gi_relative_path; omit for all matches.",
    )


class CorpusNodeEpisodeItem(BaseModel):
    """One episode whose bridge references the requested CIL id."""

    gi_relative_path: str
    kg_relative_path: str
    bridge_relative_path: str
    episode_id: str | None = None


class CorpusNodeEpisodesResponse(BaseModel):
    """Response for POST /api/corpus/node-episodes."""

    path: str
    node_id: str
    episodes: list[CorpusNodeEpisodeItem] = Field(default_factory=list)
    truncated: bool = False
    total_matched: int | None = Field(
        default=None,
        description="Pre-cap match count when truncated is True; otherwise None.",
    )


# --- Cross-layer CIL queries (GitHub #527) ---


class CilArcEpisodeBlock(BaseModel):
    """One episode slice in a position arc or topic timeline."""

    episode_id: str
    publish_date: str | None = None
    episode_title: str | None = None
    feed_title: str | None = None
    episode_number: int | None = None
    episode_image_url: str | None = None
    episode_image_local_relpath: str | None = None
    feed_image_url: str | None = None
    feed_image_local_relpath: str | None = None
    insights: list[dict[str, Any]] = Field(default_factory=list)


class CilPositionArcResponse(BaseModel):
    """Response for GET /api/persons/{person_id}/positions."""

    path: str
    person_id: str
    topic_id: str
    episodes: list[CilArcEpisodeBlock] = Field(default_factory=list)


class CilPersonProfileInsightRow(BaseModel):
    """One insight row inside ``CilPersonProfileResponse.topics``."""

    episode_id: str
    insight: dict[str, Any]
    insight_type: str
    position_hint: float | None = None


class CilPersonProfileQuoteRow(BaseModel):
    """Quote evidence row for a person profile."""

    episode_id: str
    quote: dict[str, Any]


class CilPersonProfileResponse(BaseModel):
    """Response for GET /api/persons/{person_id}/brief."""

    path: str
    person_id: str
    topics: dict[str, list[CilPersonProfileInsightRow]] = Field(default_factory=dict)
    quotes: list[CilPersonProfileQuoteRow] = Field(default_factory=list)


class CilTopicTimelineResponse(BaseModel):
    """Response for GET /api/topics/{topic_id}/timeline."""

    path: str
    topic_id: str
    episodes: list[CilArcEpisodeBlock] = Field(default_factory=list)


class CilTopicTimelineMergeRequest(BaseModel):
    """Body for POST /api/topics/timeline — merged topic timeline (cluster scope)."""

    topic_ids: list[str] = Field(
        min_length=1,
        description="Topic ids (e.g. topic:…); same normalization as GET single-topic timeline.",
    )
    path: str | None = Field(
        default=None,
        description="Corpus root. Omit when server default output_dir is set.",
    )
    insight_types: str | None = Field(
        default=None,
        description="Comma-separated insight_type filter; omit for all; ``all`` or ``*`` for all.",
    )


class CilTopicTimelineMergedResponse(BaseModel):
    """Response for POST /api/topics/timeline."""

    path: str
    topic_ids: list[str]
    episodes: list[CilArcEpisodeBlock] = Field(default_factory=list)


class CilIdListResponse(BaseModel):
    """Response for GET /api/persons/{id}/topics and GET /api/topics/{id}/persons."""

    path: str
    anchor_id: str
    ids: list[str] = Field(default_factory=list)
