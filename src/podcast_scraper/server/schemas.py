"""Pydantic models for viewer API responses."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class TranscriptSegment(BaseModel):
    """One transcript segment in the player ``segments.json`` contract (PRD-036)."""

    id: str = Field(description="Stable segment id within the episode.")
    start: float = Field(ge=0, description="Segment start time in seconds.")
    end: float = Field(ge=0, description="Segment end time in seconds.")
    text: str = Field(description="Segment transcript text.")
    speaker: str | None = Field(
        default=None,
        description="Speaker label/id when diarized (canonical person:{slug} when "
        "resolved, else raw label); omitted when unknown.",
    )


class SegmentsResponse(BaseModel):
    """Response for GET /api/app/episodes/{slug}/segments (the segments.json contract)."""

    version: str = Field(default="1.0", description="Segments contract version.")
    episode_slug: str = Field(description="Stable episode slug this transcript belongs to.")
    segments: list[TranscriptSegment] = Field(default_factory=list)


class AudioSourceResponse(BaseModel):
    """Response for GET /api/app/episodes/{slug}/audio-source (bridge, never rehost)."""

    episode_slug: str = Field(description="Stable episode slug.")
    url: str = Field(description="Origin-host enclosure URL the client plays directly.")
    mime: str | None = Field(
        default=None, description="Enclosure MIME type when known (content.media_type)."
    )
    duration_seconds: int | None = Field(
        default=None, ge=0, description="Episode duration in seconds when known."
    )
    media_id: str | None = Field(
        default=None, description="Stable media identifier (content.media_id) when present."
    )
    strategy: Literal["direct", "proxy"] = Field(
        default="direct",
        description="'direct' = client streams the origin URL; 'proxy' (future) = "
        "no-store pass-through when a host blocks direct play.",
    )
    resolved_url: str | None = Field(
        default=None,
        description="Final URL after following redirects (only set when validate=true).",
    )
    verified: bool | None = Field(
        default=None,
        description="HEAD reachability when validate=true; null when not validated.",
    )
    content_length: int | None = Field(
        default=None, ge=0, description="Content-Length from validation when available."
    )


class AppEpisodeDetail(BaseModel):
    """Response for GET /api/app/episodes/{slug} — consumer episode detail."""

    slug: str = Field(description="Stable episode slug.")
    title: str = Field(description="Episode title.")
    feed_id: str = Field(description="Owning feed id.")
    podcast_title: str | None = Field(default=None, description="Feed/show display title.")
    publish_date: str | None = Field(
        default=None, description="Publish date (YYYY-MM-DD) when known."
    )
    duration_seconds: int | None = Field(
        default=None, ge=0, description="Episode duration when known."
    )
    episode_image_url: str | None = Field(
        default=None, description="Remote (feed-hosted) episode artwork URL — fallback only."
    )
    feed_image_url: str | None = Field(
        default=None, description="Remote (feed-hosted) feed artwork URL — fallback only."
    )
    artwork_url: str | None = Field(
        default=None,
        description="Preferred artwork: our locally-stored copy (large size for the player) "
        "when present. Clients use this, falling back to the remote image URLs.",
    )
    summary_title: str | None = Field(default=None, description="Summary title when present.")
    summary_bullets: list[str] = Field(default_factory=list, description="Summary bullet points.")
    summary_text: str | None = Field(
        default=None, description="Full summary paragraph when present."
    )
    has_transcript: bool = Field(description="Whether a transcript file is referenced.")
    has_summary: bool = Field(description="Whether any summary content is present.")
    has_gi: bool = Field(description="Whether a grounded-insight artifact exists.")
    has_kg: bool = Field(description="Whether a knowledge-graph artifact exists.")
    has_bridge: bool = Field(description="Whether a canonical-identity bridge artifact exists.")


class AppEpisodeSummary(BaseModel):
    """One episode card in the consumer catalog (PRD-038; list-item shape).

    Lightweight by design: it carries the fields the catalog scan already produces.
    Per-artifact depth counts (insight_count, speaker_count) are intentionally NOT
    computed here (they would cost an artifact load per row) — the client reads them
    lazily from the detail / insights / entities endpoints. Artifact-presence flags
    (``has_gi`` / ``has_kg``) are the cheap depth signal for cards.
    """

    slug: str = Field(description="Stable episode slug.")
    title: str = Field(description="Episode title.")
    feed_id: str = Field(description="Owning feed id.")
    podcast_title: str | None = Field(default=None, description="Feed/show display title.")
    publish_date: str | None = Field(
        default=None, description="Publish date (YYYY-MM-DD) when known."
    )
    duration_seconds: int | None = Field(
        default=None, ge=0, description="Episode duration when known."
    )
    episode_image_url: str | None = Field(
        default=None, description="Remote (feed-hosted) episode artwork URL — fallback only."
    )
    feed_image_url: str | None = Field(
        default=None, description="Remote (feed-hosted) feed artwork URL — fallback only."
    )
    artwork_url: str | None = Field(
        default=None,
        description="Preferred artwork: our locally-stored copy (thumb size) when present. "
        "Clients use this, falling back to the remote image URLs.",
    )
    status: Literal["ready", "pending"] = Field(
        default="ready",
        description="Playability: 'ready' when a transcript exists, else 'pending'. "
        "Local-content MVP yields 'ready'; richer states arrive with scrape-on-demand (#1069).",
    )
    summary_preview: str | None = Field(
        default=None,
        description="Short, clean one-line lede for the card (summary title / first sentence) — "
        "NOT the bullets joined; the full bullets are in `summary_bullets`.",
    )
    summary_text: str | None = Field(
        default=None,
        description="The full prose summary, for the card's hover/expand preview (null if absent).",
    )
    summary_bullets: list[str] = Field(
        default_factory=list,
        description="Full summary bullet points, for the card's expand-on-demand insights view "
        "(so the card stays compact while the complete summary stays one tap/hover away).",
    )
    topics: list[str] = Field(
        default_factory=list, description="Short topic labels for card pills (from summary)."
    )
    has_transcript: bool = Field(description="Whether a transcript file is referenced.")
    has_summary: bool = Field(description="Whether any summary content is present.")
    has_gi: bool = Field(description="Whether a grounded-insight artifact exists.")
    has_kg: bool = Field(description="Whether a knowledge-graph artifact exists.")
    has_bridge: bool = Field(description="Whether a canonical-identity bridge artifact exists.")


class AppEpisodesResponse(BaseModel):
    """Paginated episode list for GET /api/app/episodes and /podcasts/{id}/episodes."""

    items: list[AppEpisodeSummary] = Field(default_factory=list)
    page: int = Field(ge=1, description="1-based page index for this response.")
    page_size: int = Field(ge=1, description="Requested page size.")
    total: int = Field(ge=0, description="Total episodes matching the filter.")
    has_more: bool = Field(description="Whether more pages exist after this one.")


class AppDiscoverClickBody(BaseModel):
    """A click on a discovery-feed episode — ranking-experiment telemetry (#11)."""

    slug: str = Field(description="The clicked episode's slug, as shown in the feed.")
    position: int = Field(ge=0, description="0-based rank position where it was shown.")


class AppGraphEventsBody(BaseModel):
    """A fire-and-forget batch of graph-analytics events (usage / size-dynamics / breakage).

    Each event is a free-form object carrying at least an ``action`` (e.g. ``node_tap``,
    ``rail_nav``, ``redraw``, ``handoff_failed``); the rest of the payload is open so new event
    kinds don't need a schema change.
    """

    events: list[dict[str, Any]] = Field(
        default_factory=list, description="Graph events to append to the user's log."
    )


class AppQuote(BaseModel):
    """A verbatim quote supporting an insight."""

    text: str = Field(description="Verbatim quote text.")
    speaker: str | None = Field(default=None, description="Speaker name/id when attributed.")
    char_start: int | None = Field(default=None, description="Transcript char offset start.")
    char_end: int | None = Field(default=None, description="Transcript char offset end.")
    start_ms: int | None = Field(default=None, description="Quote start timestamp (ms).")
    end_ms: int | None = Field(default=None, description="Quote end timestamp (ms).")


class AppInsight(BaseModel):
    """A grounded insight with its supporting quotes (GIL projection)."""

    id: str = Field(description="Insight node id.")
    text: str = Field(description="Insight text.")
    grounded: bool = Field(description="Whether the insight has >=1 supporting quote.")
    insight_type: str | None = Field(
        default=None, description="claim/recommendation/observation/... when set."
    )
    confidence: float | None = Field(default=None, description="Extractor confidence when set.")
    position_hint: str | None = Field(default=None, description="Temporal position hint when set.")
    quotes: list[AppQuote] = Field(default_factory=list)


class AppInsightsResponse(BaseModel):
    """Response for GET /api/app/episodes/{slug}/insights."""

    episode_slug: str = Field(description="Stable episode slug.")
    insights: list[AppInsight] = Field(default_factory=list)


class AppEntity(BaseModel):
    """A KG person/org entity mentioned in an episode."""

    id: str = Field(description="Canonical entity id (person:{slug} / org:{slug}).")
    name: str = Field(description="Display name.")
    kind: Literal["person", "org"] = Field(description="Entity kind.")


class AppTopic(BaseModel):
    """A KG topic discussed in an episode (RFC-102: cluster fields for cluster-first grouping)."""

    id: str = Field(description="Canonical topic id (topic:{slug}).")
    label: str = Field(description="Topic display label.")
    cluster_id: str | None = Field(
        default=None,
        description="Corpus topic-cluster id (graph_compound_parent_id) when the topic belongs to "
        "a multi-member cluster; null for singletons. From search/topic_clusters.json.",
    )
    cluster_label: str | None = Field(
        default=None, description="Canonical label of the topic's cluster, when clustered."
    )
    cluster_size: int = Field(
        default=0, ge=0, description="Cross-corpus member count of the topic's cluster (0 if none)."
    )
    theme_cluster_id: str | None = Field(
        default=None,
        description="Corpus THEME-cluster id (thc:{slug}) — topics discussed together "
        "(co-occurrence), distinct from the semantic cluster_id. Null when not in a theme. "
        "From enrichments/topic_theme_clusters.json.",
    )
    theme_cluster_label: str | None = Field(
        default=None, description="Canonical label of the topic's theme cluster, when in one."
    )
    theme_cluster_size: int = Field(
        default=0, ge=0, description="Member count of the topic's theme cluster (0 if none)."
    )


class AppEntitiesResponse(BaseModel):
    """Response for GET /api/app/episodes/{slug}/entities."""

    episode_slug: str = Field(description="Stable episode slug.")
    persons: list[AppEntity] = Field(default_factory=list)
    orgs: list[AppEntity] = Field(default_factory=list)
    topics: list[AppTopic] = Field(default_factory=list)


class AppEntityRef(BaseModel):
    """A resolved person/topic reference for the entity-in-search result (PRD-043 FR3 / 3.4)."""

    id: str = Field(description="Canonical entity id (person:{slug} / topic:{slug}).")
    kind: Literal["person", "topic"] = Field(description="Which card to open.")
    label: str = Field(description="Display name / topic label.")


class AppEntitySearchResponse(BaseModel):
    """Response for GET /api/app/entities/search — at most one exact/near-exact match."""

    query: str = Field(description="The query that was resolved.")
    entity: AppEntityRef | None = Field(
        default=None, description="The matched person/topic, or null when nothing matches."
    )


class AppPersonCard(BaseModel):
    """Person profile card (PRD-043 FR2; GET /api/app/persons/{id}).

    KG-grounded over the whole corpus: ``episodes`` are those whose KG asserts this person's
    node; ``related_people`` / ``related_topics`` are the entities co-occurring most often
    within those episodes (descending). Deliberately lean — no biography, no LLM (consumer
    scope). Empty/404 when the person appears in no episode's KG.
    """

    id: str = Field(description="Canonical person id (person:{slug}).")
    label: str = Field(description="Display name.")
    episode_count: int = Field(ge=0, description="Episodes this person appears in.")
    episodes: list[AppEpisodeSummary] = Field(
        default_factory=list, description="Appears-in episode cards (newest-first)."
    )
    related_people: list[AppEntity] = Field(
        default_factory=list, description="People co-appearing most often (descending)."
    )
    related_topics: list[AppTopic] = Field(
        default_factory=list,
        description="Topics co-occurring most often (descending); cluster-enriched.",
    )


class AppTopicCard(BaseModel):
    """Topic card (PRD-043 FR3; GET /api/app/topics/{id}).

    KG-grounded: ``episodes`` are those whose KG asserts this topic; ``sibling_topics`` are the
    other members of this topic's corpus cluster (same theme, from ``topic_clusters.json``);
    ``related_people`` are the people co-occurring most often within the episodes-about
    (descending). Empty/404 when the topic appears in no episode's KG.
    """

    id: str = Field(description="Canonical topic id (topic:{slug}).")
    label: str = Field(description="Topic display label.")
    cluster_id: str | None = Field(
        default=None, description="Corpus cluster id (graph_compound_parent_id) when clustered."
    )
    cluster_label: str | None = Field(
        default=None, description="Canonical label of the topic's cluster, when clustered."
    )
    cluster_size: int = Field(
        default=0, ge=0, description="Cross-corpus member count of the topic's cluster (0 if none)."
    )
    sibling_topics: list[AppTopic] = Field(
        default_factory=list,
        description="Other topics in the same SEMANTIC ('Similar') cluster.",
    )
    theme_cluster_id: str | None = Field(
        default=None,
        description="Corpus THEME-cluster id (thc:{slug}) — topics discussed together "
        "(co-occurrence), distinct from the semantic cluster_id.",
    )
    theme_cluster_label: str | None = Field(
        default=None, description="Canonical label of the topic's theme cluster, when in one."
    )
    theme_cluster_size: int = Field(
        default=0, ge=0, description="Member count of the topic's theme cluster (0 if none)."
    )
    theme_sibling_topics: list[AppTopic] = Field(
        default_factory=list,
        description="Other topics in the same THEME ('discussed together') cluster.",
    )
    episode_count: int = Field(ge=0, description="Episodes this topic is discussed in.")
    episodes: list[AppEpisodeSummary] = Field(
        default_factory=list, description="Episodes-about cards (newest-first)."
    )
    related_people: list[AppEntity] = Field(
        default_factory=list, description="People co-occurring most often (descending)."
    )


class AppTopicPerspective(BaseModel):
    """One speaker's take on a topic — their grounded insights (#1146)."""

    person_id: str = Field(description="Speaker person id (person:{slug}).")
    person_name: str = Field(description="Speaker display name.")
    insight_count: int = Field(ge=0, description="Number of this speaker's insights on the topic.")
    episode_count: int = Field(ge=0, description="Episodes in which they spoke on the topic.")
    insights: list[AppInsight] = Field(
        default_factory=list, description="Their insights on the topic (position-ordered)."
    )


class AppTopicPerspectivesResponse(BaseModel):
    """Multi-perspective synthesis: each speaker's take on a topic (#1146)."""

    topic_id: str = Field(description="Canonical topic id.")
    topic_label: str = Field(description="Topic display label.")
    perspective_count: int = Field(ge=0, description="Number of distinct speakers with a take.")
    perspectives: list[AppTopicPerspective] = Field(
        default_factory=list, description="Speakers' takes, most-insights first."
    )


class AppInterestCluster(BaseModel):
    """One selectable interest cluster for the discovery picker (PRD-043 FR4 / 3.5)."""

    id: str = Field(description="Cluster id (graph_compound_parent_id, e.g. 'tc:…').")
    label: str = Field(description="Cluster canonical label.")
    size: int = Field(ge=0, description="Cross-corpus member count (prevalence).")


class AppInterestClustersResponse(BaseModel):
    """Top interest clusters for the picker (GET /api/app/clusters)."""

    items: list[AppInterestCluster] = Field(default_factory=list)


class AppStoryline(BaseModel):
    """One THEME cluster ("storyline") — topics discussed together (co-occurrence lift).

    Followable as an interest (``id`` is the ``thc:`` token) and browsable on Home: tapping opens
    ``anchor_topic_id``'s card, whose "discussed together" set is the whole storyline.
    """

    id: str = Field(description="Theme-cluster id (graph_compound_parent_id, 'thc:{slug}').")
    label: str = Field(description="Storyline canonical label.")
    size: int = Field(ge=0, description="Member topic count.")
    anchor_topic_id: str = Field(description="Most-central member topic id — the card to open.")


class AppStorylinesResponse(BaseModel):
    """Top storylines (theme clusters) for the Home rail + picker (GET /api/app/theme-clusters)."""

    items: list[AppStoryline] = Field(default_factory=list)


class AppTrendingEntity(BaseModel):
    """One trending entity (RFC-103 momentum) — velocity (rising) + volume (recent level)."""

    entity_id: str = Field(description="Namespaced id (topic:/tc:/thc:/person:, slug, or feed_id).")
    kind: str = Field(description="topic|cluster|storyline|person|episode|show|insight.")
    label: str = Field(description="Display label.")
    velocity: float = Field(description="Rising signal: fast÷slow EWMA (>1 rising, <1 cooling).")
    volume: float = Field(description="Recent activity level (fast EWMA).")
    heating_up: bool = Field(description="velocity ≥ τ AND total ≥ floor.")
    total: int = Field(description="Total events over the lookback window.")
    series: list[int] = Field(default_factory=list, description="Weekly counts (the sparkline).")


class AppTrendingResponse(BaseModel):
    """Trending entities of one kind (GET /api/app/trending) — read-time momentum vs today."""

    kind: str
    scope: str = Field(description="corpus | mine.")
    as_of_week: str = Field(description="ISO reference week the momentum is anchored to.")
    items: list[AppTrendingEntity] = Field(default_factory=list)


class AppCorpusTrendingResponse(BaseModel):
    """Operator global view (GET /api/corpus/trending) — top momentum per kind, corpus-wide."""

    as_of_week: str
    kinds: dict[str, list[AppTrendingEntity]] = Field(default_factory=dict)


class FavoriteAdd(BaseModel):
    """Body for PUT /api/app/favorites — save a polymorphic item (idempotent on kind+ref)."""

    kind: Literal["episode", "insight", "person", "topic"] = Field(description="Saveable kind.")
    ref: str = Field(description="Stable id within the kind (episode→slug; insight→slug#id).")
    label: str | None = Field(default=None, description="Display label (title / insight text).")
    sublabel: str | None = Field(default=None, description="Secondary label (show / episode).")
    slug: str | None = Field(default=None, description="Episode slug to open (episode/insight).")
    start_ms: int | None = Field(default=None, description="Jump target for an insight (ms).")


class AppFavoriteInsight(BaseModel):
    """A saved insight in the favorites list (snapshot — insights have no global detail route)."""

    ref: str = Field(description="slug#insightId.")
    text: str = Field(description="Insight text.")
    episode_slug: str | None = Field(default=None, description="Episode to open.")
    podcast_title: str | None = Field(default=None, description="Show / episode label.")
    start_ms: int | None = Field(default=None, description="Jump-to-moment (ms).")


class AppFavoritesResponse(BaseModel):
    """The user's favorites, grouped by kind (GET/PUT/DELETE /api/app/favorites)."""

    episodes: list[AppEpisodeSummary] = Field(default_factory=list)
    insights: list[AppFavoriteInsight] = Field(default_factory=list)


class InterestsResponse(BaseModel):
    """The user's saved interest cluster ids (GET /api/app/interests)."""

    items: list[str] = Field(default_factory=list, description="Ordered cluster ids (tc:…).")


class InterestsUpdate(BaseModel):
    """Body for PUT /api/app/interests."""

    items: list[str] = Field(default_factory=list, description="Cluster ids to save.")


# --- P2 Capture: highlights + notes (PRD-040 / RFC-098 §7) ---


class HighlightCreate(BaseModel):
    """Body for POST /api/app/highlights — capture a moment, span, or saved insight."""

    episode_slug: str = Field(description="Episode the highlight belongs to.")
    kind: Literal["span", "moment", "insight"] = Field(description="Capture kind.")
    start_ms: int | None = Field(
        default=None, ge=0, description="Anchor start (ms); the stable key."
    )
    end_ms: int | None = Field(
        default=None, ge=0, description="Anchor end (ms); None for a moment."
    )
    char_start: int | None = Field(default=None, ge=0, description="Transcript char offset start.")
    char_end: int | None = Field(default=None, ge=0, description="Transcript char offset end.")
    segment_ids: list[str] = Field(default_factory=list, description="Overlapping segment ids.")
    quote_text: str | None = Field(default=None, description="Captured verbatim text (spans).")
    speaker: str | None = Field(default=None, description="Speaker label when known.")
    source_insight_id: str | None = Field(
        default=None, description="GIL insight id (insight kind)."
    )
    color: str | None = Field(default=None, description="Highlight colour/label token.")


class HighlightUpdate(BaseModel):
    """Body for PATCH /api/app/highlights/{id} — edit colour / captured text (all optional)."""

    color: str | None = Field(default=None, description="New colour/label token.")
    quote_text: str | None = Field(default=None, description="Edited captured text.")


class Highlight(BaseModel):
    """A saved highlight (response item)."""

    id: str = Field(description="Opaque highlight id.")
    episode_slug: str = Field(description="Episode the highlight belongs to.")
    kind: Literal["span", "moment", "insight"] = Field(description="Capture kind.")
    start_ms: int | None = Field(default=None, description="Anchor start (ms).")
    end_ms: int | None = Field(default=None, description="Anchor end (ms).")
    char_start: int | None = Field(default=None, description="Transcript char offset start.")
    char_end: int | None = Field(default=None, description="Transcript char offset end.")
    segment_ids: list[str] = Field(default_factory=list, description="Overlapping segment ids.")
    quote_text: str | None = Field(default=None, description="Captured verbatim text.")
    speaker: str | None = Field(default=None, description="Speaker label when known.")
    source_insight_id: str | None = Field(
        default=None, description="GIL insight id (insight kind)."
    )
    color: str | None = Field(default=None, description="Highlight colour/label token.")
    created_at: int = Field(description="Unix time captured.")
    anchor_status: str | None = Field(
        default=None, description="'anchored' | 'drifted' after a re-anchor; None until re-scraped."
    )


class HighlightsResponse(BaseModel):
    """The user's highlights (GET/POST/PATCH/DELETE /api/app/highlights)."""

    items: list[Highlight] = Field(default_factory=list)


class NoteCreate(BaseModel):
    """Body for POST /api/app/notes — attach free text to a highlight, insight, or episode."""

    target: Literal["highlight", "insight", "episode"] = Field(description="What the note is on.")
    target_id: str = Field(description="Id/slug of the target.")
    text: str = Field(min_length=1, description="Note body.")


class NoteUpdate(BaseModel):
    """Body for PATCH /api/app/notes/{id}."""

    text: str = Field(min_length=1, description="Edited note body.")


class Note(BaseModel):
    """A saved note (response item)."""

    id: str = Field(description="Opaque note id.")
    target: Literal["highlight", "insight", "episode"] = Field(description="What the note is on.")
    target_id: str = Field(description="Id/slug of the target.")
    text: str = Field(description="Note body.")
    created_at: int = Field(description="Unix time created.")
    updated_at: int = Field(description="Unix time last edited.")


class NotesResponse(BaseModel):
    """The user's notes (GET/POST/PATCH/DELETE /api/app/notes)."""

    items: list[Note] = Field(default_factory=list)


# --- P3 Consolidation: consumer enrichment read surface (RFC-088 envelopes / #1121) ---


class AppEpisodeEnrichmentResponse(BaseModel):
    """Per-episode enrichment signals for the consumer (GET /api/app/episodes/{slug}/enrichment)."""

    slug: str = Field(description="Episode slug.")
    signals: dict[str, Any] = Field(
        default_factory=dict,
        description="Enricher id → its envelope `data` payload (only enrichers that ran OK).",
    )


class AppCorpusEnrichmentResponse(BaseModel):
    """Corpus-scope enrichment signals for the consumer (GET /api/app/corpus/enrichment)."""

    signals: dict[str, Any] = Field(
        default_factory=dict, description="Enricher id → its envelope `data` payload."
    )


# --- P3 Consolidation: spaced resurfacing + derived interests (RFC-101 §5-6 / #1123) ---


class ResurfacingItem(BaseModel):
    """A highlight due to resurface, with a reflection prompt (GET /api/app/resurfacing)."""

    highlight: Highlight = Field(description="The due highlight (jump-to-moment via its anchor).")
    reflection_prompt: str = Field(description="A deterministic, no-LLM reflection prompt.")


class ResurfacingResponse(BaseModel):
    """Due resurfacing items, most-overdue first (empty when paused / nothing due)."""

    items: list[ResurfacingItem] = Field(default_factory=list)
    paused: bool = Field(default=False, description="Whether the user has paused resurfacing.")


class ResurfacingSettings(BaseModel):
    """Pacing settings (GET/PUT /api/app/resurfacing/settings)."""

    paused: bool = Field(default=False, description="Pause all resurfacing.")


class DerivedInterest(BaseModel):
    """An implicit interest token derived from the user's corpus (RFC-101 §6)."""

    token: str = Field(description="`person:<id>` / `topic:<id>` — same scheme as explicit follows")
    kind: Literal["person", "topic"] = Field(description="Entity kind.")
    label: str = Field(description="Display label.")
    count: int = Field(ge=1, description="How many heard∪captured episodes it occurs in.")


class DerivedInterestsResponse(BaseModel):
    """Ranked implicit interests (GET /api/app/interests/derived)."""

    items: list[DerivedInterest] = Field(default_factory=list)


class PlaybackPosition(BaseModel):
    """Per-user playback position for one episode."""

    slug: str = Field(description="Episode slug.")
    position_seconds: float = Field(ge=0, description="Saved playback position in seconds.")
    updated_at: int | None = Field(default=None, description="Unix time of the last save.")


class PlaybackUpdate(BaseModel):
    """Body for PUT /api/app/playback/{slug}."""

    position_seconds: float = Field(ge=0, description="Playback position in seconds.")


class PlaybackListResponse(BaseModel):
    """All saved playback positions (Home 'Continue listening')."""

    items: list[PlaybackPosition] = Field(default_factory=list)


class StatPoint(BaseModel):
    """One day bucket of a listening sparkline (UXS-014)."""

    date: str = Field(description="UTC calendar day, ISO 'YYYY-MM-DD'.")
    count: int = Field(ge=0, description="Opens on that day.")


class UserStatsResponse(BaseModel):
    """The signed-in user's own listening summary — GET /api/app/me/stats (PRD-043 / RFC-102)."""

    episodes: int = Field(ge=0, description="Distinct episodes opened / in progress.")
    shows: int = Field(ge=0, description="Distinct shows listened to.")
    listening_seconds: float = Field(
        ge=0, description="Estimated time invested (sum of furthest playback positions)."
    )
    active_days: int = Field(ge=0, description="Distinct days with at least one open.")
    day_streak: int = Field(ge=0, description="Current consecutive-day listening run.")
    daily: list[StatPoint] = Field(default_factory=list, description="Daily opens sparkline.")


class EpisodeStatsResponse(BaseModel):
    """Cross-user reach for one episode — GET /api/app/episodes/{slug}/stats (PRD-043 / RFC-102)."""

    slug: str = Field(description="Episode slug.")
    listeners: int = Field(ge=0, description="Distinct people who have opened this episode.")
    opens: int = Field(ge=0, description="Total opens across everyone.")
    insights: int = Field(ge=0, description="Grounded insights available for the episode.")
    daily: list[StatPoint] = Field(default_factory=list, description="Daily opens sparkline.")


class AppPodcastItem(BaseModel):
    """One show in the user's library (Home 'Your shows')."""

    feed_id: str = Field(description="Feed id.")
    title: str | None = Field(default=None, description="Show display title.")
    artwork_url: str | None = Field(
        default=None, description="Preferred (local) show artwork, thumb size."
    )
    image_url: str | None = Field(default=None, description="Remote feed image URL — fallback.")
    description: str | None = Field(default=None, description="Show description/blurb when known.")
    episode_count: int = Field(ge=0, default=0, description="Episodes available for this show.")


class AppPodcastsResponse(BaseModel):
    """Response for GET /api/app/podcasts — distinct shows in the corpus."""

    items: list[AppPodcastItem] = Field(default_factory=list)


class QueueResponse(BaseModel):
    """The user's play queue (ordered episode slugs)."""

    items: list[str] = Field(default_factory=list)


class QueueUpdate(BaseModel):
    """Body for PUT /api/app/queue."""

    items: list[str] = Field(default_factory=list, description="Ordered episode slugs.")


class LibraryItem(BaseModel):
    """A subscribed podcast in the user's library."""

    feed_id: str = Field(description="Feed id.")
    feed_url: str | None = Field(default=None, description="RSS feed URL.")
    title: str | None = Field(default=None, description="Show title.")
    added_at: int | None = Field(default=None, description="Unix time subscribed.")


class LibraryAdd(BaseModel):
    """Body for POST /api/app/library."""

    feed_id: str = Field(description="Feed id to subscribe.")
    feed_url: str | None = Field(default=None, description="RSS feed URL.")
    title: str | None = Field(default=None, description="Show title.")


class LibraryResponse(BaseModel):
    """The user's library (subscribed podcasts)."""

    items: list[LibraryItem] = Field(default_factory=list)


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


class CorpusProducedBy(BaseModel):
    """Corpus stamp from ``corpus_manifest.json`` ``produced_by`` (GitHub #796)."""

    code_version: str = Field(description="Semver of podcast_scraper that produced the corpus.")
    git_sha: str = Field(description="Short git commit SHA from the producing pipeline run.")
    produced_at: str = Field(description="UTC ISO-8601 timestamp when the stamp was written.")


class HealthResponse(BaseModel):
    """Response for GET /api/health."""

    status: Literal["ok"] = "ok"
    code_version: str = Field(
        default="",
        description="Running server package version (``podcast_scraper.__version__``).",
    )
    min_supported_corpus_code_version: str = Field(
        default="",
        description="Minimum ``produced_by.code_version`` the server supports without warning.",
    )
    corpus_produced_by: CorpusProducedBy | None = Field(
        default=None,
        description="Stamp from default corpus manifest when ``output_dir`` is configured.",
    )
    corpus_code_version: str | None = Field(
        default=None,
        description="Effective corpus semver (from ``produced_by`` or legacy ``tool_version``).",
    )
    corpus_version_warning: str | None = Field(
        default=None,
        description="Non-fatal mismatch between on-disk corpus and server expectations.",
    )
    artifacts_api: bool = Field(
        default=True,
        description="True when GET /api/artifacts (GI/KG list + load for graph) is mounted.",
    )
    search_api: bool = Field(
        default=True,
        description="True when GET /api/search (semantic search) is mounted.",
    )
    enriched_search_available: bool = Field(
        default=False,
        description=(
            "True when optional semantic-search enrichment (e.g. LLM join) is configured; "
            "viewer shows Enhanced chip when true."
        ),
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
    feeds_api: bool = Field(
        default=False,
        description=(
            "True when GET/PUT /api/feeds is mounted " "(``feeds.spec.yaml`` under corpus root)."
        ),
    )
    operator_config_api: bool = Field(
        default=False,
        description="True when GET/PUT /api/operator-config is mounted (non-secret YAML only).",
    )
    jobs_api: bool = Field(
        default=False,
        description="True when POST/GET /api/jobs and related pipeline job routes are mounted.",
    )


class FeedsListResponse(BaseModel):
    """Response for GET/PUT /api/feeds (structured ``{ feeds: [...] }`` on disk)."""

    path: str = Field(description="Resolved absolute corpus root path.")
    file_relpath: str = Field(description="Feeds spec file relative to corpus root (POSIX).")
    feeds: list[str | dict[str, Any]] = Field(
        default_factory=list,
        description="Feed entries in file order (URL strings or objects with url + overrides).",
    )


class FeedsPutBody(BaseModel):
    """Body for PUT /api/feeds."""

    feeds: list[str | dict[str, Any]] = Field(
        default_factory=list,
        description="Feed entries to persist (deduped by url, first-seen order kept).",
    )


class OperatorConfigGetResponse(BaseModel):
    """Response for GET/PUT /api/operator-config."""

    corpus_path: str = Field(description="Resolved corpus root from the path query parameter.")
    operator_config_path: str = Field(
        description="Absolute path of the operator YAML file on disk."
    )
    content: str = Field(
        description=(
            "Full file contents (UTF-8). GET may copy a packaged overrides-only starter "
            "when the file was missing or whitespace-only; ``profile:`` is chosen in the "
            "viewer (or CLI ``--profile``), not seeded into that starter."
        )
    )
    available_profiles: list[str] = Field(
        default_factory=list,
        description=(
            "Sorted packaged preset names (no .yaml) unioned from cwd and repo "
            "config/profiles/ (same roots as Config profile load); excludes *.example.yaml."
        ),
    )
    default_profile: str | None = Field(
        default=None,
        description=(
            "Preferred profile when the corpus has no ``profile:`` saved yet. Sourced "
            "from the ``PODCAST_DEFAULT_PROFILE`` env var; the viewer preselects this "
            "in the operator dropdown. ``null`` when unset (dev / CI default — no "
            "preselection, dropdown opens with 'None'). Always one of "
            "``available_profiles`` or ``null`` (validated server-side)."
        ),
    )


class OperatorConfigPutBody(BaseModel):
    """Body for PUT /api/operator-config."""

    content: str = Field(default="", description="Full YAML file contents (UTF-8).")


class PackagedProfile(BaseModel):
    """One packaged pipeline profile and its YAML body."""

    name: str = Field(description="Profile name (no .yaml).")
    content: str = Field(description="Raw profile YAML (providers / models / rationale).")


class OperatorProfilesResponse(BaseModel):
    """Response for GET /api/operator-config/profiles — packaged profiles + bodies."""

    profiles: list[PackagedProfile] = Field(
        default_factory=list,
        description="Packaged profiles (same allowlist as available_profiles) with content.",
    )


class PipelineJobRecord(BaseModel):
    """One row from the JSONL job registry (GET list/detail, cancel response)."""

    model_config = ConfigDict(extra="ignore")

    job_id: str
    command_type: str = Field(default="full_incremental_pipeline")
    status: str
    created_at: str
    started_at: str | None = None
    ended_at: str | None = None
    pid: int | None = None
    argv_summary: str = ""
    exit_code: int | None = None
    log_relpath: str = ""
    error_reason: str | None = None
    cancel_requested: bool = False
    queue_position: int | None = Field(
        default=None,
        description="1-based position among queued jobs for this corpus (omit when not queued).",
    )


class PipelineJobAccepted(BaseModel):
    """Response for POST /api/jobs (202 Accepted)."""

    job_id: str
    status: str
    corpus_path: str
    queue_position: int | None = None


class PipelineJobsListResponse(BaseModel):
    """Response for GET /api/jobs."""

    path: str
    jobs: list[PipelineJobRecord] = Field(default_factory=list)


class ScheduledJobItem(BaseModel):
    """One scheduled feed-sweep entry (#708)."""

    name: str
    cron: str
    enabled: bool
    kind: str = Field(
        default="pipeline",
        description="Job fired on the schedule: ``pipeline`` (ingestion) or ``enrichment``.",
    )
    next_run_at: str | None = Field(
        default=None,
        description=(
            "Next scheduled fire time (UTC ISO-8601). ``null`` when the job is "
            "disabled, the cron expression is invalid, or the scheduler hasn't "
            "started yet."
        ),
    )


class ScheduledJobsListResponse(BaseModel):
    """Response for GET /api/scheduled-jobs."""

    path: str
    scheduler_running: bool
    timezone: str
    jobs: list[ScheduledJobItem] = Field(default_factory=list)


class PipelineJobLogTailResponse(BaseModel):
    """Tail of a job subprocess log (UTF-8) for dashboard previews."""

    text: str = ""
    truncated: bool = Field(
        default=False,
        description=(
            "True when the file was larger than ``max_bytes`` and leading bytes were skipped."
        ),
    )


class PipelineJobReconcileResponse(BaseModel):
    """Response for POST /api/jobs/reconcile."""

    path: str
    updated: int = Field(ge=0)
    details: list[str] = Field(default_factory=list)


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


class IndexTimeseriesMonth(BaseModel):
    """One month bucket of indexed documents, keyed by doc_type."""

    month: str = Field(description="Publish month, YYYY-MM.")
    doc_types: dict[str, int] = Field(default_factory=dict)


class IndexTimeseriesResponse(BaseModel):
    """Response for GET /api/index/timeseries — indexed docs by publish month × doc_type."""

    available: bool = True
    by_month: list[IndexTimeseriesMonth] = Field(default_factory=list)
    doc_types: list[str] = Field(
        default_factory=list,
        description="All doc_type keys present across buckets, sorted (stable series order).",
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
    source_tier: str = Field(
        default="aux",
        description=(
            "Retrieval tier for this hit (PRD-033 FR1.1): 'insight' (synthesized), "
            "'segment' (raw transcript), or 'aux' (kg_entity/kg_topic/quote/summary). "
            "Derived from metadata.doc_type; stable across all retrieval paths."
        ),
    )
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
    query_type: str | None = Field(
        default=None,
        description=(
            "Detected query intent (PRD-033 FR1.4): entity_lookup / raw_evidence / "
            "temporal_tracking / cross_show_synthesis / semantic. From the rules router "
            "(RFC-090 §3.6); null on error responses."
        ),
    )
    lift_stats: CorpusSearchLiftStatsModel | None = Field(
        default=None,
        description="Transcript lift coverage for this response page (#528).",
    )


class RelatedNodeModel(BaseModel):
    """One corpus-graph node projected for a relational-query result (RFC-094 / #882)."""

    id: str
    type: str
    text: str = ""
    show_id: str = ""
    episode_id: str = ""


class RelationalListResponse(BaseModel):
    """Response for the flat relational endpoints (positions / insights-about / …)."""

    subject: str = Field(description="The queried subject id (person/entity/insight/podcast).")
    results: list[RelatedNodeModel] = Field(default_factory=list)
    error: str | None = None


class RelationalGroupedResponse(BaseModel):
    """Response for the grouped relational endpoints (who-said / cross-show).

    ``groups`` maps a grouping key (person id for who-said, podcast/show id for
    cross-show) to the related nodes in that group.
    """

    subject: str = Field(description="The queried topic id.")
    groups: dict[str, list[RelatedNodeModel]] = Field(default_factory=dict)
    error: str | None = None


class InsightDetailResponse(BaseModel):
    """An insight's own content, resolved from the full corpus graph (out-of-slice).

    Lets a viewer render an insight whose node isn't in the loaded artifact — e.g.
    drilling into a corpus-wide timeline mention. ``error`` is set (results empty)
    when no corpus is configured or the id isn't an insight.
    """

    subject: str = Field(description="The queried insight id.")
    text: str = ""
    insight_type: str = ""
    grounded: bool = False
    episode_id: str = ""
    show_id: str = ""
    quotes: list[RelatedNodeModel] = Field(default_factory=list)
    topics: list[RelatedNodeModel] = Field(default_factory=list)
    entities: list[RelatedNodeModel] = Field(default_factory=list)
    error: str | None = None


class QueryActivityBucket(BaseModel):
    """One day's search count (PRD-033 FR6.2)."""

    date: str = Field(description="UTC calendar day, YYYY-MM-DD.")
    count: int = Field(ge=0)


class QueryActivityResponse(BaseModel):
    """Response for GET /api/corpus/query-activity — daily search volume."""

    total: int = Field(default=0, ge=0)
    buckets: list[QueryActivityBucket] = Field(default_factory=list)
    error: str | None = None


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
    # RFC-088 chunk-8 follow-up: per-episode enrichment availability flags.
    # Populated from `metadata/enrichments/{stem}.{id}.json` presence for
    # each episode-scope enricher (currently insight_density). Cheap probe —
    # caller knows whether a drill-down to
    # `/api/corpus/episode/enrichments/{id}` will return a 200 or 404.
    enrichments_available: dict[str, bool] = Field(
        default_factory=dict,
        description=(
            "Per-enricher availability flags for episode-scope enrichers " "(e.g. insight_density)."
        ),
    )


class CorpusEpisodesResponse(BaseModel):
    """Response for GET /api/corpus/episodes."""

    path: str
    feed_id: str | None = Field(
        default=None,
        description="Echo of filter when set; null when listing all feeds.",
    )
    items: list[CorpusEpisodeListItem] = Field(default_factory=list)
    next_cursor: str | None = None
    total: int = Field(
        default=0,
        ge=0,
        description=(
            "Total cumulative-unique episode count across all runs that match the "
            "request's filters (feed_id, q, topic_q, since, until, has_gi, "
            "topic_cluster_only). Independent of pagination ``limit``/``cursor`` — "
            "callers can render 'X of Y' counts (v2.6.1 #818/#819)."
        ),
    )


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
    bridge_partition: "BridgePartitionSummary | None" = Field(
        default=None,
        description=(
            "Per-episode ``{gi_only, kg_only, both}`` identity partition counts "
            "derived from ``bridge.json::identities``. ``None`` when the bridge "
            "file is missing or unreadable. Meaningful for the first time "
            "post-#654 (previous threshold produced a mechanical ``both = 10 × "
            "episode_count`` distribution)."
        ),
    )


class BridgePartitionSummary(BaseModel):
    """Counts of identity membership across GI / KG layers in bridge.json.

    #656 Stage B: drives the per-episode bridge indicator on
    ``EpisodeDetailPanel``. Each identity (topic / person / org) is
    classified from its ``sources: {gi, kg}`` flags:

      * ``gi_only`` — ``gi=True``, ``kg=False``.
      * ``kg_only`` — ``gi=False``, ``kg=True``.
      * ``both``    — both flags ``True``.
      * ``total``   — sum of the three above (identities present in
        neither layer are never emitted by the builder, so this also
        equals ``len(identities)``).

    Per-type splits (topic / person / org) are reserved for a follow-up
    surface (hover detail) — intentionally omitted here.
    """

    gi_only: int = Field(ge=0)
    kg_only: int = Field(ge=0)
    both: int = Field(ge=0)
    total: int = Field(ge=0)


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
    # #656 Stage B: surface the #652 Part B post-extraction filter counters so
    # the dashboard can render a "Pipeline cleanup" summary per run. Each
    # counter is a total across the run; ``None`` means the field was absent
    # (legacy run or metrics.json schema mismatch).
    ads_filtered_count: int | None = None
    dialogue_insights_dropped_count: int | None = None
    topics_normalized_count: int | None = None
    entity_kinds_repaired_count: int | None = None
    # #656 Stage D: pre-extraction ad-region excision counters (#663). Each
    # is a run total; ``None`` on runs predating the counters.
    ad_chars_excised_preroll: int | None = None
    ad_chars_excised_postroll: int | None = None
    ad_episodes_with_excision_count: int | None = None


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
    summary_title: str | None = None
    summary_text: str | None = None
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


class CilTopicPerspectiveLeader(BaseModel):
    """A multi-perspective topic for the dashboard (#1146)."""

    topic_id: str
    topic_label: str
    speaker_count: int = Field(ge=0)
    insight_count: int = Field(ge=0)


class CilTopicPerspectiveLeadersResponse(BaseModel):
    """Response for GET /api/topics/perspective-leaders (#1146)."""

    path: str
    topics: list[CilTopicPerspectiveLeader] = Field(default_factory=list)


class CilTopicPerspective(BaseModel):
    """One speaker's take on a topic — their grounded insights (operator CIL view; #1146)."""

    person_id: str
    person_name: str
    insight_count: int = Field(ge=0)
    episode_count: int = Field(ge=0)
    insights: list[dict[str, Any]] = Field(default_factory=list)  # raw GI Insight nodes


class CilTopicPerspectivesResponse(BaseModel):
    """Response for GET /api/topics/{topic_id}/perspectives (#1146)."""

    path: str
    topic_id: str
    perspectives: list[CilTopicPerspective] = Field(default_factory=list)


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
