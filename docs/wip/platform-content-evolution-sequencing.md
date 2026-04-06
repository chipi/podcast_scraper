# Platform ↔ Content Evolution: Sequencing and Anticipation Guide

**Status:** Cross-analysis of two architecture documents — identifies what to
do first, what to anticipate, and what to defer.

**Date:** 2026-04-05

**Input documents:**

- [Platform Architecture Blueprint](../architecture/PLATFORM_ARCHITECTURE_BLUEPRINT.md)
  — infrastructure evolution (tenancy, workers, queues, Postgres, deployment)
- [Content Evolution Note](platform-extensibility-architecture-note.md) —
  content-type evolution (generic core, pluggable content types + transports)

**Purpose:** The platform blueprint and content evolution are two separate
efforts with different timelines. This document maps their dependencies and
identifies the **specific design decisions** during platform work (v2.x) that
must anticipate content evolution (v3.0) to avoid painful migrations later.

---

## Executive Summary

**Do platform first, content evolution later.** But make four specific
anticipations during platform work — each costs almost nothing now (a column
name, a field in a payload) but saves schema migrations and rework later.

| # | Anticipation | Where It Applies | Cost Now | Cost If Skipped |
| --- | --- | --- | --- | --- |
| 1 | Generic content identity in Postgres | RFC-051 schema | Column naming | `ALTER TABLE` on every projected table + all queries |
| 2 | Generic catalog/source model | Part A catalog tables | One extra column | New table + migration + subscription model rework |
| 3 | Generic job payload envelope | Part B worker jobs | JSON field naming | Job schema migration + worker code changes |
| 4 | Generic pipeline fingerprint key | A.4 dedup logic | Variable naming | Dedup key migration + reprocessing risk |

---

## Sequencing: What Goes When

### Phase 1: v2.x — Platform Infrastructure (Podcast-Only)

All of these proceed as planned. Content evolution does not block or change
them, except for the four anticipations noted below.

| Item | Blueprint Ref | Status | Content Evolution Impact |
| --- | --- | --- | --- |
| GI/KG Viewer v2 | RFC-062, Phase D | Current work | **None.** Pure podcast UI. |
| Semantic search | RFC-061 | In progress | **None.** Operates on text + embeddings, already generic. |
| Evaluation framework | RFC-041/057 | In progress | **None.** Needed later for content-type prompt tuning. |
| Postgres projection | RFC-051, Phase C | Next major | **ANTICIPATE** — see Anticipation #1 and #2 below. |
| Catalog + subscriptions | Part A, Phase A | Planned | **ANTICIPATE** — see Anticipation #2 below. |
| Workers + queues | Part B, simple tier | Planned | **ANTICIPATE** — see Anticipation #3 below. |
| Pipeline fingerprinting | A.4 | Planned | **ANTICIPATE** — see Anticipation #4 below. |
| Adaptive routing | RFC-053 | Planned | **Design for content type input** — add `content_type` as a routing dimension in the routing interface, even if only `"podcast"` exists at first. |
| Docker Compose | Part B | Planned | **None.** Container topology is content-agnostic. |
| Observability | Part E | Planned | **None.** Metrics and logging are content-agnostic. |
| Auth (stages 1-2) | A.12 | Planned | **None.** API key auth is content-agnostic. |
| Alembic migrations | B.16 | Planned | **None.** Migration tooling is content-agnostic. Having Alembic from day one makes the v3.0 schema changes easier. |
| Digest features | Part C | Planned | **None for v0/v1.** Digest operates on projected tables — if those tables use generic naming (Anticipation #1), digest queries generalize automatically. |

### Phase 2: v3.0 — Content Evolution Refactoring

Happens after platform v1 is operational. Depends on platform infrastructure
being in place.

| Item | Content Evolution Ref | Depends On (Platform) |
| --- | --- | --- |
| Core protocols + registry | Phase 1-2 | Postgres schema (to know what `ContentItem` maps to in DB) |
| Podcast + RSS as plugins | Phase 2-3 | Nothing — internal refactoring |
| Module reorganization | Phase 4 | Nothing — internal refactoring |
| ProcessingConfig extraction | Phase 5 | Worker implementation (to know what config workers consume) |
| Multi-source config (`sources:` field) | Phase 5 | Catalog model in Postgres (to know how sources are stored) |
| CLI plugin commands | Phase 6 | Registry implementation (Phase 1) |

### Phase 3: v3.x — Cross-Content Features

Happens after at least one external module (e.g., news-ingest) is built.

| Item | Content Evolution Ref | Depends On |
| --- | --- | --- |
| Cross-content-type KG | F.3 | Postgres projection + semantic search + at least 2 content types producing KG |
| Entity resolution | F.3 | Cross-content KG + canonical entity registry design |
| Viewer v3 (mixed content) | F.2 | Multiple content types in production + UX spec |
| Content-type-aware scheduling | F.5 | Workers operational + metrics showing contention across types |
| Dedicated pipelines | F.5 | Distributed tier (D.7 v2) + content-type queue routing |

---

## Anticipation #1: Postgres Schema — Generic Content Identity

**When:** During RFC-051 implementation (Postgres projection).

**What the blueprint currently implies:**

The blueprint (A.9 Phase C, ADR-054) defines projected tables for GI/KG data.
The natural naming based on the current codebase would be:

```sql
-- What you'd write if only thinking about podcasts:
CREATE TABLE insights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    episode_id TEXT NOT NULL,
    feed_id TEXT NOT NULL,
    insight_text TEXT NOT NULL,
    category TEXT,
    grounding_status TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE quotes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    episode_id TEXT NOT NULL,
    feed_id TEXT NOT NULL,
    quote_text TEXT NOT NULL,
    speaker TEXT,
    start_offset INTEGER,
    end_offset INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE insight_support (
    insight_id UUID NOT NULL REFERENCES insights(id),
    quote_id UUID NOT NULL REFERENCES quotes(id),
    confidence FLOAT,
    PRIMARY KEY (insight_id, quote_id)
);

CREATE TABLE kg_nodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    episode_id TEXT NOT NULL,
    feed_id TEXT NOT NULL,
    label TEXT NOT NULL,
    node_type TEXT NOT NULL,
    properties JSONB
);

CREATE TABLE kg_edges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    episode_id TEXT NOT NULL,
    source_node_id UUID NOT NULL REFERENCES kg_nodes(id),
    target_node_id UUID NOT NULL REFERENCES kg_nodes(id),
    relation_type TEXT NOT NULL,
    properties JSONB
);
```

**What to do instead:**

Replace `episode_id` and `feed_id` with generic names. Add `content_type`
and `source_transport` columns for filtering and faceting.

```sql
CREATE TABLE insights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_item_id TEXT NOT NULL,       -- was: episode_id
    source_id TEXT NOT NULL,             -- was: feed_id
    content_type TEXT NOT NULL DEFAULT 'podcast',
    source_transport TEXT NOT NULL DEFAULT 'rss',
    insight_text TEXT NOT NULL,
    category TEXT,
    grounding_status TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_insights_content_item ON insights(content_item_id);
CREATE INDEX idx_insights_source ON insights(source_id);
CREATE INDEX idx_insights_content_type ON insights(content_type);

CREATE TABLE quotes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_item_id TEXT NOT NULL,       -- was: episode_id
    source_id TEXT NOT NULL,             -- was: feed_id
    content_type TEXT NOT NULL DEFAULT 'podcast',
    source_transport TEXT NOT NULL DEFAULT 'rss',
    quote_text TEXT NOT NULL,
    speaker TEXT,
    start_offset INTEGER,
    end_offset INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_quotes_content_item ON quotes(content_item_id);
CREATE INDEX idx_quotes_source ON quotes(source_id);

CREATE TABLE insight_support (
    insight_id UUID NOT NULL REFERENCES insights(id),
    quote_id UUID NOT NULL REFERENCES quotes(id),
    confidence FLOAT,
    PRIMARY KEY (insight_id, quote_id)
);

CREATE TABLE kg_nodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_item_id TEXT NOT NULL,       -- was: episode_id
    source_id TEXT NOT NULL,             -- was: feed_id
    content_type TEXT NOT NULL DEFAULT 'podcast',
    label TEXT NOT NULL,
    node_type TEXT NOT NULL,
    properties JSONB
);

CREATE INDEX idx_kg_nodes_content_item ON kg_nodes(content_item_id);
CREATE INDEX idx_kg_nodes_source ON kg_nodes(source_id);
CREATE INDEX idx_kg_nodes_content_type ON kg_nodes(content_type);

CREATE TABLE kg_edges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_item_id TEXT NOT NULL,       -- was: episode_id
    source_node_id UUID NOT NULL REFERENCES kg_nodes(id),
    target_node_id UUID NOT NULL REFERENCES kg_nodes(id),
    relation_type TEXT NOT NULL,
    properties JSONB
);

CREATE INDEX idx_kg_edges_content_item ON kg_edges(content_item_id);
```

**Also apply to these tables** (if they exist in the projection schema):

```sql
CREATE TABLE content_items (
    content_item_id TEXT PRIMARY KEY,    -- was: episode_id / episodes.id
    source_id TEXT NOT NULL,             -- was: feed_id
    content_type TEXT NOT NULL DEFAULT 'podcast',
    source_transport TEXT NOT NULL DEFAULT 'rss',
    title TEXT NOT NULL,
    published_at TIMESTAMPTZ,
    metadata JSONB,                      -- content-type-specific metadata
    pipeline_fingerprint TEXT,           -- for dedup (A.4)
    processing_status TEXT NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_content_items_source ON content_items(source_id);
CREATE INDEX idx_content_items_type ON content_items(content_type);
CREATE INDEX idx_content_items_status ON content_items(processing_status);

CREATE TABLE summaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_item_id TEXT NOT NULL REFERENCES content_items(content_item_id),
    summary_text TEXT NOT NULL,
    provider TEXT NOT NULL,              -- "openai", "ml", "hybrid_ml"
    model TEXT,
    pipeline_fingerprint TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_summaries_content_item ON summaries(content_item_id);
```

**What this costs now:** Nothing functional. It's column names and two extra
columns (`content_type`, `source_transport`) with defaults. All queries work
the same — `WHERE content_item_id = ?` instead of `WHERE episode_id = ?`.
The defaults mean existing podcast data populates correctly without changes.

**What skipping this costs later:** `ALTER TABLE` on every projected table to
rename columns. Update every SQL query, every ORM model, every API response
schema. Backfill `content_type` column on existing rows. Risk of breaking
the viewer, digest queries, and any downstream consumers during migration.

**Python code naming convention:**

In the projection code (the Python that writes to these tables), use the
same generic naming:

```python
# Instead of:
def project_episode(episode_id: str, feed_id: str, gi_data: dict) -> None:

# Write:
def project_content_item(
    content_item_id: str,
    source_id: str,
    content_type: str = "podcast",
    gi_data: dict | None = None,
) -> None:
```

---

## Anticipation #2: Catalog Table — Generic Source Model

**When:** During Part A Phase A implementation (catalog + subscriptions).

**What the blueprint currently implies:**

```sql
CREATE TABLE feeds (
    feed_id TEXT PRIMARY KEY,            -- hash of rss_url (ADR-003)
    rss_url TEXT NOT NULL UNIQUE,
    title TEXT,
    description TEXT,
    last_polled_at TIMESTAMPTZ,
    poll_interval_seconds INTEGER DEFAULT 3600,
    enabled BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE subscriptions (
    tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id),
    feed_id TEXT NOT NULL REFERENCES feeds(feed_id),
    subscribed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, feed_id)
);
```

**What to do instead:**

```sql
CREATE TABLE sources (
    source_id TEXT PRIMARY KEY,          -- was: feed_id
    source_type TEXT NOT NULL DEFAULT 'rss_feed',
        -- 'rss_feed', 'web_scrape_target', 'social_account', 'folder_watch'
    transport TEXT NOT NULL DEFAULT 'rss',
        -- 'rss', 'web_scrape', 'mastodon', 'reddit', 'filesystem'
    content_type TEXT NOT NULL DEFAULT 'podcast',
        -- 'podcast', 'news_article', 'social_post', 'document'
    title TEXT,
    description TEXT,
    config JSONB NOT NULL,
        -- Transport-specific config. For RSS: {"rss_url": "https://..."}
        -- For web scrape: {"urls": [...], "selector": "article.main"}
        -- Validated against transport manifest's config_schema (v3.0)
    last_polled_at TIMESTAMPTZ,
    poll_interval_seconds INTEGER DEFAULT 3600,
    enabled BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_sources_type ON sources(source_type);
CREATE INDEX idx_sources_transport ON sources(transport);
CREATE INDEX idx_sources_enabled ON sources(enabled) WHERE enabled = true;

-- For backwards compatibility, a view that looks like the old feeds table:
CREATE VIEW feeds AS
SELECT
    source_id AS feed_id,
    config->>'rss_url' AS rss_url,
    title,
    description,
    last_polled_at,
    poll_interval_seconds,
    enabled,
    created_at
FROM sources
WHERE source_type = 'rss_feed';

CREATE TABLE subscriptions (
    tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id),
    source_id TEXT NOT NULL REFERENCES sources(source_id),  -- was: feed_id
    subscribed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, source_id)
);
```

**What this costs now:** The `sources` table is slightly more generic than
`feeds` — one extra column (`source_type`), `config` as JSONB instead of
`rss_url` as a dedicated column, and generic naming. The `feeds` view
provides backwards compatibility for any code that expects the old shape.

**What skipping this costs later:** New `sources` table + data migration from
`feeds`. Update `subscriptions` foreign key from `feed_id` to `source_id`.
Update all catalog API endpoints. Update the scheduler that polls feeds.
Update the viewer's feed list. Potentially break the subscription model if
`feed_id` is baked into multiple layers.

**Subscription model stays the same** — it's tenant ↔ source. The only
change is the column name (`source_id` instead of `feed_id`).

**Scheduler impact:** The scheduler (B.7.1) that polls feeds becomes
"poll sources." The query changes from:

```sql
-- Before:
SELECT * FROM feeds WHERE enabled = true AND last_polled_at < now() - interval '1 hour';

-- After:
SELECT * FROM sources WHERE enabled = true AND last_polled_at < now() - poll_interval_seconds * interval '1 second';
```

The scheduler doesn't need to know about content types — it just polls
enabled sources and enqueues jobs. The worker determines what to do based
on `source_type`, `transport`, and `content_type`.

---

## Anticipation #3: Job Payload — Generic Envelope

**When:** During Part B implementation (workers + queues, arq job schema).

**What the blueprint currently implies:**

The job payload for the `heavy` queue would naturally look like:

```python
@dataclass
class PipelineJob:
    feed_id: str
    episode_guid: str
    rss_url: str
    config: dict  # serialized Config fields
```

**What to do instead:**

```python
@dataclass
class PipelineJob:
    # Generic content identity
    content_item_id: str                 # was: episode_guid
    source_id: str                       # was: feed_id
    content_type: str = "podcast"        # "podcast", "news_article", etc.
    source_transport: str = "rss"        # "rss", "web_scrape", etc.

    # Source-specific config (transport needs this to fetch/process)
    source_config: dict = field(default_factory=dict)
        # For RSS: {"rss_url": "https://...", "episode_guid": "..."}
        # For web scrape: {"url": "https://...", "selector": "..."}

    # Processing config (what to do with the content)
    processing_config: dict = field(default_factory=dict)
        # {"summary_provider": "openai", "gi_enabled": true, ...}

    # Pipeline fingerprint for dedup (A.4)
    pipeline_fingerprint: str | None = None

    # Job metadata
    created_at: str = ""                 # ISO 8601
    retry_count: int = 0
    max_retries: int = 3
```

**JSON representation (what goes into Redis via arq):**

```json
{
  "content_item_id": "npr-planet-money-ep1234",
  "source_id": "rss_feeds.npr.org_abc123",
  "content_type": "podcast",
  "source_transport": "rss",
  "source_config": {
    "rss_url": "https://feeds.npr.org/510289/podcast.xml",
    "episode_guid": "urn:uuid:abc-123"
  },
  "processing_config": {
    "summary_provider": "openai",
    "gi_enabled": true,
    "kg_enabled": true,
    "whisper_model": "base"
  },
  "pipeline_fingerprint": "sha256:abc123...",
  "created_at": "2026-04-05T12:00:00Z",
  "retry_count": 0,
  "max_retries": 3
}
```

**Worker code that consumes this:**

```python
async def process_job(ctx: dict, job: PipelineJob) -> None:
    """Worker job handler — content-type-agnostic envelope."""
    # Build Config from job payload (v2.x: always podcast)
    cfg = build_config_from_job(job)

    # Run the pipeline
    result = run_pipeline(cfg)

    # Project results to Postgres
    project_content_item(
        content_item_id=job.content_item_id,
        source_id=job.source_id,
        content_type=job.content_type,
        artifacts=result.artifacts,
    )
```

In v2.x, `build_config_from_job` always builds a podcast `Config` from the
job payload. In v3.0, it dispatches based on `content_type` and
`source_transport` to build the right config via the plugin registry.

**What this costs now:** The job payload has four extra string fields
(`content_type`, `source_transport`, `source_config`, `pipeline_fingerprint`)
that are always `"podcast"`, `"rss"`, `{rss_url, episode_guid}`, and a hash.
The worker ignores `content_type` and `source_transport` in v2.x — they're
just metadata. The `source_config` dict replaces what would have been
top-level `rss_url` and `episode_guid` fields.

**What skipping this costs later:** Job payload schema migration. All
in-flight jobs in Redis become incompatible. Worker code needs to handle
both old and new payload formats during rollout. Dead-letter queue items
from the old format need manual reprocessing or a migration script.

---

## Anticipation #4: Pipeline Fingerprint — Generic Dedup Key

**When:** During A.4 implementation (process once, serve many).

**What the blueprint currently implies:**

The dedup key is `(episode_key, pipeline_fingerprint)` where `episode_key`
is derived from `feed_url + episode_guid` (ADR-007).

```python
def make_dedup_key(feed_url: str, episode_guid: str, fingerprint: str) -> str:
    episode_key = generate_episode_id(feed_url=feed_url, guid=episode_guid)
    return f"{episode_key}:{fingerprint}"
```

**What to do instead:**

```python
def make_dedup_key(content_item_id: str, fingerprint: str) -> str:
    """Content-type-agnostic dedup key."""
    return f"{content_item_id}:{fingerprint}"
```

Where `content_item_id` is:

- For podcasts: same as current `episode_key` (feed_url hash + episode_guid)
- For news articles: `web_<domain_hash>_<article_url_hash>`
- For social posts: `<platform>_<post_id>`

The podcast module's `generate_episode_id` function produces the
`content_item_id` for podcast content. Other modules produce their own
IDs following the same pattern.

**What this costs now:** A function signature change. Instead of
`make_dedup_key(feed_url, episode_guid, fingerprint)`, it's
`make_dedup_key(content_item_id, fingerprint)`. The caller (podcast code)
computes `content_item_id` from `feed_url + episode_guid` before calling.

**What skipping this costs later:** The dedup logic is baked into the
worker and scheduler. Changing the key format means existing dedup records
in Postgres/Redis don't match new keys. Risk of reprocessing already-
processed content (wasting compute) or failing to process new content
(if old keys collide with new ones).

---

## Anticipation #5: Adaptive Routing Interface (RFC-053)

**When:** During RFC-053 implementation.

**What to do:** Design the routing interface to accept `content_type` as an
input dimension, even though only `"podcast"` exists at first.

```python
@dataclass
class RoutingInput:
    """Input to the adaptive router for strategy selection."""
    content_type: str                    # "podcast", "news_article", etc.
    content_length_tokens: int           # Approximate token count
    has_media: bool                      # Audio/video present?
    language: str                        # "en", "es", etc.
    quality_target: str                  # "fast", "balanced", "quality"

    # Content-type-specific hints (from ContentTypeHandler)
    prompt_profile: str                  # "long_form", "article", "short_text"
    stages_requested: frozenset[str]     # {"transcription", "summarization", ...}
```

```python
@dataclass
class RoutingDecision:
    """Output of the adaptive router."""
    transcription_provider: str | None   # None if not needed
    transcription_model: str | None
    summary_provider: str
    summary_model: str
    gi_provider: str | None
    kg_provider: str | None
    prompt_variant: str                  # "long_v1", "article_v1", etc.
```

In v2.x, the router only handles `content_type="podcast"`. The routing
rules are podcast-specific. In v3.0, external modules register routing
rules for their content types, and the router dispatches based on
`content_type`.

**What this costs now:** One extra field (`content_type`) in the routing
input dataclass. The routing logic ignores it in v2.x (all inputs are
podcast).

**What skipping this costs later:** The routing interface needs to be
extended, and all existing routing rules need to be updated to handle
the new field. If the interface is a protocol that external modules
implement, changing it is a breaking change.

---

## Summary: What to Do During Each Platform RFC

| Platform RFC/Feature | Specific Anticipation | Concrete Action |
| --- | --- | --- |
| **RFC-051** (Postgres projection) | Anticipation #1 | Use `content_item_id`, `source_id`, `content_type`, `source_transport` in all table schemas. Add `content_type` index. |
| **Part A** (catalog + subscriptions) | Anticipation #2 | Name the table `sources` (not `feeds`). Add `source_type`, `transport`, `content_type` columns. Use `config JSONB` for transport-specific fields. Create `feeds` view for backwards compat. |
| **Part B** (workers + queues) | Anticipation #3 | Job payload uses `content_item_id`, `source_id`, `content_type`, `source_transport`, `source_config` dict. Worker dispatches on these fields (v2.x: always podcast). |
| **A.4** (pipeline fingerprinting) | Anticipation #4 | Dedup key is `content_item_id:fingerprint`. Caller computes `content_item_id` (podcast module uses existing `generate_episode_id`). |
| **RFC-053** (adaptive routing) | Anticipation #5 | Routing input includes `content_type` and `prompt_profile`. Routing rules are content-type-scoped. |
| **RFC-062** (viewer/server) | No change needed | Viewer already operates on artifacts. When projected tables use generic naming, viewer queries generalize automatically. |
| **Part E** (observability) | No change needed | Add `content_type` as a label/dimension on metrics (e.g., `pipeline_duration{content_type="podcast"}`). Free metadata. |
| **B.16** (Alembic migrations) | No change needed | Having Alembic from day one makes v3.0 schema changes (if any) trivial to manage. |

---

## What Does NOT Need Anticipation

These platform items are already content-agnostic or don't interact with
content identity:

- **Docker Compose topology** — container layout is about resource profiles,
  not content types
- **Redis / arq setup** — queue infrastructure is content-agnostic; the job
  payload is what matters (Anticipation #3)
- **Auth evolution** (A.12) — API keys and JWT are content-agnostic
- **Observability stack** (Part E) — Prometheus, Grafana, structured logging
  work the same regardless of content types (just add `content_type` label)
- **Deployment lifecycle** (Part F) — CI/CD, rollback, secrets management
  are content-agnostic
- **Digest features** (Part C) — operate on projected tables; if those tables
  use generic naming (Anticipation #1), digest queries work for any content
  type without changes
- **Semantic search** (RFC-061) — vector store indexes documents with metadata;
  adding `content_type` to metadata is trivial and can be done at any time
- **Evaluation framework** (RFC-041/057) — runs experiments on pipeline output;
  content-type-aware evaluation is a v3.x concern

---

## Naming Convention Reference

For consistency across all platform code, use these names:

| Generic Name | Replaces | Used In |
| --- | --- | --- |
| `content_item_id` | `episode_id`, `episode_key`, `episode_guid` | Postgres tables, job payloads, dedup keys, API responses |
| `source_id` | `feed_id`, `feed_url_hash` | Postgres tables, job payloads, catalog |
| `content_type` | *(new)* | Postgres columns, job payloads, routing input, metrics labels |
| `source_transport` | *(new)* | Postgres columns, job payloads, catalog |
| `source_config` | `rss_url` (as top-level field) | Job payloads, catalog `config` JSONB |
| `content_item` | `episode` (in generic/platform code) | Python variable names, API schemas |
| `source` | `feed` (in generic/platform code) | Python variable names, API schemas |

**Exception:** Inside the podcast module itself (`content_types/podcast/`,
`transports/rss/`), continue using `Episode`, `RssFeed`, `episode_id`,
`feed_url` — these are the domain-specific names. The generic names are
for the platform layer (Postgres, workers, API, routing) that sits above
content-type-specific code.

```python
# Podcast module (content-type-specific) — uses domain language:
episode = Episode(title="Planet Money #1234", item=xml_element, ...)
episode_id = generate_episode_id(feed_url=rss_url, guid=guid)

# Platform layer (generic) — uses generic language:
content_item_id = episode_id  # same value, different name at this layer
source_id = generate_source_id(rss_url)
project_content_item(content_item_id=content_item_id, source_id=source_id, ...)
```

---

## PR Roadmap

Concrete PR boundaries for the full journey from current state through
platform v1 to content evolution v3.0. PRs are grouped into streams that
can run in parallel where noted. Each PR is self-contained: tests pass,
CI green, no broken intermediate states.

PRs are intentionally heavy — each delivers a coherent chunk of value.

### Stream A: Finish Podcast Capabilities (Current Work)

These are in-flight or next-up. They complete the podcast feature set and
are prerequisites for everything else.

```text
PR-A1  GI/KG Viewer v2 completion (RFC-062 remaining milestones)
       ─────────────────────────────────────────────────────────
       Scope: Remaining viewer milestones, Playwright E2E, polish.
       Delivers: Complete viewer for podcast GI/KG data.
       Depends on: Nothing (current work).

PR-A2  Semantic search completion (RFC-061)
       ─────────────────────────────────────
       Scope: Vector store, embedding indexing, `podcast search` CLI,
              search API endpoint, viewer search panel.
       Delivers: Full-text + semantic search over podcast corpus.
       Depends on: Nothing (in progress, parallel with A1).

PR-A3  Evaluation framework maturity (RFC-041/057)
       ────────────────────────────────────────────
       Scope: Scorer, baseline comparison, experiment runner,
              golden dataset management.
       Delivers: Ability to benchmark prompt/model combinations.
              Needed later for content-type-aware routing (F.4).
       Depends on: Nothing (in progress, parallel with A1/A2).
```

### Stream B: Platform Infrastructure (v2.x)

These introduce the platform layer. **Anticipations #1-5 apply here** —
the specific generic naming decisions from this document.

```text
PR-B1  Postgres foundation + Alembic + projection schema
       ──────────────────────────────────────────────────
       Scope:
         - Add `db/` package with SQLAlchemy models (or raw SQL)
         - Alembic setup (`db/migrations/`, `env.py`, initial migration)
         - Projected tables: `content_items`, `insights`, `quotes`,
           `insight_support`, `summaries`, `kg_nodes`, `kg_edges`
         - ALL tables use generic naming (Anticipation #1):
           `content_item_id`, `source_id`, `content_type`,
           `source_transport` columns with podcast defaults
         - Projection logic: read gi.json/kg.json from filesystem,
           write to Postgres tables
         - `podcast db upgrade` CLI command
         - Integration tests with testcontainers Postgres
       Delivers: RFC-051 core. Postgres as query surface for GI/KG data.
       Depends on: Nothing (can start now).
       Anticipations: #1 (generic schema naming).
       Size: Large — new package, schema, migrations, projection logic,
             CLI command, tests. ~1500-2500 lines.

PR-B2  Catalog + subscriptions + sources table
       ────────────────────────────────────────
       Scope:
         - `sources` table (not `feeds`) with `source_type`, `transport`,
           `content_type`, `config JSONB` (Anticipation #2)
         - `tenants` table (single row for v1)
         - `subscriptions` table (`tenant_id` + `source_id`)
         - `feeds` backwards-compat view over `sources`
         - Alembic migration for new tables
         - Python models for catalog CRUD
         - `podcast catalog` CLI subcommands:
           `podcast catalog add <rss_url>`
           `podcast catalog list`
           `podcast catalog remove <source_id>`
         - Unit + integration tests
       Delivers: Part A Phases A-B. Catalog of sources with subscriptions.
       Depends on: PR-B1 (Postgres + Alembic in place).
       Anticipations: #2 (generic catalog model).
       Size: Medium-large — new tables, models, CLI commands. ~1000-1500 lines.

PR-B3  Workers + queues (simple tier) + job payload
       ─────────────────────────────────────────────
       Scope:
         - arq integration (Redis-backed job queue)
         - `PipelineJob` dataclass with generic envelope (Anticipation #3):
           `content_item_id`, `source_id`, `content_type`,
           `source_transport`, `source_config`, `processing_config`,
           `pipeline_fingerprint`
         - `podcast worker --queue heavy` CLI command
         - Worker consumes `heavy` queue, builds `Config` from job payload,
           calls `run_pipeline(cfg)`, projects results to Postgres
         - Dedup logic using `content_item_id:fingerprint` key
           (Anticipation #4)
         - `ingest` queue: poll sources from catalog, enqueue `heavy` jobs
         - `projection` queue: post-pipeline Postgres projection
         - arq cron for periodic RSS polling
         - Redis health check
         - Integration tests with fakeredis + real arq
         - Docker Compose update: add `redis`, `worker` services
       Delivers: Part B simple tier. Long-lived worker processing feeds
                 from catalog.
       Depends on: PR-B1 (Postgres), PR-B2 (catalog).
       Anticipations: #3 (generic job payload), #4 (generic dedup key).
       Size: Large — new worker process, queue integration, CLI, Compose,
             tests. ~2000-3000 lines.

PR-B4  Platform API routes + server integration
       ─────────────────────────────────────────
       Scope:
         - Platform routes in `server/routes/platform/`:
           `GET/POST /api/sources` (catalog CRUD)
           `GET/POST /api/subscriptions`
           `GET /api/jobs` (job status)
           `POST /api/jobs` (enqueue manual job)
           `GET /api/status` (system health + queue depths)
         - Routes behind `--platform` flag on `podcast serve`
         - Pydantic response schemas using generic naming
           (`content_item_id`, `source_id`, etc.)
         - Integration tests
       Delivers: Part A.7 API layer. Platform management via REST API.
       Depends on: PR-B2 (catalog), PR-B3 (workers/jobs).
       Size: Medium — route handlers, schemas, tests. ~800-1200 lines.

PR-B5  Observability foundation
       ────────────────────────
       Scope:
         - Prometheus metrics endpoint (`/metrics`)
         - Key metrics: `pipeline_duration_seconds{content_type}`,
           `queue_depth{queue}`, `jobs_processed_total{status}`,
           `episodes_processed_total{content_type}`
         - Structured JSON logging with correlation IDs (API → worker)
         - Health check endpoints for all services
         - Grafana dashboard JSON (importable)
         - Docker Compose update: add Grafana + Prometheus services
       Delivers: Part E foundation. Visibility into platform operations.
       Depends on: PR-B3 (workers exist to emit metrics).
       Anticipations: `content_type` as metric label from day one.
       Size: Medium — metrics, logging, dashboard, Compose. ~800-1200 lines.

PR-B6  Auth stage 2 (API key)
       ──────────────────────
       Scope:
         - `AUTH_API_KEY` env var
         - FastAPI middleware: reject requests without valid
           `Authorization: Bearer <key>` header
         - Exempt: health check, static files (viewer)
         - Worker uses same key for internal API calls
         - Tests for auth middleware
       Delivers: A.12 stage 2. Secure API for network-exposed deployments.
       Depends on: PR-B4 (platform routes to protect).
       Size: Small — middleware, env var, tests. ~200-400 lines.

PR-B7  Docker Compose production profile + deployment
       ───────────────────────────────────────────────
       Scope:
         - `docker-compose.prod.yml` with all services:
           postgres, redis, api, worker, caddy
         - Caddy config (TLS termination, reverse proxy)
         - `.env.example` with all config vars
         - `podcast db upgrade` runs before API starts
         - Volume mounts for artifacts, model cache, Postgres data
         - `deploy.sh` script (pull, migrate, restart)
         - Documentation: deployment guide
       Delivers: Part F. Deployable platform on a single host.
       Depends on: PR-B1 through PR-B6 (all platform components).
       Size: Medium — Compose files, Caddy config, scripts, docs.
             ~500-800 lines.
```

### Stream C: Adaptive Routing (Can Parallel with Late Stream B)

```text
PR-C1  Adaptive routing framework (RFC-053)
       ────────────────────────────────────
       Scope:
         - `RoutingInput` dataclass with `content_type` field
           (Anticipation #5)
         - `RoutingDecision` dataclass
         - Router interface + deterministic rule-based implementation
         - Podcast routing rules (the only content type)
         - Integration with orchestration: router selects providers
           based on episode characteristics
         - Config: `routing_strategy: auto | manual`
         - Tests for routing logic
       Delivers: RFC-053. Provider selection based on content characteristics.
                 `content_type` is a routing dimension from day one.
       Depends on: Nothing (can start after podcast capabilities are solid).
       Anticipations: #5 (content type in routing interface).
       Size: Medium — routing logic, integration, tests. ~800-1200 lines.
```

### Stream D: Content Evolution (v3.0)

These happen after platform v1 is operational (Stream B complete).
Each PR is a refactoring phase from the Content Evolution Note.

```text
PR-D1  Core protocols, manifests, and registry
       ────────────────────────────────────────
       Scope:
         - New `core/` package:
           `core/__init__.py`
           `core/protocols.py` — ContentItem, ContentSource,
             ContentTypeHandler, ProcessingConfig protocols
           `core/manifests.py` — ContentTypeManifest, TransportManifest
             dataclasses (frozen, with JSON Schema fields)
           `core/registry.py` — PluginRegistry (entry point discovery,
             get/list content types and transports, validate wiring)
           `core/testing.py` — assert_valid_content_item,
             assert_valid_manifest, assert_pipeline_compatible helpers
         - Episode satisfies ContentItem (structural typing — no changes
           to Episode itself, just verify protocol compliance in tests)
         - RSS parser satisfies ContentSource (same — verify, don't change)
         - Unit tests for registry, manifests, protocol compliance
         - NO behavior changes — existing pipeline untouched
       Delivers: Content Evolution Phase 1. The interface layer exists.
       Depends on: PR-B1 (Postgres schema informs ContentItem fields).
       Size: Medium — new package, protocols, registry, tests.
             ~800-1200 lines.

PR-D2  Podcast + RSS as plugins + module reorganization
       ─────────────────────────────────────────────────
       Scope:
         - Create `content_types/podcast/`:
           `handler.py` — PodcastContentType, PodcastContentHandler
           `manifest.py` — PODCAST_MANIFEST
           `entities.py` — move Episode, RssFeed from `models/entities.py`
             (keep re-exports in old location for backwards compat)
           `config.py` — podcast-specific config fields (extracted later
             in PR-D4, but namespace created here)
         - Create `transports/rss/`:
           `source.py` — RssTransport, RssSource (wraps existing parser)
           `manifest.py` — RSS_MANIFEST
           Move `rss/parser.py`, `rss/downloader.py` here
             (keep re-exports in old `rss/` for backwards compat)
         - Register both via entry points in `pyproject.toml`
         - Registry discovers them in tests
         - Update imports across codebase (the noisy part — but mechanical)
         - Backwards-compat re-exports so external code importing from
           old paths still works
         - ALL existing tests pass unchanged
       Delivers: Content Evolution Phases 2 + 4. Podcast and RSS are
                 plugins. Module structure reflects the architecture.
       Depends on: PR-D1 (protocols and registry exist).
       Size: Large — module moves, import updates, entry points,
             backwards-compat re-exports. ~1500-2500 lines of changes
             (mostly import path updates, not new logic).

PR-D3  Generic orchestration + split run_pipeline
       ───────────────────────────────────────────
       Scope:
         - Extract generic `run_pipeline(items, config, handler)` from
           current `run_pipeline(cfg)` in `workflow/orchestration.py`
         - New `core/pipeline.py` with the generic entry point
         - Current `run_pipeline(cfg)` becomes a thin wrapper:
           1. Create RssSource, fetch items
           2. Convert to ContentItem list
           3. Create PodcastContentHandler
           4. Call generic run_pipeline
         - `service.run()` and CLI call the podcast-specific wrapper
           (no change to external interface)
         - Worker (PR-B3) can optionally call the generic entry point
           (or continue using the wrapper — both work)
         - Integration tests: generic pipeline with mock content items
         - ALL existing tests pass unchanged (they call the wrapper)
       Delivers: Content Evolution Phase 3. The pipeline core is generic.
                 External modules can call run_pipeline with their own
                 content items.
       Depends on: PR-D2 (podcast plugin provides ContentTypeHandler).
       Size: Medium — orchestration refactoring, new generic entry point,
             tests. ~800-1500 lines.

PR-D4  ProcessingConfig extraction + multi-source config
       ──────────────────────────────────────────────────
       Scope:
         - Extract `ProcessingConfig` from `Config`:
           `core/config.py` — ProcessingConfig Pydantic model with
             summary_provider, summary_model, gi_enabled, kg_enabled,
             output_dir, workers, provider API keys/models, etc.
         - `Config` composes `ProcessingConfig` (has-a, not is-a)
         - `Config.processing` property returns ProcessingConfig
         - Add `sources` field to Config (optional list of SourceConfig):
           ```yaml
           sources:
             - transport: rss
               content_type: podcast
               config:
                 rss_url: "https://..."
           ```
         - Backwards compat: if `rss_url` present and no `sources`,
           synthesize implicit source entry
         - Config validation: registry validates transport/content_type
           wiring and source config against manifest schemas
         - External modules can construct ProcessingConfig independently
         - Update worker (PR-B3) to use ProcessingConfig
         - ALL existing Config YAML files work unchanged
       Delivers: Content Evolution Phase 5. Config is layered.
                 External modules have a clean config surface.
       Depends on: PR-D3 (generic pipeline accepts ProcessingConfig),
                   PR-B3 (worker uses config — update to ProcessingConfig).
       Size: Large — Config refactoring, new model, validation, backwards
             compat, worker update, tests. ~1500-2000 lines.

PR-D5  CLI plugin commands + plugin developer docs
       ────────────────────────────────────────────
       Scope:
         - `podcast-scraper plugins` CLI subcommand:
           `podcast-scraper plugins --list` — list installed content types
             and transports with manifest details
           `podcast-scraper plugins --validate config.yaml` — validate
             config wiring against installed plugins
         - `core/testing.py` enhancements: full test helpers for plugin
           authors (assert_valid_content_item, assert_valid_manifest,
           assert_pipeline_compatible, mock_content_item factory)
         - Documentation:
           `docs/guides/PLUGIN_DEVELOPMENT_GUIDE.md` — how to build an
             external module (package structure, manifest, entry points,
             testing, publishing)
           Update `docs/architecture/ARCHITECTURE.md` with content
             evolution section
           Promote content evolution note from `docs/wip/` to
             `docs/architecture/CONTENT_EVOLUTION_BLUEPRINT.md`
       Delivers: Content Evolution Phase 6. Plugin ecosystem is ready
                 for external module development.
       Depends on: PR-D4 (multi-source config, full registry integration).
       Size: Medium — CLI commands, test helpers, documentation.
             ~800-1200 lines.
```

### Stream E: First External Module (v3.0 Validation)

```text
PR-E1  news-ingest module (separate repo: content-modules)
       ───────────────────────────────────────────────────
       Scope (in content-modules private repo):
         - `packages/news-ingest/` package structure
         - NewsArticleContentType + NEWS_ARTICLE_MANIFEST
         - WebScrapeTransport + WEB_SCRAPE_MANIFEST
         - WebScrapeSource using trafilatura
         - NewsArticle entity satisfying ContentItem
         - NewsContentHandler (no transcription, summarization + GI + KG)
         - Entry points in pyproject.toml
         - Unit + integration tests
         - CI workflow testing against published podcast-scraper
       Scope (back in podcast-scraper repo, if needed):
         - Any protocol/manifest adjustments discovered during integration
         - Prompt templates for article-length content
       Delivers: First external module. Validates the entire plugin
                 architecture end-to-end.
       Depends on: PR-D5 (plugin ecosystem ready).
       Size: Medium — new package in separate repo. ~1000-1500 lines.
```

### Stream F: Cross-Content Features (v3.x)

```text
PR-F1  Cross-content-type KG + entity resolution (RFC needed)
       ──────────────────────────────────────────────────────
       Scope: Entity deduplication across content types, canonical entity
              registry, cross-source KG linking.
       Depends on: PR-E1 (at least 2 content types producing KG data).
       Size: Large — new RFC, entity resolution logic, KG merging.

PR-F2  Viewer v3 — mixed content UI (UXS needed)
       ──────────────────────────────────────────
       Scope: Content-type faceting, unified timeline, content-type-specific
              detail views, cross-content KG visualization.
       Depends on: PR-F1 (cross-content KG exists to visualize).
       Size: Large — new UX spec, frontend components, API changes.

PR-F3  Content-type-aware scheduling + dedicated pipelines
       ───────────────────────────────────────────────────
       Scope: Queue routing by content type, worker pool tuning,
              high-volume content batching.
       Depends on: PR-B3 (workers operational), PR-E1 (multiple content
              types in production), metrics showing contention.
       Size: Medium — queue routing logic, Compose topology changes.
```

### Visual: PR Dependency Graph

```text
Stream A (podcast features — current work)
  A1 (viewer v2) ──────────────────────────────────────────────┐
  A2 (semantic search) ────────────────────────────────────────┤
  A3 (eval framework) ─────────────────────────────────────────┤
                                                               │
Stream B (platform infrastructure — v2.x)                      │
  B1 (Postgres + projection) ──┬───────────────────────────────┤
                               │                               │
  B2 (catalog + sources) ──────┤                               │
                               │                               │
  B3 (workers + queues) ───────┤                               │
                               │                               │
  B4 (platform API) ───────────┤                               │
                               │                               │
  B5 (observability) ──────────┤                               │
                               │                               │
  B6 (auth) ───────────────────┤                               │
                               │                               │
  B7 (Docker prod) ────────────┘                               │
                                                               │
Stream C (adaptive routing — parallel with late B)             │
  C1 (routing framework) ─────────────────────────────────────┤
                                                               │
                                                               ▼
                                                    Platform v1 operational
                                                               │
Stream D (content evolution — v3.0)                            │
  D1 (core protocols + registry) ◄─────────────────────────────┘
       │
  D2 (podcast + RSS as plugins + module reorg)
       │
  D3 (generic orchestration)
       │
  D4 (ProcessingConfig + multi-source config)
       │
  D5 (CLI plugins + docs)
       │
       ▼
  Content evolution v3.0 complete
       │
Stream E (first external module — v3.0 validation)
  E1 (news-ingest in content-modules repo)
       │
       ▼
Stream F (cross-content features — v3.x)
  F1 (cross-content KG)
  F2 (viewer v3)
  F3 (content-type scheduling)
```

### PR Summary Table

| PR | Stream | Name | Size | Depends On | Anticipations |
| --- | --- | --- | --- | --- | --- |
| A1 | A | Viewer v2 completion | Large | — | — |
| A2 | A | Semantic search | Large | — | — |
| A3 | A | Eval framework | Medium | — | — |
| B1 | B | Postgres + projection | Large | — | #1 (generic schema) |
| B2 | B | Catalog + sources | Medium-large | B1 | #2 (generic catalog) |
| B3 | B | Workers + queues | Large | B1, B2 | #3 (generic job), #4 (generic dedup) |
| B4 | B | Platform API routes | Medium | B2, B3 | — |
| B5 | B | Observability | Medium | B3 | `content_type` metric label |
| B6 | B | Auth stage 2 | Small | B4 | — |
| B7 | B | Docker prod profile | Medium | B1-B6 | — |
| C1 | C | Adaptive routing | Medium | — | #5 (content type in routing) |
| D1 | D | Core protocols + registry | Medium | B1 | — |
| D2 | D | Podcast/RSS as plugins | Large | D1 | — |
| D3 | D | Generic orchestration | Medium | D2 | — |
| D4 | D | ProcessingConfig + multi-source | Large | D3, B3 | — |
| D5 | D | CLI plugins + docs | Medium | D4 | — |
| E1 | E | news-ingest (separate repo) | Medium | D5 | — |
| F1 | F | Cross-content KG | Large | E1 | — |
| F2 | F | Viewer v3 | Large | F1 | — |
| F3 | F | Content-type scheduling | Medium | B3, E1 | — |

**Total: 20 PRs across 6 streams.** Streams A, B, and C can run in
parallel. Stream D is sequential (each PR depends on the previous).
Streams E and F follow D.

---

## Decision: Separate Architecture Documents

This analysis confirms that the Platform Blueprint and Content Evolution Note
should remain **separate documents** (not merged):

| Document | Scope | Timeline |
| --- | --- | --- |
| Platform Blueprint | How to run at scale (infra) | v2.x (now → near-term) |
| Content Evolution Note | What to process (content types) | v3.0 (after platform v1) |
| **This document** | How they interact (sequencing) | Reference during v2.x implementation |

**Cross-references to add:**

- Platform Blueprint → add a note in Part A (catalog), Part B (job schema),
  and RFC-051 referencing this sequencing guide for generic naming conventions
- Content Evolution Note → already references the blueprint; add a note that
  Anticipations #1-5 are tracked in this document
