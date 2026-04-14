# Content Evolution Blueprint

**Status:** Architecture blueprint -- companion to the Platform Architecture Blueprint.

**Date:** 2026-04-14

**Input documents:**

- [Platform Architecture Blueprint](PLATFORM_ARCHITECTURE_BLUEPRINT.md)
  -- infrastructure evolution (tenancy, workers, queues, Postgres, deployment)
- [RFC-072: Canonical Identity Layer + Bridge](../rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md)
  -- cross-layer identity foundation. Shipped.
- [RFC-073: Enrichment Layer Architecture](../rfc/RFC-073-enrichment-layer-architecture.md)
  -- fourth artifact tier for derived signals. In progress.

**Purpose:** Describes how the system evolves beyond podcasts via a generic
core with pluggable content types and transports, plus the sequencing and
anticipations needed during platform work to avoid costly migrations.

---

## Part 1: Vision and Core Architecture

### Core Principle: Generic Core, Pluggable Content Types and Transports

The pipeline core is **content-type-agnostic and transport-agnostic**. It processes
content items through configurable stages (transcription, summarization, GI, KG,
search indexing) without knowing where the content came from or what kind it is.

Two concepts are separated cleanly:

| Concept | Responsibility | Examples |
| --- | --- | --- |
| **Content type** | Defines what the content is, what processing stages apply, what metadata it carries | Podcast episode (long audio + transcript), news article (medium text), social post (short text) |
| **Transport** | Defines how content is acquired -- fetching, parsing, normalizing into `ContentItem` | RSS feed polling, web scraping, social API, filesystem watcher, email ingest |

**Podcast** (content type) and **RSS** (transport) are the two default modules that
ship with this repo. They are implemented against the same interfaces that any
external module would use -- no privileged internal access.

### Why Generic Core with Podcast as Default Module

#### Advantages over "expose seams later"

1. **Interfaces are honest from day one.** Podcast + RSS must satisfy the generic
   protocols, which means those protocols are tested and proven by a real
   implementation -- not theoretical contracts designed in a vacuum.

2. **Forces discovery of real abstractions.** Today `Episode.item` is an
   `ET.Element` because nobody had to question it. When podcast becomes a module
   conforming to a generic interface, you discover what the pipeline actually needs
   vs. what is RSS baggage. That discovery happens once during the refactor.

3. **External modules are true peers.** If podcast is a plugin and news is a plugin,
   they share the same integration surface. No "podcast gets internal APIs that
   external modules can't reach."

4. **Refactoring scope is the same either way.** Whether you expose clean seams or
   make the core generic, you touch the same four areas (entry point, entity model,
   config, orchestration). The generic approach does it more thoroughly upfront but
   doesn't touch more code.

5. **Mitigates the speculation risk.** The usual danger of premature generalization
   is designing without a consumer. Here, podcast + RSS is the first consumer that
   validates every interface. You're extracting from a working system, not guessing.

#### External modules still own their dependencies

Even though the core is generic, content-type-specific and transport-specific
dependencies stay out of the core:

- News scraping needs `trafilatura` -- lives in the news module's repo
- Social connectors need OAuth libraries -- lives in the social module's repo
- Podcast needs `openai-whisper`, `spacy` -- stays in this repo as part of the
  podcast content-type module (already optional deps)

### Architecture Layers

```text
┌─────────────────────────────────────────────────────────┐
│                    CLI / Server / API                     │
│         (podcast-scraper CLI, FastAPI server)             │
├─────────────────────────────────────────────────────────┤
│                  Orchestration Layer                      │
│     run_pipeline(content_items, processing_config)        │
│     Stage routing based on content type + config          │
├──────────────┬──────────────┬───────────────────────────┤
│  Content     │  Transport   │  Processing Stages         │
│  Types       │  Modules     │  (generic)                 │
│              │              │                            │
│  ┌────────┐  │  ┌────────┐  │  ┌─────────────────────┐  │
│  │Podcast │  │  │  RSS   │  │  │ Transcription       │  │
│  │(default)│  │  │(default)│  │  │ Summarization       │  │
│  └────────┘  │  └────────┘  │  │ GI Extraction        │  │
│              │              │  │ KG Extraction        │  │
│  ┌────────┐  │  ┌────────┐  │  │ Bridge (CIL)        │  │
│  │ News   │  │  │  Web   │  │  │ Search Indexing      │  │
│  │(external)│ │  │Scrape  │  │  │ Projection           │  │
│  └────────┘  │  │(external)│ │  └─────────────────────┘  │
│              │  └────────┘  │                            │
│  ┌────────┐  │  ┌────────┐  │  ┌─────────────────────┐  │
│  │Social  │  │  │Folder  │  │  │ Enrichment Layer     │  │
│  │(external)│ │  │Watch   │  │  │ (RFC-073: optional   │  │
│  └────────┘  │  │(external)│ │  │  derived signals)    │  │
│              │  └────────┘  │  └─────────────────────┘  │
│              │              │                            │
│              │              │  ┌─────────────────────┐  │
│              │              │  │ Provider System      │  │
│              │              │  │ (9 providers,        │  │
│              │              │  │  prompt templates,   │  │
│              │              │  │  model registry)     │  │
├──────────────┴──────────────┴───────────────────────────┤
│                Storage / Serving Layer                    │
│    Filesystem, Postgres projection, Vector search,       │
│    Viewer (Vue SPA)                                      │
└─────────────────────────────────────────────────────────┘
```

---

## Part 2: Core Protocols

### ContentItem -- what the pipeline processes

```python
class ContentItem(Protocol):
    """A single piece of content to process through the pipeline."""
    id: str                          # Unique identifier
    title: str                       # Human-readable title
    content_type: str                # e.g. "podcast", "news_article", "social_post"
    source_text: str | None          # Text content (None if audio-only)
    media_path: Path | None          # Audio/video path (None if text-only)
    metadata: dict[str, Any]         # Content-type-specific metadata
    source_transport: str            # e.g. "rss", "web_scrape", "filesystem"
```

The podcast module's `Episode` satisfies this protocol (via adapter or direct
implementation). External modules provide their own types that also satisfy it.

### ContentSource -- how content is acquired

```python
class ContentSource(Protocol):
    """A transport that fetches content and yields ContentItems."""
    def fetch(self, config: SourceConfig) -> list[ContentItem]: ...
    def supports_incremental(self) -> bool: ...  # cursor-based polling
```

RSS becomes the first `ContentSource`. Web scraping, social APIs, filesystem
watchers are future implementations in external repos.

### ContentTypeHandler -- content-type-specific processing decisions

```python
class ContentTypeHandler(Protocol):
    """Declares which pipeline stages apply to a content type."""
    content_type: str
    def needs_transcription(self, item: ContentItem) -> bool: ...
    def needs_summarization(self, item: ContentItem) -> bool: ...
    def get_prompt_profile(self, item: ContentItem) -> str: ...
    # e.g. "long_form" for podcasts, "short_text" for tweets
```

This integrates with RFC-053 (Adaptive Routing) -- the content type handler is
the routing input.

### ProcessingConfig -- config for the generic pipeline

```python
class ProcessingConfig(Protocol):
    """Processing configuration independent of content source."""
    summary_provider: str
    summary_model: str
    gi_enabled: bool
    kg_enabled: bool
    enrichment_enabled: bool  # RFC-073 enrichment pass
    output_dir: Path
    # ... processing-relevant fields only
```

The existing `Config` composes `ProcessingConfig` (plus RSS-specific fields).
External modules build `ProcessingConfig` from their own config surface.

---

## Part 3: Registry, Schemas, and Discovery

The core needs three mechanisms to work with pluggable content types and transports:

1. **Manifests** -- each content type and transport declares what it is, what it
   needs, and what it provides (self-describing schema)
2. **Registry** -- runtime lookup of installed content types and transports
3. **Wiring** -- config-driven mapping of "use transport X to feed content type Y"

### Content Type Manifest

Every content type (bundled or external) provides a manifest declaring its
identity, processing requirements, and metadata shape.

```python
@dataclass(frozen=True)
class ContentTypeManifest:
    """Self-describing declaration of a content type."""

    # Identity
    name: str                        # "podcast", "news_article", "social_post"
    display_name: str                # "Podcast Episode", "News Article"
    version: str                     # Semver for the manifest schema itself

    # Processing stages this content type uses
    # Core checks these before running each stage
    stages: frozenset[str]           # {"transcription", "summarization", "gi", "kg"}

    # Content characteristics (drives adaptive routing / RFC-053)
    has_media: bool                  # Audio/video content? (True for podcast)
    has_source_text: bool            # Text content available? (True for news)
    typical_length: str              # "short" | "medium" | "long"

    # Metadata contract
    # JSON Schema defining what metadata dict looks like for this type
    # e.g. podcast: {feed_url, episode_guid, published_date, authors, ...}
    # e.g. news: {source_url, publication, author, publish_date, ...}
    metadata_schema: dict[str, Any]

    # Config extension
    # JSON Schema for content-type-specific config fields beyond ProcessingConfig
    # e.g. podcast: {whisper_model, auto_speakers, screenplay, ...}
    # e.g. news: {extract_images, min_article_length, ...}
    config_schema: dict[str, Any] | None

    # Prompt engineering
    default_prompt_profile: str      # "long_form", "article", "short_text"
    # Maps to prompt template variants in PromptStore
    # e.g. "long_form" -> prompts/<provider>/summarization/long_form_v1.j2
```

**Podcast example:**

```python
PODCAST_MANIFEST = ContentTypeManifest(
    name="podcast",
    display_name="Podcast Episode",
    version="1.0.0",
    stages=frozenset({"transcription", "summarization", "gi", "kg"}),
    has_media=True,
    has_source_text=False,  # text comes from transcription
    typical_length="long",
    metadata_schema={
        "type": "object",
        "properties": {
            "feed_url": {"type": "string", "format": "uri"},
            "episode_guid": {"type": "string"},
            "published_date": {"type": "string", "format": "date-time"},
            "authors": {"type": "array", "items": {"type": "string"}},
            "duration_seconds": {"type": "integer"},
        },
        "required": ["feed_url", "episode_guid"],
    },
    config_schema={
        "type": "object",
        "properties": {
            "whisper_model": {"type": "string", "default": "base"},
            "auto_speakers": {"type": "boolean", "default": True},
            "screenplay": {"type": "boolean", "default": False},
        },
    },
    default_prompt_profile="long_form",
)
```

### Transport Manifest

Every transport declares its identity, capabilities, compatible content types,
and config shape.

```python
@dataclass(frozen=True)
class TransportManifest:
    """Self-describing declaration of a transport."""

    # Identity
    name: str                        # "rss", "web_scrape", "filesystem"
    display_name: str                # "RSS Feed", "Web Scraper"
    version: str

    # What content types can this transport produce?
    # Registry validates that wired content types are in this list
    compatible_content_types: frozenset[str]  # {"podcast"} for RSS

    # Capabilities
    supports_incremental: bool       # Cursor-based polling (RSS: yes)
    supports_batch: bool             # Fetch N items at once (RSS: yes)
    requires_auth: bool              # Needs credentials (social APIs: yes)

    # Config contract
    # JSON Schema for transport-specific config
    # e.g. RSS: {rss_url (required), max_episodes, prefer_types, ...}
    # e.g. web_scrape: {urls (required), selector, ...}
    config_schema: dict[str, Any]

    # Scheduling hints for platform mode
    default_poll_interval_seconds: int | None  # None = one-shot, 3600 = hourly
```

**RSS example:**

```python
RSS_MANIFEST = TransportManifest(
    name="rss",
    display_name="RSS Feed",
    version="1.0.0",
    compatible_content_types=frozenset({"podcast"}),
    supports_incremental=True,
    supports_batch=True,
    requires_auth=False,
    config_schema={
        "type": "object",
        "properties": {
            "rss_url": {"type": "string", "format": "uri"},
            "max_episodes": {"type": "integer", "minimum": 1},
            "prefer_types": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["rss_url"],
    },
    default_poll_interval_seconds=3600,
)
```

### Registry -- Runtime Discovery

The registry discovers installed content types and transports via **Python entry
points** -- the standard plugin mechanism. This is how pytest discovers plugins,
Flask discovers extensions, and pip discovers build backends.

**Registration (in `pyproject.toml`):**

```toml
# This repo -- bundled defaults
[project.entry-points."content_engine.content_types"]
podcast = "podcast_scraper.content_types.podcast:PodcastContentType"

[project.entry-points."content_engine.transports"]
rss = "podcast_scraper.transports.rss:RssTransport"
```

External content-type modules are standard Python packages discovered via entry
points. Any developer can create a module by implementing the protocols and
registering via `pyproject.toml` entry points. `pip install <module>` is enough --
the core discovers it automatically at runtime. No manual config listing required.

**Registry class:**

```python
class PluginRegistry:
    """Discovers and provides access to installed content types and transports."""

    def __init__(self) -> None:
        self._content_types: dict[str, ContentTypePlugin] = {}
        self._transports: dict[str, TransportPlugin] = {}
        self._discover()

    def _discover(self) -> None:
        """Load all installed plugins via entry points."""
        for ep in entry_points(group="content_engine.content_types"):
            plugin_cls = ep.load()
            plugin = plugin_cls()
            self._content_types[plugin.manifest.name] = plugin

        for ep in entry_points(group="content_engine.transports"):
            plugin_cls = ep.load()
            plugin = plugin_cls()
            self._transports[plugin.manifest.name] = plugin

    def get_content_type(self, name: str) -> ContentTypePlugin:
        """Get a content type by name. Raises KeyError if not installed."""
        return self._content_types[name]

    def get_transport(self, name: str) -> TransportPlugin:
        """Get a transport by name. Raises KeyError if not installed."""
        return self._transports[name]

    def list_content_types(self) -> list[ContentTypeManifest]:
        """List all installed content type manifests."""
        return [p.manifest for p in self._content_types.values()]

    def list_transports(self) -> list[TransportManifest]:
        """List all installed transport manifests."""
        return [p.manifest for p in self._transports.values()]

    def validate_wiring(
        self, transport: str, content_type: str
    ) -> None:
        """Validate that a transport supports the given content type."""
        t = self.get_transport(transport)
        if content_type not in t.manifest.compatible_content_types:
            raise ValueError(
                f"Transport '{transport}' does not support "
                f"content type '{content_type}'. "
                f"Supported: {t.manifest.compatible_content_types}"
            )
```

**Plugin base classes** (what content types and transports implement):

```python
class ContentTypePlugin(Protocol):
    """What a content type module must provide."""
    manifest: ContentTypeManifest
    handler: ContentTypeHandler  # processing decisions

class TransportPlugin(Protocol):
    """What a transport module must provide."""
    manifest: TransportManifest
    source: ContentSource  # fetch logic
```

### Wiring -- Config-Driven Source Mapping

The wiring connects transports to content types in user config. Two modes:

**Implicit (backwards compatible):** Current `rss_url` in config implies
`transport=rss, content_type=podcast`. Existing configs work unchanged.

```yaml
# Current config -- works as before, implicit wiring
rss_url: "https://feeds.npr.org/510289/podcast.xml"
max_episodes: 5
```

**Explicit (multi-source):** New `sources` field for multi-source configs.
Each source declares its transport, content type, and transport-specific config.
The registry validates that the transport supports the content type and that
the config matches the transport's `config_schema`.

```yaml
# Future multi-source config
sources:
  - transport: rss
    content_type: podcast
    config:
      rss_url: "https://feeds.npr.org/510289/podcast.xml"
      max_episodes: 5

  - transport: web_scrape
    content_type: news_article
    config:
      urls:
        - "https://example.com/news"
        - "https://other.com/articles"
      selector: "article.main"

# Processing config applies to all sources
summary_provider: openai
gi_enabled: true
kg_enabled: true
```

**Validation flow:**

```text
Config YAML loaded
  -> If rss_url present and no sources: synthesize implicit source entry
  -> For each source entry:
      1. Registry.get_transport(source.transport) -- is it installed?
      2. Registry.get_content_type(source.content_type) -- is it installed?
      3. Registry.validate_wiring(transport, content_type) -- compatible?
      4. Validate source.config against transport.manifest.config_schema
  -> Build ProcessingConfig from top-level fields
  -> Pass to pipeline
```

### CLI Discovery Commands

The registry enables introspection commands:

```bash
# List installed content types
podcast-scraper plugins --content-types
# Output:
# podcast    (bundled)  Podcast Episode -- stages: transcription, summarization, gi, kg
# news_article          News Article -- stages: summarization, gi, kg

# List installed transports
podcast-scraper plugins --transports
# Output:
# rss        (bundled)  RSS Feed -- content types: podcast
# web_scrape            Web Scraper -- content types: news_article, blog_post

# Validate a config file's wiring
podcast-scraper plugins --validate config.yaml
# Output:
# [ok] source[0]: transport=rss, content_type=podcast -- OK
# source[1]: transport=web_scrape -- not installed (pip install news-ingest)
```

### Design Constraints

1. **Manifests are static and declarative.** They describe capabilities, not
   runtime state. No "call this function to find out what stages I need" -- the
   manifest says it upfront. This enables validation before any processing starts.

2. **JSON Schema for config and metadata contracts.** Using JSON Schema (not
   Python types) for `config_schema` and `metadata_schema` means external modules
   in other languages could theoretically participate, and config validation
   works without importing the plugin code.

3. **Entry points for discovery, not config files.** `pip install` is the
   registration mechanism. No manual listing of module paths. This is the Python
   ecosystem standard.

4. **Compatibility validation is transport's responsibility.** The transport
   declares which content types it can produce (`compatible_content_types`).
   The registry enforces this at wiring time.

5. **Manifest versioning.** The `version` field on manifests enables the core to
   detect incompatible plugins (e.g., a plugin built for manifest schema v2 when
   the core only understands v1). Semver on the manifest schema itself.

6. **Start minimal, extend from real needs.** The manifest fields above are what
   podcast + RSS actually need. When the second content type arrives, it will
   likely need fields not listed here. That's fine -- extend the manifest
   dataclass. The `version` field tracks schema evolution.

---

## Part 4: What Changes and What Doesn't

### What Changes in This Repo

#### 1. Orchestration becomes generic

**Current:** `run_pipeline(cfg)` hardwires RSS fetch -> `Episode` list -> process.

**Target:** `run_pipeline` accepts `list[ContentItem]` + `ProcessingConfig`. A
higher-level `run_podcast_pipeline(cfg)` (or the CLI) does RSS fetch -> convert
to `ContentItem` -> call generic `run_pipeline`.

```python
# Generic entry point (core)
def run_pipeline(
    items: list[ContentItem],
    config: ProcessingConfig,
    content_handler: ContentTypeHandler,
) -> PipelineResult: ...

# Podcast-specific entry point (podcast module, CLI uses this)
def run_podcast_pipeline(cfg: Config) -> PipelineResult:
    source = RssSource()
    items = source.fetch(cfg)
    handler = PodcastContentHandler()
    return run_pipeline(items, cfg.processing, handler)
```

#### 2. Entity model gains a generic layer

**Current:** `Episode` with `item: ET.Element` is the only entity.

**Target:** `ContentItem` protocol is the generic layer. `Episode` stays as the
podcast-specific implementation that satisfies `ContentItem`. The `ET.Element` stays
on `Episode` -- it's podcast/RSS-specific detail that the generic pipeline never
touches.

#### 3. Config splits into layers

**Current:** One `Config` with 100+ fields mixing RSS + processing.

**Target:**

```text
Config (podcast CLI, full surface)
├── rss_url, max_episodes, prefer_types, ...  (RSS/podcast-specific)
├── ProcessingConfig                           (generic, reusable)
│   ├── summary_provider, summary_model
│   ├── gi_enabled, kg_enabled
│   ├── output_dir, workers, ...
│   └── provider configs (API keys, models)
└── whisper_model, auto_speakers, ...          (podcast content-type-specific)
```

External modules construct `ProcessingConfig` directly. The full `Config` stays
for podcast CLI use.

#### 4. Module organization

```text
src/podcast_scraper/
├── core/                          # Generic pipeline (NEW)
│   ├── protocols.py               # ContentItem, ContentSource, ContentTypeHandler
│   ├── pipeline.py                # Generic run_pipeline
│   ├── config.py                  # ProcessingConfig
│   └── types.py                   # Shared types
├── content_types/                 # Content type handlers (NEW)
│   └── podcast/                   # Podcast content type (MOVED from top-level)
│       ├── handler.py             # PodcastContentHandler
│       ├── entities.py            # Episode, RssFeed (from models/)
│       └── config.py              # Podcast-specific config fields
├── transports/                    # Transport implementations (NEW)
│   └── rss/                       # RSS transport (MOVED from rss/)
│       ├── parser.py
│       ├── downloader.py
│       └── source.py              # RssSource implementing ContentSource
├── providers/                     # Provider system (UNCHANGED)
├── gi/                            # GI extraction (UNCHANGED, uses ContentItem)
├── kg/                            # KG extraction (UNCHANGED, uses ContentItem)
├── builders/                      # Bridge builder (RFC-072, UNCHANGED)
├── enrichment/                    # Enrichment layer (RFC-073, UNCHANGED)
│   ├── protocol.py                # Enricher, EnricherManifest, EpisodeArtifactBundle
│   ├── registry.py                # Builtin enricher registry
│   ├── enrichment_pass.py         # Two-phase runner
│   └── builtin/                   # Deterministic, embedding, ML, LLM enrichers
├── summarization/                 # Summarization (UNCHANGED)
├── transcription/                 # Transcription (UNCHANGED)
├── search/                        # Vector search (UNCHANGED)
├── server/                        # FastAPI server (UNCHANGED -- enrichment routes added)
├── workflow/                      # Orchestration (REFACTORED to use core/)
├── cli.py                         # CLI (calls podcast-specific entry point)
├── config.py                      # Full Config (composes ProcessingConfig)
└── service.py                     # Service API (calls podcast-specific entry point)
```

### What Does NOT Change

- **Provider system** -- already content-agnostic, operates on text
- **GI / KG extraction** -- operates on text + metadata, not RSS
- **Summarization** -- operates on text
- **Search / vector indexing** -- operates on documents with metadata
- **Enrichment layer (RFC-073)** -- the enricher protocol (`Enricher`,
  `EnricherManifest`, `EpisodeArtifactBundle`) is already content-agnostic.
  Enrichers read core artifacts (GIL, KG, bridge) and produce derived signals.
  When the pipeline processes non-podcast content types, enrichers work
  unchanged as long as the content type produces the same artifact shapes.
  Content-type-specific enrichers (if needed) register via the same registry.
- **Viewer / server** -- displays artifacts, already generic enough. The
  enrichment layer adds new server routes (e.g. `GET /api/topics/{slug}` for
  PRD-026, extended `POST /api/search` for PRD-027) and viewer surfaces
  (Topic Entity View, Enriched Search panel), but these consume artifacts
  generically and do not hardcode podcast assumptions.
- **Prompt templates** -- `PromptStore` structure unchanged; content-type-aware
  variants are a natural extension
- **CLI** -- stays `podcast-scraper <rss_url>`, calls podcast-specific entry point
- **All existing tests and acceptance configs** -- podcast behavior is preserved

---

## Part 5: Refactoring Sequence

The refactoring is **not speculative** -- podcast + RSS validate every interface.

### Phase 1: Define core protocols and manifests

- Create `core/protocols.py` with `ContentItem`, `ContentSource`,
  `ContentTypeHandler`, `ProcessingConfig`
- Create `core/manifests.py` with `ContentTypeManifest`, `TransportManifest`
- Create `core/registry.py` with `PluginRegistry`
- Podcast `Episode` satisfies `ContentItem` (adapter or structural typing)
- RSS parser satisfies `ContentSource`
- No behavior changes -- just interface and schema definitions

### Phase 2: Implement podcast + RSS as plugins

- Create `content_types/podcast/` with `PodcastContentType` plugin class,
  `PodcastContentHandler`, and `PODCAST_MANIFEST`
- Create `transports/rss/` with `RssTransport` plugin class, `RssSource`,
  and `RSS_MANIFEST`
- Register both via entry points in `pyproject.toml`
- Registry discovers them; existing behavior unchanged

### Phase 3: Split orchestration

- Extract generic `run_pipeline(items, config, handler)` from current
  `run_pipeline(cfg)`
- Current `run_pipeline(cfg)` becomes a thin wrapper that does RSS fetch ->
  convert to `ContentItem` -> call generic pipeline
- All tests pass unchanged

### Phase 4: Reorganize modules

- Move RSS code under `transports/rss/`
- Move podcast entities under `content_types/podcast/`
- Create `core/` package
- Update imports (this is the noisiest part but mechanically simple)

### Phase 5: Extract ProcessingConfig

- Split `Config` into `ProcessingConfig` (generic) + podcast-specific fields
- `Config` composes `ProcessingConfig`
- External modules can construct `ProcessingConfig` independently
- Add `sources` config field (optional, alongside `rss_url` for backwards compat)
- Config validation uses registry + manifest schemas

### Phase 6: CLI plugin commands

- Add `podcast-scraper plugins` subcommand for listing installed plugins
  and validating config wiring
- Useful for debugging "is my external module installed correctly?"

**Each phase is independently shippable and all tests pass after each phase.**

---

## Part 6: Platform Anticipations

### Executive Summary

**Do platform first, content evolution later.** But make four specific
anticipations during platform work -- each costs almost nothing now (a column
name, a field in a payload) but saves schema migrations and rework later.

Two v2.x efforts -- RFC-072 (Canonical Identity Layer) and RFC-073
(Enrichment Layer) -- are already content-agnostic by design. Their patterns
(canonical IDs, bridge artifact, enricher protocol, plugin registry) are
**implemented precursors** that the v3.0 content evolution will generalise.
The anticipations below should leverage these existing patterns.

| # | Anticipation | Where It Applies | Cost Now | Cost If Skipped |
| --- | --- | --- | --- | --- |
| 1 | Generic content identity in Postgres | RFC-051 schema | Column naming + CIL projection tables | `ALTER TABLE` on every projected table + all queries |
| 2 | Generic catalog/source model | Part A catalog tables | One extra column | New table + migration + subscription model rework |
| 3 | Generic job payload envelope | Part B worker jobs | JSON field naming + `enrichment_enabled` | Job schema migration + worker code changes |
| 4 | Generic pipeline fingerprint key | A.4 dedup logic | Variable naming | Dedup key migration + reprocessing risk |

### Anticipation #1: Postgres Schema -- Generic Content Identity

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

**Canonical identity columns (RFC-072 alignment):** The Postgres schema should
also include columns for RFC-072 canonical IDs. The bridge artifact
(`bridge.json`) already emits `person:{slug}`, `org:{slug}`, `topic:{slug}`
identities per episode. When projecting to Postgres, these canonical IDs
become the natural join keys for cross-layer and cross-content-type queries:

```sql
CREATE TABLE canonical_identities (
    canonical_id TEXT PRIMARY KEY,       -- e.g. "person:lex-fridman", "topic:ai-regulation"
    identity_type TEXT NOT NULL,         -- "person", "org", "topic" (from CIL)
    display_name TEXT NOT NULL,
    aliases TEXT[],                      -- from bridge.json aliases
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_canonical_identities_type ON canonical_identities(identity_type);

CREATE TABLE content_item_identities (
    content_item_id TEXT NOT NULL,       -- FK to content_items
    canonical_id TEXT NOT NULL,          -- FK to canonical_identities
    source_gi BOOLEAN DEFAULT false,     -- from bridge.json sources.gi
    source_kg BOOLEAN DEFAULT false,     -- from bridge.json sources.kg
    PRIMARY KEY (content_item_id, canonical_id)
);

CREATE INDEX idx_cii_canonical ON content_item_identities(canonical_id);
```

This is the Postgres projection of `bridge.json` -- the same canonical IDs
that the filesystem bridge provides for cross-layer joins become SQL-queryable.
When non-podcast content types arrive, their bridge artifacts produce the same
canonical IDs, and the join table works unchanged.

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

**Enrichment outputs (RFC-073 alignment):** If projecting enrichment outputs
to Postgres, the same generic naming applies. Enricher outputs reference
`content_item_id` and canonical IDs from the bridge:

```sql
CREATE TABLE enrichment_outputs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_item_id TEXT NOT NULL,       -- episode or other content item
    enricher_name TEXT NOT NULL,         -- e.g. "topic_cooccurrence", "temporal_velocity"
    enricher_scope TEXT NOT NULL,        -- "episode" or "corpus"
    content_type TEXT NOT NULL DEFAULT 'podcast',
    output JSONB NOT NULL,              -- enricher-specific output
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_enrichment_content_item ON enrichment_outputs(content_item_id);
CREATE INDEX idx_enrichment_enricher ON enrichment_outputs(enricher_name);
```

**What this costs now:** Nothing functional. It's column names and two extra
columns (`content_type`, `source_transport`) with defaults. All queries work
the same -- `WHERE content_item_id = ?` instead of `WHERE episode_id = ?`.
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

### Anticipation #2: Catalog Table -- Generic Source Model

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
`feeds` -- one extra column (`source_type`), `config` as JSONB instead of
`rss_url` as a dedicated column, and generic naming. The `feeds` view
provides backwards compatibility for any code that expects the old shape.

**What skipping this costs later:** New `sources` table + data migration from
`feeds`. Update `subscriptions` foreign key from `feed_id` to `source_id`.
Update all catalog API endpoints. Update the scheduler that polls feeds.
Update the viewer's feed list. Potentially break the subscription model if
`feed_id` is baked into multiple layers.

**Subscription model stays the same** -- it's tenant <-> source. The only
change is the column name (`source_id` instead of `feed_id`).

**Scheduler impact:** The scheduler (B.7.1) that polls feeds becomes
"poll sources." The query changes from:

```sql
-- Before:
SELECT * FROM feeds WHERE enabled = true AND last_polled_at < now() - interval '1 hour';

-- After:
SELECT * FROM sources WHERE enabled = true AND last_polled_at < now() - poll_interval_seconds * interval '1 second';
```

The scheduler doesn't need to know about content types -- it just polls
enabled sources and enqueues jobs. The worker determines what to do based
on `source_type`, `transport`, and `content_type`.

### Anticipation #3: Job Payload -- Generic Envelope

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
    """Worker job handler -- content-type-agnostic envelope."""
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
The worker ignores `content_type` and `source_transport` in v2.x -- they're
just metadata. The `source_config` dict replaces what would have been
top-level `rss_url` and `episode_guid` fields.

**What skipping this costs later:** Job payload schema migration. All
in-flight jobs in Redis become incompatible. Worker code needs to handle
both old and new payload formats during rollout. Dead-letter queue items
from the old format need manual reprocessing or a migration script.

### Anticipation #4: Pipeline Fingerprint -- Generic Dedup Key

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

### Anticipation #5: Adaptive Routing Interface (RFC-053)

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

### Summary: What to Do During Each Platform RFC

| Platform RFC/Feature | Specific Anticipation | Concrete Action |
| --- | --- | --- |
| **RFC-051** (Postgres projection) | Anticipation #1 | Use `content_item_id`, `source_id`, `content_type`, `source_transport` in all table schemas. Add `content_type` index. Project RFC-072 bridge identities into `canonical_identities` + `content_item_identities` join table. Project RFC-073 enrichment outputs into `enrichment_outputs` table. |
| **Part A** (catalog + subscriptions) | Anticipation #2 | Name the table `sources` (not `feeds`). Add `source_type`, `transport`, `content_type` columns. Use `config JSONB` for transport-specific fields. Create `feeds` view for backwards compat. |
| **Part B** (workers + queues) | Anticipation #3 | Job payload uses `content_item_id`, `source_id`, `content_type`, `source_transport`, `source_config` dict. Worker dispatches on these fields (v2.x: always podcast). Include `enrichment_enabled` in processing config. |
| **A.4** (pipeline fingerprinting) | Anticipation #4 | Dedup key is `content_item_id:fingerprint`. Caller computes `content_item_id` (podcast module uses existing `generate_episode_id`). |
| **RFC-053** (adaptive routing) | Anticipation #5 | Routing input includes `content_type` and `prompt_profile`. Routing rules are content-type-scoped. |
| **RFC-062** (viewer/server) | No change needed | Viewer already operates on artifacts with UXS hub-and-spoke model. New content types add feature UXS files; shared design system (UXS-001) is reused. |
| **RFC-072** (CIL + bridge) | Already shipped | Canonical IDs (`person:`, `org:`, `topic:`) are content-agnostic. Bridge artifact pattern works for any content type. Postgres projection should use these as join keys (see Anticipation #1). |
| **RFC-073** (enrichment layer) | Already content-agnostic | Enricher protocol reads core artifacts, produces derived signals. No anticipation needed -- the protocol works for any content type that produces GIL/KG/bridge artifacts. |
| **Part E** (observability) | No change needed | Add `content_type` as a label/dimension on metrics (e.g., `pipeline_duration{content_type="podcast"}`). Free metadata. |
| **B.16** (Alembic migrations) | No change needed | Having Alembic from day one makes v3.0 schema changes (if any) trivial to manage. |

### What Does NOT Need Anticipation

These platform items are already content-agnostic or don't interact with
content identity:

- **Docker Compose topology** -- container layout is about resource profiles,
  not content types
- **Redis / arq setup** -- queue infrastructure is content-agnostic; the job
  payload is what matters (Anticipation #3)
- **Auth evolution** (A.12) -- API keys and JWT are content-agnostic
- **Observability stack** (Part E) -- Prometheus, Grafana, structured logging
  work the same regardless of content types (just add `content_type` label)
- **Deployment lifecycle** (Part F) -- CI/CD, rollback, secrets management
  are content-agnostic
- **Digest features** (Part C) -- operate on projected tables; if those tables
  use generic naming (Anticipation #1), digest queries work for any content
  type without changes
- **Semantic search** (RFC-061) -- vector store indexes documents with metadata;
  adding `content_type` to metadata is trivial and can be done at any time
- **Canonical Identity Layer** (RFC-072) -- `person:{slug}`, `org:{slug}`,
  `topic:{slug}` are already content-agnostic. The bridge artifact and CIL
  patterns work unchanged for any content type that produces GIL/KG artifacts.
  The slugifier is a shared utility.
- **Enrichment layer** (RFC-073) -- enricher protocol reads core artifacts
  (GIL, KG, bridge) and produces derived signals. Content-agnostic by design.
  Enrichers that operate on transcript text (e.g. `insight_density`) will need
  content-type-specific variants, but the protocol and registry are generic.
- **GI/KG viewer** (RFC-062) -- now uses a hub-and-spoke UXS model (UXS-001
  shared design system + feature-specific UXS files). New content types add
  feature UXS files; shared tokens and layout primitives are reused. New
  server routes for enrichment (PRD-026, PRD-027) and CIL queries
  (`/api/persons/*`, `/api/topics/*`) are content-agnostic.
- **Evaluation framework** (RFC-041/057) -- runs experiments on pipeline output;
  content-type-aware evaluation is a v3.x concern

---

## Part 7: Sequencing and Dependencies

### Phase 1: v2.x -- Platform Infrastructure (Podcast-Only)

All of these proceed as planned. Content evolution does not block or change
them, except for the four anticipations noted in Part 6.

| Item | Blueprint Ref | Status | Content Evolution Impact |
| --- | --- | --- | --- |
| GI/KG Viewer v2 | RFC-062, Phase D | Shipped | **None.** Pure podcast UI. UXS split (UXS-001 through UXS-008) already separates shared design system from feature-specific contracts -- extends naturally to v3 content-type views. |
| Semantic search | RFC-061 | Shipped | **None.** Operates on text + embeddings, already generic. |
| Canonical Identity Layer + bridge | RFC-072 | Shipped | **PRECURSOR.** `person:{slug}`, `org:{slug}`, `topic:{slug}` are content-agnostic canonical IDs. The bridge artifact and CIL patterns are the **implemented foundation** for cross-content-type entity resolution (Phase 3 / F.3). Postgres schema (Anticipation #1) should use these canonical IDs directly. |
| Enrichment layer | RFC-073 | In progress | **None.** Enricher protocol is content-agnostic -- reads core artifacts (GIL, KG, bridge), produces derived signals. Works unchanged for any content type that produces the same artifact shapes. First consumers: PRD-026 (Topic Entity View), PRD-027 (Enriched Search). |
| Evaluation framework | RFC-041/057 | In progress | **None.** Needed later for content-type prompt tuning. |
| Postgres projection | RFC-051, Phase C | Next major | **ANTICIPATE** -- see Anticipation #1 and #2 in Part 6. RFC-072 canonical IDs (`person:{slug}`, `topic:{slug}`) should be the identity columns, not ad-hoc slugs. |
| Catalog + subscriptions | Part A, Phase A | Planned | **ANTICIPATE** -- see Anticipation #2 in Part 6. |
| Workers + queues | Part B, simple tier | Planned | **ANTICIPATE** -- see Anticipation #3 in Part 6. |
| Pipeline fingerprinting | A.4 | Planned | **ANTICIPATE** -- see Anticipation #4 in Part 6. |
| Adaptive routing | RFC-053 | Planned | **Design for content type input** -- add `content_type` as a routing dimension in the routing interface, even if only `"podcast"` exists at first. |
| Docker Compose | Part B | Planned | **None.** Container topology is content-agnostic. |
| Observability | Part E | Planned | **None.** Metrics and logging are content-agnostic. |
| Auth (stages 1-2) | A.12 | Planned | **None.** API key auth is content-agnostic. |
| Alembic migrations | B.16 | Planned | **None.** Migration tooling is content-agnostic. Having Alembic from day one makes the v3.0 schema changes easier. |
| Digest features | Part C | Planned | **None for v0/v1.** Digest operates on projected tables -- if those tables use generic naming (Anticipation #1), digest queries generalize automatically. |

### Phase 2: v3.0 -- Content Evolution Refactoring

Happens after platform v1 is operational. Depends on platform infrastructure
being in place.

| Item | Content Evolution Ref | Depends On (Platform) |
| --- | --- | --- |
| Core protocols + registry | Phase 1-2 | Postgres schema (to know what `ContentItem` maps to in DB). **Note:** RFC-072 CIL patterns (`person:{slug}`, `topic:{slug}`, bridge artifact) and RFC-073 enricher protocol are already content-agnostic -- the core protocols should reference these as proven patterns for identity and enrichment. |
| Podcast + RSS as plugins | Phase 2-3 | Nothing -- internal refactoring |
| Module reorganization | Phase 4 | Nothing -- internal refactoring |
| ProcessingConfig extraction | Phase 5 | Worker implementation (to know what config workers consume). Should include `enrichment_enabled` (RFC-073). |
| Multi-source config (`sources:` field) | Phase 5 | Catalog model in Postgres (to know how sources are stored) |
| CLI plugin commands | Phase 6 | Registry implementation (Phase 1) |

### Phase 3: v3.x -- Cross-Content Features

Happens after at least one external module (e.g., news-ingest) is built.

| Item | Content Evolution Ref | Depends On |
| --- | --- | --- |
| Cross-content-type KG | F.3 | Postgres projection + semantic search + at least 2 content types producing KG. **Note:** RFC-072 CIL canonical IDs (`person:{slug}`, `topic:{slug}`) and bridge artifact are the **implemented foundation** -- the same IDs produced by different content types automatically resolve to the same canonical entity. |
| Entity resolution | F.3 | Cross-content KG + canonical entity registry design. RFC-072 `slugify` handles deterministic cases; ambiguous cases (name variations) need embedding similarity or LLM canonicalization. |
| Viewer v3 (mixed content) | F.2 | Multiple content types in production + UX spec. Current UXS hub-and-spoke model (UXS-001 shared design system + feature UXS files) extends naturally. |
| Content-type-aware scheduling | F.5 | Workers operational + metrics showing contention across types |
| Dedicated pipelines | F.5 | Distributed tier (D.7 v2) + content-type queue routing |

### Naming Convention Reference

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
| `person:{slug}` | `speaker:{slug}` (GIL), `entity:person:{slug}` (KG) | CIL canonical IDs (RFC-072), bridge.json, Postgres `canonical_identities`, API responses, cross-layer joins |
| `org:{slug}` | `entity:organization:{slug}` (KG) | CIL canonical IDs (RFC-072), bridge.json, Postgres `canonical_identities`, API responses |
| `topic:{slug}` | *(already shared, now formalised)* | CIL canonical IDs (RFC-072), bridge.json, Postgres `canonical_identities`, enrichment outputs (RFC-073) |

**Exception:** Inside the podcast module itself (`content_types/podcast/`,
`transports/rss/`), continue using `Episode`, `RssFeed`, `episode_id`,
`feed_url` -- these are the domain-specific names. The generic names are
for the platform layer (Postgres, workers, API, routing) that sits above
content-type-specific code.

```python
# Podcast module (content-type-specific) -- uses domain language:
episode = Episode(title="Planet Money #1234", item=xml_element, ...)
episode_id = generate_episode_id(feed_url=rss_url, guid=guid)

# Platform layer (generic) -- uses generic language:
content_item_id = episode_id  # same value, different name at this layer
source_id = generate_source_id(rss_url)
project_content_item(content_item_id=content_item_id, source_id=source_id, ...)
```

---

## Part 8: PR Roadmap

Concrete PR boundaries for the full journey from current state through
platform v1 to content evolution v3.0. PRs are grouped into streams that
can run in parallel where noted. Each PR is self-contained: tests pass,
CI green, no broken intermediate states.

PRs are intentionally heavy -- each delivers a coherent chunk of value.

### Stream A: Podcast Capabilities

These complete the podcast feature set and are prerequisites for everything
else. Items marked "shipped" are implemented and exercised in CI.

```text
PR-A1  GI/KG Viewer v2 (RFC-062)                              [SHIPPED]
       ──────────────────────────
       Scope: Complete viewer with Playwright E2E, UXS hub-and-spoke
              model (UXS-001 shared design system + UXS-002 through
              UXS-006 for feature surfaces).
       Delivers: Complete viewer for podcast GI/KG data.

PR-A2  Semantic search (RFC-061)                               [SHIPPED]
       ─────────────────────────
       Scope: Vector store, embedding indexing, search API endpoint,
              viewer search panel, chunk-to-Insight lift.
       Delivers: Full-text + semantic search over podcast corpus.

PR-A3  Canonical Identity Layer + bridge (RFC-072)             [SHIPPED]
       ───────────────────────────────────────────
       Scope: Canonical slugifier, CIL namespace (person:/org:/topic:),
              GIL ontology migration (speaker: -> person:), KG ontology
              migration (entity:person: -> person:), bridge.json emission,
              GIL v1.1 fields (insight_type, position_hint), CIL read
              routes (/api/persons/*, /api/topics/*), chunk-to-Insight
              lift in semantic search, viewer CIL awareness.
       Delivers: Cross-layer identity foundation. Enables cross-episode
              person intelligence, topic threading, enriched search lift.
       Impact on content evolution: person:{slug}, org:{slug}, topic:{slug}
              are content-agnostic -- the same canonical IDs will work for
              any content type that produces GIL/KG artifacts.

PR-A4  Enrichment layer (RFC-073)                           [IN PROGRESS]
       ──────────────────────────
       Scope: Enricher protocol, registry, enrichment pass in pipeline,
              deterministic enrichers (topic_cooccurrence, temporal_velocity,
              grounding_rate, guest_coappearance, insight_density),
              LLM enrichers (llm_search_synthesis, llm_position_narration,
              profile_synthesis), server routes for enrichment outputs.
       Delivers: Fourth artifact tier for derived signals. First consumers:
              PRD-026 (Topic Entity View), PRD-027 (Enriched Search).
       Impact on content evolution: Enricher protocol is content-agnostic --
              reads core artifacts via EpisodeArtifactBundle, produces
              derived signals. Works unchanged for any content type.

PR-A5  Evaluation framework maturity (RFC-041/057)
       ────────────────────────────────────────────
       Scope: Scorer, baseline comparison, experiment runner,
              golden dataset management.
       Delivers: Ability to benchmark prompt/model combinations.
              Needed later for content-type-aware routing (F.4).
       Depends on: Nothing (in progress, parallel with A3/A4).
```

### Stream B: Platform Infrastructure (v2.x)

These introduce the platform layer. **Anticipations #1-5 apply here** --
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
       Size: Large -- new package, schema, migrations, projection logic,
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
       Size: Medium-large -- new tables, models, CLI commands. ~1000-1500 lines.

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
       Size: Large -- new worker process, queue integration, CLI, Compose,
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
       Size: Medium -- route handlers, schemas, tests. ~800-1200 lines.

PR-B5  Observability foundation
       ────────────────────────
       Scope:
         - Prometheus metrics endpoint (`/metrics`)
         - Key metrics: `pipeline_duration_seconds{content_type}`,
           `queue_depth{queue}`, `jobs_processed_total{status}`,
           `episodes_processed_total{content_type}`
         - Structured JSON logging with correlation IDs (API -> worker)
         - Health check endpoints for all services
         - Grafana dashboard JSON (importable)
         - Docker Compose update: add Grafana + Prometheus services
       Delivers: Part E foundation. Visibility into platform operations.
       Depends on: PR-B3 (workers exist to emit metrics).
       Anticipations: `content_type` as metric label from day one.
       Size: Medium -- metrics, logging, dashboard, Compose. ~800-1200 lines.

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
       Size: Small -- middleware, env var, tests. ~200-400 lines.

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
       Size: Medium -- Compose files, Caddy config, scripts, docs.
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
       Size: Medium -- routing logic, integration, tests. ~800-1200 lines.
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
           `core/protocols.py` -- ContentItem, ContentSource,
             ContentTypeHandler, ProcessingConfig protocols
           `core/manifests.py` -- ContentTypeManifest, TransportManifest
             dataclasses (frozen, with JSON Schema fields)
           `core/registry.py` -- PluginRegistry (entry point discovery,
             get/list content types and transports, validate wiring)
           `core/testing.py` -- assert_valid_content_item,
             assert_valid_manifest, assert_pipeline_compatible helpers
         - Episode satisfies ContentItem (structural typing -- no changes
           to Episode itself, just verify protocol compliance in tests)
         - RSS parser satisfies ContentSource (same -- verify, don't change)
         - **RFC-072 alignment:** ContentItem protocol should include a
           method or property for emitting canonical CIL identities
           (person:/org:/topic: slugs). The existing `identity/slugify.py`
           and bridge builder patterns are the reference implementation.
           PluginRegistry should validate that content types can produce
           bridge-compatible identity declarations.
         - **RFC-073 alignment:** ProcessingConfig protocol should include
           `enrichment_enabled: bool`. The enricher protocol and registry
           are already content-agnostic -- they read artifacts via
           EpisodeArtifactBundle and produce derived signals. The core
           PluginRegistry can reuse the enricher registry pattern for
           plugin discovery via entry points.
         - Unit tests for registry, manifests, protocol compliance
         - NO behavior changes -- existing pipeline untouched
       Delivers: Content Evolution Phase 1. The interface layer exists.
       Depends on: PR-B1 (Postgres schema informs ContentItem fields).
              RFC-072 CIL patterns and RFC-073 enricher protocol are
              reference implementations for identity and enrichment.
       Size: Medium -- new package, protocols, registry, tests.
             ~800-1200 lines.

PR-D2  Podcast + RSS as plugins + module reorganization
       ─────────────────────────────────────────────────
       Scope:
         - Create `content_types/podcast/`:
           `handler.py` -- PodcastContentType, PodcastContentHandler
           `manifest.py` -- PODCAST_MANIFEST
           `entities.py` -- move Episode, RssFeed from `models/entities.py`
             (keep re-exports in old location for backwards compat)
           `config.py` -- podcast-specific config fields (extracted later
             in PR-D4, but namespace created here)
         - Create `transports/rss/`:
           `source.py` -- RssTransport, RssSource (wraps existing parser)
           `manifest.py` -- RSS_MANIFEST
           Move `rss/parser.py`, `rss/downloader.py` here
             (keep re-exports in old `rss/` for backwards compat)
         - Register both via entry points in `pyproject.toml`
         - Registry discovers them in tests
         - **RFC-072 alignment:** PodcastContentHandler declares that it
           produces bridge.json (CIL identities). The manifest includes
           `produces_bridge: true`. The existing bridge_builder.py is
           the reference implementation.
         - **RFC-073 alignment:** PodcastContentHandler declares that it
           supports enrichment (enricher protocol). The manifest includes
           `supports_enrichment: true`. Existing enrichers work unchanged.
         - Update imports across codebase (the noisy part -- but mechanical)
         - Backwards-compat re-exports so external code importing from
           old paths still works
         - ALL existing tests pass unchanged
       Delivers: Content Evolution Phases 2 + 4. Podcast and RSS are
                 plugins. Module structure reflects the architecture.
       Depends on: PR-D1 (protocols and registry exist).
       Size: Large -- module moves, import updates, entry points,
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
           (or continue using the wrapper -- both work)
         - Integration tests: generic pipeline with mock content items
         - ALL existing tests pass unchanged (they call the wrapper)
       Delivers: Content Evolution Phase 3. The pipeline core is generic.
                 External modules can call run_pipeline with their own
                 content items.
       Depends on: PR-D2 (podcast plugin provides ContentTypeHandler).
       Size: Medium -- orchestration refactoring, new generic entry point,
             tests. ~800-1500 lines.

PR-D4  ProcessingConfig extraction + multi-source config
       ──────────────────────────────────────────────────
       Scope:
         - Extract `ProcessingConfig` from `Config`:
           `core/config.py` -- ProcessingConfig Pydantic model with
             summary_provider, summary_model, gi_enabled, kg_enabled,
             enrichment_enabled (RFC-073), output_dir, workers,
             provider API keys/models, etc.
         - `Config` composes `ProcessingConfig` (has-a, not is-a)
         - `Config.processing` property returns ProcessingConfig
         - **RFC-073 alignment:** Include `enrichment` section in config
           with `enabled`, `opt_in` list, and per-enricher overrides.
           The existing enrichment config pattern is the reference.
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
                   PR-B3 (worker uses config -- update to ProcessingConfig).
       Size: Large -- Config refactoring, new model, validation, backwards
             compat, worker update, tests. ~1500-2000 lines.

PR-D5  CLI plugin commands + plugin developer docs
       ────────────────────────────────────────────
       Scope:
         - `podcast-scraper plugins` CLI subcommand:
           `podcast-scraper plugins --list` -- list installed content types
             and transports with manifest details
           `podcast-scraper plugins --validate config.yaml` -- validate
             config wiring against installed plugins
         - `core/testing.py` enhancements: full test helpers for plugin
           authors (assert_valid_content_item, assert_valid_manifest,
           assert_pipeline_compatible, mock_content_item factory)
         - Documentation:
           `docs/guides/PLUGIN_DEVELOPMENT_GUIDE.md` -- how to build an
             external module (package structure, manifest, entry points,
             testing, publishing)
           Update `docs/architecture/ARCHITECTURE.md` with content
             evolution section
       Delivers: Content Evolution Phase 6. Plugin ecosystem is ready
                 for external module development.
       Depends on: PR-D4 (multi-source config, full registry integration).
       Size: Medium -- CLI commands, test helpers, documentation.
             ~800-1200 lines.
```

### Visual: PR Dependency Graph

```text
Stream A (podcast capabilities)
  A1 (viewer v2, RFC-062)          [SHIPPED] ──────────────────┐
  A2 (semantic search, RFC-061)    [SHIPPED] ──────────────────┤
  A3 (CIL + bridge, RFC-072)      [SHIPPED] ──────────────────┤
  A4 (enrichment layer, RFC-073)   [IN PROGRESS] ─────────────┤
  A5 (eval framework)              [IN PROGRESS] ─────────────┤
                                                               │
Stream B (platform infrastructure -- v2.x)                     │
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
Stream C (adaptive routing -- parallel with late B)            │
  C1 (routing framework) ─────────────────────────────────────┤
                                                               │
                                                               ▼
                                                    Platform v1 operational
                                                               │
Stream D (content evolution -- v3.0)                           │
  D1 (core protocols + registry) ◄─────────────────────────────┘
       │  (references RFC-072 CIL patterns + RFC-073 enricher protocol)
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
```

### PR Summary Table

| PR | Stream | Name | Size | Status | Depends On | Anticipations |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | A | Viewer v2 (RFC-062) | Large | Shipped | -- | -- |
| A2 | A | Semantic search (RFC-061) | Large | Shipped | -- | -- |
| A3 | A | CIL + bridge (RFC-072) | Large | Shipped | -- | -- |
| A4 | A | Enrichment layer (RFC-073) | Large | In progress | -- | -- |
| A5 | A | Eval framework | Medium | In progress | -- | -- |
| B1 | B | Postgres + projection | Large | -- | -- | #1 (generic schema + CIL projection) |
| B2 | B | Catalog + sources | Medium-large | -- | B1 | #2 (generic catalog) |
| B3 | B | Workers + queues | Large | -- | B1, B2 | #3 (generic job + enrichment_enabled), #4 (generic dedup) |
| B4 | B | Platform API routes | Medium | -- | B2, B3 | -- |
| B5 | B | Observability | Medium | -- | B3 | `content_type` metric label |
| B6 | B | Auth stage 2 | Small | -- | B4 | -- |
| B7 | B | Docker prod profile | Medium | -- | B1-B6 | -- |
| C1 | C | Adaptive routing | Medium | -- | -- | #5 (content type in routing) |
| D1 | D | Core protocols + registry | Medium | -- | B1 | RFC-072 CIL patterns, RFC-073 enricher protocol |
| D2 | D | Podcast/RSS as plugins | Large | -- | D1 | bridge + enrichment manifest flags |
| D3 | D | Generic orchestration | Medium | -- | D2 | -- |
| D4 | D | ProcessingConfig + multi-source | Large | -- | D3, B3 | -- |
| D5 | D | CLI plugins + docs | Medium | -- | D4 | -- |

**Total: 18 PRs across 4 streams.** Stream A items A1-A3 are shipped;
A4-A5 are in progress. Streams A, B, and C can run in parallel. Stream D
is sequential (each PR depends on the previous).

---

## Part 9: Future Design Areas

These are known areas that need concrete design work before or during the v3.0
refactoring. Each will likely become its own RFC or section in this document.
Captured here to track the thinking; answers come from doing, not more planning.

### F.1 Storage and output layout for multiple content types

**Problem:** Today everything goes into `output/rss_<host>_<hash>/` -- a layout
derived from RSS feed identity (ADR-003). When news articles and social posts
exist alongside podcasts, the filesystem layout needs to accommodate multiple
content types and transports without breaking existing podcast output.

**Considerations:**

- Postgres (RFC-051) is the next major infrastructure addition. Once projection
  exists, the filesystem layout becomes less critical for reads (DB is the query
  surface) but still matters for canonical blob storage and CLI output.
- The layout should work for both filesystem-only (CLI, no DB) and
  filesystem + DB (platform mode) deployments.
- Existing podcast output paths must not change -- backwards compatibility.

**Sketch (needs RFC):**

```text
output/
├── rss_<host>_<hash>/              # Existing podcast layout (unchanged)
│   ├── 0001 - Episode Title.txt
│   ├── gi.json
│   └── kg.json
├── web_<domain>_<hash>/            # News articles by source domain
│   ├── <article_id>/
│   │   ├── content.txt
│   │   ├── gi.json
│   │   └── kg.json
│   └── ...
└── social_<platform>_<hash>/       # Social posts by platform + account/feed
    └── ...
```

Or a more generic layout:

```text
output/
├── <transport>_<source_hash>/
│   └── <content_item_id>/
│       ├── content.{txt,json}
│       ├── metadata.json
│       ├── gi.json
│       └── kg.json
```

**Dependency:** Postgres introduction (RFC-051). Design the filesystem layout
and DB schema together so they're consistent.

### F.2 Viewer / server for mixed content (v3 spec needed)

**Problem:** The viewer (RFC-062) currently shows podcast episodes with GI/KG
data. With multiple content types, the UI needs to present a unified view across
podcasts, articles, and social posts.

**This needs its own spec** -- probably a UXS (UX Spec) or RFC for viewer v3.

**Current state (v2.x):** The viewer UX documentation has been refactored into a
hub-and-spoke model:

- [UXS-001](../uxs/UXS-001-gi-kg-viewer.md) -- shared design system (tokens,
  typography, layout primitives, states)
- Feature-specific UXS files: [UXS-002](../uxs/UXS-002-corpus-digest.md)
  (Digest), [UXS-003](../uxs/UXS-003-corpus-library.md) (Library),
  [UXS-004](../uxs/UXS-004-graph-exploration.md) (Graph),
  [UXS-005](../uxs/UXS-005-semantic-search.md) (Search),
  [UXS-006](../uxs/UXS-006-dashboard.md) (Dashboard)
- New feature UXS for enrichment consumers:
  [UXS-007](../uxs/UXS-007-topic-entity-view.md) (PRD-026),
  [UXS-008](../uxs/UXS-008-enriched-search.md) (PRD-027)

This structure already separates shared design tokens from feature-specific
contracts. A v3 viewer for mixed content types would extend the same pattern --
new feature UXS files for content-type-specific views, shared tokens in UXS-001.

**Key questions:**

- Unified timeline vs content-type tabs vs faceted search?
- How does the KG visualization show cross-content-type connections?
- Content-type-specific detail views (podcast has audio player, article has
  reading view, social post has thread view)?
- Does the viewer need to know about content types at all, or does it just
  display artifacts generically?

**Likely approach:** The viewer operates on artifacts (GI, KG, bridge,
enrichments) which are content-type-agnostic. Content-type-specific rendering
(audio player, etc.) is handled by optional UI components that register per
content type -- mirroring the backend plugin architecture. The enrichment layer
(RFC-073) and its consumers (PRD-026 Topic Entity View, PRD-027 Enriched Search)
demonstrate this: they consume derived signals from any content type that
produces GIL/KG/bridge artifacts, without hardcoding podcast assumptions.

### F.3 Cross-content-type KG and entity resolution

**This is the core value proposition** of the multi-content-type vision. A person
mentioned in a podcast, quoted in a news article, and discussed in a social
thread should be the same entity in the knowledge graph.

**Precursor work (v2.x):** RFC-072 (Canonical Identity Layer) has already
established the foundation for cross-episode entity resolution within a single
content type. The bridge artifact (`*.bridge.json`) assigns canonical IDs
(`person:{slug}`, `topic:{slug}`) to entities across episodes, linking GIL and
KG nodes to stable identities. This is Phase 2 in the approach below -- and it
is **implemented**. The same canonical ID scheme and bridge structure are
designed to extend to cross-content-type resolution (Phase 3).

**The hard problems:**

- **Entity identification:** How do you know "Elon Musk" in a podcast transcript,
  "Mr. Musk" in a news article, and "@elonmusk" in a social post are the same
  entity? NER gives you mentions, not identities.
- **Entity resolution / record linkage:** Matching mentions to canonical entities.
  Options: Wikidata IDs, custom entity registry, embedding-based similarity.
  RFC-072's `slugify` approach (deterministic slug from display name) handles
  the simple case; ambiguous cases need the strategies above.
- **Scale:** A popular entity might have thousands or millions of links across
  content items. The KG data model and query patterns need to handle this
  without degrading. Postgres with proper indexing, or a dedicated graph DB
  (Neo4j, etc.) for the cross-source KG layer?
- **Incremental updates:** When a new article mentions a known entity, how do
  you link it without reprocessing the entire corpus?

**Phased approach:**

1. **Within-source KG** -- entities are per-episode, no cross-linking
2. **Within-type corpus KG (implemented -- RFC-072)** -- canonical identity
   resolution across episodes of the same podcast via bridge artifact.
   `person:{slug}` and `topic:{slug}` IDs are stable across episodes.
   Enrichers (RFC-073) consume these identities to compute cross-episode
   signals (topic co-occurrence, temporal velocity, grounding rate).
3. **Cross-type corpus KG** -- entity resolution across content types. This is
   where the real value is and where the hard problems live. The bridge
   structure from Phase 2 extends naturally -- a `person:elon-musk` in a
   podcast bridge and a `person:elon-musk` in a news bridge are the same
   canonical entity.
4. **Canonical entity registry** -- a maintained registry of known entities with
   stable IDs, aliases, and metadata. Possibly backed by Wikidata or a custom
   ontology.

**This is the most ambitious part of the vision and should be its own RFC.**

### F.4 Prompt strategy and content routing per content type

**Problem:** The same ML task (summarization, GI extraction, KG extraction)
needs very different approaches depending on content length and type:

| Content Type | Typical Length | Summarization Strategy | GI/KG Strategy |
| --- | --- | --- | --- |
| Podcast | 10k-50k words | MAP-REDUCE, long-form prompts | Full extraction pipeline |
| News article | 500-2000 words | Single-pass, article-tuned prompts | Lighter extraction, more entities |
| Social post | 10-280 chars | Skip or aggregate (summarize threads) | Entity + sentiment only |
| Video transcript | 5k-30k words | Similar to podcast | Similar to podcast |

**This connects to two existing planned features:**

- **RFC-053 (Adaptive Routing):** Already planned to select strategies based on
  content characteristics. Content type becomes a primary routing dimension.
- **`data/eval/` experiments:** The evaluation framework can benchmark different
  prompt/model combinations per content type. Results feed into the router's
  decision table.

**Sequence:**

1. Build the evaluation framework for podcasts (current work)
2. When the first non-podcast content type arrives, run eval experiments to find
   the right prompts/models for that type
3. Feed results into the adaptive router as content-type-aware routing rules
4. Over time, the router learns the best model for each (content_type, task,
   quality_target) combination

### F.5 Platform blueprint impact: content-type-aware scheduling

**Problem:** Different content types have very different processing costs and
volumes:

| Content Type | Processing Cost | Volume | Priority Pattern |
| --- | --- | --- | --- |
| Podcast | High (Whisper: minutes per episode) | Low (weekly per feed) | Batch, can wait |
| News article | Low (text only, seconds) | Medium (daily per source) | Near-real-time |
| Social post | Very low (tiny text) | High (continuous stream) | Aggregate, batch |

**Impact on the platform blueprint (Part B and D):**

- **Queue design (B.7):** Content-type-aware queue routing. Podcasts go to
  `heavy` queue (GPU). News articles go to `light` queue (CPU only, high
  concurrency). Social posts might aggregate before processing.
- **Worker pools (D.5):** Worker resource profiles should consider content type
  mix, not just pipeline stage. A deployment processing mostly news articles
  needs fewer GPU workers and more CPU workers than a podcast-heavy deployment.
- **Capacity planning (D.4):** The hardware sizing in Part D assumes podcast
  workloads. Need parallel capacity models for mixed workloads.

**Approach:** Don't redesign the blueprint now. Instead:

1. Add a section to the blueprint acknowledging content-type-aware scheduling
   as a future concern (cross-reference this document)
2. When the first non-podcast content type arrives, measure actual processing
   costs and adjust queue/worker design based on data
3. The two-tier queue model (simple -> distributed) in B.7 already supports
   this -- content type becomes another dimension in the distributed tier's
   queue routing

**Dedicated pipelines vs shared pipeline:** For high-volume content types
(social posts), it may make sense to run dedicated pipeline instances with
tuned concurrency and batching, rather than mixing them with podcast jobs
in the same queue. The platform blueprint's Compose topology (D.5) supports
this -- different `command` / queue subscription per worker service. Fine-tune
capacity by scaling workers per content-type queue independently.

### F.6 Timing: v3.0 refactoring after podcast capabilities are solid

**This is a v3.0 effort.** The refactoring to generic core + pluggable content
types happens after podcast capabilities are fully nailed down:

**Prerequisites (must be done first):**

- GI/KG viewer v2 complete (RFC-062) -- shipped
- Canonical Identity Layer + bridge (RFC-072) -- shipped (precursor to
  cross-content-type entity resolution)
- Enrichment layer (RFC-073) -- in progress (content-agnostic enricher
  protocol; first consumers: PRD-026, PRD-027)
- Postgres projection (RFC-051) -- next major infrastructure
- Semantic search (RFC-061) -- shipped
- Adaptive routing (RFC-053) -- planned
- Evaluation framework mature -- needed for content-type-aware routing
- Platform blueprint v1 operational -- workers, queues, basic multi-feed

**v3.0 refactoring sequence:**

1. Introduce core protocols and registry (Phase 1-2 from Refactoring Sequence)
2. Refactor podcast + RSS into plugin form (Phase 3-4)
3. Extract ProcessingConfig, add multi-source config (Phase 5-6)
4. Build first external module (news-ingest) to validate the architecture
5. Iterate on protocols based on real integration experience

**Parallel work during v2.x (before v3.0):**

- Keep module boundaries clean in current work -- don't add more podcast
  coupling to the processing pipeline
- RFC-072 (CIL/bridge) and RFC-073 (enrichment) are already content-agnostic
  by design -- their protocols work for any content type that produces
  GIL/KG/bridge artifacts
- When introducing Postgres, design the schema to be content-type-extensible
  (use generic `content_item_id` instead of `episode_id` in new tables)
- When building adaptive routing, design the routing interface to accept
  content type as an input dimension
- Enricher registry (RFC-073) uses the same module-attribute discovery pattern
  proposed for content-type plugins -- validates the approach before v3.0

This way, the v3.0 refactoring is smaller because v2.x work already
anticipated it.

### Envisioned module roadmap

| Module | Content Type | Transport(s) | Key Dependencies | Notes |
| --- | --- | --- | --- | --- |
| *(bundled)* | Podcast episode | RSS | whisper, spacy | Ships with this repo |
| `news-ingest` | News article | Web scraping | trafilatura | Medium text, no audio |
| `social-ingest` | Social post | Mastodon, Reddit, Bluesky | mastodon.py, asyncpraw, atproto | Short text, high volume |
| `video-ingest` | Video transcript | YouTube API | yt-dlp, whisper | Similar to podcast |
| `folder-ingest` | Mixed documents | Filesystem watcher | watchdog, pypdf | Local files, PDFs |
| `email-ingest` | Email / newsletter | IMAP, webhook | imaplib | Medium text, attachments |

---

## Part 10: Relation to Other Documents

| Document | Scope | Answers |
| --- | --- | --- |
| [Architecture](ARCHITECTURE.md) | Current state | How does the system work today? |
| [Platform Blueprint](PLATFORM_ARCHITECTURE_BLUEPRINT.md) | Infrastructure evolution | How do we scale to a platform? (tenancy, workers, queues, DB, deployment) |
| **This document** | Content evolution | How do we evolve beyond podcasts? (content types, transports, plugins, cross-source KG) |
| [RFC-072](../rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md) | Identity layer | How do entities get canonical IDs across episodes? (precursor to cross-content-type resolution) |
| [RFC-073](../rfc/RFC-073-enrichment-layer-architecture.md) | Enrichment layer | How do derived signals layer on top of core artifacts? (content-agnostic protocol) |
| [PRD-026](../prd/PRD-026-topic-entity-view.md) / [PRD-027](../prd/PRD-027-enriched-search.md) | Enrichment consumers | First features consuming enricher outputs (topic view, enriched search) |
| [Non-Functional Requirements](NON_FUNCTIONAL_REQUIREMENTS.md) | Quality attributes | What are the performance, reliability, security targets? |

**Cross-references needed:**

- Platform Blueprint should reference this doc in Part B (queue design) and
  Part D (worker sizing) for content-type-aware scheduling considerations
- This doc references the Platform Blueprint for infrastructure it depends on
  (Postgres, workers, queues)
- Architecture doc should reference this doc in the "Architecture Evolution"
  section as the v3.0 direction
- RFC-073 enrichment layer is content-agnostic and extends naturally to
  non-podcast content types (no changes needed when new types arrive)
- RFC-072 CIL/bridge is the implemented precursor to cross-content-type entity
  resolution (F.3) -- the canonical ID scheme (`person:{slug}`, `topic:{slug}`)
  is designed to work across content types

---

## Resolved Design Decisions

1. **Plugin discovery:** Python entry points via `importlib.metadata`. `pip install`
   is the registration mechanism. See Registry section in Part 3.

2. **Migration path:** Existing `Config` YAML files with `rss_url` keep working.
   The core synthesizes an implicit `sources` entry from `rss_url` for backwards
   compatibility. No user-facing changes required.

---

## Open Questions

1. **Package naming:** The package is called `podcast_scraper` but the core is
   generic. Options: (a) rename to something like `content_engine` (breaking),
   (b) keep `podcast_scraper` as the package name since podcast is the primary use
   case and the entry point, (c) publish `core/` as a separate thin package
   (`content-engine-core`) that both this repo and external modules depend on.
   Leaning toward (b) for now -- rename is high cost, low immediate value.

2. **Entry point group namespace:** Using `content_engine.content_types` and
   `content_engine.transports` as entry point groups. Should this match the
   package name (`podcast_scraper.content_types`) or use a neutral namespace?
   Neutral is better if external modules shouldn't appear to be sub-packages of
   `podcast_scraper`.

3. **Shared viewer:** External modules push artifacts into the same
   storage/projection layer. The viewer shows all content types with faceting.
   Cross-content-type KG (entity resolution across sources) is the long-term
   goal -- defer until at least two content types produce KG data.

4. **Versioning contract:** `core/protocols.py` and manifest dataclasses become
   the stable public API with semver guarantees. Internal implementation details
   (how the podcast module works) are not part of the public contract. Need to
   define what "breaking change" means for manifest schemas.

5. **Manifest schema evolution:** When a new content type needs manifest fields
   that don't exist yet, how do we extend without breaking existing plugins?
   Options: (a) optional fields with defaults (simplest), (b) manifest schema
   versioning with migration, (c) `extra: dict` escape hatch. Likely (a) for
   most cases, with (b) reserved for structural changes.

6. **Transport <-> content type cardinality:** Can one transport produce multiple
   content types? (e.g., a "web_scrape" transport might produce both
   "news_article" and "blog_post"). The manifest supports this via
   `compatible_content_types: frozenset`. But does the transport decide the
   content type per item, or does the user declare it in config? Probably
   config-declared for simplicity.
