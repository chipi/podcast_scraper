# Platform Extensibility Architecture Note

**Status:** Architectural note — captures direction for making the pipeline a
generic content-processing engine with podcast + RSS as the default bundled modules.

**Date:** 2026-04-05

**Context:** Thinking about evolving the platform beyond podcasts into news, social
feeds, and other content types. The core pipeline becomes generic; podcast logic and
RSS transport become two default modules that ship with the repo. Future content
types and transports live in separate repos as peer plugins.

---

## Core Principle: Generic Core, Pluggable Content Types and Transports

The pipeline core is **content-type-agnostic and transport-agnostic**. It processes
content items through configurable stages (transcription, summarization, GI, KG,
search indexing) without knowing where the content came from or what kind it is.

Two concepts are separated cleanly:

| Concept | Responsibility | Examples |
| --- | --- | --- |
| **Content type** | Defines what the content is, what processing stages apply, what metadata it carries | Podcast episode (long audio + transcript), news article (medium text), social post (short text) |
| **Transport** | Defines how content is acquired — fetching, parsing, normalizing into `ContentItem` | RSS feed polling, web scraping, social API, filesystem watcher, email ingest |

**Podcast** (content type) and **RSS** (transport) are the two default modules that
ship with this repo. They are implemented against the same interfaces that any
external module would use — no privileged internal access.

---

## Why Generic Core with Podcast as Default Module

### Advantages over "expose seams later"

1. **Interfaces are honest from day one.** Podcast + RSS must satisfy the generic
   protocols, which means those protocols are tested and proven by a real
   implementation — not theoretical contracts designed in a vacuum.

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

### External modules still own their dependencies

Even though the core is generic, content-type-specific and transport-specific
dependencies stay out of the core:

- News scraping needs `trafilatura` — lives in the news module's repo
- Social connectors need OAuth libraries — lives in the social module's repo
- Podcast needs `openai-whisper`, `spacy` — stays in this repo as part of the
  podcast content-type module (already optional deps)

---

## Architecture Layers

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

## Core Protocols

### ContentItem — what the pipeline processes

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

### ContentSource — how content is acquired

```python
class ContentSource(Protocol):
    """A transport that fetches content and yields ContentItems."""
    def fetch(self, config: SourceConfig) -> list[ContentItem]: ...
    def supports_incremental(self) -> bool: ...  # cursor-based polling
```

RSS becomes the first `ContentSource`. Web scraping, social APIs, filesystem
watchers are future implementations in external repos.

### ContentTypeHandler — content-type-specific processing decisions

```python
class ContentTypeHandler(Protocol):
    """Declares which pipeline stages apply to a content type."""
    content_type: str
    def needs_transcription(self, item: ContentItem) -> bool: ...
    def needs_summarization(self, item: ContentItem) -> bool: ...
    def get_prompt_profile(self, item: ContentItem) -> str: ...
    # e.g. "long_form" for podcasts, "short_text" for tweets
```

This integrates with RFC-053 (Adaptive Routing) — the content type handler is
the routing input.

### ProcessingConfig — config for the generic pipeline

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

## Registry, Schemas, and Discovery

The core needs three mechanisms to work with pluggable content types and transports:

1. **Manifests** — each content type and transport declares what it is, what it
   needs, and what it provides (self-describing schema)
2. **Registry** — runtime lookup of installed content types and transports
3. **Wiring** — config-driven mapping of "use transport X to feed content type Y"

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
    # e.g. "long_form" → prompts/<provider>/summarization/long_form_v1.j2
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

### Registry — Runtime Discovery

The registry discovers installed content types and transports via **Python entry
points** — the standard plugin mechanism. This is how pytest discovers plugins,
Flask discovers extensions, and pip discovers build backends.

**Registration (in `pyproject.toml`):**

```toml
# This repo — bundled defaults
[project.entry-points."content_engine.content_types"]
podcast = "podcast_scraper.content_types.podcast:PodcastContentType"

[project.entry-points."content_engine.transports"]
rss = "podcast_scraper.transports.rss:RssTransport"
```

```toml
# External repo (e.g. news-ingest)
[project.entry-points."content_engine.content_types"]
news_article = "news_ingest.content_types.news:NewsArticleContentType"

[project.entry-points."content_engine.transports"]
web_scrape = "news_ingest.transports.web:WebScrapeTransport"
```

`pip install news-ingest` is enough — the core discovers it automatically at
runtime. No manual config listing required.

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

### Wiring — Config-Driven Source Mapping

The wiring connects transports to content types in user config. Two modes:

**Implicit (backwards compatible):** Current `rss_url` in config implies
`transport=rss, content_type=podcast`. Existing configs work unchanged.

```yaml
# Current config — works as before, implicit wiring
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
  → If rss_url present and no sources: synthesize implicit source entry
  → For each source entry:
      1. Registry.get_transport(source.transport) — is it installed?
      2. Registry.get_content_type(source.content_type) — is it installed?
      3. Registry.validate_wiring(transport, content_type) — compatible?
      4. Validate source.config against transport.manifest.config_schema
  → Build ProcessingConfig from top-level fields
  → Pass to pipeline
```

### CLI Discovery Commands

The registry enables introspection commands:

```bash
# List installed content types
podcast-scraper plugins --content-types
# Output:
# podcast    (bundled)  Podcast Episode — stages: transcription, summarization, gi, kg
# news_article          News Article — stages: summarization, gi, kg

# List installed transports
podcast-scraper plugins --transports
# Output:
# rss        (bundled)  RSS Feed — content types: podcast
# web_scrape            Web Scraper — content types: news_article, blog_post

# Validate a config file's wiring
podcast-scraper plugins --validate config.yaml
# Output:
# [ok] source[0]: transport=rss, content_type=podcast — OK
# source[1]: transport=web_scrape — not installed (pip install news-ingest)
```

### Design Constraints

1. **Manifests are static and declarative.** They describe capabilities, not
   runtime state. No "call this function to find out what stages I need" — the
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
   likely need fields not listed here. That's fine — extend the manifest
   dataclass. The `version` field tracks schema evolution.

---

## What Changes in This Repo

### 1. Orchestration becomes generic

**Current:** `run_pipeline(cfg)` hardwires RSS fetch → `Episode` list → process.

**Target:** `run_pipeline` accepts `list[ContentItem]` + `ProcessingConfig`. A
higher-level `run_podcast_pipeline(cfg)` (or the CLI) does RSS fetch → convert
to `ContentItem` → call generic `run_pipeline`.

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

### 2. Entity model gains a generic layer

**Current:** `Episode` with `item: ET.Element` is the only entity.

**Target:** `ContentItem` protocol is the generic layer. `Episode` stays as the
podcast-specific implementation that satisfies `ContentItem`. The `ET.Element` stays
on `Episode` — it's podcast/RSS-specific detail that the generic pipeline never
touches.

### 3. Config splits into layers

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

### 4. Module organization

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

---

## What Does NOT Change

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

## Refactoring Sequence

The refactoring is **not speculative** — podcast + RSS validate every interface.

### Phase 1: Define core protocols and manifests

- Create `core/protocols.py` with `ContentItem`, `ContentSource`,
  `ContentTypeHandler`, `ProcessingConfig`
- Create `core/manifests.py` with `ContentTypeManifest`, `TransportManifest`
- Create `core/registry.py` with `PluginRegistry`
- Podcast `Episode` satisfies `ContentItem` (adapter or structural typing)
- RSS parser satisfies `ContentSource`
- No behavior changes — just interface and schema definitions

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
- Current `run_pipeline(cfg)` becomes a thin wrapper that does RSS fetch →
  convert to `ContentItem` → call generic pipeline
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

## Relation to Existing Architecture

- **Platform Blueprint:** Fully compatible. The multi-tenant platform (catalog,
  subscriptions, workers) becomes content-type-aware. The catalog holds feeds AND
  other sources. Workers process `ContentItem`s regardless of origin.

- **RFC-053 (Adaptive Routing):** `ContentTypeHandler` IS the routing input.
  Adaptive routing selects strategies based on content type + item characteristics.
  This is the natural integration point.

- **RFC-062 (Server/Viewer):** The viewer displays artifacts from any content type.
  The Postgres projection layer (RFC-051) is already keyed by episode/artifact IDs
  -- generalizing to content item IDs is straightforward.

- **RFC-072 (CIL / Bridge):** The Canonical Identity Layer and bridge artifact
  (`*.bridge.json`) establish cross-layer identity resolution (`person:{slug}`,
  `topic:{slug}`). This is a **direct precursor** to cross-content-type entity
  resolution (F.3 below) -- the same canonical IDs and bridge structure can link
  entities across content types, not just across episodes.

- **RFC-073 (Enrichment Layer):** The enricher protocol is content-agnostic by
  design -- enrichers read core artifacts (GIL, KG, bridge) and produce derived
  signals. When non-podcast content types produce the same artifact shapes, all
  existing enrichers work unchanged. Content-type-specific enrichers register via
  the same registry and protocol. The first consumers are PRD-026 (Topic Entity
  View) and PRD-027 (Enriched Search).

- **Provider System:** No changes. Providers operate on text/audio, not on content
  types. LLM-tier enrichers (RFC-073) and query-time enrichers (PRD-027) use the
  existing provider system -- no separate provider setup needed.

---

## External Modules — Packaging and Structure

External modules are **standard Python packages** — each with its own repo,
`pyproject.toml`, tests, and release cycle. `pip install` is the only integration
step. This is the same pattern used by pytest plugins, Flask extensions, and
flake8 checkers.

### What an external module looks like

Each module is a pip-installable package that:

1. Depends on `podcast-scraper` (for core protocols and processing)
2. Implements `ContentTypePlugin` and/or `TransportPlugin` protocols
3. Registers via entry points in `pyproject.toml`
4. Owns its own transport-specific and content-type-specific dependencies

### Example: news-ingest

```text
news-ingest/
├── pyproject.toml
├── README.md
├── src/news_ingest/
│   ├── __init__.py
│   ├── content_types/
│   │   └── news/
│   │       ├── __init__.py         # Exports NewsArticleContentType
│   │       ├── handler.py          # NewsContentHandler (ContentTypeHandler)
│   │       ├── manifest.py         # NEWS_ARTICLE_MANIFEST (ContentTypeManifest)
│   │       └── entities.py         # NewsArticle dataclass (satisfies ContentItem)
│   └── transports/
│       └── web_scrape/
│           ├── __init__.py         # Exports WebScrapeTransport
│           ├── source.py           # WebScrapeSource (ContentSource)
│           ├── manifest.py         # WEB_SCRAPE_MANIFEST (TransportManifest)
│           └── extractors.py       # trafilatura / readability wrappers
└── tests/
    ├── test_handler.py
    ├── test_source.py
    └── test_manifest.py
```

```toml
# news-ingest/pyproject.toml
[project]
name = "news-ingest"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "podcast-scraper>=2.0",         # Core protocols + processing
    "trafilatura>=1.6",             # Article extraction
    "readability-lxml>=0.8",        # Fallback extraction
]

[project.optional-dependencies]
dev = ["pytest", "pytest-cov"]

[project.entry-points."content_engine.content_types"]
news_article = "news_ingest.content_types.news:NewsArticleContentType"

[project.entry-points."content_engine.transports"]
web_scrape = "news_ingest.transports.web_scrape:WebScrapeTransport"
```

### Example: social-ingest (multiple transports, one content type)

A single package can ship multiple transports that all produce the same content
type. Each transport gets its own entry point.

```text
social-ingest/
├── pyproject.toml
├── src/social_ingest/
│   ├── __init__.py
│   ├── content_types/
│   │   └── social_post/
│   │       ├── __init__.py
│   │       ├── handler.py          # SocialPostContentHandler
│   │       ├── manifest.py         # SOCIAL_POST_MANIFEST
│   │       └── entities.py         # SocialPost (satisfies ContentItem)
│   └── transports/
│       ├── mastodon/
│       │   ├── __init__.py
│       │   ├── source.py           # MastodonSource (ContentSource)
│       │   └── manifest.py         # MASTODON_MANIFEST
│       ├── reddit/
│       │   ├── __init__.py
│       │   ├── source.py           # RedditSource
│       │   └── manifest.py         # REDDIT_MANIFEST
│       └── bluesky/
│           ├── __init__.py
│           ├── source.py           # BlueskySource
│           └── manifest.py         # BLUESKY_MANIFEST
└── tests/
```

```toml
# social-ingest/pyproject.toml
[project]
name = "social-ingest"
version = "0.1.0"
dependencies = [
    "podcast-scraper>=2.0",
    "mastodon.py>=1.8",
    "asyncpraw>=7.7",               # Reddit
    "atproto>=0.0.46",              # Bluesky AT Protocol
]

[project.entry-points."content_engine.content_types"]
social_post = "social_ingest.content_types.social_post:SocialPostContentType"

[project.entry-points."content_engine.transports"]
mastodon = "social_ingest.transports.mastodon:MastodonTransport"
reddit = "social_ingest.transports.reddit:RedditTransport"
bluesky = "social_ingest.transports.bluesky:BlueskyTransport"
```

### This repo's entry points (bundled defaults)

Podcast and RSS are registered the same way — no special treatment.

```toml
# podcast-scraper/pyproject.toml (this repo)
[project.entry-points."content_engine.content_types"]
podcast = "podcast_scraper.content_types.podcast:PodcastContentType"

[project.entry-points."content_engine.transports"]
rss = "podcast_scraper.transports.rss:RssTransport"
```

### User experience

```bash
# Core only — podcast + RSS bundled
pip install podcast-scraper
podcast-scraper "https://feeds.npr.org/510289/podcast.xml"

# Add news support
pip install news-ingest

# Check what's installed
podcast-scraper plugins --list
# Content types:
# podcast        (bundled)   Podcast Episode
# news_article               News Article
# Transports:
# rss            (bundled)   RSS Feed → podcast
# web_scrape                 Web Scraper → news_article

# Use multi-source config
podcast-scraper --config multi.yaml
```

```yaml
# multi.yaml
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
        - "https://example.com/tech"
        - "https://other.com/science"
      selector: "article.main"

summary_provider: openai
gi_enabled: true
kg_enabled: true
```

### Why standard Python packages (not a custom plugin format)

1. **Entry points just work.** `pip install` registers the plugin. `pip uninstall`
   removes it. No config files to edit, no manual registration.

2. **Dependency isolation.** Each module declares its own dependencies. Users who
   only want podcasts never install `trafilatura` or `mastodon.py`.

3. **Independent versioning and release cycles.** Release `news-ingest` v0.3
   without touching the core. A social API breaking change doesn't block a
   podcast pipeline release.

4. **Standard tooling.** pytest, CI/CD, PyPI publishing (or private index),
   `pip install -e .` for development — all works the same way.

5. **Proven pattern.** pytest plugins, Flask extensions, flake8 checkers,
   Sphinx extensions, tox environments — the Python ecosystem uses entry points
   extensively. Well-understood, well-documented, no framework magic.

### Testing external modules

External modules test against the core's protocols:

```python
# tests/test_manifest.py — validate manifest is well-formed
from podcast_scraper.core.registry import PluginRegistry

def test_news_article_discovered():
    registry = PluginRegistry()
    ct = registry.get_content_type("news_article")
    assert ct.manifest.name == "news_article"
    assert "summarization" in ct.manifest.stages

def test_web_scrape_compatible_with_news():
    registry = PluginRegistry()
    registry.validate_wiring("web_scrape", "news_article")  # no error
```

```python
# tests/test_source.py — validate transport produces valid ContentItems
from news_ingest.transports.web_scrape import WebScrapeSource

def test_fetch_returns_content_items(mock_html):
    source = WebScrapeSource()
    items = source.fetch(config)
    for item in items:
        assert item.id
        assert item.title
        assert item.content_type == "news_article"
        assert item.source_text  # news articles always have text
```

The core can also provide test utilities:

```python
# podcast_scraper.core.testing — helpers for plugin authors
from podcast_scraper.core.testing import (
    assert_valid_content_item,
    assert_valid_manifest,
    assert_pipeline_compatible,
)
```

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

## Repo Strategy, CI, and Development Workflow

### The problem: public core + private modules

The core (`podcast-scraper` with podcast + RSS) is open source. Future content
modules (news, social, etc.) are private — not published, not open-sourced.
This creates a repo structure question.

### Options considered

| Option | Layout | Pros | Cons |
| --- | --- | --- | --- |
| **Multi-repo** | N separate repos, one per module | Clean separation, per-module access control | Cross-repo dependency pain, N separate CIs, no atomic changes, need private PyPI |
| **Full monorepo (private)** | One private repo, all packages | One CI, atomic changes, trivial local dev | Can't accept open-source contributions to the core |
| **Public core + private monorepo** | Public repo for core; one private repo for all private modules | Open-source core, monorepo benefits for private modules, only 2 repos | Slight cross-repo friction (but only 2 repos, not N) |
| **Public monorepo, selective publishing** | One public repo, only publish core to PyPI | Simplest, one CI | Private module code is visible on GitHub |

### Recommendation: Public core + private module monorepo (Option B)

```text
github.com/you/podcast-scraper/        (PUBLIC — open source)
├── src/podcast_scraper/
│   ├── core/                           # Generic protocols, registry
│   ├── content_types/podcast/          # Bundled podcast content type
│   ├── transports/rss/                 # Bundled RSS transport
│   ├── providers/, gi/, kg/, ...       # Processing pipeline
│   └── ...
├── pyproject.toml                      # Published to PyPI as podcast-scraper
└── tests/

github.com/you/content-modules/        (PRIVATE — your modules)
├── packages/
│   ├── news-ingest/
│   │   ├── pyproject.toml
│   │   ├── src/news_ingest/
│   │   └── tests/
│   ├── social-ingest/
│   │   ├── pyproject.toml
│   │   ├── src/social_ingest/
│   │   └── tests/
│   └── video-ingest/
│       ├── pyproject.toml
│       ├── src/video_ingest/
│       └── tests/
├── Makefile
├── .github/workflows/ci.yml
└── pyproject.toml                      # Workspace root (tooling config)
```

**Why this works:**

- `podcast-scraper` stays public, accepts open-source contributions, published
  to PyPI
- All private modules live in **one** private repo (not N repos) — you get
  monorepo benefits: one CI, atomic changes across modules, shared tooling
- Cross-repo friction is minimal because it's only 2 repos, not 5
- No private PyPI needed — private modules are installed from the repo checkout
- Adding a new private module is just `mkdir packages/email-ingest/` — no new
  repo, no new CI, no new infrastructure

### CI for the public core (podcast-scraper)

Same as today — no changes. Tests cover the core pipeline, podcast content type,
and RSS transport. Published to PyPI on release.

```yaml
# podcast-scraper/.github/workflows/ci.yml (unchanged from current)
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -e .[dev]
      - run: make ci
```

### CI for the private monorepo (content-modules)

One workflow tests all private modules against the published core (or a pinned
git ref for pre-release testing).

```yaml
# content-modules/.github/workflows/ci.yml
name: CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        # Test against stable and latest core
        core-version: ["podcast-scraper>=2.0", "podcast-scraper@git+https://github.com/you/podcast-scraper.git@main"]
    steps:
      - uses: actions/checkout@v4

      - name: Install core
        run: pip install "${{ matrix.core-version }}"

      - name: Install all private modules (editable)
        run: |
          pip install -e packages/news-ingest
          pip install -e packages/social-ingest
          pip install -e packages/video-ingest

      - name: Plugin registration smoke test
        run: |
          python -c "
          from podcast_scraper.core.registry import PluginRegistry
          r = PluginRegistry()
          types = [m.name for m in r.list_content_types()]
          transports = [m.name for m in r.list_transports()]
          assert 'news_article' in types, f'Missing news_article in {types}'
          assert 'social_post' in types, f'Missing social_post in {types}'
          assert 'web_scrape' in transports, f'Missing web_scrape in {transports}'
          print(f'OK: {len(types)} content types, {len(transports)} transports')
          "

      - name: Test all modules
        run: |
          pytest packages/news-ingest/tests
          pytest packages/social-ingest/tests
          pytest packages/video-ingest/tests

      - name: Lint all modules
        run: ruff check packages/
```

The **matrix strategy** tests against both the stable PyPI release and the latest
`main` of the core. This catches breaking changes early — if the core's `main`
breaks a private module, you see it before the core is released.

### Makefile for the private monorepo

```makefile
# content-modules/Makefile

PACKAGES := news-ingest social-ingest video-ingest

# Install everything for local development
install-dev:
	pip install -e ../podcast-scraper[dev]
	$(foreach pkg,$(PACKAGES),pip install -e packages/$(pkg);)

# Install from PyPI core (CI mode)
install-ci:
	pip install "podcast-scraper>=2.0"
	$(foreach pkg,$(PACKAGES),pip install -e packages/$(pkg);)

# Test all modules
test:
	$(foreach pkg,$(PACKAGES),pytest packages/$(pkg)/tests;)

# Test a single module
test-%:
	pytest packages/$*/tests

# Lint all modules
lint:
	ruff check packages/

# Format all modules
format:
	ruff format packages/

# Verify all plugins register correctly
verify-plugins:
	python -c "\
	from podcast_scraper.core.registry import PluginRegistry; \
	r = PluginRegistry(); \
	print('Content types:', [m.name for m in r.list_content_types()]); \
	print('Transports:', [m.name for m in r.list_transports()])"
```

### Local development workflow

Two repos side by side, one virtualenv:

```bash
# Clone both repos
~/Projects/
├── podcast-scraper/          # public core (your existing repo)
└── content-modules/          # private modules

# Set up development environment
cd ~/Projects/content-modules
python -m venv .venv
source .venv/bin/activate

# Install core in editable mode (local changes reflected immediately)
pip install -e ../podcast-scraper[dev]

# Install all private modules in editable mode
make install-dev

# Now everything is live-editable:
# - Edit a core protocol in ../podcast-scraper/src/ → reflected immediately
# - Edit a news handler in packages/news-ingest/src/ → reflected immediately
# - Run tests across everything
make test

# Test just one module
make test-news-ingest

# Verify plugins are wired correctly
make verify-plugins
```

### When you change a core protocol

The workflow for a breaking change to `ContentItem` or a manifest:

```text
1. Edit the protocol in podcast-scraper (local checkout)
2. Run podcast-scraper's own tests → fix podcast + RSS modules
3. Run content-modules tests → fix news, social, etc.
4. Both repos green locally
5. PR + merge to podcast-scraper
6. PR + merge to content-modules (CI matrix catches if core main is broken)
7. Release podcast-scraper to PyPI
8. content-modules CI auto-tests against new PyPI release (matrix)
```

Because both repos are checked out locally with editable installs, step 1-4
happens in one terminal session. No publishing, no waiting for CI, no version
juggling.

### Adding a new private module

```bash
cd ~/Projects/content-modules

# Create the package structure
mkdir -p packages/email-ingest/src/email_ingest/{content_types/email,transports/imap}
mkdir -p packages/email-ingest/tests

# Create pyproject.toml
cat > packages/email-ingest/pyproject.toml << 'EOF'
[project]
name = "email-ingest"
version = "0.1.0"
dependencies = ["podcast-scraper>=2.0"]

[project.entry-points."content_engine.content_types"]
email = "email_ingest.content_types.email:EmailContentType"

[project.entry-points."content_engine.transports"]
imap = "email_ingest.transports.imap:ImapTransport"
EOF

# Install and verify
pip install -e packages/email-ingest
make verify-plugins
# Content types: ['podcast', 'news_article', 'social_post', 'email']
# Transports: ['rss', 'web_scrape', 'mastodon', 'reddit', 'bluesky', 'imap']
```

No new repo. No new CI config. No new infrastructure. Just a new directory.

### Deployment with private modules

For deploying a server/worker that uses private modules:

```dockerfile
# content-modules/Dockerfile
FROM python:3.12-slim

# Install public core from PyPI
RUN pip install podcast-scraper>=2.0

# Copy and install private modules
COPY packages/ /app/packages/
RUN pip install /app/packages/news-ingest \
                /app/packages/social-ingest

# Plugins auto-register via entry points
CMD ["podcast-scraper", "serve", "--config", "/app/config.yaml"]
```

Or with a private PyPI / artifact registry if you prefer not to copy source
into the image:

```dockerfile
# Alternative: install from private registry
RUN pip install --index-url https://your-registry/simple/ \
    podcast-scraper news-ingest social-ingest
```

---

## Future Work: Areas That Need Design

These are known areas that need concrete design work before or during the v3.0
refactoring. Each will likely become its own RFC or section in this document.
Captured here to track the thinking; answers come from doing, not more planning.

### F.1 Storage and output layout for multiple content types

**Problem:** Today everything goes into `output/rss_<host>_<hash>/` — a layout
derived from RSS feed identity (ADR-003). When news articles and social posts
exist alongside podcasts, the filesystem layout needs to accommodate multiple
content types and transports without breaking existing podcast output.

**Considerations:**

- Postgres (RFC-051) is the next major infrastructure addition. Once projection
  exists, the filesystem layout becomes less critical for reads (DB is the query
  surface) but still matters for canonical blob storage and CLI output.
- The layout should work for both filesystem-only (CLI, no DB) and
  filesystem + DB (platform mode) deployments.
- Existing podcast output paths must not change — backwards compatibility.

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
3. The two-tier queue model (simple → distributed) in B.7 already supports
   this — content type becomes another dimension in the distributed tier's
   queue routing

**Dedicated pipelines vs shared pipeline:** For high-volume content types
(social posts), it may make sense to run dedicated pipeline instances with
tuned concurrency and batching, rather than mixing them with podcast jobs
in the same queue. The platform blueprint's Compose topology (D.5) supports
this — different `command` / queue subscription per worker service. Fine-tune
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

---

## Relationship to Other Architecture Documents

This document is a **peer** of the Platform Architecture Blueprint, not a
section within it.

| Document | Scope | Answers |
| --- | --- | --- |
| [Architecture](../architecture/ARCHITECTURE.md) | Current state | How does the system work today? |
| [Platform Blueprint](../architecture/PLATFORM_ARCHITECTURE_BLUEPRINT.md) | Infrastructure evolution | How do we scale to a platform? (tenancy, workers, queues, DB, deployment) |
| **This document** | Content evolution | How do we evolve beyond podcasts? (content types, transports, plugins, cross-source KG) |
| [RFC-072](../rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md) | Identity layer | How do entities get canonical IDs across episodes? (precursor to cross-content-type resolution) |
| [RFC-073](../rfc/RFC-073-enrichment-layer-architecture.md) | Enrichment layer | How do derived signals layer on top of core artifacts? (content-agnostic protocol) |
| [PRD-026](../prd/PRD-026-topic-entity-view.md) / [PRD-027](../prd/PRD-027-enriched-search.md) | Enrichment consumers | First features consuming enricher outputs (topic view, enriched search) |
| [Non-Functional Requirements](../architecture/NON_FUNCTIONAL_REQUIREMENTS.md) | Quality attributes | What are the performance, reliability, security targets? |

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

**When to promote from `docs/wip/`:** When the v3.0 refactoring is approved
and scheduled, move this to `docs/architecture/CONTENT_EVOLUTION_BLUEPRINT.md`
(or similar) and add cross-references.

---

## Resolved Design Decisions

1. **Plugin discovery:** Python entry points via `importlib.metadata`. `pip install`
   is the registration mechanism. See Registry section above.

2. **Migration path:** Existing `Config` YAML files with `rss_url` keep working.
   The core synthesizes an implicit `sources` entry from `rss_url` for backwards
   compatibility. No user-facing changes required.

3. **Repo strategy:** Public core repo (`podcast-scraper`) + private module
   monorepo (`content-modules`). See Repo Strategy section above.

---

## Open Questions

1. **Package naming:** The package is called `podcast_scraper` but the core is
   generic. Options: (a) rename to something like `content_engine` (breaking),
   (b) keep `podcast_scraper` as the package name since podcast is the primary use
   case and the entry point, (c) publish `core/` as a separate thin package
   (`content-engine-core`) that both this repo and external modules depend on.
   Leaning toward (b) for now — rename is high cost, low immediate value.

2. **Entry point group namespace:** Using `content_engine.content_types` and
   `content_engine.transports` as entry point groups. Should this match the
   package name (`podcast_scraper.content_types`) or use a neutral namespace?
   Neutral is better if external modules shouldn't appear to be sub-packages of
   `podcast_scraper`.

3. **Shared viewer:** External modules push artifacts into the same
   storage/projection layer. The viewer shows all content types with faceting.
   Cross-content-type KG (entity resolution across sources) is the long-term
   goal — defer until at least two content types produce KG data.

4. **Versioning contract:** `core/protocols.py` and manifest dataclasses become
   the stable public API with semver guarantees. Internal implementation details
   (how the podcast module works) are not part of the public contract. Need to
   define what "breaking change" means for manifest schemas.

5. **Manifest schema evolution:** When a new content type needs manifest fields
   that don't exist yet, how do we extend without breaking existing plugins?
   Options: (a) optional fields with defaults (simplest), (b) manifest schema
   versioning with migration, (c) `extra: dict` escape hatch. Likely (a) for
   most cases, with (b) reserved for structural changes.

6. **Transport ↔ content type cardinality:** Can one transport produce multiple
   content types? (e.g., a "web_scrape" transport might produce both
   "news_article" and "blog_post"). The manifest supports this via
   `compatible_content_types: frozenset`. But does the transport decide the
   content type per item, or does the user declare it in config? Probably
   config-declared for simplicity.
