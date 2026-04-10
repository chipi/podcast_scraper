# API Documentation

## Purpose

This directory contains comprehensive API documentation for `podcast_scraper`, including programmatic interfaces, configuration options, data models, and migration guides.

## API Documentation Index

### Core APIs

| Document | Description |
| :--- | :--- |
| [Core API](CORE.md) | Primary public API (`run_pipeline`, `Config`, package information) |
| [Service API](SERVICE.md) | Non-interactive service interface for daemons and process management |
| [CLI Interface](CLI.md) | Command-line interface documentation |
| [Configuration API](CONFIGURATION.md) | Configuration model, environment variables, and file formats |
| [Data Models](MODELS.md) | Core data structures (`Episode`, `RssFeed`, `TranscriptionJob`) |
| [Multi-feed corpus artifacts](CORPUS_MULTI_FEED_ARTIFACTS.md) | `corpus_manifest.json` / `corpus_run_summary.json` contracts (#506); links to RFC-063 |

### HTTP / viewer API (`[server]` extra) {: #http-viewer-api-server-extra }

The **FastAPI** server and **Vue 3** GI/KG viewer are optional. They are **not** imported from `podcast_scraper` top-level exports; stability is documented in the [Server Guide](../guides/SERVER_GUIDE.md) and OpenAPI (`/docs` when `uvicorn` is running).

| Document | Description |
| :--- | :--- |
| [Server Guide](../guides/SERVER_GUIDE.md) | Endpoints under `/api/*`, static SPA mounting, dev workflow (`make serve`), testing pointers |
| [RFC-062: GI/KG viewer v2](../rfc/RFC-062-gi-kg-viewer-v2.md) | Viewer milestones and architecture |
| [RFC-067: Corpus Library](../rfc/RFC-067-corpus-library-api-viewer.md) | `/api/corpus/*` catalog, Library tab, search handoffs |
| [RFC-068: Corpus Digest](../rfc/RFC-068-corpus-digest-api-viewer.md) | `GET /api/corpus/digest`, Digest tab, Library 24h glance (PRD-023); capability flag `corpus_digest_api` on `GET /api/health` |

**v2.6.0** ships Corpus Library routes, index rebuild, and related viewer UX; see [Release v2.6.0](../releases/RELEASE_v2.6.0.md).

### API Reference & Guides

| Document | Description |
| :--- | :--- |
| [API Reference](REFERENCE.md) | Complete API reference documentation |
| [API Boundaries](BOUNDARIES.md) | Public vs. private API boundaries and stability guarantees |
| [API Versioning](VERSIONING.md) | API versioning strategy and compatibility policies |
| [API Migration Guide](MIGRATION_GUIDE.md) | Migration guides for API changes and breaking changes |

## Quick Start

**For programmatic usage:**

1. Start with [Core API](CORE.md) — Main entry point and functions.
2. Review [Configuration API](CONFIGURATION.md) — Setup and configuration options.
3. See [Data Models](MODELS.md) — Understand data structures.

**For command-line usage:**

1. See [CLI Interface](CLI.md) — Command-line options and examples.

**For service/daemon usage:**

1. See [Service API](SERVICE.md) — Non-interactive service interface.

**For the HTTP viewer (`podcast serve`, `/api/*`):**

1. See [Server Guide](../guides/SERVER_GUIDE.md) — Install `[server]` extra, endpoint reference, CORS and static assets.

**For API stability and migration:**

1. See [API Boundaries](BOUNDARIES.md) — What's public vs. private.
2. See [API Versioning](VERSIONING.md) — Versioning strategy.
3. See [API Migration Guide](MIGRATION_GUIDE.md) — Breaking changes.

## Quick Links

- **[Architecture](../architecture/ARCHITECTURE.md)** — System design and module responsibilities.
- **[Development Guide](../guides/DEVELOPMENT_GUIDE.md)** — Development practices and guidelines.
- **[Testing Guide](../guides/TESTING_GUIDE.md)** — Testing strategies and examples.
