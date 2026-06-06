# API Documentation

## Purpose

`podcast_scraper` exposes **three API surfaces**, documented here:

1. **Programmatic (library) API** — the public Python package: `run_pipeline`, `Config`,
   the data models, the CLI, and the non-interactive service interface. Importable from
   `podcast_scraper`, with stability guarantees.
2. **HTTP / Viewer API** — the optional **FastAPI** server (`podcast serve`, all routes
   under `/api/*`) that backs the Vue 3 GI/KG viewer. **Not** part of the importable
   package surface; ships with the `[dev]` extra.
3. **MCP / Agent tools** — the optional generic **MCP server** (`podcast mcp`) exposing
   the platform's read capabilities (search, the relational-query layer, CIL, catalog) as
   composable, read-only tools for agentic clients. Library-wrapped, stdio transport.
   Spec: [PRD-034](../prd/PRD-034-generic-mcp-server.md) / [RFC-095](../rfc/RFC-095-generic-mcp-server.md);
   usage: [Server Guide — MCP server](../guides/SERVER_GUIDE.md#mcp-server-agent-tools).

## Programmatic (library) API

| Document | Description |
| :--- | :--- |
| [Core API](CORE.md) | Primary public API (`run_pipeline`, `Config`, package information) |
| [Service API](SERVICE.md) | Non-interactive service interface for daemons and process management |
| [CLI Interface](CLI.md) | Command-line interface documentation |
| [Configuration API](CONFIGURATION.md) | Configuration model, environment variables, and file formats |
| [Data Models](MODELS.md) | Core data structures (`Episode`, `RssFeed`, `TranscriptionJob`) |
| [Multi-feed corpus artifacts](CORPUS_MULTI_FEED_ARTIFACTS.md) | `corpus_manifest.json` / `corpus_run_summary.json` contracts (#506); links to RFC-063 |
| [API Reference](REFERENCE.md) | Complete public-API reference (`run_pipeline`, `Config`, Service) |
| [API Boundaries](BOUNDARIES.md) | Public vs. private API boundaries and stability guarantees |
| [API Versioning](VERSIONING.md) | API versioning strategy and compatibility policies |
| [API Migration Guide](MIGRATION_GUIDE.md) | Migration guides for API changes and breaking changes |

## HTTP / Viewer API (`[dev]` extra) {: #http-viewer-api-server-extra }

The **FastAPI** server and **Vue 3** GI/KG viewer are optional and **not** imported from
`podcast_scraper` top-level exports. The **[HTTP API Reference](HTTP_API.md)** is the
endpoint catalogue; the live, always-current spec is the server's own OpenAPI at `/docs`
(when `uvicorn` is running).

| Document | Description |
| :--- | :--- |
| [HTTP API Reference](HTTP_API.md) | **Endpoint catalogue + response models** for all `/api/*` routes (the source of truth alongside live OpenAPI) |
| [Server Guide](../guides/SERVER_GUIDE.md) | Running the server (`make serve`), app/route architecture, CORS, static SPA mounting, and **extending** it |
| [RFC-062: GI/KG viewer v2](../rfc/RFC-062-gi-kg-viewer-v2.md) | Viewer milestones and architecture |
| [RFC-067: Corpus Library](../rfc/RFC-067-corpus-library-api-viewer.md) | `/api/corpus/*` catalog, Library tab, search handoffs |
| [RFC-068: Corpus Digest](../rfc/RFC-068-corpus-digest-api-viewer.md) | `GET /api/corpus/digest`, Digest tab, Library 24h glance (PRD-023); capability flag `corpus_digest_api` on `GET /api/health` |
| [RFC-094: Search-powered surfaces query layer](../rfc/RFC-094-search-powered-surfaces-query-layer.md) | `/api/relational/*` relational-query layer (#882) + `/api/corpus/query-activity` (FR6.2); consumed by the PRD-033 surfaces |

**v2.6.0** ships Corpus Library routes, index rebuild, and related viewer UX; see [Release v2.6.0](../releases/RELEASE_v2.6.0.md).

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

1. See the [HTTP API Reference](HTTP_API.md) — the endpoint catalogue and response models.
2. See the [Server Guide](../guides/SERVER_GUIDE.md) — install the `[dev]` extra, run the server, CORS/static assets, and add routes.

**For API stability and migration:**

1. See [API Boundaries](BOUNDARIES.md) — What's public vs. private.
2. See [API Versioning](VERSIONING.md) — Versioning strategy.
3. See [API Migration Guide](MIGRATION_GUIDE.md) — Breaking changes.

## Quick Links

- **[Architecture](../architecture/ARCHITECTURE.md)** — System design and module responsibilities.
- **[Development Guide](../guides/DEVELOPMENT_GUIDE.md)** — Development practices and guidelines.
- **[Testing Guide](../guides/TESTING_GUIDE.md)** — Testing strategies and examples.
