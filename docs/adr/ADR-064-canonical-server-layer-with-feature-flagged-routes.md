# ADR-064: Canonical Server Layer with Feature-Flagged Route Groups

- **Status**: Accepted
- **Date**: 2026-04-03
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-062](../rfc/RFC-062-gi-kg-viewer-v2.md)
- **Related Issues**: [#50](https://github.com/chipi/podcast_scraper/issues/50), [#347](https://github.com/chipi/podcast_scraper/issues/347), [#489](https://github.com/chipi/podcast_scraper/issues/489), [#626](https://github.com/chipi/podcast_scraper/issues/626) (RFC-077 viewer feeds / operator / jobs on `serve`)

## Context & Problem Statement

The project needs a server layer for two purposes arriving at different times: the
viewer v2 (v2.6) needs API endpoints for search, explore, and artifact loading; the
platform (#50, #347) needs API endpoints for feed management, episode browsing, and
job submission (v2.7). Building two separate servers would duplicate middleware, config,
and deployment patterns. Building one server named "viewer" would require a rename
when platform work starts.

## Decision

We create a **canonical server layer** in `src/podcast_scraper/server/`:

1. **Module naming**: `server/`, not `viewer/` or `api/`. This is the project's single
   HTTP server module.
2. **Feature-flagged route groups**: `create_app(enable_viewer=True,
   enable_platform=False)`. Route groups mount or not based on flags.
3. **`podcast serve` CLI command**: New top-level subcommand. Supports `--output-dir`,
   `--port`, and future `--platform` flag.
4. **Viewer routes (v2.6)**: `/api/health`, `/api/artifacts`, `/api/search`,
   `/api/explore`, `/api/index/stats`. Mounted by default.
5. **Platform route stubs (v2.7)**: Empty files for `/api/feeds`, `/api/episodes`,
   `/api/jobs`. Present in tree but not mounted until `enable_platform=True`.
6. **FastAPI app factory pattern**: One `create_app()` function, routers registered
   conditionally, shared dependencies (config, output_dir, vector_store).

## Rationale

- **Megasketch alignment**: Constraint A.2 — "One pipeline core, multiple shells." The
  CLI is one shell, the server is another. Not two servers.
- **Avoid rename tax**: Naming it `server/` now avoids a migration when platform routes
  land. Every import path, test path, and config reference would need updating.
- **Additive growth**: v2.7 adds routes and views to the existing server, not a new
  module. No architectural restructuring between versions.
- **Shared infrastructure**: Middleware (CORS, future auth), dependencies (config,
  vector store), and error handling are defined once.

## Alternatives Considered

1. **`src/podcast_scraper/viewer/` naming**: Rejected; forces rename when platform
   routes arrive. Implies the server exists only for the viewer.
2. **Separate viewer and platform servers**: Rejected; duplicates middleware, config
   loading, deployment. Two processes for one application.
3. **No server — CLI-only with subprocess calls from frontend**: Rejected; loses async,
   loses shared state (loaded vector index), poor developer experience.
4. **Flask instead of FastAPI**: Rejected; FastAPI provides async, auto-docs, Pydantic
   validation, and typed route parameters natively.

## Consequences

- **Positive**: One server, one CLI command, one deployment unit. Additive growth path
  from viewer to platform. Shared dependencies and middleware.
- **Negative**: Viewer routes and platform routes share a module; must maintain clean
  separation between route groups (solved by separate files and feature flags).
- **Neutral**: `podcast serve` is a new CLI subcommand that replaces
  `scripts/gi_kg_viz_server.py`.

## Implementation Notes

- **Module**: `src/podcast_scraper/server/`
- **Entry point**: `podcast serve --output-dir ./output [--port 8000]`
- **App factory**: `server.app.create_app(output_dir, static_dir=...,
  enable_platform=False)`
- **Makefile**: `make serve` (combined), `make serve-api` (backend only)
- **Relationship to `service.py`**: `service.run()` is one-shot pipeline execution.
  `server/` is long-lived HTTP. Platform jobs will call `service.run()` internally.

## References

- [RFC-062: GI/KG Viewer v2](../rfc/RFC-062-gi-kg-viewer-v2.md)
- [Platform Megasketch](../architecture/PLATFORM_ARCHITECTURE_BLUEPRINT.md) — constraint A.2
- [#50: Simple UI + server](https://github.com/chipi/podcast_scraper/issues/50)
- [#347: UI for DB output](https://github.com/chipi/podcast_scraper/issues/347)

## Addendum (2026-04): RFC-077 operator surfaces (separate flags)

The **first** HTTP surfaces for **corpus RSS list file**, **viewer-safe operator YAML**,
and **pipeline subprocess jobs** ship as **top-level route modules** next to viewer
corpus routes — **not** behind the historical `enable_platform` stub package:

| Surface | Router module | `create_app` flag | Serve / env (reload factory) |
| ------- | ------------- | ----------------- | ---------------------------- |
| Structured feeds file (`feeds.spec.yaml`) | `routes/feeds.py` | `enable_feeds_api` | `--enable-feeds-api` / `PODCAST_SERVE_ENABLE_FEEDS_API` |
| Operator YAML (non-secret) | `routes/operator_config.py` | `enable_operator_config_api` | `--enable-operator-config-api` / `PODCAST_SERVE_ENABLE_OPERATOR_CONFIG_API` |
| Pipeline jobs (subprocess + registry) | `routes/jobs.py` | `enable_jobs_api` | `--enable-jobs-api` / `PODCAST_SERVE_ENABLE_JOBS_API` |

`enable_platform` remains **reserved** for future megasketch work (#50, #347) such as
catalog CRUD or DB-backed corpus APIs. See [RFC-077](../rfc/RFC-077-viewer-feeds-and-serve-pipeline-jobs.md)
and [SERVER_GUIDE — Platform evolution](../guides/SERVER_GUIDE.md#platform-evolution).
