# Server Guide

The FastAPI server layer behind the Workbench.

## Overview

`podcast_scraper` follows a **"one pipeline core, multiple shells"** philosophy:
the same Python library that powers the CLI, service API, and batch workflows
also backs an HTTP server.
The server exposes corpus artifacts, **semantic search**, the **relational query**
routes, **Corpus Library** and digest APIs, **vector index** stats / rebuild controls, the
operator surfaces (feeds, operator config, pipeline jobs, enrichment, ops) and the
platform's own sign-in / account routes (`/api/app/*`)
through a JSON API consumed by the Vue 3 SPA
([`web/workbench/`][viewer-readme]).

**What this repository ships.** Public's scope is set by
[ADR-170](../adr/ADR-170-private-repo-split-platform-common-apps.md). Public registers no
built-in pipeline stages and no built-in read hooks
(`src/podcast_scraper/builtin_stage_hooks.py`, `builtin_read_hooks.py`): speakers and
diarization, grounded insights (GI), the identity ledger and the advanced KG passes are
provided by an installed extension ([ADR-168](../adr/ADR-168-installed-extensions.md)). An
extension can also add routes and CLI subcommands; they mount under the same serve postures
as the routes below ([Extension routes](#extension-routes)). Without one, the routes in this
guide are the whole API.

**Corpus lifecycle:** routes read **current** files under the configured corpus root.
Re-running extraction or corpus topic clustering can change canonical ids
and cluster compounds; responses are not pinned to an older ingest.

**Repo layout:** Python at the root, Node UI in `web/workbench/`. See
[Polyglot repository guide](POLYGLOT_REPO_GUIDE.md) for env files and Makefile targets (`make serve`,
`make test-ui`, etc.).

The server is described in
[Workbench](../rfc/RFC-062-gi-kg-viewer-v2.md) and is implemented in
[`src/podcast_scraper/server/`][server-pkg].
Route groups are additive: the **operator read routes** (artifacts,
search, relational, index stats/rebuild, corpus library, corpus digest, enrichments, ops, …) and
the **`/api/app/*`** routes ship by default. The **RFC-077** operator routes (**feeds**,
**operator-config**, **pipeline jobs** with their **enrichment** and **scheduled-jobs** companions)
mount separately via `create_app` flags (see [RFC-077](../rfc/RFC-077-viewer-feeds-and-serve-pipeline-jobs.md)
and [ADR-064 addendum](../adr/ADR-064-canonical-server-layer-with-feature-flagged-routes.md#addendum-2026-04-rfc-077-operator-surfaces-separate-flags)).

[viewer-readme]: https://github.com/chipi/podcast_scraper/blob/main/web/workbench/README.md
[server-pkg]: https://github.com/chipi/podcast_scraper/blob/main/src/podcast_scraper/server/

## Quick start

### 1. Install the `[dev]` extra

```bash
pip install -e '.[dev]'
```

This adds `fastapi` and `uvicorn[standard]`.

### 2. Build the frontend (once)

```bash
cd web/workbench && npm install && npm run build
```

The build produces `web/workbench/dist/`.
When that directory exists, the server mounts it at `/` automatically.

### 3. Start the server

```bash
python -m podcast_scraper.cli serve --output-dir /path/to/corpus/output
```

The `/api/app/*` routes keep their state in Postgres
([ADR-169](../adr/ADR-169-the-database-is-the-source-of-truth.md)): set `PODCAST_DATABASE_URL`
(see [Configuration — Database](../api/CONFIGURATION.md#database-app-state)), or start the
server with `make serve-api`, which runs a local dev PostgreSQL for the checkout.

### 4. Open the browser

Navigate to <http://127.0.0.1:8000>.
The served shell carries the `--output-dir` corpus path; if the UI asks for a **Corpus root
folder**, enter the same path,
then **List files**, select artifacts, and **Load selected into graph**.
Use the **Library** tab to browse feeds and episodes (Corpus Library).

## Architecture

### App factory

[`app.py`][app-py] exposes two entry points:

| Function | Purpose |
| -------- | ------- |
| `create_app(output_dir, *, static_dir, enable_feeds_api, enable_operator_config_api, enable_jobs_api, operator_config_file)` | Build the `FastAPI` instance with the always-on routes, the flagged operator routes and optional static assets. (`enable_platform` is a retained no-op.) |
| `create_app_for_uvicorn()` | Factory for `uvicorn --factory` (reload mode and `serve --workers N`). Reads `PODCAST_SERVE_OUTPUT_DIR` and the `PODCAST_SERVE_*` flags from the environment. |

`create_app` stores `output_dir` on `app.state` so route handlers can
fall back to it when the caller omits the `?path=` query parameter.

When `output_dir` is set, any `?path=` value must resolve to that directory
or a subdirectory (see `server/pathutil.py`). Overrides are rejected if the
server has no default corpus root, so callers cannot aim the API at arbitrary
paths on disk.

[app-py]: https://github.com/chipi/podcast_scraper/blob/main/src/podcast_scraper/server/app.py

### Route groups

Operator-plane routers are included with `prefix="/api"`, the platform's own routers with
`prefix="/api/app"`, organized by domain (the endpoint-level catalogue is the
[HTTP API Reference](../api/HTTP_API.md)):

```text
routes/
  health.py          # always — GET /health
  version.py         # always — GET /version (also at /api/app/version)
  app_auth.py        # always, /api/app — sign-in (OAuth, email link/code), /me, logout
  app_admin.py       # always, /api/app — users, access policy, release (admin)
  app_user_preferences.py  # always, /api/app — GET/PUT/PATCH /preferences
  app_profile.py     # always, /api/app — display name, avatar
  app_graph_events.py # always, /api/app — graph-event log (POST + admin reads)
  app_og.py          # always, mounted at /og/... (unauthenticated social cards)
  internal_outbox.py # always, mounted at /internal (token-gated delivery outbox)
  artifacts.py       # operator read — list / load KG (and GI) JSON
  index_stats.py     # operator read — LanceDB index metrics + staleness, timeseries
  index_rebuild.py   # operator read — POST /index/rebuild, /corpus/topic-clusters/rebuild
  search.py          # operator read — GET /search, POST /search/compare
  relational.py      # operator read — /relational/* relational-query layer (RFC-094, #882)
  query_activity.py  # operator read — GET /corpus/query-activity (search-volume; FR6.2)
  corpus_library.py  # operator read — /corpus/feeds, episodes, detail, similar, node-episodes
  corpus_metrics.py  # operator read — /corpus/stats, documents/manifest, documents/run-summary, runs/summary
  corpus_digest.py   # operator read — GET /corpus/digest
  corpus_coverage.py # operator read — GET /corpus/coverage
  corpus_enrichments.py # operator read — /corpus/enrichments, entity-signals
  corpus_text_file.py # operator read — GET /corpus/text-file (transcript / caption files under corpus root)
  corpus_binary.py   # operator read — GET /corpus/binary (downloaded podcast artwork)
  corpus_media.py    # operator read — GET /corpus/media (persisted audio)
  corpus_rollback.py # operator read — DELETE /corpus/runs/{id}, /corpus/episodes/{id} (guarded)
  usage.py           # operator read — GET /usage
  resilience.py      # operator read — GET /resilience, POST /ops/resilience/reset
  ops.py             # operator read — /ops/summary, cache-stats, secrets/status, corpus/*
  llm_gateway.py     # operator read — GET /ops/llm-gateway
  feeds.py           # optional — GET/PUT /feeds (feeds.spec.yaml); enable_feeds_api
  feed_overrides.py  # optional — /feeds/overrides/* (per-feed and per-episode); enable_feeds_api
  operator_config.py # optional — GET/PUT /operator-config; enable_operator_config_api
  jobs.py            # optional — POST/GET /jobs, cancel, reconcile, stop, resume; enable_jobs_api
  scheduled_jobs.py  # optional — GET /scheduled-jobs (cron list + next-run); enable_jobs_api
  enrichment.py      # optional — /enrichment/*, POST /jobs/enrichment; enable_jobs_api
  enrichment_config.py # optional — /enrichment/config*; enable_jobs_api
  platform/          # reserved stub package (#50, #347); not mounted
```

**RFC-077** routers (`feeds` with `feed_overrides`, `operator_config`, `jobs` with
`scheduled_jobs`, `enrichment` and `enrichment_config`) mount when their
matching **`enable_*_api`** flags are true in `create_app` (or the equivalent
`PODCAST_SERVE_ENABLE_*` env vars for `uvicorn --factory`: reload mode and `serve --workers N`).

The **`routes/platform/`** package is **not** mounted today and `enable_platform` does nothing
([ADR-064](../adr/ADR-064-canonical-server-layer-with-feature-flagged-routes.md)).

#### Serve postures

`create_app` reads two environment flags and mounts different sets of the operator read plane:

| Posture | Flag | Mounted |
| ------- | ---- | ------- |
| **full** (default) | none | health, version, `/api/app/*`, every operator read router above, ungated: the network around the server is the gate. |
| **app-only** | `PODCAST_SERVE_APP_ONLY=1` | health, version and `/api/app/*` only; the flagged operator routers are forced off. |
| **operator-public** | `PODCAST_SERVE_OPERATOR_PUBLIC=1` | the curated read-only subset (not `index_rebuild`, `ops`, `resilience` or `corpus_rollback`), each behind a role gate (`require_viewer_access`); the flagged operator routers are forced off, and boot is refused under open signup. |

#### Extension routes

An installed extension adds routers through `Extension.routers()`; `_mount_extension_routers`
in `app.py` mounts them after the core routes, under the same postures (operator-plane routers
follow the table above). Public ships none of its own. Routes that exist only with an
extension, for example grounded-insight explore and person and cross-layer identity queries, are
the extension's to document and are not part of this guide or of public's API. Public has no
`GET /api/corpus/topic-clusters`; `POST /api/corpus/topic-clusters/rebuild` (in
`index_rebuild.py`) only re-derives `search/topic_clusters.json`.

### Static file mounting

When a built SPA exists at `web/workbench/dist/`, `create_app` mounts
it at `/` via `StaticFiles(directory=..., html=True)`.
Pass `--no-static` (or `static_dir=False` in code) to skip mounting and
run the API standalone.

### CORS

The CORS middleware allows credentialed calls from the web origins in
`PODCAST_SERVE_CORS_ORIGINS` (comma-separated) when it is set. When it is unset, only a dev box
(`PODCAST_ENV` unset or `dev`) trusts the local Vite dev servers
(`http://127.0.0.1:5173` / `http://localhost:5173`, and the E2E port 5174); a deployed server
(`PODCAST_ENV` other than `dev`) trusts no other page. The native-shell origins
(`capacitor://localhost`, `https://localhost`, `http://localhost`) are always CORS-allowed
(`web_origins` / `cors_allowed_origins` in `app.py`).

## HTTP API

Endpoints live under the `/api` prefix (the platform's own under `/api/app`), apart from the public
social cards under `/og` and the token-gated delivery outbox under `/internal`. The full **endpoint catalogue**, request parameters, and **response models** are documented in the dedicated **[HTTP API Reference](../api/HTTP_API.md)** (under `docs/api/`, alongside the programmatic library API). With the server running, the live OpenAPI spec is at **`/docs`** (Swagger UI) and **`/openapi.json`**.

This guide covers **running and extending** the server (below); the HTTP API Reference is the source of truth for *what each endpoint does*.

## MCP server (agent tools)

Public ships no MCP server: there is no `src/podcast_scraper/mcp/` package, and
`python -m podcast_scraper.cli mcp` answers `no command named 'mcp'; an installed extension may
provide it`. A downstream build can provide the `mcp` subcommand (and the `make serve-mcp`
target, which calls it) through an installed extension; its tools, transports and auth are
documented with that extension.

## Route conventions

1. **One file per domain** — `routes/<domain>.py`
   (e.g., `routes/health.py`, `routes/search.py`).
2. **Each file exports a `router`** — an `APIRouter` instance with a `tags`
   list for OpenAPI grouping.
3. **No prefix on the router itself** — the `/api` prefix is applied in
   `app.py` via `app.include_router(router, prefix="/api")`.
4. **Corpus resolution helper** — routes that accept an optional `?path=`
   parameter resolve it against `request.app.state.output_dir` as a
   fallback.
   The private `_resolve_corpus_root(path, fallback)` pattern is repeated
   in `index_stats.py`, `index_rebuild.py`, `search.py`, `relational.py`,
   `corpus_library.py`, `corpus_digest.py`, `corpus_metrics.py`, `jobs.py` and others.
5. **RFC-077 routes** live as top-level `routes/*.py` files (same router pattern).
   **`routes/platform/`** is a reserved, unmounted stub.
6. **Classify a new operator read router.** Add it to `_OPERATOR_READ_ROUTES` in `app.py` and
   decide, consciously, whether it also belongs in `_OPERATOR_PUBLIC_READ_ROUTES` (read-only
   routes only; anything that mutates stays out).

## Adding new routes

### Step-by-step

1. **Create the route file** — `src/podcast_scraper/server/routes/<domain>.py`:

   ```python
   from fastapi import APIRouter

   router = APIRouter(tags=["<domain>"])

   @router.get("/<domain>")
   async def my_endpoint() -> dict:
       return {"hello": "world"}
   ```

2. **Add Pydantic response model(s)** to `schemas.py`:

   ```python
   class MyResponse(BaseModel):
       hello: str
   ```

3. **Mount the router in `app.py`**: add the module to `_OPERATOR_READ_ROUTES` (see convention 6),
   or include it behind its own flag as `feeds` and `jobs` are:

   ```python
   from podcast_scraper.server.routes import my_domain

   app.include_router(my_domain.router, prefix="/api")
   ```

4. **Write tests** in
   `tests/unit/podcast_scraper/server/test_<domain>.py` (or
   `tests/integration/server/` for a wired-app test)
   using `FastAPI`'s `TestClient`:

   ```python
   pytest.importorskip("fastapi")
   from fastapi.testclient import TestClient
   from podcast_scraper.server.app import create_app

   def test_my_endpoint(tmp_path):
       app = create_app(tmp_path, static_dir=False)
       client = TestClient(app)
       resp = client.get("/api/<domain>")
       assert resp.status_code == 200
   ```

5. **Document it** — add the new endpoint and any response model to the
   [HTTP API Reference](../api/HTTP_API.md) (endpoint table + Response models list).

## Configuration

### CLI flags

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--output-dir` | *(required)* | Corpus output directory containing `metadata/*.kg.json` (and `*.gi.json` where an extension wrote them) and `search/`. |
| `--host` | `127.0.0.1` | Bind address. |
| `--port` | `8000` | TCP port. |
| `--reload` | off | Dev only: restart workers when Python files change (uses `uvicorn --factory`). |
| `--workers N` | `1` (or `PODCAST_SERVE_WORKERS`) | Worker processes. With more than one, the cron scheduler runs in one leader per host, and metrics need `PROMETHEUS_MULTIPROC_DIR`. |
| `--no-static` | off | Do not mount built SPA assets even if `web/workbench/dist/` exists. |
| `--enable-feeds-api` | off | Mount `GET/PUT /api/feeds` and `/api/feeds/overrides/*`. |
| `--enable-operator-config-api` | off | Mount `GET/PUT /api/operator-config`. |
| `--enable-jobs-api` | off | Mount `/api/jobs`, `/api/scheduled-jobs` and the `/api/enrichment/*` routes. |
| `--config-file PATH` | none | Pin one operator YAML for all corpora (default: `<corpus>/viewer_operator.yaml`). |

### Environment variable

| Variable | Purpose |
| -------- | ------- |
| `PODCAST_SERVE_OUTPUT_DIR` | Set automatically by `run_serve()`. Used by `create_app_for_uvicorn()` in reload mode. Can also be set manually when running uvicorn directly. |
| `PODCAST_SERVE_ENABLE_FEEDS_API`, `PODCAST_SERVE_ENABLE_OPERATOR_CONFIG_API`, `PODCAST_SERVE_ENABLE_JOBS_API` | The flag equivalents of `--enable-*-api`, read by `create_app_for_uvicorn()`. |
| `PODCAST_SERVE_CONFIG_FILE`, `PODCAST_SERVE_NO_STATIC` | Equivalents of `--config-file` and `--no-static` for the factory. |
| `PODCAST_SERVE_APP_ONLY`, `PODCAST_SERVE_OPERATOR_PUBLIC` | Serve postures ([Serve postures](#serve-postures)). |
| `PODCAST_SERVE_CORS_ORIGINS` | Comma-separated web origins trusted for credentialed calls ([CORS](#cors)). |
| `PODCAST_SERVE_WARM_SEARCH` | Set to `0` to skip the best-effort search warm-up at startup. |
| `PODCAST_SERVE_WORKERS` | Default for `--workers`. |
| `PODCAST_SCHEDULER_TZ` | Timezone for the in-process feed-sweep scheduler (#708). Defaults to `TZ` env, then `UTC`. |

### CLI entry point

```bash
python -m podcast_scraper.cli serve --output-dir /path/to/output [--host 0.0.0.0] [--port 9000] [--reload]
```

The `serve` sub-command is handled by
[`cli_handlers.py`][cli-handlers] (`parse_serve_argv` + `run_serve`).

[cli-handlers]: https://github.com/chipi/podcast_scraper/blob/main/src/podcast_scraper/server/cli_handlers.py

## Scheduled feed sweeps (#708)

Optional in-process cron scheduler that fires the same pipeline-job path
as `POST /api/jobs` on the operator's chosen schedule. Built on
[APScheduler](https://github.com/agronholm/apscheduler) (3.x; in the
`[dev]` extra). Mounts whenever `enable_jobs_api=True` and the operator
YAML contains `scheduled_jobs:`.

### Why API-level (not host-side cron)

Works in dev containers and servers alike (no systemd needed), persists with
the corpus (no host state to migrate on redeploy), and routes failures
through the existing job-state webhook surface so your alerting and dashboards
already see the events. See [GH #708](https://github.com/chipi/podcast_scraper/issues/708).

### Schedule definition

Add a top-level `scheduled_jobs:` list to `viewer_operator.yaml` (the
packaged example ships a commented hint):

```yaml
scheduled_jobs:
  - name: morning-feed-sweep
    cron: "0 4 * * *"      # standard 5-field cron (m h dom mon dow)
    enabled: true
  - name: evening-sweep
    cron: "0 20 * * *"
    enabled: false           # loaded but not fired
```

Each schedule reuses the corpus's standing `feeds.spec.yaml` + this same
operator YAML — there is no per-schedule profile / feeds / max_episodes
override in V1 (use multiple schedules if you need different cadences for
different feed sets, V2 will revisit per-schedule overrides).

`name` is used as the job id, the Prometheus label, and shows up in logs
and any Slack alerts; keep it short and stable. Allowed characters:
letters, digits, `-`, `_`. Names must be unique within a corpus.

### Reload behavior

- **App startup** (FastAPI lifespan): scheduler starts if at least one
  enabled job exists.
- **`PUT /api/operator-config`**: scheduler reloads the YAML and rebuilds
  triggers in-process. No restart needed — operators can add/remove
  schedules from the viewer Configuration tab and they take effect on
  Save.
- **App shutdown**: scheduler stops cleanly via the same lifespan hook.

An optional `kind` per schedule picks what it runs: `pipeline` (the
default, the same job as `POST /api/jobs`, enqueued with trigger `nightly`)
or `enrichment` (the same job as `POST /api/jobs/enrichment`).

A slot the scheduler never fired is reported, not skipped silently.
APScheduler keeps no memory of slots that pass while the API is down, so
the scheduler records each fire in `<corpus>/.viewer/scheduler_state.json`.
On every start it compares that record with the cron and emits one
`pipeline_run.skipped` event (fields `job`, `kind`, `slot_at`, `reason`,
`detected_at`) for the latest slot that never ran, with reason
`scheduler_not_running_at_slot`. A slot that is reached while the API is up
but runs past APScheduler's 1-hour misfire grace is reported the same way,
with reason `misfire`. Each slot is reported once. A schedule seen for the
first time reports nothing older than that start.

### Inspecting state

`GET /api/scheduled-jobs?path=<corpus>` returns the parsed schedule list
with each job's next-run preview:

```json
{
  "path": "/corpora/main",
  "scheduler_running": true,
  "timezone": "UTC",
  "jobs": [
    {"name": "morning-feed-sweep", "cron": "0 4 * * *", "enabled": true,
     "next_run_at": "2026-05-03T04:00:00Z"},
    {"name": "evening-sweep", "cron": "0 20 * * *", "enabled": false,
     "next_run_at": null}
  ]
}
```

### Failure handling

A scheduled fire is indistinguishable from a manual `POST /api/jobs` once
it lands — the job appears in the same JSONL registry, Slack/HA webhooks
fire on terminal state through `emit_job_state_change`, and the row is
visible at `GET /api/jobs`. Two scheduler-specific Prometheus counters
flank this:

| Counter | Labels | Increment trigger |
| ------- | ------ | ----------------- |
| `podcast_scheduled_jobs_triggered_total` | `name` | Cron fired (job_id may or may not have been issued yet) |
| `podcast_scheduled_jobs_failed_total` | `name`, `reason` | Spawn raised before a job was registered (e.g. invalid cron, event-loop unavailable) |

Both are no-ops when `prometheus_client` is unavailable (i.e. without
the `[dev]` extra).

### Limitations (V1)

- Per-schedule overrides (profile / feeds / max_episodes) are not wired —
  use multiple schedules with different operator YAMLs if you need them.
- No calendar-aware schedules (`every 3rd Tuesday`); standard cron only.
- The viewer's **Configuration → Scheduled** section (#709) lists schedules
  with next-run previews and an enable/disable toggle (writes back via
  `PUT /api/operator-config`, which reloads the scheduler). Adding or editing
  schedules still happens in the **Job Configuration** YAML editor
  (`scheduled_jobs:`).

## Development workflow

### Three Makefile targets

| Target | What it does | When to use |
| ------ | ------------ | ----------- |
| `make serve` | Runs `serve-api` and `serve-ui` in parallel (`make -j2`). | Day-to-day full-stack development. |
| `make serve-api` | Starts the FastAPI server on port **8000**. | Backend-only work, or when running Vite separately. |
| `make serve-ui` | Runs `npm run dev` inside `web/workbench/` (Vite on port **5173**). | Frontend-only work, or when running the API separately. |

Override the corpus directory:

```bash
make serve SERVE_OUTPUT_DIR=/path/to/output
```

### Vite proxy

During development, Vite proxies `/api/*` requests to
`http://127.0.0.1:8000`.
Open **<http://127.0.0.1:5173>** in the browser (not port 8000) to get
hot module replacement for Vue components while the API runs separately.

### Hot reload (Python)

Pass `--reload` to restart uvicorn workers when Python files change:

```bash
python -m podcast_scraper.cli serve --output-dir ./output --reload
```

In reload mode, uvicorn uses `create_app_for_uvicorn` as a factory and
reads `PODCAST_SERVE_OUTPUT_DIR` from the environment.

## Testing

### Unit tests

Located in `tests/unit/podcast_scraper/server/`:

| File | Coverage |
| ---- | -------- |
| `test_scheduler_parse.py`, `test_scheduler_missed_slots.py` | `scheduled_jobs:` parsing and the skipped-slot reports. |
| `test_serve_workers.py`, `test_leader.py` | `serve --workers N` and the per-host leader. |
| `test_app_operator_public_guard.py` | The operator-public serve posture. |
| `test_index_staleness_provider_mismatch.py` | Index staleness helpers used by index stats. |
| `test_catalog_page_limit.py`, `test_catalog_cache_sharing.py` | Corpus Library catalog paging and cache. |
| `test_job_trigger.py`, `test_jobs_episode_selection.py`, `test_queue_sweeper.py` | Pipeline job spawn, selection and queue. |

Tests use `FastAPI`'s synchronous `TestClient` and `tmp_path` fixtures.

Run them with (requires **`[dev]`** so FastAPI is present):

```bash
pip install -e '.[dev]'   # if needed
pytest tests/unit/podcast_scraper/server -q
```

### Integration tests

`tests/integration/server/test_server_api.py` exercises the **wired** app with real filesystem
artifacts (no mocking of route internals). Tests cover health, artifact listing/loading,
path-traversal blocking, index stats, search (no-index graceful error), and app factory
edge cases. Marked `@pytest.mark.integration`.

Additional integration modules under `tests/integration/server/` include
`test_viewer_corpus_library.py` (Corpus Library routes),
`test_viewer_corpus_digest.py` (`GET /api/corpus/digest`),
`test_index_rebuild.py` (`POST /api/index/rebuild` gate and acceptance),
`test_viewer_index_stats.py`, `test_viewer_api.py`, `test_server_app.py`, and
`test_server_package_init.py`.

```bash
pytest tests/integration/server/test_server_api.py -v
```

### Frontend unit tests (Vitest)

Pure TypeScript utility logic (parsing, merge, metrics, formatting, colors, visual groups,
search-focus mapping) is covered by **Vitest** unit tests co-located with source:

```bash
make test-ui          # Vitest + coverage gate, no browser
```

Tests: `web/workbench/src/**/*.test.ts` (for example `src/utils/`). Config: `vite.config.ts` `test` block.

### Browser E2E (Playwright)

[ADR-066](../adr/ADR-066-playwright-for-ui-e2e-testing.md) chose
Playwright for browser-level E2E testing.
Tests live in `web/workbench/e2e/` and run against a Vite dev server
on port **5174** (separate from the dev port 5173), talking to a real API that
`playwright.config.ts` starts (`db upgrade --plane player`, then `cli serve` over a seeded
corpus) unless one is already listening on the API port.

```bash
make test-ui-e2e
```

This target installs npm dependencies, installs the Firefox browser, and
runs `npm run test:e2e` inside `web/workbench/`.

**E2E vs `serve`:** Playwright serves the SPA from **Vite**, so it does not exercise the
**built `dist/`** that `cli serve` mounts. It is **not** a substitute for smoke-testing **`cli serve`**
(or **`make serve`**) against a corpus. See [Testing Guide — Browser E2E](TESTING_GUIDE.md#browser-e2e-workbench).

See the [E2E Testing Guide](E2E_TESTING_GUIDE.md) and
[Testing Guide](TESTING_GUIDE.md) for more detail.

## Platform evolution

**Mounted by default:** the operator read routers (`health`, `artifacts`, `index_stats`,
`index_rebuild`, `search`, `relational`, `corpus_library`, and the related **`/api/corpus/*`**
routes) and the `/api/app/*` routers as wired in `create_app`, subject to the
[serve postures](#serve-postures).

**RFC-077 (opt-in):** `routes/feeds.py`, `routes/operator_config.py`, and `routes/jobs.py`
mount when `enable_feeds_api`, `enable_operator_config_api`, and `enable_jobs_api` are
true (CLI flags on `serve` or `PODCAST_SERVE_ENABLE_*` for the factory). These are
**not** the historical `routes/platform/*` stubs — see [RFC-077](../rfc/RFC-077-viewer-feeds-and-serve-pipeline-jobs.md)
and [ADR-064 addendum](../adr/ADR-064-canonical-server-layer-with-feature-flagged-routes.md#addendum-2026-04-rfc-077-operator-surfaces-separate-flags).

**Seeding.** `POST /api/jobs` seeds `<corpus>/viewer_operator.yaml` with the starter the Job
Configuration tab writes when the file is missing, because the pipeline reads it
(`_ensure_default_viewer_operator_yaml` in `routes/operator_config.py`, called from `routes/jobs.py`).

**Job runner operations note:** the job registry is `<corpus>/.viewer/jobs.jsonl`, guarded by a
file lock, so the worker processes of one `serve --workers N` share it; the cron scheduler and
the queue sweeper run in one leader per host (`server/leader.py`). Hosts that do not share the
corpus directory do not share job state. **Cancel** is **POSIX-first** (`SIGTERM` to the child);
Windows semantics differ.

**Still reserved:** `routes/platform/` and `enable_platform` are an unmounted stub and a
retained no-op ([ADR-064](../adr/ADR-064-canonical-server-layer-with-feature-flagged-routes.md)).

## Batch pipeline JSONL and log shipping (#746)

Long **CLI / docker** batch runs can stream structured events through
`workflow/jsonl_emitter.py` (`run_started`, `episode_finished`, `run_finished`)
to `run.jsonl` when `jsonl_metrics_enabled` is true. For a log store such as **Loki**, each
line should be ingested as **one JSON log line**. The optional flag
`jsonl_metrics_echo_stdout` duplicates every JSONL line to **stdout** so the
existing docker log pipeline of your log shipper
can ship them with stable labels (for example **env**, **release**, **app**=`podcast_scraper`,
compose **service** as **service_name**). Use LogQL
`| json | event_type="run_finished"` in Explore or the imported pipeline-execution dashboard (its JSON is in the infrastructure
repository's Grafana dashboards). Prometheus **`/metrics`** on the API container remains
separate from this JSONL path.

## Related docs

| Document | Description |
| -------- | ----------- |
| [Workbench](../rfc/RFC-062-gi-kg-viewer-v2.md) | Viewer design (milestones, success criteria). |
| [Corpus Library](../rfc/RFC-067-corpus-library-api-viewer.md) | Catalog API and viewer integration. |
| [Corpus Digest](../rfc/RFC-068-corpus-digest-api-viewer.md) | Digest API & viewer (`GET /api/corpus/digest`, Digest tab, Library glance). |
| [ADR-064](../adr/ADR-064-canonical-server-layer-with-feature-flagged-routes.md) | Canonical server layer with feature-flagged route groups. |
| [RFC-077](../rfc/RFC-077-viewer-feeds-and-serve-pipeline-jobs.md) | Opt-in feeds file, operator YAML, and HTTP pipeline jobs on `serve`. |
| [ADR-065](../adr/ADR-065-vue3-vite-cytoscape-frontend-stack.md) | Vue 3 + Vite + Cytoscape.js frontend stack decision. |
| [ADR-066](../adr/ADR-066-playwright-for-ui-e2e-testing.md) | Playwright for UI E2E testing. |
| [Testing Guide](TESTING_GUIDE.md) | Commands, markers, and browser E2E section. |
| [E2E Testing Guide](E2E_TESTING_GUIDE.md) | Playwright browser E2E details. |
| [Development Guide](DEVELOPMENT_GUIDE.md) | Dev environment, `make serve` / `serve-api` / `serve-ui`. |
| [Configuration API](../api/CONFIGURATION.md) | Env vs YAML, [twelve-factor alignment (config)](../api/CONFIGURATION.md#twelve-factor-app-alignment-config). |
| [Viewer README][viewer-readme] | Frontend SPA setup, dev workflow, build instructions. |
| [`src/podcast_scraper/server/`][server-pkg] | Server source code. |

---

**Version:** 1.4
**Created:** 2026-04-04
**Updated:** 2026-11-17 — Route inventory and serve postures checked against `server/app.py`; extension-provided routes removed
