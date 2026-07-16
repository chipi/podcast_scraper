# Server Guide

The FastAPI server layer for the GI/KG viewer and future platform APIs.

## Overview

`podcast_scraper` follows a **"one pipeline core, multiple shells"** philosophy:
the same Python library that powers the CLI, service API, and batch workflows
also backs an HTTP server.
The server exposes corpus artifacts, **semantic search**, **GI explore**,
**Cross-layer CIL queries** (person/topic arcs over `*.bridge.json` + GI/KG siblings),
**Corpus Library** catalog APIs, and **vector index** stats / rebuild controls
through a JSON API consumed by the Vue 3 SPA
([`web/gi-kg-viewer/`][viewer-readme]).

**Corpus lifecycle:** routes read **current** files under the configured corpus root.
Re-running extraction, the bridge builder, or corpus topic clustering can change canonical ids
and cluster compounds; responses are not pinned to an older ingest. See [Operational note (canonical identity)](../rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md#operational-note-re-pipeline-enrichment-and-read-path-stance).

**Repo layout:** Python at the root, Node UI in `web/gi-kg-viewer/`. See
[Polyglot repository guide](POLYGLOT_REPO_GUIDE.md) for env files and Makefile targets (`make serve`,
`make test-ui`, etc.).

The server is described in
[GI/KG viewer v2](../rfc/RFC-062-gi-kg-viewer-v2.md) and is implemented in
[`src/podcast_scraper/server/`][server-pkg].
Route groups are additive: **viewer routes** ship by default (artifacts,
search, explore, index stats/rebuild, corpus library, corpus digest, CIL, …).
**RFC-077** opt-in routes (**feeds**, **operator-config**, **pipeline jobs**)
mount separately via `create_app` flags (see [RFC-077](../rfc/RFC-077-viewer-feeds-and-serve-pipeline-jobs.md)
and [ADR-064 addendum](../adr/ADR-064-canonical-server-layer-with-feature-flagged-routes.md#addendum-2026-04-rfc-077-operator-surfaces-separate-flags)).
Future **megasketch** platform routes (#50, #347) remain behind `enable_platform` when implemented.

[viewer-readme]: https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/README.md
[server-pkg]: https://github.com/chipi/podcast_scraper/blob/main/src/podcast_scraper/server/

## Quick start

### 1. Install the `[dev]` extra

```bash
pip install -e '.[dev]'
```

This adds `fastapi` and `uvicorn[standard]`.

### 2. Build the frontend (once)

```bash
cd web/gi-kg-viewer && npm install && npm run build
```

The build produces `web/gi-kg-viewer/dist/`.
When that directory exists, the server mounts it at `/` automatically.

### 3. Start the server

```bash
podcast serve --output-dir /path/to/corpus/output
```

### 4. Open the browser

Navigate to <http://127.0.0.1:8000>.
Set **Corpus root folder** in the UI to the same `--output-dir` path,
then **List files**, select artifacts, and **Load selected into graph**.
Use the **Library** tab to browse feeds and episodes (Corpus Library).

## Architecture

### App factory

[`app.py`][app-py] exposes two entry points:

| Function | Purpose |
| -------- | ------- |
| `create_app(output_dir, *, static_dir)` | Build the `FastAPI` instance with viewer routes and optional static assets. |
| `create_app_for_uvicorn()` | Factory for `uvicorn --factory` (reload mode). Reads `PODCAST_SERVE_OUTPUT_DIR` from the environment. |

`create_app` stores `output_dir` on `app.state` so route handlers can
fall back to it when the caller omits the `?path=` query parameter.

When `output_dir` is set, any `?path=` value must resolve to that directory
or a subdirectory (see `server/pathutil.py`). Overrides are rejected if the
server has no default corpus root, so callers cannot aim the API at arbitrary
paths on disk.

[app-py]: https://github.com/chipi/podcast_scraper/blob/main/src/podcast_scraper/server/app.py

### Route groups

Routers are included with `prefix="/api"` and organized by domain:

```text
routes/
  health.py          # viewer — always available
  artifacts.py       # viewer — list / load GI & KG JSON
  index_stats.py     # viewer — LanceDB index metrics + staleness
  index_rebuild.py   # viewer — POST /index/rebuild (background job)
  search.py          # viewer — semantic corpus search (+ query-activity logging)
  relational.py      # viewer — /relational/* relational-query layer (RFC-094, #882)
  query_activity.py  # viewer — GET /corpus/query-activity (search-volume; FR6.2)
  explore.py         # viewer — GI explore + UC4 NL query
  cil.py             # viewer — CIL position arc, person profile, topic timeline (#527)
  corpus_library.py  # viewer — /corpus/* catalog + similar episodes
  corpus_text_file.py # viewer — GET /corpus/text-file (transcript / caption files under corpus root)
  corpus_digest.py   # viewer — GET /corpus/digest
  corpus_topic_clusters.py   # viewer — GET /corpus/topic-clusters
  feeds.py           # optional — GET/PUT /feeds (feeds.spec.yaml); enable_feeds_api
  operator_config.py # optional — GET/PUT /operator-config; enable_operator_config_api
  jobs.py            # optional — POST/GET /jobs, cancel, reconcile; enable_jobs_api
  scheduled_jobs.py  # optional — GET /scheduled-jobs (cron list + next-run); enable_jobs_api
  platform/          # reserved stub package (#50, #347); not used for RFC-077
```

**RFC-077** routers (`feeds`, `operator_config`, `jobs`) mount when their
matching **`enable_*_api`** flags are true in `create_app` (or the equivalent
`PODCAST_SERVE_ENABLE_*` env vars for `uvicorn --factory` reload).

The **`routes/platform/`** package is **not** mounted today; it is reserved for
future megasketch platform work
([ADR-064](../adr/ADR-064-canonical-server-layer-with-feature-flagged-routes.md)).

### Static file mounting

When a built SPA exists at `web/gi-kg-viewer/dist/`, `create_app` mounts
it at `/` via `StaticFiles(directory=..., html=True)`.
Pass `--no-static` (or `static_dir=False` in code) to skip mounting and
run the API standalone.

### CORS

The CORS middleware allows `http://127.0.0.1:5173` and
`http://localhost:5173` so the Vite dev server can call the API
during development.

## HTTP API

All endpoints live under the `/api` prefix. The full **endpoint catalogue**, request parameters, and **response models** are documented in the dedicated **[HTTP API Reference](../api/HTTP_API.md)** (under `docs/api/`, alongside the programmatic library API). With the server running, the live OpenAPI spec is at **`/docs`** (Swagger UI) and **`/openapi.json`**.

This guide covers **running and extending** the server (below); the HTTP API Reference is the source of truth for *what each endpoint does*.

## MCP server (agent tools)

Separate from the HTTP API, the **generic MCP server** (PRD-034 / RFC-095) exposes the
platform's read capabilities as composable, read-only tools for agentic clients (Claude
Desktop/Code, Cursor). It is **stdio** transport and **library-wrapped** — no HTTP server
required; the corpus directory is the read context.

```bash
pip install -e '.[dev,search]'        # dev includes the MCP SDK; search = ML retrieval deps
podcast mcp --corpus /path/to/corpus  # stdio server; point your agent client at this
```

**Tools (RFC-095):** `resolve_entity` (name → canonical id — call first) and
`search_corpus` (hybrid two-tier; tiers + intent + grounded evidence) from slice 1; plus
the relational traversals (slice 2): `person_positions`, `who_said_about_topic`,
`cross_show_synthesis`, `insights_about_entity`, `topic_entities`, `related_insights`,
`show_episodes`; the CIL intelligence tools (slice 3): `person_profile`,
`topic_timeline`, `position_arc`; and the catalog/navigation tools: `list_feeds`,
`list_episodes`, `episode_detail`, `top_people`. **16 tools.** (`corpus_digest` is
intentionally omitted — agents compose `search_corpus` + `list_episodes` instead of a
recency view.) Full catalogue: [RFC-095](../rfc/RFC-095-generic-mcp-server.md).

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
   in `index_stats.py`, `index_rebuild.py`, `search.py`, `explore.py`,
   `cil.py`, `corpus_library.py`, `corpus_digest.py`, and `corpus_metrics.py`.
5. **RFC-077 routes** live as top-level `routes/*.py` files (same router pattern).
   **`routes/platform/`** is reserved for future #50/#347 surfaces, not feeds/jobs v1.

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

3. **Include the router in `app.py`**:

   ```python
   from podcast_scraper.server.routes import my_domain

   app.include_router(my_domain.router, prefix="/api")
   ```

4. **Write unit tests** in
   `tests/unit/podcast_scraper/server/test_viewer_<domain>.py`
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
| `--output-dir` | *(required)* | Corpus output directory containing `metadata/*.gi.json`, `metadata/*.kg.json`, and `search/`. |
| `--host` | `127.0.0.1` | Bind address. |
| `--port` | `8000` | TCP port. |
| `--reload` | off | Dev only: restart workers when Python files change (uses `uvicorn --factory`). |
| `--no-static` | off | Do not mount built SPA assets even if `web/gi-kg-viewer/dist/` exists. |

### Environment variable

| Variable | Purpose |
| -------- | ------- |
| `PODCAST_SERVE_OUTPUT_DIR` | Set automatically by `run_serve()`. Used by `create_app_for_uvicorn()` in reload mode. Can also be set manually when running uvicorn directly. |
| `PODCAST_SCHEDULER_TZ` | Timezone for the in-process feed-sweep scheduler (#708). Defaults to `TZ` env, then `UTC`. |

### CLI entry point

```bash
podcast serve --output-dir /path/to/output [--host 0.0.0.0] [--port 9000] [--reload]
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

Works on Codespace pre-prod (no systemd) and VPS prod alike, persists with
the corpus (no host state to migrate on redeploy), and routes failures
through the existing job-state webhook surface so Slack and Grafana
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

Misfire grace is **1 hour**: a schedule whose trigger time was missed
(host suspended / rebooting) will fire on wakeup if within 1 h of the
nominal time, then skip silently if not.

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
| `make serve-ui` | Runs `npm run dev` inside `web/gi-kg-viewer/` (Vite on port **5173**). | Frontend-only work, or when running the API separately. |

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
podcast serve --output-dir ./output --reload
```

In reload mode, uvicorn uses `create_app_for_uvicorn` as a factory and
reads `PODCAST_SERVE_OUTPUT_DIR` from the environment.

## Testing

### Unit tests

Located in `tests/unit/podcast_scraper/server/`:

| File | Coverage |
| ---- | -------- |
| `test_viewer_api_m1.py` | Health endpoint, artifact listing, artifact loading, path traversal rejection. |
| `test_viewer_index_stats.py` | Index stats with no corpus, no index, and a mocked LanceDB index. |
| `test_viewer_search.py` | Search with no corpus, mocked `run_corpus_search` results. |
| `test_viewer_explore.py` | Explore filter mode and UC4 natural-language mode with mocked GI functions. |
| `test_cil_queries.py` | CIL query helpers over synthetic bridge + GI/KG bundles (#527). |
| `test_corpus_catalog.py` | Catalog rows, filters, and pagination helpers for Corpus Library. |
| `test_index_rebuild_gate.py` | Per-corpus rebuild mutex (`CorpusRebuildGate`). |
| `test_index_staleness.py` | Index staleness helpers used by index stats. |

**Lift and offsets (search):** `tests/unit/podcast_scraper/search/test_transcript_chunk_lift.py`
exercises **chunk-to-Insight lift** (#528); `test_gil_chunk_offset_verify.py` covers **Quote vs
indexed chunk** alignment helpers used by **`verify-gil-chunk-offsets`**.

All test files guard on `pytest.importorskip("fastapi")` so they stay
runnable in intentionally stripped environments (the default `test-unit`
job installs ``.[dev]``, which includes FastAPI).
Tests use `FastAPI`'s synchronous `TestClient` and `tmp_path` fixtures.

Run them with (requires **`[dev]`** so FastAPI is present):

```bash
pip install -e '.[dev]'   # if needed
make test-unit -k server
```

### Integration tests

`tests/integration/server/test_server_api.py` exercises the **wired** app with real filesystem
artifacts (no mocking of route internals). Tests cover health, artifact listing/loading,
path-traversal blocking, index stats, search (no-index graceful error), explore (filter +
NL modes), and app factory edge cases. Marked `@pytest.mark.integration`.

Additional integration modules under `tests/integration/server/` include
`test_viewer_corpus_library.py` (Corpus Library routes),
`test_viewer_corpus_digest.py` (`GET /api/corpus/digest`),
`test_cil_api.py` (CIL `/api/persons/*` and `/api/topics/*` routes),
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
make test-ui          # ~150 ms, no browser
```

Tests: `web/gi-kg-viewer/src/utils/*.test.ts`. Config: `vite.config.ts` `test` block.

### Browser E2E (Playwright)

[ADR-066](../adr/ADR-066-playwright-for-ui-e2e-testing.md) chose
Playwright for browser-level E2E testing.
Tests live in `web/gi-kg-viewer/e2e/` and run against a Vite dev server
on port **5174** (separate from the dev port 5173).

```bash
make test-ui-e2e
```

This target installs npm dependencies, installs the Firefox browser, and
runs `npm run test:e2e` inside `web/gi-kg-viewer/`.

**CI vs `serve`:** Playwright in CI runs the SPA on **Vite** with **mocked or spec-level `/api/*`**
handling — fast and deterministic. It is **not** a substitute for smoke-testing **`cli serve`**
(or **`make serve`**), where FastAPI serves **built `dist/`** and **real** viewer APIs against a
corpus. See [Testing Guide — Browser E2E](TESTING_GUIDE.md#browser-e2e-gi-kg-viewer-v2).

See the [E2E Testing Guide](E2E_TESTING_GUIDE.md) and
[Testing Guide](TESTING_GUIDE.md) for more detail.

## Platform evolution

**Mounted by default:** **viewer** routers (`health`, `artifacts`, `index_stats`,
`index_rebuild`, `search`, `explore`, `cil`, `corpus_library`, and related **`/api/corpus/*`**
routes) as wired in `create_app`.

**RFC-077 (opt-in):** `routes/feeds.py`, `routes/operator_config.py`, and `routes/jobs.py`
mount when `enable_feeds_api`, `enable_operator_config_api`, and `enable_jobs_api` are
true (CLI flags on `podcast serve` or `PODCAST_SERVE_ENABLE_*` for reload). These are
**not** the historical `routes/platform/*` stubs — see [RFC-077](../rfc/RFC-077-viewer-feeds-and-serve-pipeline-jobs.md)
and [ADR-064 addendum](../adr/ADR-064-canonical-server-layer-with-feature-flagged-routes.md#addendum-2026-04-rfc-077-operator-surfaces-separate-flags).

**Job runner operations note:** the in-process job registry + spawn path targets
**single `uvicorn` worker** local use. Multiple workers or hosts do not share job state;
use one worker for predictable `POST /api/jobs` behavior unless a future design adds
a broker. **Cancel** is **POSIX-first** (`SIGTERM` to the child); Windows semantics differ.

**Still reserved:** `routes/platform/` and `enable_platform` for future megasketch
catalog / DB-backed APIs ([#50][i50], [#347][i347]) — not the RSS-list or RFC-077 job runner.

[i50]: https://github.com/chipi/podcast_scraper/issues/50
[i347]: https://github.com/chipi/podcast_scraper/issues/347

## Batch pipeline JSONL and Grafana Loki (#746)

Long **CLI / docker** batch runs can stream structured events through
`workflow/jsonl_emitter.py` (`run_started`, `episode_finished`, `run_finished`)
to `run.jsonl` when `jsonl_metrics_enabled` is true. For **Grafana Loki**, each
line should be ingested as **one JSON log line**. The optional flag
`jsonl_metrics_echo_stdout` duplicates every JSONL line to **stdout** so the
existing **Grafana Agent** docker log pipeline (`compose/grafana-agent.yaml`)
ships them with stable labels (**env**, **release**, **app**=`podcast_scraper`,
compose **service** surfaced as **service_name** in Grafana Cloud). Use LogQL
`| json | event_type="run_finished"` in Explore or the imported dashboard
`config/grafana/dashboards/common/grafana-dashboard-pipeline-execution.json` (also linked from
`config/manual/`). Prometheus **`/metrics`** on the API container remains
separate from this JSONL path.

## Related docs

| Document | Description |
| -------- | ----------- |
| [GI/KG viewer v2](../rfc/RFC-062-gi-kg-viewer-v2.md) | Viewer design (milestones, success criteria). |
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

**Version:** 1.3
**Created:** 2026-04-04
**Updated:** 2026-05-05 — Batch pipeline JSONL / Loki (#746)
