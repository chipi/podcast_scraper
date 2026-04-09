# Server Guide

The FastAPI server layer for the GI/KG viewer and future platform APIs.

## Overview

`podcast_scraper` follows a **"one pipeline core, multiple shells"** philosophy:
the same Python library that powers the CLI, service API, and batch workflows
also backs an HTTP server.
The server exposes corpus artifacts, semantic search, and GI explore
through a JSON API consumed by the Vue 3 SPA
([`web/gi-kg-viewer/`][viewer-readme]).

The server was introduced by
[RFC-062](../rfc/RFC-062-gi-kg-viewer-v2.md) and is implemented in
[`src/podcast_scraper/server/`][server-pkg].
Route groups are additive: **viewer routes** ship today; **platform routes**
(feeds, episodes, jobs, status) will follow the same pattern
([ADR-064](../adr/ADR-064-canonical-server-layer-with-feature-flagged-routes.md)).

[viewer-readme]: https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/README.md
[server-pkg]: https://github.com/chipi/podcast_scraper/blob/main/src/podcast_scraper/server/

## Quick start

### 1. Install the `[server]` extra

```bash
pip install -e '.[server]'
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
  index_stats.py     # viewer — FAISS vector index metrics
  search.py          # viewer — semantic corpus search
  explore.py         # viewer — GI explore + UC4 NL query
  platform/          # placeholder stubs (feeds, episodes, jobs, status)
```

Platform routers (`routes/platform/`) are **not mounted** in the current
milestone.
When they are ready, they will be included in `create_app` behind a
feature-flag check
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

## API reference

All endpoints live under the `/api` prefix.

| Method | Path | Tag | Description | Key query params |
| ------ | ---- | --- | ----------- | ---------------- |
| GET | `/api/health` | health | Server health check. Returns `{"status": "ok"}`. | — |
| GET | `/api/artifacts` | artifacts | List `*.gi.json` and `*.kg.json` files under a corpus directory (recursive). | `path` (required) — corpus output directory |
| GET | `/api/artifacts/{path}` | artifacts | Load and return a single artifact JSON by relative path. | `path` (required) — corpus root for the relative lookup |
| GET | `/api/index/stats` | index | FAISS vector index aggregate statistics. | `path` (optional) — corpus dir; omit to use server default |
| GET | `/api/search` | search | Semantic corpus search via FAISS + sentence embeddings. | `q` (required), `path`, `type`, `feed`, `since`, `speaker`, `grounded_only`, `top_k`, `embedding_model` |
| GET | `/api/explore` | explore | GI cross-episode explore (filter mode) or UC4 natural-language query. | `path`, `question` / `q`, `topic`, `speaker`, `grounded_only`, `min_confidence`, `sort_by`, `limit`, `strict` |

### Response models

Pydantic response schemas are defined in
[`schemas.py`][schemas-py]:

- `HealthResponse`
- `ArtifactListResponse` / `ArtifactItem`
- `IndexStatsEnvelope` / `IndexStatsBody`
- `CorpusSearchApiResponse` / `SearchHitModel`
- `ExploreApiResponse`

[schemas-py]: https://github.com/chipi/podcast_scraper/blob/main/src/podcast_scraper/server/schemas.py

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
   in `index_stats.py`, `search.py`, and `explore.py`.
5. **Platform routes** use a sub-package (`routes/platform/`) and will
   follow the same conventions when mounted.

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

5. **Update this guide** — add the new endpoint to the API reference table.

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

### CLI entry point

```bash
podcast serve --output-dir /path/to/output [--host 0.0.0.0] [--port 9000] [--reload]
```

The `serve` sub-command is handled by
[`cli_handlers.py`][cli-handlers] (`parse_serve_argv` + `run_serve`).

[cli-handlers]: https://github.com/chipi/podcast_scraper/blob/main/src/podcast_scraper/server/cli_handlers.py

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
| `test_viewer_index_stats.py` | Index stats with no corpus, no index, and mocked FAISS store. |
| `test_viewer_search.py` | Search with no corpus, mocked `run_corpus_search` results. |
| `test_viewer_explore.py` | Explore filter mode and UC4 natural-language mode with mocked GI functions. |

All test files guard on `pytest.importorskip("fastapi")` so they are
skipped when the `[server]` extra is not installed.
Tests use `FastAPI`'s synchronous `TestClient` and `tmp_path` fixtures.

Run them with (requires **`[server]`** so tests are not skipped):

```bash
pip install -e '.[dev,server]'   # if needed
make test-unit -k server
```

### Integration tests

`tests/integration/test_server_api.py` exercises the **wired** app with real filesystem
artifacts (no mocking of route internals). Tests cover health, artifact listing/loading,
path-traversal blocking, index stats, search (no-index graceful error), explore (filter +
NL modes), and app factory edge cases. Marked `@pytest.mark.integration`.

```bash
pytest tests/integration/test_server_api.py -v
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

The `routes/platform/` sub-package contains placeholder stubs for future
platform routes:

| Module | Future endpoint | Tracking |
| ------ | --------------- | -------- |
| `feeds.py` | `GET/POST /api/feeds` | [#50][i50] |
| `episodes.py` | `GET /api/episodes` | [#50][i50] |
| `jobs.py` | `GET/POST /api/jobs` | [#347][i347] |
| `status.py` | `GET /api/status` | [#347][i347] |

These will be added using the same pattern described in
[Adding new routes](#adding-new-routes):
create the router, add schemas, include in `app.py`, write tests.
Feature flags
([ADR-064](../adr/ADR-064-canonical-server-layer-with-feature-flagged-routes.md))
will gate platform routes so they can be enabled independently of the
viewer routes.

[i50]: https://github.com/chipi/podcast_scraper/issues/50
[i347]: https://github.com/chipi/podcast_scraper/issues/347

## Related docs

| Document | Description |
| -------- | ----------- |
| [RFC-062](../rfc/RFC-062-gi-kg-viewer-v2.md) | GI/KG viewer v2 design (milestones, success criteria). |
| [ADR-064](../adr/ADR-064-canonical-server-layer-with-feature-flagged-routes.md) | Canonical server layer with feature-flagged route groups. |
| [ADR-065](../adr/ADR-065-vue3-vite-cytoscape-frontend-stack.md) | Vue 3 + Vite + Cytoscape.js frontend stack decision. |
| [ADR-066](../adr/ADR-066-playwright-for-ui-e2e-testing.md) | Playwright for UI E2E testing. |
| [Testing Guide](TESTING_GUIDE.md) | Commands, markers, and browser E2E section. |
| [E2E Testing Guide](E2E_TESTING_GUIDE.md) | Playwright browser E2E details. |
| [Development Guide](DEVELOPMENT_GUIDE.md) | Dev environment, `make serve` / `serve-api` / `serve-ui`. |
| [Viewer README][viewer-readme] | Frontend SPA setup, dev workflow, build instructions. |
| [`src/podcast_scraper/server/`][server-pkg] | Server source code. |

---

**Version:** 1.0
**Created:** 2026-04-04
