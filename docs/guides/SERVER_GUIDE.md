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
  index_stats.py     # viewer — FAISS vector index metrics + staleness
  index_rebuild.py   # viewer — POST /index/rebuild (background job)
  search.py          # viewer — semantic corpus search
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

## API reference

All endpoints live under the `/api` prefix. With the server running, OpenAPI is at **`/docs`** (Swagger UI) and **`/openapi.json`**.

### Authentication

Local **dev** server: no auth. Treat **production** deployments as out-of-scope for this guide unless you add your own reverse proxy or middleware.

| Method | Path | Tag | Description | Key query params |
| ------ | ---- | --- | ----------- | ---------------- |
| GET | `/api/health` | health | Liveness and **capability flags**: `status`; core viewer `artifacts_api`, `search_api`, `explore_api`, `index_routes_api`, `corpus_metrics_api`, `cil_queries_api` (default **true** when mounted); catalog `corpus_library_api`, `corpus_digest_api`, `corpus_binary_api`. **RFC-077 (default false):** `feeds_api`, `operator_config_api`, `jobs_api` when those route groups are mounted. Omit digest flag on older builds → clients treat digest as unavailable. | — |
| GET, PUT | `/api/feeds` | feeds | Read/write structured **`feeds.spec.yaml`** under the resolved corpus root (JSON **`{ "feeds": [...] }`** on PUT). | `path` |
| GET, PUT | `/api/operator-config` | operator_config | Read/write **viewer-safe** operator YAML at the server-resolved path. **GET** returns `content`, `operator_config_path`, and **`available_profiles`** (union of packaged `config/profiles/*.yaml` names from **cwd** and **repo** roots, same as `Config` preset resolution; excluding `*.example.yaml`). When the file is **missing or whitespace-only** and the packaged preset **`cloud_balanced`** exists, **GET** **writes** a minimal `profile: cloud_balanced` file first (idempotent if already that content). **PUT** rejects forbidden **secret** keys and top-level **feed** keys (`rss`, `rss_url`, `rss_urls`, `feeds`) — use `/api/feeds` / **`feeds.spec.yaml`** for feeds. See [RFC-077](../rfc/RFC-077-viewer-feeds-and-serve-pipeline-jobs.md) (Phase 1b) and [PRD-030](../prd/PRD-030-viewer-feed-sources-and-pipeline-jobs.md). | `path` |
| GET, POST | `/api/jobs` | jobs | List (`GET`) or enqueue (`POST`) pipeline jobs for the corpus; **`GET /api/jobs/{id}`**, **`POST /api/jobs/{id}/cancel`**, **`POST /api/jobs/reconcile`**. | `path` |
| GET | `/api/scheduled-jobs` | scheduled-jobs | List in-process cron schedules from `viewer_operator.yaml` `scheduled_jobs:` ([#708](https://github.com/chipi/podcast_scraper/issues/708)). Each row carries `name`, `cron`, `enabled`, `next_run_at` (UTC ISO; `null` when disabled or invalid cron). Mounts only when **`enable_jobs_api`** is on. Operators add/remove schedules by editing the YAML via **`PUT /api/operator-config`**, which triggers a scheduler reload in-process. | `path` |
| GET | `/api/artifacts` | artifacts | List `*.gi.json`, `*.kg.json`, and `*.bridge.json` (recursive); each item includes `mtime_utc` (#507) and `publish_date` (YYYY-MM-DD from episode metadata when present, else UTC calendar day from file mtime). | `path` (required) — corpus output directory |
| GET | `/api/artifacts/{path}` | artifacts | Load and return a single artifact JSON by relative path. | `path` (required) — corpus root for the relative lookup |
| GET | `/api/index/stats` | index | FAISS index stats, staleness heuristics, and rebuild job flags (`rebuild_in_progress`, `rebuild_last_error`; #507). | `path`, `embedding_model` (optional; compare index to this id, else `Config` default) |
| POST | `/api/index/rebuild` | index | Queue background `index_corpus` (202); mutex per corpus. Poll `GET /api/index/stats`. | `path`, `rebuild`, `embedding_model`, `vector_index_path`, `vector_faiss_index_mode`, `vector_index_types` (comma-separated) |
| GET | `/api/search` | search | Semantic corpus search via FAISS + sentence embeddings. **Transcript** hits may include optional **`lifted`**: overlapping **Quote** → **Insight** plus **speaker** / **topic** display names from **`bridge.json`** when present ([Semantic Search Guide — lift](SEMANTIC_SEARCH_GUIDE.md#chunk-to-insight-lift-and-offset-verification-rfc-072--528)). **Insight** hits may include **`supporting_quotes`** (from indexer enrichment): quote **`speaker_id`** / **`speaker_name`** mirror **`.gi.json`** — often **`null`** / absent when segments lack diarization labels (GitHub [#541](https://github.com/chipi/podcast_scraper/issues/541); canonical rules: [Development Guide — GI quote `speaker_id`](DEVELOPMENT_GUIDE.md#gi-quote-speaker-id)). Successful responses include optional **`lift_stats`**: **`transcript_hits_returned`** and **`lift_applied`** for the returned page (after `top_k`). | `q` (required), `path`, `type`, `feed`, `since`, `speaker`, `grounded_only`, `top_k`, `embedding_model`, `dedupe_kg_surfaces` (default `true`: merge same-text `kg_entity` / `kg_topic` rows) |
| GET | `/api/explore` | explore | GI cross-episode explore (filter mode) or UC4 natural-language query. Insight rows may include **`supporting_quotes`** with **`speaker_id`** / **`speaker_name`** mirroring **`.gi.json`** (often absent without diarization — GitHub [#541](https://github.com/chipi/podcast_scraper/issues/541); [Development Guide — GI quote `speaker_id`](DEVELOPMENT_GUIDE.md#gi-quote-speaker-id)). | `path`, `question` / `q`, `topic`, `speaker`, `grounded_only`, `min_confidence`, `sort_by`, `limit`, `strict` |
| GET | `/api/persons/{person_id}/positions` | cil | Position arc — chronological insights for a **person** and **topic** across episodes. Scans `**/*.bridge.json` with sibling GI/KG. | `topic` (required), `path`, `insight_types` (comma-separated; omit → `claim` only; `all` / `*` → no filter) |
| GET | `/api/persons/{person_id}/brief` | cil | Person profile — insights grouped by topic plus quotes for that person. | `path` |
| GET | `/api/persons/{person_id}/topics` | cil | Distinct topic ids for that person (from brief keys). | `path` |
| GET | `/api/topics/{topic_id}/timeline` | cil | Topic timeline — insights about the topic per episode. Each episode may include `episode_title`, `feed_title`, `episode_number`, and artwork fields (`episode_image_url`, `episode_image_local_relpath`, `feed_image_url`, `feed_image_local_relpath`) from sibling `*.metadata.json`. | `path`, `insight_types` (omit → all types; `all` / `*` → all) |
| POST | `/api/topics/timeline` | cil | **Merged** topic timeline — same Pattern C rules as GET, but **one** corpus scan for **multiple** `topic_ids` in the JSON body (cluster scope). Response includes `topic_ids` (deduped, canonical order) and `episodes` (merged per `episode_id`, insight nodes deduped). | JSON body: `topic_ids` (required), `path` (optional if default output dir), `insight_types` (optional; same semantics as GET) |
| GET | `/api/topics/{topic_id}/persons` | cil | Distinct `person:` ids that discuss the topic via grounded quotes. | `path` |
| POST | `/api/corpus/resolve-episode-artifacts` | corpus | Map logical **`episode_id`** values (from metadata) to corpus-relative **`gi_relative_path`** / **`kg_relative_path`** / **`bridge_relative_path`** from one catalog scan. Rows without GI on disk are listed in **`missing_episode_ids`**. The GI/KG viewer’s topic-cluster **sibling auto-load** builds candidate ids from **`topic_clusters.json`** → **`clusters[].members[].episode_ids`** (union per touched cluster). | JSON body: `episode_ids` (required), `path` (optional if server default set) |
| POST | `/api/corpus/node-episodes` | corpus | **Progressive graph expansion** (cross-episode): for a canonical **`node_id`** (`person:`, `org:`, or `topic:`), scan **`*.bridge.json`** bundles (no full triple-read), then resolve sibling **`*.gi.json`** / **`*.kg.json`** by stem. Returns **`episodes`** (metadata-relative paths when present, optional **`episode_id`**, GI/KG flags) and **`truncated`** when the cap is hit. The GI/KG viewer uses this on eligible nodes (**double-tap** expand). | JSON body: `node_id` (required), `path` (optional if server default set), `limit` (optional cap) |
| GET | `/api/corpus/feeds` | corpus | Aggregate feeds from episode metadata under the corpus root. | `path` (optional if server default set) |
| GET | `/api/corpus/episodes` | corpus | Paginated episode list (newest-first scan); optional filters. Each item includes **`cil_digest_topics`** as an **empty** array (CIL pills are returned on **digest** rows and **episode detail** only, to avoid per-row bridge reads in the list). GI/KG path fields support graph loads after opening an episode. When **`topic_cluster_only=true`**, the server keeps rows whose **`bridge.json`** topic appears on a multi-member topic-cluster **member** that lists this episode's **`episode_id`** in **`members[].episode_ids`** (reads **`search/topic_clusters.json`** per request; pipeline-built clusters include those ids). | `path`, `feed_id`, `q` (title substring), `topic_q`, `since` (`YYYY-MM-DD`), `topic_cluster_only` (boolean), `limit` (1–200), `cursor` |
| GET | `/api/corpus/episodes/detail` | corpus | Episode row + summary bullets + GI/KG/bridge paths + **`cil_digest_topics`** (same shape as list / digest). | `path`, `metadata_relpath` (required) |
| GET | `/api/corpus/episodes/similar` | corpus | FAISS semantic peers for an episode; **200** with `error` when index missing. | `path`, `metadata_relpath` (required), `top_k` (1–25) |
| GET | `/api/corpus/digest` | corpus | Feed-diverse **recent episodes** (metadata + GI/KG flags), optional **`cil_digest_topics`** per row (bridge + topic-cluster order; omitted when `compact=true`), and optional **semantic topic** bands. `compact=true` forces 24h, smaller cap, no topic bands and no CIL pill enrichment. | `path`, `window` (`24h` / `7d` / `1mo` / `since`), `since` (required if `window=since`; `1mo` = previous **calendar month**, UTC), `compact`, `include_topics`, `max_rows` |
| GET | `/api/corpus/topic-clusters` | corpus | **Topic clustering** artifact: returns `search/topic_clusters.json` when present (**404** with `available: false` when missing). | `path` (optional if server default set) |
| GET | `/api/corpus/text-file` | corpus | Inline file under the corpus root for browser viewing (`.txt`, `.md`, `.vtt`, `.srt`, `.json`). Graph **Quote** node detail uses this for the in-app transcript viewer and the **Open raw transcript in new tab** header link. Pipeline runs that **direct-download** WebVTT/SubRip normalize to **`transcripts/… .txt`** (plus `*.segments.json` for GI timing); metadata usually points at that `.txt`. If `relpath` ends with `.txt` (and is not already `*.cleaned.txt`) and that file is missing, the server tries the sibling `stem.cleaned.txt` (metadata often still references the raw Whisper path). | `path` (optional if server default set), `relpath` (required) |
| GET | `/api/corpus/stats` | corpus | **Publish-month** histogram (`YYYY-MM` → episode count) from one catalog scan; GI/KG Dashboard. | `path` |
| GET | `/api/corpus/documents/manifest` | corpus | Return `corpus_manifest.json` at corpus root (**404** if missing). | `path` |
| GET | `/api/corpus/documents/run-summary` | corpus | Return `corpus_run_summary.json` at corpus root (**404** if missing). | `path` |
| GET | `/api/corpus/runs/summary` | corpus | Discover `run.json` under the tree (capped), compact metrics per file for Dashboard. | `path` |
| GET | `/api/corpus/coverage` | corpus | GI/KG sibling-file presence per episode; aggregates by publish month and feed (Dashboard). | `path` |
| GET | `/api/corpus/persons/top` | corpus | Top speakers by grounded insight count (scans `*.gi.json` under catalog). | `path`, `limit` |

Design and response field semantics: [Corpus Digest](../rfc/RFC-068-corpus-digest-api-viewer.md). Topic strings: repo `config/digest_topics.yaml`.

### Response models

Pydantic response schemas are defined in
[`schemas.py`][schemas-py]:

- `HealthResponse`
- `ArtifactListResponse` / `ArtifactItem`
- `IndexStatsEnvelope` / `IndexStatsBody` / `IndexRebuildAccepted`
- `CorpusSearchApiResponse` / `SearchHitModel` (optional **`lifted`** on transcript rows when lift applies; optional **`supporting_quotes`** on insight rows — quote speaker fields follow GI segment/diarization rules, issue **#541**)
- `ExploreApiResponse` (insight **`supporting_quotes`** speaker fields follow GI segment/diarization rules, issue **#541**)
- `CilArcEpisodeBlock` / `CilPositionArcResponse` / `CilPersonProfileInsightRow` / `CilPersonProfileQuoteRow` / `CilPersonProfileResponse` / `CilTopicTimelineResponse` / `CilTopicTimelineMergeRequest` / `CilTopicTimelineMergedResponse` / `CilIdListResponse`
- `CorpusResolveEpisodesRequest` / `CorpusResolveEpisodesResponse` / `CorpusResolvedEpisodeArtifact`
- `CorpusFeedsResponse` / `CorpusFeedItem`
- `CorpusEpisodesResponse` / `CorpusEpisodeListItem`
- `CorpusEpisodeDetailResponse`
- `CorpusSimilarEpisodesResponse` / `CorpusSimilarEpisodeItem`
- `CorpusDigestResponse` / `CorpusDigestRow` / `CorpusDigestTopicBand` / `CorpusDigestTopicHit`
  (digest rows: **`summary_bullet_graph_topic_ids`** parallel to **`summary_bullets_preview`**;
  topic bands: **`graph_topic_id`** — `topic:{slug}` from the band label for Graph focus)
- `CorpusStatsResponse` / `CorpusRunsSummaryResponse` / `CorpusRunSummaryItem`

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
`[server]` extra). Mounts whenever `enable_jobs_api=True` and the operator
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
the `[server]` extra).

### Limitations (V1)

- Per-schedule overrides (profile / feeds / max_episodes) are not wired —
  use multiple schedules with different operator YAMLs if you need them.
- No calendar-aware schedules (`every 3rd Tuesday`); standard cron only.
- No Scheduled tab in the viewer — operators edit `scheduled_jobs:` via
  the existing Configuration YAML editor. The `GET /api/scheduled-jobs`
  endpoint exists so a future viewer tab can list schedules + next-run
  previews; that UI is a follow-up of #708.

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
| `test_cil_queries.py` | CIL query helpers over synthetic bridge + GI/KG bundles (#527). |
| `test_corpus_catalog.py` | Catalog rows, filters, and pagination helpers for Corpus Library. |
| `test_index_rebuild_gate.py` | Per-corpus rebuild mutex (`CorpusRebuildGate`). |
| `test_index_staleness.py` | Index staleness helpers used by index stats. |

**Lift and offsets (search):** `tests/unit/podcast_scraper/search/test_transcript_chunk_lift.py`
exercises **chunk-to-Insight lift** (#528); `test_gil_chunk_offset_verify.py` covers **Quote vs
indexed chunk** alignment helpers used by **`verify-gil-chunk-offsets`**.

All test files guard on `pytest.importorskip("fastapi")` so they are
skipped when the `[server]` extra is not installed.
Tests use `FastAPI`'s synchronous `TestClient` and `tmp_path` fixtures.

Run them with (requires **`[server]`** so tests are not skipped):

```bash
pip install -e '.[dev,server]'   # if needed
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
`config/grafana/grafana-dashboard-pipeline-execution.json` (also linked from
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
