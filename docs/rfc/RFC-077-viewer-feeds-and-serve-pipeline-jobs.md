# RFC-077: Viewer feeds API, operator config, job runner, and process hygiene (`serve`)

- **Status**: Draft
- **Authors**: Podcast Scraper Team
- **Stakeholders**: Maintainers; viewer operators
- **Tracking**: [GitHub #626](https://github.com/chipi/podcast_scraper/issues/626)
- **Related PRDs**:
  - [PRD-030](../prd/PRD-030-viewer-feed-sources-and-pipeline-jobs.md)
  - [PRD-003](../prd/PRD-003-user-interface-config.md)
  - [PRD-025](../prd/PRD-025-corpus-intelligence-dashboard-viewer.md)
- **Related ADRs**:
  - [ADR-064](../adr/ADR-064-canonical-server-layer-with-feature-flagged-routes.md) — one server; feature-flagged routes (see **§ ADR** for whether a *new* ADR is needed for subprocess jobs)
- **Related RFCs**:
  - [RFC-008](RFC-008-config-model.md) — `Config` / validation
  - [RFC-062](RFC-062-gi-kg-viewer-v2.md) — viewer + `serve`
  - [RFC-063](RFC-063-multi-feed-corpus-append-resume.md)
  - [RFC-065](RFC-065-live-pipeline-monitor.md) — **CLI** monitor only; not authoritative for HTTP jobs
  - [RFC-007](RFC-007-cli-interface.md)
- **Related UX specs**:
  - [VIEWER_IA.md](../uxs/VIEWER_IA.md)
  - [UXS-001](../uxs/UXS-001-gi-kg-viewer.md)
  - [UXS-006](../uxs/UXS-006-dashboard.md)
- **Related Documents**:
  - [SERVER_GUIDE.md](../guides/SERVER_GUIDE.md)
  - [CONFIGURATION.md](../api/CONFIGURATION.md)
  - [GitHub #593](https://github.com/chipi/podcast_scraper/issues/593) — optional root **`profile:`** merges packaged `config/profiles/<name>.yaml` defaults (`Config._resolve_profile`)

## Abstract

This RFC specifies **Phase 1**: (a) opt-in **`GET`/`PUT /api/feeds`** for structured corpus **`feeds.spec.yaml`** (root **`{ feeds: [...] }`** — JSON on the wire; extension selects YAML vs JSON on disk when using **`--feeds-spec`**); (b) opt-in **`GET`/`PUT /api/operator-config`** for a **viewer-safe operator YAML** file whose location follows **precedence**: path from **`podcast serve --config-file`** when the server was started with it, else a **fixed basename under the resolved corpus root** (e.g. `viewer_operator.yaml`). **Secrets never belong in that file** — only environment variables; **PUT** and (optionally) **GET** enforce **forbidden secret keys** and **forbidden feed-list keys** at the YAML root (see § Phase 1b), aligned with [RFC-008](RFC-008-config-model.md). **`GET`** also returns **`available_profiles`** for the viewer profile picker (#593).

**Phase 2** specifies **HTTP-triggered pipeline jobs** (**child OS subprocess** of `serve` — see **§ Phase 2 — Architecture decision** for why and what we rejected), a **durable job registry**, **stale and orphan detection**, **cancel** semantics, and **operator-facing reconciliation** so many runs do not leave ambiguous background state. **`GET /api/health`** gains only **capability booleans** (`feeds_api`, `operator_config_api`, `jobs_api`); job payloads never live on health.

**Product choices (from stakeholder, #626):** secrets **only in env**; config path **serve `--config-file` else corpus default**.

## Problem Statement

Operators need **feeds + config + job clarity** without leaving the viewer. Without hygiene, **many subprocess jobs** create **zombie/stale records**, **orphan PIDs**, and **unclear “what is still running”** at end of day. Without config in the UI, operators **over-type** paths and **paste secrets into YAML** — we forbid the latter in the operator file and steer env instead.

## Goals

1. Feeds API + health flag + serve opt-in (**Phase 1a**).
2. Operator config API + validation + health flag + serve opt-in (**Phase 1b** — may ship same or next PR).
3. Job runner + registry + cancel + stale/reconcile + Dashboard surfacing (**Phase 2**).
4. Document **forbidden keys** and **path resolution** in SERVER_GUIDE.

## Constraints & Assumptions

- Corpus path resolution reuses `resolve_corpus_path_param` in `src/podcast_scraper/server/pathutil.py` (same as other viewer routes).
- **Forbidden keys** on PUT: maintain `FORBIDDEN_OPERATOR_CONFIG_KEYS` (or generate from `Config` fields documented as secrets in CONFIGURATION.md / field names matching `*_api_key`, `api_key`, `openai_api_key`, etc. — implementation picks one approach; tests must cover rejection).
- **GET** may return `409` / structured error if on-disk file **already** contains forbidden keys (legacy) — operator fixes out-of-band, or server offers one-time **strip** behind explicit `?force_sanitize=1` (optional — default safe).

## Design & Implementation

### Phase 1a — Feeds (structured spec)

- **Canonical basename:** **`feeds.spec.yaml`** at corpus root (writer default). **`--feeds-spec PATH`** accepts **`.yaml`**, **`.yml`**, or **`.json`**; parser choice follows the file extension (same pattern as main **`--config`** / `load_config_file`).
- **Document schema:** top-level **object** with a single **`feeds`** array (required). Each element is either a **string** (RSS URL) or an **object** with required **`url`** (http/https) and optional keys that are an **explicit allowlist** of per-inner-run `Config` fields (download resilience, timeouts, user agent, episode window — see `podcast_scraper.rss.feeds_spec.RSS_FEED_ENTRY_OVERRIDE_KEYS`). Unknown keys on feed objects are rejected (**`extra="forbid"`**). Unknown top-level keys besides **`feeds`** and optional **`_comment*`** are rejected.
- **`GET`/`PUT /api/feeds?path=`** — JSON: **`GET`** returns **`path`**, **`file_relpath`**, **`feeds`** (mixed strings and objects; URL-only entries may be returned as bare strings). **`PUT`** body **`{ "feeds": [...] }`**; server validates, dedupes by URL (first-seen order), writes **UTF-8 YAML** atomically to **`feeds.spec.yaml`**.
- **Merge order (CLI and jobs):** packaged **`profile:`** (if any) → global operator **`Config`** → **per-feed entry** fields from the spec file for overlapping keys on each inner `run_pipeline` config (`model_copy` update).
- **`feeds_api`** on **`GET /api/health`** reflects `app.state.feeds_api_enabled` (from `create_app(..., enable_feeds_api=...)`).
- `create_app(..., enable_feeds_api=False)`; `PODCAST_SERVE_ENABLE_FEEDS_API` for the uvicorn reload factory (`serve_feature_kwargs_from_environ`).
- Fix typo in any prior notes: health route is **`GET /api/health`**, not `/api/handler`.

### Phase 1b — Operator config file

**Path precedence (computed once at server startup — not chosen by the browser):**

1. If `podcast serve` was started with **`--config-file <PATH>`** (wired through `parse_serve_argv` → `create_app(..., operator_config_file=PATH)` and `PODCAST_SERVE_CONFIG_FILE` for reload factory), set  
   `app.state.operator_config_path = Path(PATH).expanduser().resolve()`.
2. Else set `app.state.operator_config_path = output_dir.resolve() / "viewer_operator.yaml"` (basename documented in SERVER_GUIDE).

**Security:** `GET`/`PUT` accept the same **`path=`** corpus query as other viewer routes (for **anchor checks** and consistency), but the **YAML bytes are read/written only at `operator_config_path`** — the client **never** supplies a filesystem path for the config file itself. That honors “prefer `--config-file` when set, else corpus default” **without** path injection.

**Wire `podcast serve`:** add optional `--config-file`; set env for reload; pass into `create_app`.

**HTTP:**

- **`GET /api/operator-config?path=<corpus>`** — resolve corpus with `resolve_corpus_path_param`; JSON matches **`OperatorConfigGetResponse`**: `corpus_path`, **`operator_config_path`** (absolute path string), `content`, **`available_profiles`** (sorted list of packaged preset **names** without `.yaml` extension). **Side effect:** if the operator file is **missing or whitespace-only** and **`cloud_balanced`** is in **`available_profiles`**, the handler **creates** the file with **`profile: cloud_balanced`** before returning (so the viewer profile picker has a sane default without an extra **PUT**). If **`cloud_balanced`** is not packaged in the environment, **`content`** may stay empty until the operator **PUT**s.
- **`PUT /api/operator-config?path=<corpus>`** — same corpus resolution; body = YAML string; **reject** forbidden secret keys **and** forbidden **feed-list** top-level keys (see **§ Feeds vs operator YAML**); atomic write to **`operator_config_path`** only.

**Health:** `operator_config_api: bool` reflects `app.state.operator_config_api_enabled`.

**Viewer:** status bar **Config** next to **Feeds**; shared **modal with tabs** (Feeds | Config) per UXS-001; Config tab = **Profile** `<select>` (from `available_profiles` + “None”) + monospace **overrides** YAML + Save; surface validation errors from `422` / forbidden-key `400` bodies.

#### Profile composition (#593)

Operator YAML may include a single top-level **`profile: <preset>`** line. At pipeline load, `Config` merges **`config/profiles/<preset>.yaml`** as **defaults**, then applies explicit keys from the same operator file (explicit wins). The `profile` key is consumed and is not a persisted `Config` field. **`available_profiles`** on **`GET /api/operator-config`** lists packaged presets by **unioning** `*.yaml` stems from every **`config/profiles`** directory that exists: **(1)** cwd-relative `config/profiles` (same as `Config._resolve_profile`’s first candidate), **(2)** repo-root `config/profiles` next to the installed sources, de-duplicated by resolved path. Excludes **`*.example.yaml`** (stem ends with `.example`). If no directories exist, **`available_profiles`** is an empty array.

#### Feeds vs operator YAML

Canonical **viewer** workflow: **feeds** live in corpus **`feeds.spec.yaml`** (Feeds API; CLI **`--feeds-spec`** with the same resolved path). Operator YAML **must not** duplicate feed sources at the **root** mapping. **`PUT /api/operator-config`** rejects these top-level keys (normalized): **`rss`**, **`rss_url`**, **`rss_urls`**, **`feeds`** — response `detail.error` = **`forbidden_operator_feed_keys`** when **every** forbidden key is a feed-list key; **`forbidden_operator_keys`** when any other forbidden key appears (**including** a mix of feed keys and secrets — clients should use `detail.keys` for specifics; same HTTP **400** shape with `keys` list).

**Phase 2 jobs:** when **`feeds.spec.yaml`** exists under the corpus anchor, child argv passes **`--feeds-spec <absolute path>`** and **`--config`** (same flag the main CLI uses) to `operator_config_path`; effective pipeline config includes **`profile:`** merge as above. Legacy **`--rss-file`** remains on the main CLI for line lists but is not what the job runner or Feeds API emit.

### Phase 2 — Jobs and subprocess

- **`POST /api/jobs`** (enqueue), **`GET /api/jobs`** (list), **`GET /api/jobs/{id}`** (detail), **`POST /api/jobs/{id}/cancel`**, **`POST /api/jobs/reconcile`** (registry + PID hygiene).
- **`jobs_api`** on **`GET /api/health`** reflects `app.state.jobs_api_enabled`.

### Phase 2 — Architecture decision: where the pipeline runs (review this)

**Chosen approach (v1):** start the pipeline as an **OS child process** of the `uvicorn` parent — typically **`asyncio.create_subprocess_exec`** with **`sys.executable`** and **`python -m podcast_scraper.cli …`** (exact argv mirrors what operators would type). The HTTP handler returns **202** + **`job_id`** immediately; the parent **records PID**, **redirects child stdout/stderr** to a corpus-local log file, and a **background asyncio task** `await`s the child to update the job registry (`succeeded` / `failed` / exit code).

**Why we decided this (fit for `podcast serve` + ADR-064):**

1. **CLI parity & debuggability** — failures reproduce with the **same entrypoint** as manual runs; support and docs already center on `podcast …` argv.
2. **Crash and memory isolation** — pipeline OOM or hard abort in native code **should not** tear down the HTTP server that serves search, artifacts, and the SPA.
3. **CPU / event loop** — the scraper is **CPU- and I/O-heavy**; running it synchronously inside an `async def` route **blocks** the event loop. Starlette **`BackgroundTasks`** still run **in-process** and are widely treated as appropriate only for **lightweight** work; they do **not** add crash isolation. A **separate interpreter process** avoids sharing fate with the API worker.
4. **Alignment with “one server”** ([ADR-064](../adr/ADR-064-canonical-server-layer-with-feature-flagged-routes.md)) — we add **routes + a small runner** inside the existing FastAPI app instead of introducing a **second** always-on service (broker worker) for the **local operator** story.

#### Compose / full-stack Docker (#660, RFC-079 Phase 2)

When the API runs inside **[RFC-079](RFC-079-full-stack-docker-compose.md)** compose and execution mode selects **Docker** pipeline runs, **[GitHub #660](https://github.com/chipi/podcast_scraper/issues/660)** may replace in-container `sys.executable` subprocess with a factory that invokes the **`pipeline` service** image. That path is **additive**: **native** `make serve-api` (full venv on the host) keeps the **subprocess + argv** model above unchanged. **Docker-only** operator metadata (e.g. **`pipeline_install_extras`** for **`ml`** vs **`llm`** image tier) is **validated only** when enqueueing/spawning in that Docker mode — not for laptop-only operators. See [RFC-079 §Native vs Docker](RFC-079-full-stack-docker-compose.md#native-vs-docker).

**Alternatives considered (explicit non-picks for v1):**

| Option | Pros | Cons | Verdict |
| ------ | ---- | ---- | ------- |
| **In-process** `service.run()` on a thread pool (`run_in_executor`) | Easiest to attach Python loggers; no argv marshalling | **Same process** as `serve` — OOM / fatal error can kill the viewer; **GIL** contention with request threads; **`uvicorn --reload`** restarts **abort** in-flight runs; cancel is cooperative only | **Not v1 default** — acceptable only for **very short** dev experiments |
| **Starlette `BackgroundTasks`** | Minimal code | Runs **after response** but still **in-process**; same crash/GIL/reload issues; **no** strong job lifecycle without extra plumbing | **Rejected** for multi-hour pipelines |
| **External task queue** (Celery, RQ, Dramatiq, Redis + separate worker) | Retries, persistence, horizontal workers | **New operational dependencies**; second deployable; overkill for **single-machine** `127.0.0.1` workflows; duplicates what ADR-064 deferred as “platform” until needed | **Future** if we ship a **hosted multi-tenant** product |
| **`multiprocessing.Process`** (fork/spawn API) | In-Python IPC | **Pickling** large `Config`, **spawn** cost and **macOS/Windows** quirks; worse **argv story** than “just run CLI” | **Subprocess CLI** preferred for parity |
| **External only** (cron, systemd timer, file watcher) | Zero `serve` CPU | **Disjoint** from “Run job” in the UI; harder to attach **`job_id`**, cancel, and Dashboard rows to one product | **Out of scope** for interactive Phase 2 |

**Implementation notes tied to this choice:**

- **Non-blocking spawn:** use **`create_subprocess_exec`** (not `subprocess.run()` inside `async def` without offload) so the worker does not stall the event loop while the child runs for hours.
- **Cancel (v1 shipped):** parent sends **SIGTERM** to the child PID only (no timed grace → **SIGKILL** loop yet). **Windows** differs (`TerminateProcess` / taskkill); v1 is **POSIX-first**. Future revision may add grace + SIGKILL after documented `PODCAST_*` tuning.
- **Durability caveat:** until registry + replay exist, **server restart** may orphan in-flight children or lose `queued` rows — v1 documents **local dev** expectations; production hardening may move toward **option C** later.

### Validation — why subprocess + local registry is sound (and where it stops)

This subsection records **design validation** (including informal cross-check against common FastAPI/Starlette practice). It is **not** a formal benchmark or security audit.

**What we believe is correct for v1 (`podcast serve`, local operator, single machine):**

1. **Subprocess is the right default for long, CPU-heavy, failure-prone work** sitting beside an ASGI app: the pipeline can **OOM**, hit native extension faults, or run for **hours** without taking down HTTP routes the operator still needs (health, search, artifacts).
2. **CLI-shaped argv is the right integration surface** for this repo: support and debugging already assume `python -m podcast_scraper.cli …`; reproducing a “bad job” does not require a different entrypoint than manual runs.
3. **Middle ground is intentional:** community guidance typically buckets work into (a) **in-process / BackgroundTasks** for *light* post-response tasks, (b) **separate process** for isolation, (c) **broker-backed workers** (Celery, RQ, etc.) when **persistence, retries, and horizontal workers** matter. For **local** `serve`, **(b) + a durable-enough registry** matches product scope without **(c)**’s operational cost.
4. **ADR-064 alignment:** we keep **one** long-lived server module and **feature-flag** job routes — we do **not** require a second always-on service for the first shipping slice.

**Known limits (honest scope — not hidden “gotchas”):**

1. **Durability across `serve` restart** is weaker than a broker: in-flight or `queued` jobs may be **orphaned** or **lost** until replay / reconciliation semantics exist; v1 targets **local dev** clarity, not cloud SLA.
2. **`uvicorn --workers > 1`** requires **cross-process locking** (or documented **single-worker** constraint for job submission v1).
3. **Horizontal scale-out** (many API replicas sharing one job queue) is **out of scope** for this design; escalation path is the **external queue** row in the alternatives table above.
4. **Windows** cancel and signal semantics differ from POSIX; implementation must document behavior; v1 may be **POSIX-first** with best-effort Windows.

**When we would revisit the decision:** sustained need for **cross-restart durability**, **multi-tenant** job isolation, or **multiple concurrent heavy jobs per corpus** under many uvicorn workers → prefer **broker + worker** (or a small embedded queue with a second process) and **supersede** this section with a new ADR (see below).

### ADR: do we need a separate, new ADR?

**Default: no — not required to ship.** [RFC-077](RFC-077-viewer-feeds-and-serve-pipeline-jobs.md) plus existing [ADR-064](../adr/ADR-064-canonical-server-layer-with-feature-flagged-routes.md) (“one server, feature-flagged route groups”) are sufficient **design records** for the first implementation: the **process model** and **alternatives** live in this RFC’s **§ Phase 2 — Architecture decision** and **§ Validation**.

**Add a new ADR (recommended later if any trigger applies):**

| Trigger | Why extract an ADR |
| ------- | ----------------- |
| **Cross-reference load** | Another RFC or subsystem needs a **one-line** “Accepted: pipeline jobs run as subprocess of `serve`” without importing all of RFC-077. |
| **Immutability ritual** | Stakeholders want an **Accepted** ADR row in [`docs/adr/index.md`](../adr/index.md) as the canonical “this is how jobs run” pin. |
| **Supersession** | We adopt **Celery/RQ** (or similar) later — a **new ADR** should **supersede** the subprocess-first decision and point at the broker RFC. |

**If we add an ADR:** use the next free ADR number from `docs/adr/index.md`, keep this RFC as the **detailed** spec (API, registry, hygiene), and make the ADR a **short decision + consequences** with links back here.

### Phase 2 — Job registry and hygiene (“end of day clear”)

**Child argv (reference):** subprocess includes **`--output-dir`** = resolved corpus anchor, **`--config`** = `str(app.state.operator_config_path)` (operator file may set **`profile:`** — merged with packaged preset before `Config` validation), and **`--feeds-spec`** = absolute path to **`feeds.spec.yaml`** under anchor when present (plus any agreed pipeline flags from RFC / CLI). Feed entries for the run come from that spec file, not from operator YAML root keys.

**Registry storage:** under corpus root **`.viewer/jobs.jsonl`** (or SQLite if concurrency demands — start JSONL + file lock).

**Job record fields (minimum):** `job_id`, `created_at`, `started_at`, `ended_at`, `status` (`queued` \| `running` \| `succeeded` \| `failed` \| `cancelled` \| `stale`), `pid` (nullable), `argv_summary`, `exit_code`, `log_relpath`, `last_progress_at` (optional heartbeat from worker thread writing sidecar).

**Stale detection:**

- **Wall-clock timeout** T (configurable, default e.g. 24h or product-set): if `running` and `now - started_at > T` → auto-mark **`stale`** or require **`POST /api/jobs/reconcile`** to promote — pick **auto-mark stale + visible banner** in Dashboard.
- **PID liveness:** periodic lightweight check (optional background task in server) `os.kill(pid, 0)`; if dead but status still `running` → **`orphan_reconciled` → `failed`** with reason.

**Cancel:** `POST /api/jobs/{id}/cancel` sends **SIGTERM** (v1); registry records **`cancel_requested`**; if the child is still alive the row may remain **`running`** with **`cancel_requested: true`** until the process exits and the waiter updates status — clients should treat that as “cancel pending,” not necessarily terminal **`cancelled`** yet. Idempotent if already terminal.

**Reconcile:** `POST /api/jobs/reconcile` (operator-triggered) scans registry + PIDs, returns summary `{ "updated": n, "details": [...] }` for UI toast and Pipeline refresh.

**Multi-job policy:** default **one active `running` job per corpus** (file lock); queue additional `POST`s as `queued` or return **409** — RFC recommends **queue** with visible position in Pipeline table.

**Multi-worker uvicorn:** file lock on registry append + job spawn; document **single worker** for v1 if lock complexity deferred.

### Distinction from RFC-065

| Aspect | RFC-065 | RFC-077 Phase 2 |
| ------ | ------- | ---------------- |
| Surface | CLI + terminal | HTTP + Vue + Dashboard |
| State | `.pipeline_status.json` | Job registry + API |

Optional bridge: child pipeline may still emit `.pipeline_status.json`; server **may** attach last lines to job record — not required v1.

## Documentation deliverables

**Phase 1a:** SERVER_GUIDE (feeds), VIEWER_IA, UXS-001 (Feeds).

**Phase 1b:** SERVER_GUIDE (operator-config path rules, forbidden keys), UXS-001 (tabs / Config editor).

**Phase 2:** UXS-006 (Pipeline: status, stale, cancel, reconcile), UXS-001 (optional job chip), SERVER_GUIDE (job API).

## Testing strategy

- Integration: feeds + operator-config PUT rejection on forbidden secret and forbidden feed keys; GET legacy file with secret → error path; GET includes `available_profiles`; feeds **`GET`/`PUT`** round-trip **`feeds`** JSON vs **`feeds.spec.yaml`** on disk; structured spec parser rejects unknown top-level keys and unknown per-feed keys.
- Job hygiene: unit/integration for stale transition + cancel mock subprocess.
- Playwright: mocked APIs for dialogs.

## Risks & mitigations

| Risk | Mitigation |
| ---- | ---------- |
| Secrets leak via GET | Forbidden-key policy; optional redact read |
| Config path confusion | Single `operator_config_path` on `app.state` at startup; SERVER_GUIDE path rules |
| Orphan subprocess | PID checks + cancel + reconcile |

## Revision history

| Date | Change |
| ---- | ------ |
| 2026-04-19 | Initial Draft |
| 2026-04-19 | Operator config (path precedence + no secrets in file); job registry + stale/cancel/reconcile; health flags |
| 2026-04-19 | ADR-style section: **child subprocess** for pipeline vs in-process / BackgroundTasks / broker; alternatives table + external validation note |
| 2026-04-19 | **§ Validation** expanded (soundness + limits); **§ ADR** — default RFC-only, when to add new ADR |
| 2026-04-19 | Align with shipped code: `app.state.*_enabled`, `operator_config_path`, GET JSON shape, Phase 2 route list + reconcile, cancel v1 (SIGTERM only, cancel_pending semantics) |
| 2026-04-20 | Profiles (#593): `available_profiles` on GET; PUT rejects root feed keys; viewer profile picker + overrides; § Profile composition + feeds vs operator YAML |
| 2026-04-21 | `available_profiles` union cwd + repo `config/profiles/`; mixed PUT errors documented; viewer profile select sole source of truth on save |
| 2026-04-20 | **Feeds spec:** canonical **`feeds.spec.yaml`**, **`{ feeds: [...] }`** schema, **`--feeds-spec`**, **`GET`/`PUT`** JSON shape (**`feeds`** not **`urls`**), per-feed override allowlist + merge order; jobs argv **`--config`** + **`--feeds-spec`** (replaces **`--rss-file`** / **`--config-file`** for child CLI) |
| 2026-04-22 | **§ Compose / #660:** native subprocess jobs unchanged; Docker stack (#660) additive; **`pipeline_install_extras`** Docker-path-only — cross-ref [RFC-079 §Native vs Docker](RFC-079-full-stack-docker-compose.md#native-vs-docker) |
