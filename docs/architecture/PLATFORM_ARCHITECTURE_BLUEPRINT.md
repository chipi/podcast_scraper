# Platform Vision & Architecture Blueprint

**Status:** Architecture blueprint — the **target-state vision** for platformizing
podcast_scraper. Describes where the system is going, not where it is today (for current
state, see [Architecture](ARCHITECTURE.md)). Concrete RFCs will be broken out from
individual sections as implementation begins.

**Audience:** PRD/RFC authors and implementers planning service mode, tenancy, deployment,
hardware sizing, ML workload distribution, observability, and downstream digest features.

**Relationship to other architecture docs:**

| Document | Purpose |
| --- | --- |
| [Architecture](ARCHITECTURE.md) | **Current state** — how the system works today |
| [Non-Functional Requirements](NON_FUNCTIONAL_REQUIREMENTS.md) | **Current constraints** — performance, reliability, observability targets |
| [Testing Strategy](TESTING_STRATEGY.md) | **Current testing** — test pyramid, patterns, CI integration |
| **This document** | **Target state** — where we are going, what to build next |

**Last updated:** 2026-04-03.

**How to use this doc**

- **Part A** — Catalog, subscriptions, reuse, Postgres tenancy, CLI vs platform constraints,
  **auth evolution** (A.12).
- **Part B** — Compose, API vs worker, Redis, named queues (two tiers: simple + distributed),
  **queue library decision** (B.15: arq), **DB migration strategy** (B.16: Alembic),
  **file locking** (B.14), CLI entry points (B.9.1), image split strategy (B.9.2),
  **distributed tier testing strategy** (B.17).
- **Part C** — "Many podcasts, one brain": weekly digest gaps, sequencing, **pipeline
  integration** (C.6), **resource requirements** (C.7), **viewer integration** (C.8),
  **graduation alignment** (C.9).
- **Part D** — Hardware sizing, resource profiles, container topology for true ML parallelism,
  queue-per-concern, Apple Silicon / non-GPU path, eval workloads, **concrete hardware
  configurations with pricing** (Linux PC builds, Mac Mini lineup, cloud/rental options,
  **total cost of ownership** in break-even analysis).
- **Part E** — Observability, monitoring, logging, dashboards, control plane, alerting,
  **AI agent with human-mirrored deploy workflow** (E.9.5).
- **Part F** — Deployment lifecycle: CI/CD → release → deploy → restart → rollback. Compose vs
  K8s comparison. Configuration management, secrets, infrastructure-as-code.
- **Cross-cutting:** Unified graduation path in D.7 syncs topology (D), observability (E.10),
  deployment (F.10), and auth (A.12) stages.

---

## Part A. Multi-tenant platform (catalog, subscriptions, reuse)

### A.1 Goal (target picture)

- A **small server** holds **configuration**; the primary object is a **list of podcasts
  (feeds)** the system should process — not one static YAML per machine.

- The system **continuously** pulls and processes subscribed content.
- A **small UI** lets users **manage** which shows they care about.
- **Many tenants:** processing is **central and deduplicated**; users only **see** slices of
  canonical data (summaries, GI, KG projections) they are **entitled** to.

**Design intent:** Shape the **data model and boundaries** for multi-tenancy **from the start**,
even when **v1** runs a **single** tenant — avoid a painful "add `tenant_id` everywhere"
migration later.

### A.2 Product constraints (CLI + optional platform)

These are **non-negotiable directions** for how the platform relates to the existing tool:

1. **CLI stays first-class** — Easy runs from **config** (YAML/JSON): scripts, CI, ad-hoc use,
   outputs on disk **without** running a server. Most users can stay on this path forever.

2. **Service / platform mode is optional** — Long-lived process for **continuous** pull,
   multi-feed orchestration, Postgres projection, optional **UI/API**. Headless single-user

   platform and multi-tenant UI are **graduations** of the same service layer, not a separate
   product fork.

3. **One pipeline core, two execution modes** — The core pipeline logic lives in
   `run_pipeline` and must not be duplicated. It supports two execution modes:

   - **Atomic mode (simple tier):** `run_pipeline(cfg)` — one `Config` in, all artifacts
     out. Platform wraps it: schedule, dedup, materialize `Config` per job, cursors,
     projection. This is the **v1 default** and the only mode CLI uses.
   - **Stage mode (distributed tier):** The same pipeline stages are invocable individually
     via `run_pipeline(cfg, start_stage=..., end_stage=...)` or equivalent per-stage
     functions. Platform decomposes a full run into stage-based jobs with durable state.
     This is the **v2 mode** when metrics show contention (see D.7).

   Both modes call the same underlying stage functions — stage mode is a decomposition of
   atomic mode, not a parallel implementation. The `Config` model gains optional
   `resume_from_stage` / `run_only_stage` fields; existing CLI and `service.run()` ignore
   them (defaulting to full pipeline).

4. **Avoid diverging config dialects** — Prefer **one `Config` model** and shared validation;
   the platform adds **catalog, subscriptions, tenants** in the DB and **builds** a `Config`

   (or equivalent dict) for each worker invocation instead of maintaining a parallel
   "platform-only" config schema that drifts from the CLI.

**Summary:** Library + **CLI** (simple path) + optional **service** (daemon / API / UI) on the
same engine — not "CLI tool vs platform" as two implementations.

### A.3 Separate three concepts

| Concept | Meaning |
| --- | --- |
| **Catalog** | Global **directory of feeds** (and normalized show metadata) the **platform** knows. "What exists / what we can process," not "Alice's list" alone. |
| **Subscription** | **Tenant** ↔ **catalog entity** (e.g. feed). "I want this show in my library." |
| **Entitlement / visibility** | What the tenant may **read**: episodes, GI, KG, summaries. Often: subscription + optional admin grants or feature flags. |

**Ingestion** should be driven by **catalog + policy** (what is globally enabled, what has
subscribers), not by "run this user's private config file" as the only source of truth.

### A.4 Process once, serve many (reuse)

Expensive steps (download, transcribe, summarize, GI, KG) run **once** per logical episode +
**pipeline fingerprint**, not once per user.

- **Episode identity:** Canonical feed id + episode `guid` (ADR-007).
- **Pipeline fingerprint:** Versions / hashes of models, providers, prompts, schemas that
  affect outputs (summaries, `gi.json`, KG artifact).

**Rule:** At most **one canonical artifact set** per `(episode_key, pipeline_fingerprint)`.
If tenant B subscribes to the same feed+episode, they **attach** to existing rows — no second
Whisper run for the same fingerprint.

**Workers** consume a **deduped queue** of "work needed," not independent full pipeline runs
per user.

### A.5 Storage layout (conceptual)

- **Canonical blob/object paths** for transcripts, metadata, GI, KG — **not**
  `users/<tenant>/...` as the primary store for pipeline outputs.

- **Postgres** holds: catalog, subscriptions, cursors, job state, **pointers** to blobs, and
  **projected** tables (PRD-018 / RFC-051) keyed by **global episode and artifact ids**.

**Tenant-specific** overlays (preferences, notes, stars) — **separate** tables with
`tenant_id`, not duplicate transcripts.

### A.6 Multi-tenancy model (Postgres)

**Default:** **Single database**, **shared tables**, **`tenant_id`** on tenant-scoped rows, plus
**RLS** where appropriate.

- **v1 single user:** One row in `tenants` (e.g. `default`); always set `tenant_id` in API and
  queries.

- **Catalog** (`feeds`, `episodes`, artifact registry): often **global**.
- **Subscriptions:** `tenant_id` + `feed_id`.
- **Reads:** "My library" = join subscriptions → episodes → canonical projections; enforce RLS
  or API-layer checks.

Avoid **schema-per-tenant** early unless compliance requires it.

### A.7 Service shape (logical components)

1. **API + UI** — Admin: catalog. User: subscriptions, entitled reads. **RFC-062** defines the
   initial server layer (`src/podcast_scraper/server/`) and GI/KG Viewer as the first UI
   consumer; platform CRUD routes activate via feature flag (`enable_platform`).

2. **Scheduler / worker** — Feeds with subscribers (or globally enabled); enqueue work;
  **cursors**.

3. **Projection / indexer** — Files or object storage → Postgres (RFC-051); optional search
   later. **Semantic search** (RFC-061) adds FAISS vector indexing and embedding-based retrieval
   as a parallel path alongside SQL projection.

**Pipeline core:** `run_pipeline`-class logic = **one unit of work** (config in, artifacts
out). Platform adds **queue, dedup, paths, post-run projection**.

### A.8 GI / KG / summaries under reuse

- **Store once** per `(episode, pipeline_fingerprint)` in canonical storage + projections.
- **Tenant view:** Filter by subscription / entitlements.
- **Overlays:** Tenant-scoped tables for bookmarks, labels.
- **Semantic index:** Vector embeddings (RFC-061) are per-corpus, not per-tenant; tenants query
  the shared index filtered by entitlements.

### A.9 Phased delivery (platform)

| Phase | Deliver | Concrete work |
| --- | --- | --- |
| **A** | `tenant_id`, `tenants`, subscriptions, catalog; single tenant row. | Schema + migrations |
| **B** | Long-lived worker + cursors; feeds from catalog ∩ subscriptions. | Worker process, queue, Redis/PG jobs |
| **C** | Projection to Postgres (RFC-051); API reads DB + blob pointers. | SQL projection, GIL/KG tables (ADR-054) |
| **D** | Server + UI (RFC-062). `podcast serve` CLI. GI/KG Viewer as first UI. Semantic search panel. | FastAPI `server/` module (ADR-064), Vue 3 SPA (ADR-065), Playwright E2E (ADR-066) |
| **E** | Second tenant + RLS; quotas / billing later. | RLS policies, API auth |

**Phase D is now concrete** — RFC-062 defines the server module, route groups, frontend stack,
and `podcast serve` CLI entry point. The viewer is the first consumer; platform CRUD routes
(feeds, episodes, jobs) extend the same server via feature flags. See
[RFC-062](../rfc/RFC-062-gi-kg-viewer-v2.md) and [UXS-001](../uxs/UXS-001-gi-kg-viewer.md).

Defer: billing, orgs, per-tenant pipeline overrides, legal review — but **name** shared-corpus /
ToS risks in a threat model.

### A.10 Risks / watch (platform)

- **Fingerprinting** wrong → wrong shared artifacts.
- **Ops surface:** Postgres, workers, object storage, auth, UI — scope consciously.
- **Legal:** Multi-user shared corpus → takedown, copyright, PII in logs.

### A.11 Relation to current repo

- **Today:** One `Config`, one `rss`, `run_pipeline` → filesystem; `service.run` one-shot.
  **CLI** = default mental model. GI/KG viewer v1 (`web/gi-kg-viz/`) is a vanilla-JS prototype.
  Semantic search (RFC-061 Phase 1) adds `podcast search`, `podcast index`, vector store.

- **Platform:** Catalog + subscriptions **when** operator chooses platform mode; workers replace
  "cron per feed"; Postgres = state + projections; reuse via **episode key + fingerprint**.
  RFC-062 server module is the **seed** — viewer routes land first, platform routes extend
  additively.

### A.12 Authentication and authorization evolution

Auth is not a single feature — it's a capability that grows with the deployment model. The
system starts as a single-user tool and evolves into a multi-tenant platform. Auth should
not be over-engineered early, but the architecture must not make it hard to add later.

| Stage | When | Auth model | What changes |
| --- | --- | --- | --- |
| **1. Single user** | CLI + local server (RFC-062 v2.6) | None. API is localhost-only, served behind no reverse proxy or behind Caddy with no auth. | Nothing. `podcast serve` binds to `127.0.0.1`. |
| **2. Single user + remote** | Server exposed to network (home server, VPS) | **API key** in `Authorization: Bearer <key>` header. Key stored in `.env`, checked by FastAPI middleware. Caddy terminates TLS. | Add `AUTH_API_KEY` env var. FastAPI middleware rejects requests without valid key. AI agent (E.9) uses the same key. |
| **3. Multi-user (SaaS)** | Platform mode (`--platform`), multiple users | **JWT + OAuth 2.0** (e.g., Auth0, Keycloak, or self-hosted). Tenant isolation in Postgres (`tenant_id` column). RBAC: `admin`, `editor`, `viewer`. | Add auth dependency in FastAPI (`Depends(get_current_user)`). Tenant-scoped queries. Platform routes require auth; viewer routes can be public or gated. |

**Key constraints:**

- Stage 1 → 2 is a config change (add env var, restart). No code architecture change.
- Stage 2 → 3 requires a proper auth provider and DB schema changes (add `tenant_id`).
  This is a planned RFC when multi-tenant demand is clear.
- The admin API routes (E.8, control plane) require at minimum stage 2 auth.
- The AI agent (E.9) authenticates with an API key in all stages; in stage 3 it gets a
  service account with `admin` role.

---

## Part B. Deployment, API, workers, queues, Docker

**Depends on:** Part A (especially **A.2** product constraints and **A.7** service shape).

### B.1 Control plane vs data plane

| Layer | Responsibility | Typical runtime |
| --- | --- | --- |
| **API** | Auth, tenant context, **config** (catalog, subscriptions), **read** APIs for consumption (summaries, GI, KG, **semantic search**) from **Postgres** + blob pointers + **vector index** — **not** Whisper per request. | Gunicorn + **Uvicorn** workers (ASGI) for FastAPI/Starlette. **RFC-062** defines the initial server. |
| **Worker(s)** | Pull from queue (or DB), **build `Config`**, **`run_pipeline`**, canonical files, **projection** to Postgres. See **Part D** for resource-profile-aware worker pools. | Long-lived process(es). |

**DB** = Postgres (metadata, projections, optional jobs). **Files** (or object store) first, then
**project** (RFC-051).

### B.2 End-to-end data flow (target)

```text
User/UI → API (RFC-062 server) → Postgres (catalog, subscriptions, entitlements)
                    ↓
              enqueue job (Redis or Postgres `jobs` table)
                    ↓
Worker(s) → run_pipeline(cfg) → canonical files (+ optional object store)
                    ↓                       ↓
              projection → Postgres    vector index → FAISS/Qdrant (RFC-061)
                    ↓                       ↓
User/UI → API → read Postgres       → semantic search API
              (+ signed URLs to blobs if needed)
```

**Option:** Projection **inline** after `run_pipeline` or **separate** job on `projection`
queue (see B.7–B.8).

### B.3 Docker Compose vs Kubernetes vs PaaS

| Approach | When |
| --- | --- |
| **Compose** | Single host / small VPS; **SaaS v1** — **recommended start**. |
| **K8s** | Multi-node, heavy autoscaling — **defer**. |
| **PaaS** (Fly, Railway, Render, Cloud Run, Fargate) | Containers without self-managed K8s. |

### B.4 Compose service catalog

**Minimal:** `postgres`, `api` (your image), `worker` (same image, different `command`), `caddy`
(or Traefik). Volumes: Postgres data; **artifact root** shared for workers/projection.

**Add:** `redis` (arq — see B.15), `minio` (S3-compatible blobs), `ui` (static SPA or behind
Caddy).

**Expanded (Part D):** When ML workload distribution matters, split `worker` into
`worker-gpu`, `worker-ml`, `worker-io` — same image, different queue subscriptions. See
**D.5** for the full container topology.

**Avoid early:** DIY K8s; ClickHouse/OpenSearch until needed; **one container per podcast**.

### B.5 Redis and job queue

**Role:** Broker for job library — API enqueues, returns fast; workers compete; retries /
visibility / optional dedup.

**Not:** System of record for feeds/users — **Postgres** is.

**Alternatives:** Postgres `jobs` + `SKIP LOCKED` (fewer moving parts, more DIY); in-process
queue (single worker only).

**Conclusion:** **Redis** when using standard queue libs and multiple workers; **Postgres
queue** OK for minimal early.

### B.6 Worker pools: one type vs many

| Model | Notes |
| --- | --- |
| **One worker service, many queues** | Same image; `ingest`, `heavy`, `projection` in one binary — **start here**. |
| **Multiple Compose services, same image** | Different `command` / queue subscription — when **contention** measured. **Part D** details when to graduate. |
| **Multiple images** | CUDA vs slim — when **dependencies** diverge. |

**Specialization does not fix:** shared code bugs, Postgres/Redis down, same bad deploy to all
pools.

**It helps:** GPU vs CPU isolation; **backlog** isolation across queues.

### B.7 Named queues vs pipeline *steps* — two tiers

This section defines two deployment tiers that coexist. The simple tier runs from day one; the
distributed tier activates when metrics justify it. Both tiers share the same pipeline code
(A.2 constraint #3).

#### Simple tier (v1 default)

Queues separate **job categories**, not individual `run_pipeline` stages. One job runs the full
pipeline **download → Whisper → summarize → … → index** atomically via `run_pipeline(cfg)`.
The platform layer (or CLI) enqueues one job per episode; the worker picks it up and runs
everything end-to-end.

| Queue | Payload | Worker | Concurrency |
| --- | --- | --- | --- |
| **`ingest`** | Poll RSS, enqueue `heavy` | `worker-io` | Higher (I/O) |
| **`heavy`** | Full `run_pipeline(cfg)` incl. local Whisper | `worker-gpu` (or `worker-all`) | **1** per GPU |
| **`projection`** | Files → Postgres only | `worker-io` | Medium (CPU/DB) |

This is sufficient for single-user and small-corpus deployments. The `worker-gpu` (or on
modest hardware, a single `worker-all`) runs the entire pipeline per episode. No stage state
machine, no inter-stage message passing.

#### Distributed tier (v2, when metrics trigger — see D.7)

Split the pipeline into **stage-based jobs** with durable state when the simple tier shows
contention: GPU worker idle waiting for I/O, or CPU-bound summarization blocks the next
transcription job. Each stage targets the worker pool with the right resource profile.

| Queue | Payload | Worker pool | Concurrency |
| --- | --- | --- | --- |
| **`ingest`** | Poll RSS, download audio/transcripts | `worker-io` | High (I/O-bound) |
| **`transcribe`** | Whisper / diarization (RFC-058) | `worker-gpu` | 1 per GPU |
| **`enrich`** | Summarize, GIL, KG, NLI, adaptive routing (RFC-053) | `worker-ml` | 2–4 (CPU/RAM) |
| **`index`** | Embed + FAISS upsert (RFC-061), vector index maintenance | `worker-ml` | Medium |
| **`projection`** | Files → Postgres (RFC-051) | `worker-io` | Medium |

Episode state machine: `pending → downloading → transcribing → enriching → indexing →
projecting → done` (or `failed` at any stage with retry metadata).

`run_pipeline(cfg, start_stage="transcribe", end_stage="transcribe")` executes a single stage.
The scheduler (see B.7.1 below) decomposes full-pipeline jobs into stage jobs and manages the
state machine.

#### B.7.1 Scheduler — role and design

The **scheduler** is a long-running process (separate container in D.5) that:

1. **Decomposes jobs** — When a new episode arrives on `ingest`, the scheduler creates the
   initial state record and enqueues the first stage. In simple tier, this is just
   `heavy` → `run_pipeline(cfg)`. In distributed tier, it creates a stage chain.
2. **Advances stages** — When a stage completes, the completing worker posts a completion
   event. The scheduler reads it and enqueues the next stage.
3. **Manages retries** — Failed stages are retried with exponential backoff (max 3 per stage).
   After max retries, the episode enters `failed` state and an alert fires.
4. **Reports status** — Exposes `/api/jobs` and `/api/status` endpoints (or publishes to
   Redis) for the API server to query.
5. **Enforces concurrency limits** — respects per-queue concurrency caps.

**Implementation:** In simple tier the scheduler is lightweight — it can run inside the API
server process as a background task (or omitted entirely if the queue library handles
callbacks). In distributed tier it runs as a dedicated container.

**CLI entry points (see also B.9.1):**

```text
podcast serve                  → API server + viewer (RFC-062)
podcast worker --queue heavy   → Worker consuming a specific queue
podcast scheduler              → Scheduler process (distributed tier only)
```

### B.8 Local Whisper as bottleneck

- **Simple tier:** Full episode on `heavy` queue, concurrency 1 (or 2 if VRAM allows) per GPU.
  Other queues handle RSS/projection so they don't sit behind a Whisper pile.
- **Distributed tier:** `transcribe` queue is GPU-exclusive; other stages flow freely.
- When diarization (RFC-058, pyannote) is enabled, transcription time roughly doubles —
  a strong trigger to move from simple to distributed tier.

### B.9 API + worker separation — timing

**Recommend early:** separate Compose services, **same image** (see B.9.2 on image strategy),
different `command` — security boundary, independent scale, safer deploy cadence.

The **API server** is defined by RFC-062 (`src/podcast_scraper/server/`) and started via
`podcast serve`. Workers are a separate process using the same Python package.

#### B.9.1 CLI entry points for all processes

All processes use the same Python package and Docker image. The `command` field in Compose
determines which mode runs.

| CLI command | Process | Docker `command` |
| --- | --- | --- |
| `podcast serve --output-dir ./output` | API server + viewer (RFC-062) | `podcast serve --output-dir /data/output` |
| `podcast serve --output-dir ./output --platform` | API + platform routes (v2.7) | `podcast serve --output-dir /data/output --platform` |
| `podcast worker --queue heavy` | Worker (simple tier: full pipeline) | `podcast worker --queue heavy` |
| `podcast worker --queue transcribe` | Worker (distributed tier: one stage) | `podcast worker --queue transcribe` |
| `podcast scheduler` | Scheduler (distributed tier) | `podcast scheduler` |

The `podcast worker` command is a new CLI subcommand that starts a queue consumer. It wraps
the chosen queue library (see B.13) and calls `run_pipeline(cfg)` (simple tier) or
`run_pipeline(cfg, start_stage=..., end_stage=...)` (distributed tier) for each job.

#### B.9.2 Docker image strategy

**Single image (default):** All services share one Docker image built from the same
`Dockerfile`. This keeps the build simple and ensures every container has the same code
version. The image includes all dependencies (ML models, GPU libs, Python packages).

**When to split images:** Split into `podcast-scraper-gpu` and `podcast-scraper-base` when
the GPU dependencies (CUDA runtime, cuDNN, PyTorch with CUDA) make the image > 8 GB and
`worker-io` containers are paying for 4+ GB of unused GPU libraries. Expected timing:
when adding pyannote (RFC-058) or when CUDA runtime pushes the image past the threshold.

| Image | Contains | Used by |
| --- | --- | --- |
| `podcast-scraper-gpu` | Full: CUDA + PyTorch + all ML | `worker-gpu`, `worker-ml` |
| `podcast-scraper-base` | Python + core + no CUDA | `api`, `scheduler`, `worker-io` |

Both images share a common base layer (`python:3.12-slim` + app code) to maximize Docker
layer caching.

### B.10 Postgres (deployment recap)

Target **vanilla PostgreSQL** for SaaS; extensions as needed. **Canonical files + projection**
(RFC-051). Avoid **SQLite in prod** if you want one dialect (see separate DB discussions).

Projected tables (ADR-054): `insights`, `quotes`, `insight_support`, `kg_nodes`, `kg_edges` —
keyed by episode identity (ADR-007) + pipeline fingerprint.

### B.11 Summary table (deployment)

| Topic | **Simple tier (v1)** | **Distributed tier (v2, Part D)** |
| --- | --- | --- |
| Orchestration | **Compose** (4–5 containers) | **Compose** (8+ containers) |
| Queue backend | **Redis** with **arq** (B.15) | **Redis** with **arq** — dedicated queues per profile |
| Queue library | **arq** — asyncio, typed, minimal | Same (**arq**) |
| Workers | **One `worker-all`**, 3 queues | **3 worker services** (gpu, ml, io) — see B.9.2 for image split |
| Job model | **One job = full `run_pipeline(cfg)`** | **Stage-based jobs** with durable state machine |
| Whisper | **`heavy` queue + concurrency 1** | **`transcribe` queue on `worker-gpu`** |
| Pipeline vs queue | **Job boundary only** | **Stage boundaries** match resource profiles |
| Scheduler | arq cron / omitted | Dedicated `scheduler` container (B.7.1) |
| DB migrations | **Alembic** (B.16) | Same (**Alembic**) |
| Entry points | `podcast serve`, `podcast worker` | + `podcast scheduler` (B.9.1) |

### B.12 Numbered recommendations (RFC authors)

1. **Two-tier model**: `run_pipeline(cfg)` = atomic worker unit (simple tier, v1 default);
   stage-based decomposition = distributed tier (v2, see B.7 / D.7).
2. Queues: **`ingest`** + **`heavy`** + **`projection`** in simple tier; split to 5 queues
   in distributed tier when metrics trigger it.
3. **`heavy` concurrency** ≈ **1** per GPU for local Whisper.
4. No stage splits until metrics show contention (GPU idle time, queue depth spikes).
5. Compose: **`postgres` + `api` + `worker` + `caddy`** + **`redis`**; add `scheduler`,
   `worker-gpu`, `worker-ml`, `worker-io` in distributed tier.
6. RFC: job payload schema, idempotency (`episode_key` + `pipeline_fingerprint`), dead-letter.
7. **CI:** integration tests on **Postgres** + **Redis** for platform paths.
8. **Semantic search** (RFC-061): vector index on shared volume; `api` reads for search;
   `worker-ml` writes after embedding. `index` queue in distributed tier.
9. **Server module** (RFC-062): `api` container runs `podcast serve`; viewer routes always
   on; platform routes behind `--platform` flag.
10. **Queue library**: arq (see B.15). **Database migrations**: Alembic (see B.16).

### B.13 RFC checklist (deployment / orchestration)

- Job model JSON schema; dedup; queue names; worker ↔ queue mapping; concurrency per queue; GPU
  notes; retries, DLQ, visibility; projection inline vs async; Compose reference; secrets;
  observability (correlation id API → worker); CLI entry points (`podcast serve`, `podcast
  worker`, `podcast scheduler`); image split criteria (B.9.2); file locking (B.14); migration
  strategy (B.16); auth evolution (A.12); distributed tier testing (B.17).

### B.14 Shared volume concurrency and file locking

Workers read and write shared artifact directories and the FAISS index concurrently. Without
coordination, race conditions can corrupt data (e.g., two workers writing the same episode's
artifacts, or one worker reading a FAISS index while another writes to it).

**Strategies by data type:**

| Data | Risk | Mitigation |
| --- | --- | --- |
| **Episode artifacts** (JSON files) | Two workers processing same episode | Queue dedup: `episode_key` as job unique ID prevents double-enqueue. If a duplicate slips through, the later worker finds existing output and skips (idempotent writes). |
| **FAISS index** (single `.faiss` file) | Concurrent reads + writes | **Write-aside + atomic swap:** Writer builds index in a temp file (`index.faiss.tmp`), then `os.rename()` over the live file (atomic on Linux/macOS). Readers see either old or new — never a partial write. |
| **Model cache** (`~/.cache/huggingface/`) | Multiple workers downloading same model | Mount a shared read-only volume with pre-populated models. Use `HF_HOME` env var. Model downloads are idempotent; concurrent downloads to the same path are safe (Hugging Face uses temp files + rename). |
| **Postgres** | Standard DB concurrency | Normal transaction isolation. No file-level concern. |
| **Redis** | Standard message queue | Atomic operations built into Redis. No file-level concern. |

**FAISS index update pattern:**

```python
import os, tempfile, shutil

def update_faiss_index(index, new_vectors, index_path):
    index.add(new_vectors)
    # Write to temp file in same directory (same filesystem for atomic rename)
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(index_path), suffix=".tmp")
    os.close(fd)
    faiss.write_index(index, tmp_path)
    os.rename(tmp_path, index_path)  # atomic on POSIX
```

### B.15 Queue library decision: arq

**Decision:** Use **arq** as the Redis-backed job queue library.

| Library | Async | Overhead | Features | Fit |
| --- | --- | --- | --- | --- |
| **RQ** | No (sync) | Low | Simple, mature, dashboard (rq-dashboard) | Good for simple tier but sync-only limits API integration |
| **Celery** | Optional | High | Full-featured: priority, rate limiting, canvas, monitoring (Flower) | Over-engineered for our scale; complex config; heavy dependency tree |
| **arq** | Yes (native asyncio) | Low | Typed jobs, cron, retries, health checks, job results, Redis-native | Best fit: async aligns with FastAPI; minimal config; typed |

**Why arq:**

1. **FastAPI alignment** — Both are asyncio-native. The API server can enqueue jobs without
   blocking the event loop. Workers are async too, allowing efficient I/O multiplexing.
2. **Typed jobs** — Job functions have typed signatures, matching our Pydantic-everywhere
   approach.
3. **Minimal footprint** — ~2k LOC, no C extensions, no external monitoring daemon. Monitoring
   via Prometheus metrics we already define (E.3).
4. **Redis-native** — Uses Redis streams/sorted sets directly. No broker abstraction layer.
   Inspectable via standard Redis CLI (useful for AI agent observability in E.9).
5. **Built-in retry** — Per-job retry with configurable backoff. Dead-letter via `max_tries`.

**Worker startup:**

```python
# podcast worker --queue heavy
from arq import create_pool
from arq.connections import RedisSettings

class WorkerSettings:
    functions = [run_pipeline_job]
    redis_settings = RedisSettings(host="redis")
    queue_name = "heavy"
    max_jobs = 1  # GPU concurrency
```

**Scheduler integration:** arq has built-in cron scheduling for recurring jobs (RSS polling).
The separate `podcast scheduler` command is only needed in distributed tier for stage-chain
orchestration (B.7.1); simple tier uses arq's `cron_jobs` directly.

### B.16 Database migration strategy: Alembic

**Decision:** Use **Alembic** for database schema migrations from day one.

**Why from day one:** The Postgres schema (projected tables from RFC-051/ADR-054:
`insights`, `quotes`, `insight_support`, `kg_nodes`, `kg_edges`) will evolve as new pipeline
stages land. Without migrations, every schema change requires manual DDL or table drops.

**Setup:**

```text
src/podcast_scraper/
├── server/
│   └── ...
├── db/
│   ├── __init__.py
│   ├── models.py          # SQLAlchemy models (or raw SQL table defs)
│   └── migrations/
│       ├── env.py          # Alembic environment
│       ├── script.py.mako  # Migration template
│       └── versions/       # Auto-generated migration files
│           ├── 001_initial_schema.py
│           └── ...
└── ...
```

**Migration workflow:**

1. Developer modifies schema (models or raw SQL).
2. `alembic revision --autogenerate -m "add column"` creates migration file.
3. `alembic upgrade head` applies pending migrations.
4. In Docker: `podcast db upgrade` CLI command runs before the API server starts.

**Container startup order:**

```yaml
# docker-compose.prod.yml
api:
  command: >
    sh -c "podcast db upgrade && podcast serve --output-dir /data/output"
  depends_on:
    postgres:
      condition: service_healthy
```

**Rollback:** `alembic downgrade -1` for the last migration. Each migration must have a
working `downgrade()` function. Rollback strategy per scenario in F.7.

**AI agent and migrations:** The AI agent (E.9) does **not** create or run migrations
autonomously. Migration failures trigger a P1 alert with a runbook that pages a human.
The agent can report the current schema version and pending migrations.

### B.17 Testing strategy for the distributed tier

The existing test pyramid (unit ~80%, integration ~14%, E2E ~6%) covers `run_pipeline`,
CLI, and API paths well. The distributed tier introduces new components (arq workers,
scheduler, stage decomposition, FAISS atomic swap, multi-container interaction) that need
their own testing strategy — without duplicating what's already tested.

**Principle:** The distributed tier is a **deployment concern** wrapping existing pipeline
logic. Pipeline correctness is proven by existing tests. Distributed tests verify
**orchestration, messaging, concurrency, and failure handling** — not whether Whisper
produces correct transcripts.

#### B.17.1 What to test per layer

| Layer | What to test | How | Infra needed |
| --- | --- | --- | --- |
| **Unit** | Stage decomposition logic: `run_pipeline(cfg, start_stage=X, end_stage=X)` produces correct partial output. Config validation for `resume_from_stage` / `run_only_stage` fields. Job payload serialization/deserialization. | Mock pipeline internals; test that stage boundaries are correct. | None |
| **Unit** | Scheduler state machine: episode transitions (`pending → downloading → ... → done`), retry logic (exponential backoff, max retries → `failed`), stage-chain creation from a full-pipeline request. | In-memory state; no Redis. | None |
| **Unit** | FAISS atomic-swap write pattern: temp file → rename. File locking edge cases. | `tmp_path` fixture; no real FAISS index needed (mock `faiss.write_index`). | None |
| **Integration** | arq worker picks up a job from Redis, calls `run_pipeline(cfg)` (mocked), and reports completion. Verify job lifecycle: enqueue → dequeue → execute → result. | Real Redis (Docker or `fakeredis`), real arq, mocked pipeline. | Redis |
| **Integration** | Scheduler + arq: scheduler decomposes a job into stages, enqueues stage 1, worker completes it, scheduler advances to stage 2. Full stage chain for one episode. | Real Redis, real arq, mocked pipeline stages (return canned output per stage). | Redis |
| **Integration** | Dead-letter: job fails 3 times → moves to DLQ. Verify alert would fire. | Real Redis, real arq, job that always raises. | Redis |
| **Integration** | Queue concurrency: `heavy` queue with `max_jobs=1` — second job waits while first runs. | Real Redis, real arq, two jobs, timing assertions. | Redis |
| **Integration** | Alembic migrations: `upgrade` from empty → current; `downgrade -1` → `upgrade` roundtrip. Schema matches expected tables. | Real Postgres (Docker). | Postgres |
| **E2E** | Full distributed pipeline: `podcast worker --queue heavy` processes an episode end-to-end (real `run_pipeline` with test fixtures). Verify artifacts appear on shared volume and projection in Postgres. | Docker Compose (test profile) with all containers. | Full stack |
| **E2E** | API → queue → worker → result: `POST /api/jobs` (when platform routes exist) enqueues a job; worker processes it; `GET /api/jobs/{id}` shows `done`. | Docker Compose (test profile). | Full stack |

#### B.17.2 Testing infrastructure

| Component | Approach | Notes |
| --- | --- | --- |
| **Redis for integration tests** | `fakeredis` (in-process, no Docker) for unit-like speed; real Redis via `docker compose -f docker-compose.test.yml up redis` for integration. | arq works with `fakeredis` for basic tests; real Redis for concurrency/timing tests. |
| **Postgres for integration tests** | Already exists — `conftest.py` fixtures with `testcontainers` or local Postgres. Extend for Alembic migration tests. | Add fixture that runs `alembic upgrade head` on a test database. |
| **Docker Compose test profile** | `docker-compose.test.yml` with `test` profile: smaller images, test fixtures mounted, `WHISPER_MODEL=tiny` (fast, low VRAM), test Postgres, test Redis. | E2E distributed tests run against this. CI uses this in the `docker-test` workflow. |
| **Mocked pipeline for orchestration tests** | `run_pipeline` replaced with a fast mock that sleeps briefly and writes canned output files. Tests orchestration, not ML. | Stage-mode mock returns partial output matching the requested stage range. |

#### B.17.3 CI integration

| CI stage | What runs | Time budget |
| --- | --- | --- |
| **`make ci-fast`** | Unit tests for stage decomposition, scheduler state machine, FAISS swap, job serialization. No Redis/Postgres needed. | +30s over current |
| **`make ci`** | + Integration tests with Redis (`fakeredis`) and Postgres (testcontainers). arq job lifecycle, migration roundtrip. | +2–3 min |
| **`make docker-test`** | + E2E distributed tests in Docker Compose test profile. Full job flow with `tiny` Whisper model. | +5–10 min |

**Anti-patterns to avoid:**

- Testing Whisper accuracy through the distributed pipeline — that's what existing E2E tests
  do via `run_pipeline` directly. Distributed E2E tests should use `tiny` model or mocked
  transcription.
- Testing every stage combination in E2E — unit tests cover stage boundary logic. E2E tests
  verify one representative full-chain flow.
- Flaky timing-dependent assertions in concurrency tests — use arq's built-in job result
  polling instead of `time.sleep` + assertions.

#### B.17.4 Test-first implementation order

When building the distributed tier, tests should be written **before** the implementation
they verify, in this order:

1. **Unit: stage decomposition** — Define and test `start_stage`/`end_stage` parameters
   before refactoring `run_pipeline` to support them.
2. **Unit: scheduler state machine** — Define and test state transitions before building the
   scheduler process.
3. **Integration: arq job lifecycle** — Verify enqueue/dequeue/execute with a trivial job
   before wiring `run_pipeline` as the job function.
4. **Integration: scheduler + arq chain** — Verify stage chaining before deploying the
   distributed topology.
5. **E2E: Docker Compose test profile** — Verify end-to-end flow before going to production.

---

## Part C. Corpus digest and weekly rollup

**Depends on:** Stable core (transcripts, metadata, summaries, optional GI/KG) and **queryable**
corpus (PRD-018 / RFC-051 or documented file patterns).

### C.1 Problem sketch

User with **10–50 podcasts** wants to **navigate** recent arrivals, **consume** quickly, **dig
deep** selectively, and answer **"what happened last week across my library?"** without
duplicate noise and with **trust** when claims matter.

Today: **per-episode** artifacts only; no **first-class time-scoped cross-feed digest** contract.

### C.2 Layering (summaries / KG / GI)

| Layer | Role | Home |
| --- | --- | --- |
| **Summaries** | Fast **consumption** | PRD-005, metadata |
| **KG** | **Navigation** across episodes | PRD-019, RFC-055/056 |
| **GIL** | **Value and trust** | PRD-017, RFC-049/050 |
| **Semantic search** | **Discovery** across corpus | PRD-021, RFC-061 |

Digest features should assume these layers are **stable and versioned** before hard rollups.

### C.3 Gaps (product opportunities, not promises)

1. **Time-scoped aggregation** — Window (week, last N days) across corpus; one digest artifact or
   view.

2. **Cross-feed inbox** — "New since last run"; per-feed watermarks; backlog vs delta.
3. **Story clustering / dedup** — Same story across shows. Semantic search (RFC-061) embeddings
   can power cross-episode similarity for dedup.
4. **Ranking / time budgets** — e.g. "30 minutes this week."
5. **Change detection** — "What's new on topic X **this week**" vs cumulative KG.
6. **Digest output contract** — Versioned JSON or doc (themes, GI-backed bullets, episode links).
7. **Presentation** — HTML/email/Obsidian out of core unless scoped; **contract** first. The
   GI/KG Viewer (RFC-062) can serve as the first digest consumption UI.
8. **Personalization** — Watchlists; **user config** + KG.

### C.4 Sequencing (when core is stable)

1. Stable **published** dates + episode identity (ADR-007); summaries + optional GI/KG.
2. **Queryable corpus** (RFC-051 or file patterns).
3. **Semantic search** (RFC-061) over corpus — enables discovery before structured digest.
4. **Digest v0** — Time filter + sorted episodes + summary lead + link; no clustering.
5. **Digest v1** — KG rollups, GI-highlighted bullets, semantic clustering, dedup iteration.

### C.5 Non-goals (early digest)

- Replacing listening / primary sources for high-stakes decisions.
- Full **recommender** or social graph.
- **Merging** GI into KG artifacts — join at digest/query layer if needed.

### C.6 Digest in the pipeline and container topology

Digest is a **post-pipeline** feature — it runs after individual episodes are fully processed
(transcribed, summarized, enriched, indexed, projected). It operates on **corpus-level** data,
not episode-level. This has implications for how it fits into Parts B and D.

**Where digest runs:**

| Digest feature | Data source | Runs on | Queue (B.7) | Resource profile (D.2) |
| --- | --- | --- | --- | --- |
| **Time-scoped aggregation** | Postgres projected tables (RFC-051) | `worker-io` | `projection` or new `digest` | DB-bound |
| **Cross-feed inbox / watermarks** | Postgres + episode identity (ADR-007) | `worker-io` | `projection` or `digest` | DB-bound |
| **Story clustering / dedup** | Vector index (RFC-061) — pairwise similarity | `worker-ml` | `index` or `digest` | ML-compute (embedding comparison) |
| **Ranking / time budgets** | Projected summaries + user config | `worker-io` | `digest` | DB-bound + light CPU |
| **Change detection** ("new on topic X") | Vector index (RFC-061) — temporal query | `worker-ml` | `digest` | ML-compute |
| **Digest output generation** | Aggregated data from above steps | `worker-io` | `digest` | IO-bound (JSON/HTML generation) |

**New queue (distributed tier):** When digest features are non-trivial, add a **`digest`**
queue that runs on `worker-io` (for DB-heavy digest) or `worker-ml` (for clustering/semantic
digest). In simple tier, digest runs as a post-processing step after `run_pipeline` completes
all episodes for a batch.

**Scheduling:** Digest is **not** per-episode — it's periodic or on-demand. The scheduler
(B.7.1) triggers digest jobs on a cron schedule (e.g., weekly) or when the user requests it
via the API (`POST /api/digest/generate`).

### C.7 Resource requirements for digest

Digest features have modest resource requirements compared to the per-episode pipeline:

| Feature | RAM | GPU | Duration | Notes |
| --- | --- | --- | --- | --- |
| Time-scoped aggregation | < 512 MB | None | Seconds | SQL queries on projected tables |
| Cross-feed inbox | < 512 MB | None | Seconds | SQL + watermark logic |
| Story clustering | 1–2 GB | Optional (faster with GPU) | 30s–5 min per corpus | Pairwise cosine similarity over embeddings; O(n²) on episode count |
| LLM-powered digest (cloud) | < 512 MB + API | None | 10–30s per digest | Cloud LLM summarizes top stories |
| LLM-powered digest (local) | 4–8 GB | Recommended | 1–5 min per digest | Local summarization model |

**Impact on D.3 minimum specs:** Digest does not raise the minimum hardware bar. The most
resource-intensive digest feature (story clustering) reuses the embedding model already
loaded by the `index` stage. If clustering runs on `worker-ml`, it shares that worker's
resources.

### C.8 Digest and the viewer (RFC-062)

The GI/KG Viewer (RFC-062) is the first **consumption layer** for digest output. Planned
viewer extensions for digest:

| View | What it shows | API endpoint | Phase |
| --- | --- | --- | --- |
| **Weekly digest** | Top stories, GI-highlighted bullets, episode links | `GET /api/digest/latest` | Digest v0 |
| **Cross-feed timeline** | Episodes across feeds sorted by date, with delta markers | `GET /api/digest/timeline?since=7d` | Digest v0 |
| **Topic trends** | "What's new on topic X this week" — KG diff view | `GET /api/digest/topics?since=7d` | Digest v1 |
| **Story clusters** | Grouped episodes covering the same story | `GET /api/digest/clusters?since=7d` | Digest v1 |

These are new viewer routes added in the same `server/routes/` structure from RFC-062, behind
a `--digest` feature flag (or always-on once stable).

### C.9 Digest sequencing in the graduation path

Digest features align with the graduation path (D.7):

| Graduation stage | Digest capability |
| --- | --- |
| **v0: CLI** | `podcast digest --since 7d` — prints a text summary of recent episodes. Uses existing summaries + file patterns. No clustering. |
| **v1: Simple Compose** | Digest as a scheduled arq cron job. Output as JSON artifact. Viewable in viewer (C.8). |
| **v2: Split workers** | `digest` queue on `worker-io`/`worker-ml`. Story clustering via embeddings. Delta detection via vector index. |
| **v3: SaaS** | Per-tenant digest with personalization. Email/webhook delivery. |

---

## Part D. Hardware sizing and distributed ML processing

**Depends on:** Part B (container topology, queue model). **Motivated by:** The pipeline now has
5+ ML-intensive stages that compete for GPU, VRAM, and RAM. Running them sequentially inside one
`heavy` job means a 60-minute Whisper transcription blocks summarization, GIL extraction, KG
extraction, NLI entailment checking, and vector indexing for the entire backlog. This part
specifies **minimum hardware**, **resource profiles**, **container topology**, and **queue design**
to enable true parallelism — processes do not wait for each other.

### D.1 ML component inventory (resource requirements)

These are the models and tasks the pipeline uses, with their resource profiles based on the
current codebase (`providers/ml/`, `evaluation/`, `search/`, `gi/`, `kg/`):

| Component | Models (from `model_registry.py`) | VRAM / RAM | GPU benefit | Typical duration |
| --- | --- | --- | --- | --- |
| **Whisper transcription** | `openai/whisper` (base/small/medium/large-v3) | 1–6 GB VRAM | **Critical** (10–40× faster) | 5–60 min per episode |
| **Speaker diarization** | `pyannote/speaker-diarization-3.1` (RFC-058) | 2–4 GB VRAM | **Critical** | 2–10 min per episode |
| **Summarization (local)** | BART, DistilBART, PEGASUS, LED-16384 | 1–3 GB RAM/VRAM | Helpful (2–5× faster) | 30s–5 min per episode |
| **Hybrid MAP-REDUCE** | LongT5 MAP + Ollama/FLAN-T5 REDUCE (RFC-042) | 2–6 GB RAM | Helpful | 1–5 min per episode |
| **GIL extraction** | Provider-dependent (ML tier: FLAN-T5 + QA + NLI) | 2–4 GB RAM | Moderate | 1–3 min per episode |
| **KG extraction** | Provider-dependent (LLM or summary-bullet) | 1–3 GB RAM | Moderate | 30s–2 min per episode |
| **Embedding encode** | `sentence-transformers` (all-MiniLM-L6-v2 etc.) | 0.5–1 GB RAM | Optional | 10–30s per episode |
| **NLI entailment** | CrossEncoder NLI (`providers/ml/nli_loader.py`) | 1–2 GB RAM | Optional | 5–20s per episode |
| **Extractive QA** | HF QA pipeline (`providers/ml/extractive_qa.py`) | 1–2 GB RAM | Optional | 5–15s per episode |
| **FAISS index** | `faiss-cpu` (RFC-061) | RAM proportional to corpus | No | Seconds (incremental) |
| **Evaluation scoring** | ROUGE, BLEU, WER, embedding sim (`evaluation/scorer.py`) | 0.5–1 GB RAM | No | 10–60s per run |
| **Audio preprocessing** | FFmpeg + VAD (RFC-040, ADR-036/037/038/039) | Minimal | No | 30s–2 min per episode |

### D.2 Resource profiles

Group pipeline stages by resource affinity so that stages with the **same resource needs** share
a worker pool, and stages with **different needs** never block each other:

| Profile | Stages | Bottleneck | Can share pool? |
| --- | --- | --- | --- |
| **GPU-heavy** | Whisper, pyannote diarization | VRAM (4–8 GB), GPU compute | Only with each other (serialized on GPU) |
| **ML-compute** | Summarization, GIL, KG, NLI, QA, **embedding encode** (`sentence-transformers`), adaptive routing (RFC-053) | RAM (4–16 GB), CPU cores | Yes — CPU-parallel, multiple concurrent |
| **IO-bound** | RSS fetch, audio download, transcript download, file projection, **FAISS upsert** (index write, disk-bound) | Network, disk I/O | Yes — high concurrency OK |
| **DB-bound** | Postgres projection (RFC-051), index stats | DB connection pool | Yes — separate from ML |

**Key insight:** GPU-heavy and ML-compute **must not** share a queue because a 45-minute
Whisper job blocks 15 episodes' worth of 3-minute summarizations. Separating them unlocks
**pipeline-level parallelism**: while Whisper transcribes episode N, ML-compute processes
episodes N-1, N-2, ... that already have transcripts.

**Embedding + FAISS split:** The `index` stage has two sub-steps with different profiles:
encoding text into vectors (ML-compute, needs ~1 GB RAM for the `sentence-transformers`
model) and upserting vectors into the FAISS index (IO-bound, disk write). In practice
both run in the same worker because encoding dominates runtime (seconds vs milliseconds).
The `index` queue runs on **`worker-ml`**, not `worker-io`, because the worker must have
the embedding model loaded. The FAISS write is fast enough that it doesn't warrant a
separate queue.

### D.3 Minimum hardware specifications

#### D.3.1 Development / single-user (non-cloud, self-hosted)

| Resource | Minimum | Recommended | Notes |
| --- | --- | --- | --- |
| **CPU** | 8 cores | 16 cores | 2 API + 2 IO workers + 4 ML workers |
| **RAM** | 32 GB | 64 GB | Models load 1–6 GB each; concurrent models need headroom |
| **GPU** | 1× discrete, 8 GB VRAM | 1× discrete, 12+ GB VRAM | For Whisper + diarization. NVIDIA preferred (CUDA), AMD ROCm possible |
| **Storage** | 100 GB SSD | 250 GB NVMe | Model cache (~20 GB), audio temp, corpus artifacts, FAISS index |
| **Network** | 10 Mbps | 100 Mbps | RSS/audio download; API providers if using cloud LLMs |

#### D.3.2 Apple Silicon path (MPS, no discrete GPU)

Apple Silicon (M1 Pro / M2 Pro / M3 Pro or higher) is a viable **development** target:

| Resource | Minimum | Recommended | Notes |
| --- | --- | --- | --- |
| **Unified memory** | 32 GB | 64 GB | Shared between CPU and GPU (MPS); no separate VRAM |
| **CPU cores** | 10+ (8P+2E) | 12+ (8P+4E) | M2 Pro or better |
| **Storage** | 256 GB SSD | 512 GB SSD | Internal NVMe is fast; model cache + corpus |

**Constraints on Apple Silicon:**

- ADR-046 (MPS exclusive mode) serializes GPU work to prevent memory contention — Whisper
  and summarization cannot use MPS simultaneously.
- No diarization GPU acceleration (pyannote uses CUDA; MPS support is experimental).
- Suitable for **single-user / dev**, not for production multi-tenant with continuous ingestion.

#### D.3.3 Small production server (non-cloud, 10–50 feeds)

| Resource | Specification | Notes |
| --- | --- | --- |
| **CPU** | 16–32 cores (Xeon/EPYC/Ryzen) | Dedicated ML workers need cores |
| **RAM** | 64–128 GB ECC | Multiple concurrent models + Postgres + Redis |
| **GPU** | 1–2× NVIDIA, 12–24 GB VRAM each | RTX 3090/4090 or A4000/A5000; Whisper large-v3 needs ~6 GB |
| **Storage** | 500 GB–1 TB NVMe | Hot storage for models + active corpus; cold tier for archives |
| **Network** | 1 Gbps | Podcast audio downloads at scale |

**Two GPUs unlock true parallelism:** GPU 1 for Whisper/diarization, GPU 2 for GPU-accelerated
summarization/embedding — or both on Whisper with `concurrency 2`.

### D.4 Episode processing time budget (reference)

For a **single 60-minute podcast episode** with all ML features enabled, typical wall-clock
times on recommended hardware:

| Stage | GPU (CUDA) | CPU-only | Apple Silicon (MPS) |
| --- | --- | --- | --- |
| Audio download + preprocess | 1–3 min | 1–3 min | 1–3 min |
| Whisper transcription (medium) | 3–8 min | 30–60 min | 8–15 min |
| Speaker diarization (pyannote) | 2–5 min | 15–30 min | N/A (CPU fallback) |
| Summarization (LED-16384) | 30s–2 min | 3–8 min | 1–3 min |
| GIL extraction (hybrid tier) | 1–3 min | 2–5 min | 1–4 min |
| KG extraction | 30s–2 min | 1–3 min | 1–2 min |
| Embedding + FAISS index | 10–30s | 20–60s | 15–40s |
| Postgres projection | 5–15s | 5–15s | 5–15s |
| **Total (sequential)** | **~10–25 min** | **~55–115 min** | **~15–30 min** |
| **Total (parallel, Part D topology)** | **~8–15 min** | **~35–65 min** | **~12–20 min** |

**Parallel gain:** With stage-based queues, the main bottleneck is Whisper. While Whisper works
on episode N, all post-transcription stages process the backlog — effective throughput improves
by 30–50%.

### D.5 Container topology (Compose)

#### Simple tier (v1 — 5 containers)

```text
┌─────────────────┐  ┌────────────────────────────────┐
│  caddy           │  │  api + viewer                   │
│  (reverse proxy) │  │  (FastAPI, RFC-062 server,      │
│  64 MB RAM       │  │  Vue SPA, search API)           │
│  0.5 CPU         │  │  2 CPU, 4 GB RAM                │
└─────────────────┘  └────────────────────────────────┘

┌────────────────────────────────────────┐
│  worker-all                             │
│  queue: ingest, heavy, projection       │
│  All pipeline stages (run_pipeline)     │
│  GPU + 8–12 GB VRAM                     │
│  4 CPU, 16 GB RAM                       │
│  Concurrency: heavy=1, ingest=4, proj=2 │
└────────────────────────────────────────┘

┌──────────────────┐  ┌──────────────────┐
│  postgres         │  │  redis            │
│  2 CPU, 4 GB RAM  │  │  1 CPU, 1 GB RAM  │
│  Persistent vol.  │  │  Persistent vol.  │
└──────────────────┘  └──────────────────┘
```

#### Distributed tier (v2 — 8+ containers)

```text
┌─────────────────┐  ┌────────────────────────────────┐  ┌─────────────────────┐
│  caddy           │  │  api + viewer                   │  │  scheduler           │
│  (reverse proxy) │  │  (FastAPI, RFC-062 server,      │  │  (stage-chain        │
│                  │  │  Vue SPA, search API,           │  │  orchestration,      │
│  64 MB RAM       │  │  FAISS query)                   │  │  B.7.1)              │
│  0.5 CPU         │  │  2 CPU, 4 GB RAM                │  │  1 CPU, 1 GB RAM     │
└─────────────────┘  └────────────────────────────────┘  └─────────────────────┘

┌────────────────────────┐  ┌───────────────────────────┐  ┌──────────────────────┐
│  worker-gpu             │  │  worker-ml                 │  │  worker-io            │
│  queue: transcribe      │  │  queue: enrich, index      │  │  queue: ingest,       │
│                         │  │                            │  │         projection    │
│  Whisper, pyannote      │  │  Summarization, GIL, KG,   │  │                       │
│  diarization            │  │  NLI, QA, embedding encode,│  │  Download, Postgres   │
│                         │  │  FAISS upsert (D.2)        │  │  projection           │
│  GPU + 8–12 GB VRAM     │  │  Adaptive routing (RFC-053) │  │  2 CPU, 4 GB RAM      │
│  2 CPU, 8 GB RAM        │  │                            │  │  Concurrency: 4–8     │
│  Concurrency: 1 per GPU │  │  4 CPU, 8–16 GB RAM        │  │                       │
│                         │  │  (optional GPU for 2–5×)   │  │                       │
│                         │  │  Concurrency: 2–4          │  │                       │
└────────────────────────┘  └───────────────────────────┘  └──────────────────────┘

┌──────────────────┐  ┌──────────────────┐
│  postgres         │  │  redis            │
│  2 CPU, 4 GB RAM  │  │  1 CPU, 1 GB RAM  │
│  Persistent vol.  │  │  Persistent vol.  │
└──────────────────┘  └──────────────────┘
```

**Shared volumes:** Artifact root (read/write by all workers, read by API), model cache
(read by `worker-gpu` and `worker-ml`), FAISS index directory (write by `worker-ml`,
read by `api`). See B.14 for file locking strategy on shared volumes.

**Image strategy:** See B.9.2. Single image by default; split into `podcast-scraper-gpu`
and `podcast-scraper-base` when GPU dependencies make the image > 8 GB.

### D.6 Queue-per-concern flow (stage-based pipeline)

```text
                      ┌──────────────────────┐
 New feed poll ──────►│   ingest (worker-io)  │
                      │   Download audio,     │
                      │   fetch transcript    │
                      └──────────┬───────────┘
                                 │ audio ready
                                 ▼
                      ┌──────────────────────┐
                      │ transcribe (worker-gpu)│
                      │  Whisper + diarization │
                      │  Concurrency: 1/GPU   │
                      └──────────┬───────────┘
                                 │ transcript ready
                                 ▼
                      ┌──────────────────────┐
                      │  enrich (worker-ml)   │
                      │  Summarize, GIL, KG,  │
                      │  NLI, QA              │
                      │  Concurrency: 2–4     │
                      └──────────┬───────────┘
                                 │ artifacts ready
                        ┌────────┴────────┐
                        ▼                 ▼
             ┌────────────────┐  ┌──────────────────┐
             │ index           │  │ projection        │
             │ (worker-ml)     │  │ (worker-io)       │
             │ Embed + FAISS   │  │ Files → Postgres  │
             └────────────────┘  └──────────────────┘
```

**State machine per episode:**

```text
pending → downloading → transcribing → enriching → indexing → projected → done
                                                         └──► indexed ──►┘
```

Each transition is a job on the appropriate queue. **Failed** jobs retry on the same queue
(with exponential backoff) or move to dead-letter. State is persisted in Postgres `jobs` table
or Redis sorted sets.

### D.7 Graduation path (from simple to distributed)

Not every deployment needs Part D from day one. The graduation path aligns with the
observability (E.10) and deployment (F.10) graduation paths — see the unified timeline below.

| Stage | Topology | Observability (E.10) | Deployment (F.10) | Auth (A.12) | When to graduate |
| --- | --- | --- | --- | --- | --- |
| **v0: CLI** | Single process, no containers | `--json-logs` + `metrics.json` | `make serve` / `docker compose up` | None | Dev, ad-hoc use, < 5 feeds |
| **v1: Simple Compose** | `postgres` + `api` + `worker-all` + `caddy` + `redis` | + Grafana, health checks (E.6) | `deploy.sh` via SSH; `.env` + secrets | API key (stage 2) | Service mode, 5–20 feeds, single GPU |
| **v2: Split workers** | + `worker-gpu` + `worker-ml` + `worker-io` + `scheduler` | + PLG stack, alerting, control plane | GitHub Actions CD; Watchtower | API key | Whisper backlog blocks summarization; > 20 feeds |
| **v3: Multi-GPU / SaaS** | Multiple `worker-gpu`; ML on GPU | + AI agent (shadow → active) | K8s / GitOps | JWT + multi-tenant | > 50 feeds or real-time ingestion; multiple users |

**Trigger to graduate v1 → v2:** When Whisper queue depth > 10 episodes and summarization
latency > 10 minutes (because it's waiting behind Whisper). Metrics from RFC-027 / RFC-043
should alert on this.

**Key invariant:** Never deploy a more complex topology tier without the matching
observability tier. Specifically: don't graduate D.v2 (split workers) without at least
E.v2 (PLG stack with metrics from each worker). Distributed systems without distributed
observability are invisible systems.

### D.8 Cloud LLM fallback (API providers reduce hardware needs)

If cloud API costs are acceptable, the hardware requirements drop significantly:

| Using cloud for... | Hardware saved | Trade-off |
| --- | --- | --- |
| **Transcription** (OpenAI Whisper API, Gemini) | No GPU needed for Whisper | API cost, latency, privacy |
| **Summarization** (OpenAI, Gemini, Anthropic, Mistral, etc.) | Less RAM, no GPU for summarization | API cost per episode |
| **GIL/KG extraction** (cloud LLM tier) | No local FLAN-T5/QA/NLI | API cost, higher quality |
| **All ML via API** | 4 CPU, 8 GB RAM is sufficient | Full API dependency |

This is the **existing multi-provider architecture** (9 providers, RFC-029). The per-capability
provider selection (ADR-026) already supports mixed configurations — e.g., local Whisper +
cloud summarization, or cloud transcription + local GIL.

### D.9 Evaluation workloads (`data/eval/`)

The evaluation infrastructure (`data/eval/`, `evaluation/scorer.py`, RFC-041, RFC-057) has its
own resource considerations:

| Workload | Resource needs | Where it runs |
| --- | --- | --- |
| **Experiment runs** (`make eval-run`) | Same as pipeline (model-dependent) | Dev machine or CI |
| **Scoring** (`score_run`) | ROUGE/BLEU (CPU), embedding similarity (sentence-transformers) | CPU-only OK |
| **Baseline comparison** | CPU-only (file comparison) | CI |
| **AutoResearch** (RFC-057) | Many experiment iterations; needs efficient model loading | Dev machine with GPU |

**Recommendation:** Evaluation does **not** need the distributed topology (Part D.5). It runs
on a developer's machine or in CI. However, `model_registry.py` preloading and caching
(RFC-028) should be used to avoid re-loading models between experiment iterations.

### D.10 Model memory management

Multiple models competing for RAM/VRAM is the main risk in local ML. Key patterns:

1. **Lazy loading** (ADR-005): Models load only when needed; `worker-gpu` only loads Whisper
   and pyannote, not summarization models.
2. **MPS exclusive mode** (ADR-046): On Apple Silicon, GPU work is serialized to prevent
   memory contention.
3. **Model preloading** (RFC-028): `model_loader.py` `preload_whisper_models` warms cache on
   worker start, avoiding cold-start latency on first job.
4. **Worker specialization** (Part D.5): Each worker pool loads only the models it needs —
   `worker-gpu` never loads sentence-transformers; `worker-ml` never loads Whisper.
5. **Offload after use:** For constrained systems, models can be explicitly unloaded
   (`del model; torch.cuda.empty_cache()`) after processing a batch.

### D.11 Minimum total resource allocation (Compose)

#### Simple tier (v1 — 5 containers), minimum totals

| Resource | Minimum | Recommended |
| --- | --- | --- |
| **CPU cores** | 6 | 8–12 |
| **RAM** | 16 GB | 32 GB |
| **GPU VRAM** | 8 GB (1 GPU) | 12 GB |
| **Disk** | 50 GB SSD | 100 GB NVMe |

The single `worker-all` runs pipeline stages sequentially on one GPU. This is viable for
5–20 feeds with `medium` Whisper model.

#### Distributed tier (v2 — 8+ containers), minimum totals

| Resource | Minimum | Recommended |
| --- | --- | --- |
| **CPU cores** | 12 | 20–24 |
| **RAM** | 32 GB | 64 GB |
| **GPU VRAM** | 8 GB (1 GPU) | 12–24 GB (1–2 GPUs) |
| **Disk** | 100 GB SSD | 250 GB NVMe |

Observability sidecar (PLG stack) adds ~3 GB RAM and 2 CPU on top of these numbers.

### D.12 Whisper model sizing — choosing model vs hardware

The Whisper model choice directly determines GPU requirements. Not every setup needs `large-v3`:

| Model | Parameters | VRAM (FP16) | VRAM (faster-whisper, int8) | Disk | WER (English) | Speed vs real-time |
| --- | --- | --- | --- | --- | --- | --- |
| **tiny** | 39 M | ~1 GB | ~0.5 GB | 75 MB | 7.6% | ~32× |
| **base** | 74 M | ~1 GB | ~0.7 GB | 142 MB | 5.0% | ~16× |
| **small** | 244 M | ~2 GB | ~1 GB | 466 MB | 3.4% | ~6× |
| **medium** | 769 M | ~5 GB | ~2.5 GB | 1.5 GB | 2.9% | ~2× |
| **large-v3** | 1.55 B | ~10 GB | ~4.5 GB | 2.9 GB | 2.0% | ~1× |

**Recommendation for budget builds:** Use **`medium`** (2.9% WER is excellent for podcasts,
which are typically clear English speech). This halves the VRAM requirement compared to
`large-v3` — an RTX 3060 12 GB runs `medium` with 7 GB headroom for other work. Upgrade to
`large-v3` only when processing multilingual content or noisy audio.

**`faster-whisper`** (CTranslate2 backend) reduces VRAM usage by ~50% with negligible quality
loss. Consider switching from `openai-whisper` to `faster-whisper` for production deployments —
this is an implementation decision, not a pipeline architecture change.

### D.13 GPU clarification — integrated vs discrete

**Critical distinction:** The GPU discussion in this document refers to **discrete GPUs** (a
separate card with its own VRAM), not the integrated graphics that come built into every CPU
or Mac.

| Type | Examples | Suitable for Whisper? | Suitable for summarization? |
| --- | --- | --- | --- |
| **Integrated (Intel UHD/Iris)** | Intel UHD 630/770, Intel Arc iGPU | No — no CUDA, too slow | No |
| **Integrated (AMD Radeon)** | AMD Radeon 680M/780M (in Ryzen APUs) | No — no CUDA support | No |
| **Apple Silicon (MPS)** | M1/M2/M3/M4 GPU cores | Yes (MPS) — moderate speed | Yes (MPS) — moderate speed |
| **Discrete NVIDIA (budget)** | RTX 3060 12GB, RTX 4060 8GB | **Yes — good** | Yes — helpful |
| **Discrete NVIDIA (mid)** | RTX 3090 24GB, RTX 4070 Ti 12GB | **Yes — great** | Yes — great |
| **Discrete NVIDIA (pro)** | RTX 4090 24GB, A4000/A5000 | **Yes — optimal** | Yes — optimal |
| **Discrete AMD** | RX 7900 XT/XTX | Experimental (ROCm) — not recommended | Experimental |

**Bottom line:** For a Linux PC build, you need to **buy a discrete NVIDIA GPU separately** and
install it in a PCIe x16 slot. The GPU that "comes with" a PC (integrated graphics) does
nothing for ML inference. On Mac, Apple Silicon's unified memory with MPS is the GPU — no
separate card needed, but it's slower than CUDA on a discrete NVIDIA card.

### D.14 Concrete hardware configurations and pricing

#### D.14.1 Linux PC — Budget tier (< €500)

**Strategy:** Refurbished business desktop + used GPU. Best price/performance for ML.

**Option A: Refurbished tower + used RTX 3060 (~€400–480)**

| Component | Specific model | Price (EUR, used/refurb) |
| --- | --- | --- |
| PC base | Dell OptiPlex 7090 Tower (i7-11700, 16 GB, 512 GB NVMe) or HP Z2 Tower G5 (i7-10700) | ~€180–250 |
| RAM upgrade | +16 GB DDR4 (to reach 32 GB total) | ~€25–35 |
| GPU | NVIDIA RTX 3060 12 GB (used) | ~€140–180 |
| Storage (optional) | +1 TB SATA SSD for corpus data | ~€40–50 |
| **Total** | | **~€385–515** |

**Why this works:**

- i7-10700/11700: 8 cores / 16 threads — enough for API + ML workers + IO
- 32 GB DDR4: Runs summarization models, GIL, KG, sentence-transformers concurrently
- RTX 3060 12 GB: Whisper `medium` in ~3 min per 60-min episode; `large-v3` fits with
  `faster-whisper` int8; 12 GB VRAM is the sweet spot for budget ML
- Tower form factor (not SFF/mini) required for full-height PCIe x16 GPU slot + adequate PSU

**Important: check PSU.** The refurbished PC must have a 350W+ PSU with a PCIe 6/8-pin power
connector (or budget €30–40 for a PSU upgrade). RTX 3060 draws ~170W. Many business towers
have 300W PSU — verify before buying.

**Option B: Mini PC (no GPU) + cloud API (~€250–350)**

| Component | Specific model | Price (EUR) |
| --- | --- | --- |
| Mini PC | Beelink SER7 (Ryzen 7 7840HS, 32 GB DDR5, 500 GB NVMe) or Minisforum UM790 Pro | ~€300–350 |
| Cloud API | OpenAI Whisper API + Gemini/Mistral for summarization | ~€5–15/month |
| **Total** | | **~€300–365 + API costs** |

**When this makes sense:** If you primarily use cloud providers for transcription and
summarization (the existing multi-provider architecture), you don't need a GPU at all. The mini
PC handles API orchestration, GIL/KG extraction (cloud tier), FAISS indexing, and the viewer.
Local ML (Whisper, BART) would be very slow on CPU-only.

#### D.14.2 Linux PC — Optimal tier (~€800–1200)

**Strategy:** Used workstation or new mid-range build with a stronger GPU.

| Component | Specific model | Price (EUR) |
| --- | --- | --- |
| PC base | Dell Precision 3640/3660 Tower (i7-12700, 32 GB, 512 GB) or build: Ryzen 7 7700 + B650 + 32 GB DDR5 + 1 TB NVMe + 550W PSU + case | ~€350–550 |
| GPU | NVIDIA RTX 3090 24 GB (used, ~€500) or RTX 4060 Ti 16 GB (used, ~€300) | ~€300–500 |
| RAM (if building) | 64 GB DDR5 (2×32 GB) | ~€120 (if needed) |
| **Total** | | **~€700–1100** |

**Why this is optimal:**

- RTX 3090 24 GB: Runs `large-v3` FP16 (10 GB) with 14 GB headroom for diarization; or runs
  Whisper + pyannote concurrently
- 64 GB RAM: Multiple concurrent ML workers without contention
- This is the **Part D.5 topology target** — enough resources to split into worker-gpu +
  worker-ml + worker-io

#### D.14.3 Mac Mini — from worst to best

| Model | CPU/GPU | Unified memory | Whisper `medium` | Whisper `large-v3` | Full pipeline | Price (new/refurb) |
| --- | --- | --- | --- | --- | --- | --- |
| Mac Mini 2018 (Intel i7) | 6-core i7-8700B, Intel UHD 630 | N/A (8–64 GB DDR4) | CPU-only: ~30 min/episode | CPU-only: ~60 min/episode | Very slow; no MPS | ~€200–300 (used) |
| Mac Mini M1 (2020) | 8-core (4P+4E), 8-core GPU | 16 GB | MPS: ~12 min/episode | OOM with 16 GB (tight) | Slow; 16 GB limiting | ~€350–450 (used) |
| **Mac Mini M2 (2023)** | 8-core (4P+4E), 10-core GPU | 16 or **24 GB** | MPS: ~10 min/episode | 24 GB: fits; ~18 min | **Viable minimum for dev** | ~€450–550 (24 GB, used) |
| **Mac Mini M4 (2024)** | 10-core (4P+6E), 10-core GPU | 16 / 24 / **32 GB** | MPS: ~8 min/episode | 32 GB: comfortable | **Good for dev** | €599 (16 GB new), €999 (24 GB), €1,399 (32 GB) |
| **Mac Mini M4 Pro (2024)** | 12–14-core, 16–20-core GPU | 24 / **48** / 64 GB | MPS: ~5 min/episode | 48 GB: fast + headroom | **Best Mac option** | €1,399 (24 GB), €1,999 (48 GB) |

**Verdict:**

- **Intel Mac Mini: do NOT buy for ML.** No MPS, no GPU acceleration, CPU-only Whisper is
  painfully slow. Only viable if you use 100% cloud API providers.
- **M1 (16 GB): too tight.** Models + OS eat most of 16 GB; `large-v3` may OOM.
- **M2 (24 GB): minimum viable.** Whisper medium runs well; large-v3 fits. GIL/KG/search work.
  But ADR-046 serializes GPU work — no parallel Whisper + summarization.
- **M4 (32 GB): sweet spot.** New at €1,399, refurb potentially less. 32 GB handles Whisper
  medium + summarization + embedding models with room for Postgres + Redis. Best value if you
  want Mac.
- **M4 Pro (48 GB): ideal.** Runs everything including `large-v3` + diarization models +
  concurrent ML workers. 273 GB/s memory bandwidth (vs 120 GB/s on M4) makes inference
  meaningfully faster. €1,999 is steep but it's a real ML workstation.

**Apple Silicon limitations (all models):**

- MPS exclusive mode (ADR-046) — GPU work serialized (Whisper and summarization don't
  overlap on MPS)
- pyannote diarization has limited MPS support — falls back to CPU
- Not viable for production multi-tenant continuous ingestion (use Linux + CUDA for that)
- Cannot add a discrete GPU — what you buy is what you get, forever

### D.15 Cloud / rental options

For users who don't want to own hardware, or need to scale beyond a single box:

#### D.15.1 Dedicated GPU servers (monthly rental)

| Provider | Plan | GPU | CPU | RAM | Storage | Price/month |
| --- | --- | --- | --- | --- | --- | --- |
| **Hetzner GEX44** | Dedicated | RTX 4000 SFF Ada (20 GB VRAM) | i5-13500 (14 cores) | 64 GB DDR5 | 1.875 TB NVMe | **€184/mo** + €264 setup |
| **Hetzner GEX130** | Dedicated | RTX 6000 Ada (48 GB VRAM) | Xeon Gold 5412U (24 cores) | 128 GB DDR5 ECC | 3.84 TB NVMe | **€838/mo** + €79 setup |
| **Vast.ai** | Marketplace | RTX 3090 24 GB (peer-hosted) | Varies | Varies | Varies | **~€50–100/mo** (continuous) |
| **RunPod** | On-demand | RTX 3090 24 GB | Varies | Varies | Varies | **~€200/mo** (continuous) |

**Recommendation:**

- **Hetzner GEX44 (€184/mo)** is the best value for a reliable dedicated GPU server. 20 GB
  VRAM handles all Whisper models + diarization comfortably. 64 GB RAM + 14-core CPU supports
  the full Part D.5 topology. This is a real dedicated server (not shared) in a German/Finnish
  datacenter.
- **Vast.ai (~€50–100/mo)** is cheapest but **peer-to-peer** — variable reliability, machines
  can disappear, no SLA. Fine for batch processing, not for continuous service.
- **Hetzner GEX130 (€838/mo)** is overkill for < 50 feeds but ideal for production
  multi-tenant with 48 GB VRAM.

#### D.15.2 No-GPU cloud (API-only mode)

If all ML runs via API providers (OpenAI, Gemini, Mistral, etc.):

| Provider | Type | Specs | Price/month |
| --- | --- | --- | --- |
| **Hetzner VPS (CX31)** | VPS | 8 vCPU, 32 GB RAM, 240 GB NVMe | ~€30/mo |
| **Hetzner dedicated** | Dedicated | 8-core, 64 GB RAM, 512 GB NVMe | ~€50–70/mo |
| **DigitalOcean/Linode** | VPS | 8 vCPU, 32 GB RAM | ~€50–60/mo |

Plus API costs: ~€5–20/month for 10–50 feeds depending on provider and episode length.

**This is the cheapest path to running the full platform** (API + viewer + Postgres + Redis +
workers) if you offload all ML to cloud providers.

#### D.15.3 Break-even: own hardware vs rental

**Hardware cost only (simplistic):**

| Scenario | Own hardware (upfront) | Monthly rental | Break-even |
| --- | --- | --- | --- |
| Budget Linux PC (€450) vs Hetzner GEX44 (€184/mo) | €450 one-time | €184/mo + €264 setup | ~2.5 months |
| Optimal Linux PC (€1,000) vs Hetzner GEX44 (€184/mo) | €1,000 one-time | €184/mo + €264 setup | ~5.5 months |
| Mac Mini M4 32 GB (€1,400) vs Hetzner GEX44 (€184/mo) | €1,400 one-time | €184/mo + €264 setup | ~7.5 months |

**Real-world total cost of ownership (home server):**

The break-even above compares hardware price vs rental. In reality, running a home server
has ongoing costs that the table omits:

| Cost | Monthly estimate | Notes |
| --- | --- | --- |
| **Electricity** | €10–30/mo | GPU PC at 150–300W, 24/7. Mac Mini ~€3–5/mo. |
| **Internet** (static IP or dynamic DNS) | €0–10/mo | Most home ISPs lack static IP; use DuckDNS/Cloudflare Tunnel (free) or upgrade plan. |
| **Maintenance time** | 2–4 hours/mo | OS updates, Docker upgrades, disk space, GPU driver updates, restarting after power outage. Harder to value but real. |
| **Reliability risk** | Hard to price | Home internet goes down, hardware fails, power outages. No SLA. |
| **Backup** | €0–5/mo | Local backup is free; offsite (B2/S3) adds cost for large datasets. |

**Adjusted break-even (with ops overhead):**

| Scenario | Monthly own cost (hw amortized + ops) | Monthly rent | Real break-even |
| --- | --- | --- | --- |
| Budget Linux (€450, 24-mo amortize) + ops | ~€19 hw + €20 elec + €5 misc = ~€44/mo | €184/mo | **Own wins from month 1** (but you do the ops work) |
| Optimal Linux (€1,000, 24-mo) + ops | ~€42 hw + €25 elec + €5 misc = ~€72/mo | €184/mo | **Own wins from month 1** (you're the SRE) |
| Mac Mini M4 (€1,400, 36-mo) + ops | ~€39 hw + €5 elec + €5 misc = ~€49/mo | €184/mo | **Own wins from month 1** (lowest ops burden) |

**Verdict:**

- **Home hardware is cheaper per month in all scenarios** — even with electricity and
  maintenance factored in. The Hetzner GPU rental premium (~€184/mo) is high for a
  single-user / small-scale deployment.
- **But you become the SRE.** Every power outage, driver issue, and disk full is yours
  to handle. If your time is expensive or you need guaranteed uptime, rental wins on
  **operational simplicity**, not on cost.
- **Recommended path:** Start with home hardware for development and small-corpus processing.
  Move to cloud rental when you need: (a) production uptime SLA, (b) remote access without
  managing tunnels, or (c) more GPU than your home setup provides.
- **Hybrid option:** Run the dev/staging instance at home; rent a Hetzner VPS (€10–30/mo,
  no GPU) for the API server + viewer + Postgres, and use cloud ML APIs for transcription
  and summarization. This avoids GPU rental costs entirely while still having a reliable
  production endpoint.

### D.16 Recommended configurations (decision matrix)

| Profile | Hardware | Cost | Whisper | Full pipeline | Distributed workers | Best for |
| --- | --- | --- | --- | --- | --- | --- |
| **Cheapest viable** | Mini PC + cloud API | €300 + €10/mo API | Cloud (OpenAI/Gemini) | Yes (cloud ML) | No (single process) | Experimenting, < 10 feeds, cost-sensitive |
| **Budget local ML** | Refurb tower + RTX 3060 12 GB | ~€450 | `medium` in 3–8 min | Yes (sequential) | v1 Compose | Dev, 5–20 feeds, privacy-first |
| **Mac dev** | Mac Mini M4 32 GB | ~€1,400 | `medium` in 8 min (MPS) | Yes (serialized MPS) | v1 Compose (constrained) | Mac users, dev, single-user |
| **Optimal home** | Tower + RTX 3090 24 GB + 64 GB RAM | ~€1,000 | `large-v3` in 5–8 min | Yes (parallel) | v2 Split workers | 10–50 feeds, quality-focused |
| **Cloud rental** | Hetzner GEX44 | €184/mo | `large-v3` in 3–5 min | Yes (parallel) | v2–v3 topology | Production, no hardware mgmt |
| **Production** | Hetzner GEX130 or dual-GPU build | €838/mo or ~€2,500 | `large-v3`, concurrent | Full parallel | v3 Multi-GPU | 50+ feeds, multi-tenant |

---

## Part E. Observability, monitoring, and control plane

**Depends on:** Part B (service containers exist), Part D (worker pools produce metrics).
**Motivated by:** A distributed system with 3 worker pools, Redis queues, Postgres, and a
GPU-bound pipeline is opaque without centralized logging, metrics, dashboards, and alerting.
Today the project has rich **per-run metrics** and **structured JSON logs** (Issue #379) but
no aggregation stack and no live dashboard — operators can't see queue depth, GPU utilization,
worker health, or backlog trends without SSH-ing into the box.

### E.1 What exists today (audit)

| Capability | Status | Where |
| --- | --- | --- |
| **Pipeline runtime metrics** | Rich in-memory collection, per-stage timings, per-episode durations, provider token counts | `workflow/metrics.py`, `utils/provider_metrics.py` |
| **Run manifests** | System state snapshot per run (Python, OS, GPU, model versions, git SHA, config hash) | `run.json`, `run_manifest.json`, `metrics.json` per run |
| **Structured JSON logging** | CLI `--json-logs` flag → `JSONFormatter`; designed for ELK/Splunk/CloudWatch | `cli.py`, `config.py`, `workflow/orchestration.py` |
| **CI/nightly metrics** | Metrics collection, history, dashboards on GitHub Pages | `scripts/dashboard/`, `.github/workflows/nightly.yml`, RFC-025/026 |
| **Automated alerts** | Nightly summary with metric alerts; PR comments planned | RFC-043 (partial) |
| **Operational observability PRD** | Product umbrella for health score, bottleneck visibility, dashboards | PRD-016 (open) |
| **Correlation IDs** | Not implemented — no request-to-worker trace linkage | Gap |
| **Live system dashboards** | Not implemented — no Grafana/Prometheus/Loki | Gap |
| **Queue / worker monitoring** | Not implemented — no queue depth, worker health, GPU utilization visibility | Gap |

### E.2 Observability stack (target architecture)

The observability stack runs as **additional Compose services** alongside the application
containers from Part D.5. It does not change the application code — it **collects** what the
application already emits (logs, metrics) and **visualizes** it.

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        Observability sidecar services                  │
│                                                                        │
│  ┌───────────────┐  ┌───────────────┐  ┌──────────────────────────┐   │
│  │  Prometheus    │  │  Loki          │  │  Grafana                  │  │
│  │  (metrics      │  │  (log          │  │  (dashboards,             │  │
│  │  scrape/store) │  │  aggregation)  │  │  alerts, explore)         │  │
│  │  1 CPU, 1 GB   │  │  1 CPU, 1 GB   │  │  1 CPU, 512 MB            │  │
│  └───────┬───────┘  └───────┬───────┘  └────────────┬───────────────┘ │
│          │                  │                        │                  │
│          │  scrape /metrics │  log driver / push     │  query both      │
│          ▼                  ▼                        ▼                  │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  Application containers (api, worker-gpu, worker-ml, worker-io,  │ │
│  │  scheduler, postgres, redis)                                      │ │
│  │  — expose /metrics (Prometheus format)                            │ │
│  │  — emit JSON logs to stdout (Docker log driver → Loki)            │ │
│  └───────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────┘
```

**Why Prometheus + Loki + Grafana (PLG):**

- **Lightweight** — runs on the same box; total ~3 GB RAM overhead.
- **Free / open source** — no license costs.
- **Standard** — every DevOps engineer knows this stack; rich ecosystem of exporters.
- **Docker-native** — Loki has a Docker log driver; Prometheus scrapes HTTP endpoints.
- **Composable** — add only what you need: Grafana-only is the simplest start; Prometheus
  adds metrics; Loki adds log aggregation.

**Alternative (simpler):** Skip Prometheus/Loki entirely and use **Grafana Cloud Free Tier**
(10k metrics, 50 GB logs, 50 GB traces — free). Push metrics/logs to Grafana Cloud via
Alloy agent. Zero local infrastructure for observability.

### E.3 Metrics to expose (application → Prometheus)

Each application container exposes `GET /metrics` (Prometheus text format). The FastAPI server
(RFC-062) can use `prometheus-fastapi-instrumentator` or a lightweight custom endpoint. Workers
expose metrics via a sidecar HTTP port or push to a Prometheus Pushgateway.

| Metric | Source | Type | Why it matters |
| --- | --- | --- | --- |
| **`queue_depth{queue}`** | Redis (or Postgres jobs) | Gauge | Backlog visibility; triggers Part D.7 graduation |
| **`queue_processing_time{queue}`** | Worker | Histogram | Identifies slow queues |
| **`episodes_processed_total{stage}`** | Worker | Counter | Throughput |
| **`episode_stage_duration_seconds{stage}`** | Worker (`workflow/metrics.py`) | Histogram | Bottleneck detection |
| **`model_load_time_seconds{model}`** | Worker | Histogram | Cold start impact |
| **`gpu_utilization_percent`** | `nvidia-smi` exporter or `pynvml` | Gauge | GPU saturation |
| **`gpu_memory_used_bytes`** | `nvidia-smi` exporter | Gauge | VRAM pressure |
| **`api_request_duration_seconds{path}`** | API (FastAPI middleware) | Histogram | API latency |
| **`api_requests_total{path,status}`** | API | Counter | Error rate |
| **`vector_index_size`** | API (`/api/index/stats`) | Gauge | Corpus growth |
| **`provider_tokens_total{provider,task}`** | Worker (`provider_metrics.py`) | Counter | API cost tracking |
| **`provider_errors_total{provider,error_type}`** | Worker | Counter | Provider reliability |
| **`provider_retry_total{provider}`** | Worker | Counter | Transient failure rate |
| **`postgres_connections_active`** | Postgres exporter | Gauge | DB health |
| **`redis_connected_clients`** | Redis exporter | Gauge | Queue health |

### E.4 Logging strategy (application → Loki)

**Today:** Application writes JSON logs to stdout when `--json-logs` is set. Docker captures
stdout via its log driver.

**Target:**

1. **All containers log to stdout** — Docker default. No file-based logging inside containers.
2. **Docker Compose log driver → Loki** — Loki's Docker log driver tags each log line with
   container name, service name, and compose project. Zero application changes.
3. **Structured fields** — JSON logs already include stage, episode, provider, duration. Add:
   - **`correlation_id`** — UUID generated at job creation (API → queue → worker); threaded
     through all log lines for that episode's processing. Enables tracing a single episode
     from API request to completed artifacts.
   - **`worker_id`** — which worker instance handled the job.
   - **`queue`** — which queue the job came from.
4. **Log levels** — `ERROR` for failures, `WARNING` for retries/degraded, `INFO` for stage
   transitions, `DEBUG` for model loading / detailed timings. Default: `INFO` in production.

**Loki query examples:**

```logql
{service="worker-gpu"} |= "transcription" | json | duration > 300
{service="worker-ml"} | json | level="ERROR"
{correlation_id="abc123"} -- trace one episode across all services
```

### E.5 Dashboards (Grafana)

Pre-built dashboards shipped as JSON in `deploy/grafana/dashboards/`:

| Dashboard | Panels | Purpose |
| --- | --- | --- |
| **Pipeline Overview** | Queue depths (all queues), episodes processed/hour, active workers, error rate | At-a-glance system health |
| **GPU & Worker Health** | GPU utilization, VRAM usage, worker-gpu/ml/io CPU and memory, model load times | Hardware saturation and bottleneck detection |
| **Episode Lifecycle** | Per-episode stage durations (waterfall chart), p50/p95/p99 processing times, failed episodes | Performance profiling |
| **API & Viewer** | Request latency by endpoint, error rate, active connections, search query latency | Frontend/API health |
| **Provider Costs** | Token usage by provider, estimated cost/day, error rates by provider, retries | API cost management |
| **Corpus Growth** | Total episodes, vector index size, GI/KG artifact counts, storage usage | Capacity planning |

### E.6 Per-service health checks and degradation model

Every container exposes a health endpoint that reports not just "up/down" but a **degradation
level** — enabling nuanced alerting and automated remediation (E.9).

#### E.6.1 Health endpoint contract

Each service exposes `GET /healthz` (or equivalent) returning structured JSON:

```json
{
  "status": "healthy",
  "degraded": false,
  "checks": {
    "queue_connection": {"status": "ok", "latency_ms": 2},
    "gpu_available": {"status": "ok", "vram_free_mb": 4200},
    "model_loaded": {"status": "ok", "model": "whisper-medium"},
    "disk_space": {"status": "warning", "free_gb": 12, "threshold_gb": 10}
  },
  "uptime_seconds": 84200,
  "version": "v2.7.1",
  "git_sha": "abc1234"
}
```

**Status levels:**

| Level | Meaning | Response |
| --- | --- | --- |
| **`healthy`** | All checks pass | No action |
| **`degraded`** | Service works but with reduced capability | Alert + investigate; AI agent may intervene |
| **`unhealthy`** | Service cannot process work | Alert + auto-restart; AI agent evaluates root cause |
| **`dead`** | No response to health probe (container unresponsive) | Docker/K8s restarts container automatically |

#### E.6.2 Per-service health checks

| Service | Health checks | Degradation examples |
| --- | --- | --- |
| **api** | Postgres reachable, Redis reachable, FAISS index loadable, response latency < 2s | Postgres slow (queries degraded); FAISS index missing (search disabled, graph-only mode) |
| **worker-gpu** | GPU available (`nvidia-smi`), VRAM free > model requirement, queue connection, model loadable | GPU at 95% VRAM (can't load larger model); CUDA error (falls back to CPU = severely degraded) |
| **worker-ml** | Sufficient RAM for models, queue connection, model loadable, CPU not saturated | RAM pressure (model loading slow); all models loaded but CPU at 100% (throughput degraded) |
| **worker-io** | Network connectivity, disk I/O responsive, queue connection, Postgres writable | Disk nearly full (projection fails); network flapping (download retries increasing) |
| **scheduler** | Postgres reachable, Redis reachable, can enqueue jobs, cursors advancing | Cursor stuck (feed not polling); enqueue failing (queue full or Redis down) |
| **postgres** | Connections available, replication lag (if applicable), WAL disk free | Connection pool exhausted; slow queries (missing index); disk 90% full |
| **redis** | Memory usage < 80%, connected clients < max, evictions = 0 | Memory near limit (queue performance degrades); high eviction rate (losing jobs) |

#### E.6.3 Docker Compose health checks

```yaml
services:
  worker-gpu:
    healthcheck:
      test: ["CMD", "python", "-c",
        "import requests; r=requests.get('http://localhost:9100/healthz'); r.raise_for_status(); assert r.json()['status'] != 'unhealthy'"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s  # GPU model loading takes time
    restart: unless-stopped

  worker-ml:
    healthcheck:
      test: ["CMD", "python", "-c",
        "import requests; r=requests.get('http://localhost:9101/healthz'); r.raise_for_status()"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 45s
    restart: unless-stopped
```

**Docker handles `dead`** — if a container fails its healthcheck `retries` times, Docker marks
it unhealthy and `restart: unless-stopped` restarts it. This covers basic crash recovery
without any AI agent involvement.

### E.7 Alerting

Grafana alerting (or Prometheus Alertmanager) fires on conditions that require attention.
Alerts are **tiered** to distinguish noise from emergencies:

#### E.7.1 Alert severity tiers

| Tier | Name | Response time | Notification | AI agent action |
| --- | --- | --- | --- | --- |
| **P1 — Critical** | System down or data loss risk | Immediate | Push notification + Slack + email | Auto-remediate if runbook exists (E.9) |
| **P2 — Warning** | Degraded but functional | Within 1 hour | Slack + email | Investigate and propose fix (E.9) |
| **P3 — Info** | Anomaly, no impact yet | Next business day | Email or dashboard annotation | Log for pattern analysis (E.9) |

#### E.7.2 Alert rules

| Alert | Condition | Severity | Human action | AI agent action (E.9) |
| --- | --- | --- | --- | --- |
| **Worker dead** | Health probe fails 3× in 2 min | **P1** | Check container, GPU | Restart container; if restart fails 3×, check logs for root cause |
| **Queue backlog** | `queue_depth{queue="transcribe"} > 20` for 15 min | **P2** | Scale worker or investigate | Analyze queue drain rate; if worker is healthy, adjust concurrency |
| **GPU OOM** | `gpu_memory_used_bytes / gpu_memory_total > 0.95` | **P1** | Check model loading | Switch to smaller Whisper model in config, redeploy worker |
| **Provider API down** | `provider_errors_total` spike + HTTP 5xx | **P2** | Check provider status page | Switch to fallback provider (ADR-026 per-capability selection) |
| **Provider rate-limited** | `provider_retry_total` spike + HTTP 429 | **P3** | Review API plan | Throttle concurrency for that provider; queue jobs slower |
| **Disk space low** | `node_filesystem_avail_bytes < 10 GB` | **P1** | Expand storage | Run cleanup script (old runs, temp files); alert if < 5 GB after cleanup |
| **API latency** | `api_request_duration_seconds{p95} > 5s` | **P2** | Check Postgres/index | Analyze slow query log; check FAISS index size; restart if connection leak |
| **Postgres connections** | `active_connections > max_connections * 0.8` | **P2** | Tune pool | Restart idle workers; increase pool size in config |
| **Redis memory** | `redis_memory_used > redis_maxmemory * 0.85` | **P2** | Review queue sizes | Purge completed jobs; increase maxmemory or add eviction policy |
| **Episode stuck** | Job in `transcribing` state > 120 min | **P2** | Check worker logs | Kill stuck job; re-enqueue; if pattern repeats, flag episode as problematic |
| **Model load failure** | Worker reports `model_loaded: failed` | **P1** | Check model cache | Re-download model; if cache corrupt, clear and reload; restart worker |

**Notification channels:** Email, Slack webhook, Pushover, or PagerDuty — configured per
deployment. The AI agent (E.9) receives the same alerts via webhook.

### E.8 Control plane (system-level management)

Beyond monitoring, the platform needs a **control plane** for operators (and AI agents) to
manage the system without SSH. This is a future extension of the RFC-062 server, not a
separate product.

| Capability | Implementation | Phase | AI-agent-actionable? |
| --- | --- | --- | --- |
| **Queue management** | View queue depths, pause/resume queues, retry failed jobs, inspect DLQ | Platform API routes (`/api/admin/queues`) | Yes — pause/resume, retry |
| **Worker management** | View active workers, current job per worker, restart worker (via Docker API) | Platform API + Docker socket | Yes — restart, scale |
| **Job inspection** | View job state machine per episode, re-enqueue stuck jobs, cancel jobs | Platform API routes (`/api/admin/jobs`) | Yes — re-enqueue, cancel |
| **System health** | CPU/RAM/GPU/disk per container, Postgres/Redis connection status | Grafana dashboards (E.5) + `/api/admin/health` | Yes — read-only diagnosis |
| **Configuration** | View/update runtime config (queue concurrency, model selection) without restart | Platform API + config reload mechanism | Yes — modify and apply |
| **Deployment** | Current image version, git SHA, deploy new version, rollback | `/api/admin/deploy` + Docker socket or Git-based deploy | Yes — deploy, rollback |
| **Log query** | Search structured logs by correlation ID, time range, severity, service | Loki API (proxied through admin) or direct Grafana | Yes — root cause analysis |
| **Metric query** | Query Prometheus for arbitrary metrics, time ranges, aggregations | Prometheus API (proxied through admin) | Yes — pattern detection |

**Phase E1 (with RFC-062):** Health endpoint only (`/api/health`).
**Phase E2 (with platform routes):** Admin routes under `/api/admin/` behind auth.
**Phase E3 (production):** Full control plane with Grafana dashboards and alerting.
**Phase E4 (AI-ops):** AI agent webhook integration; agent can call admin API (E.9).

### E.9 AI-agent-as-on-call (actionable observability)

**Vision:** An AI agent acts as the **P1 on-rotation engineer**. It receives alerts, reads
logs and metrics, diagnoses root causes, and executes remediation — either automatically for
known runbook scenarios or by proposing changes for human approval on novel issues.

This is not a vague "AI does stuff" aspiration — it's a concrete architecture with defined
boundaries, a code-as-config contract the agent can manipulate, and safety guardrails.

#### E.9.1 Architecture

```text
┌────────────────────────────────────────────────────────────────────────────┐
│                         AI Agent Loop                                      │
│                                                                            │
│  ┌──────────────┐     ┌──────────────────┐     ┌────────────────────────┐  │
│  │  Alert        │────►│  Observe          │────►│  Diagnose              │  │
│  │  (webhook)    │     │  - Query Loki     │     │  - LLM reasons over   │  │
│  │               │     │  - Query Prom     │     │    logs + metrics +    │  │
│  │  Source:      │     │  - Read health    │     │    health checks       │  │
│  │  Grafana /    │     │    endpoints      │     │  - Match against       │  │
│  │  Alertmanager │     │  - Read configs   │     │    runbook library     │  │
│  └──────────────┘     └──────────────────┘     └──────────┬─────────────┘  │
│                                                           │                 │
│                        ┌──────────────────────────────────┘                 │
│                        ▼                                                    │
│           ┌──────────────────────────┐                                      │
│           │  Decide + Act             │                                      │
│           │                          │                                      │
│           │  Auto-remediate:         │  Propose to human:                   │
│           │  - Restart container     │  - Config change (PR)                │
│           │  - Switch model/provider │  - Architecture change               │
│           │  - Run cleanup script    │  - Novel failure pattern             │
│           │  - Retry failed jobs     │  - Cost/quality trade-off            │
│           │  - Adjust concurrency    │                                      │
│           └──────────┬───────────────┘                                      │
│                      │                                                      │
│                      ▼                                                      │
│           ┌──────────────────────────┐                                      │
│           │  Execute                  │                                      │
│           │  - Call admin API (E.8)   │                                      │
│           │  - Commit config to Git   │                                      │
│           │  - Trigger deploy (F.3)   │                                      │
│           │  - Post incident summary  │                                      │
│           └──────────────────────────┘                                      │
└────────────────────────────────────────────────────────────────────────────┘
```

#### E.9.2 The agent's toolkit (what it can read and write)

The AI agent is an **API consumer** of the platform. It does not have SSH access. Its
capabilities are bounded by what the admin API (E.8) exposes:

**Read (observe + diagnose):**

| Source | API | What the agent sees |
| --- | --- | --- |
| Alerts | Grafana webhook → agent endpoint | Alert name, severity, affected service, metric values |
| Logs | Loki API (`/loki/api/v1/query_range`) | Structured JSON logs, filterable by service, level, correlation ID, time |
| Metrics | Prometheus API (`/api/v1/query`) | All metrics from E.3 — queue depth, GPU, latency, errors, etc. |
| Health | `/api/admin/health` (all services) | Per-service health check results (E.6) |
| Jobs | `/api/admin/jobs` | Job states, stuck jobs, failed jobs with error messages |
| Queues | `/api/admin/queues` | Queue depths, processing rates, DLQ contents |
| Config | `/api/admin/config` | Current runtime configuration (models, concurrency, providers) |
| Deploy | `/api/admin/deploy` | Current image version, git SHA, container states |

**Write (act):**

| Action | API | Safety level |
| --- | --- | --- |
| Restart container | `POST /api/admin/workers/{id}/restart` | **Auto** — safe, idempotent |
| Retry failed jobs | `POST /api/admin/jobs/retry` | **Auto** — safe, idempotent |
| Pause/resume queue | `POST /api/admin/queues/{name}/pause` | **Auto** — reversible |
| Adjust concurrency | `PUT /api/admin/config/concurrency` | **Auto** — reversible, bounded |
| Switch provider | `PUT /api/admin/config/provider/{capability}` | **Auto** — uses existing ADR-026 fallback |
| Switch Whisper model | `PUT /api/admin/config/whisper_model` | **Approve** — quality trade-off |
| Run cleanup script | `POST /api/admin/maintenance/cleanup` | **Auto** — safe, bounded |
| Deploy new version | `POST /api/admin/deploy` | **Approve** — requires human confirmation |
| Rollback | `POST /api/admin/deploy/rollback` | **Auto** for P1 (last known good); **Approve** otherwise |
| Commit config to Git | GitHub API → PR | **Approve** — always via PR, never direct push |

#### E.9.3 Safety guardrails

The AI agent operates under strict constraints:

1. **Auto vs Approve:** Each action is classified as "Auto" (agent executes immediately) or
   "Approve" (agent creates a proposal — Slack message, GitHub PR, or admin UI notification —
   and waits for human confirmation). Classification is configurable per deployment.

2. **Rate limiting:** Maximum 3 auto-remediations per alert per hour. If the same alert fires
   and the agent has already remediated 3 times, it escalates to human with a summary of what
   it tried.

3. **Blast radius:** Agent can only affect the podcast_scraper deployment. No access to host
   OS, other services, or infrastructure outside the Compose/K8s boundary.

4. **Audit trail:** Every agent action is logged to a dedicated `agent-actions` log stream
   (in Loki) with: timestamp, alert trigger, diagnosis reasoning, action taken, outcome.
   Humans can review the full decision chain.

5. **Kill switch:** `PUT /api/admin/agent/enabled` — human can disable the agent at any time.
   Agent reverts to observe-only mode and sends all alerts to human channels.

6. **Runbook-only auto-remediation:** The agent only auto-remediates when it matches a
   **known runbook** (E.9.4). Novel failure patterns always escalate to human.

7. **Dry-run mode:** Agent can run in "shadow mode" — it diagnoses and proposes actions but
   never executes them. Useful for building trust before enabling auto-remediation.

#### E.9.4 Runbook library (code-as-config)

Runbooks are **structured YAML files** in `deploy/runbooks/` that the AI agent matches against
alerts. Each runbook defines: trigger conditions, diagnostic steps, remediation actions, and
escalation criteria.

```text
deploy/runbooks/
├── worker-dead.yml
├── gpu-oom.yml
├── queue-backlog.yml
├── provider-down.yml
├── disk-full.yml
├── episode-stuck.yml
├── model-load-failure.yml
├── postgres-connections.yml
├── api-latency.yml
└── _template.yml
```

**Example runbook (`deploy/runbooks/gpu-oom.yml`):**

```yaml
name: GPU OOM Recovery
trigger:
  alert: gpu_oom
  severity: P1
  service: worker-gpu

diagnose:
  - query_prometheus: "gpu_memory_used_bytes{service='worker-gpu'}"
  - query_loki: "{service='worker-gpu'} |= 'CUDA out of memory' | json"
  - check_health: worker-gpu
  - check_config: whisper_model  # what model is currently configured?

remediate:
  - action: switch_whisper_model
    from: large-v3
    to: medium
    safety: auto  # VRAM reduction is safe; quality trade-off is known
    reason: "large-v3 requires ~10 GB VRAM; medium requires ~5 GB"

  - action: restart_worker
    service: worker-gpu
    safety: auto
    wait_seconds: 30  # wait for model to reload

  - action: verify_health
    service: worker-gpu
    expect: healthy
    timeout_seconds: 120

escalate_if:
  - health_still_unhealthy_after: 300  # 5 min after remediation
  - remediation_failed: true
  - same_alert_in_last_hour: 3  # recurring despite fix

notify:
  on_auto_remediate: slack  # "Agent switched worker-gpu to whisper-medium due to OOM"
  on_escalate: slack + email + pushover
```

**The agent's decision process:**

1. Alert arrives via webhook.
2. Agent matches alert to runbook(s) by `trigger.alert` + `trigger.service`.
3. Agent executes `diagnose` steps — queries Loki, Prometheus, health endpoints.
4. Agent checks `remediate` actions: if `safety: auto`, execute immediately; if
   `safety: approve`, create proposal and wait.
5. After remediation, agent runs `verify_health`. If still unhealthy, check
   `escalate_if` conditions.
6. Agent posts incident summary to notification channel with full reasoning chain.

**No matching runbook?** Agent collects diagnostics (logs, metrics, health) and posts a
**structured incident report** to Slack/email for human analysis. It does NOT guess.

#### E.9.5 Config-as-code for agent-driven changes

For changes that need to survive container restarts (model selection, provider fallback,
concurrency), the agent follows the same workflow a **human on-call engineer** would — just
faster. The approach depends on the deployment model:

**Two paths — matching what a human would do:**

| Change type | Human workflow | Agent workflow | Deploy mechanism |
| --- | --- | --- | --- |
| **Runtime config** (env vars, concurrency) | SSH → edit `.env` → `docker compose up -d` | Call `/api/admin/config` to change value → service restarts | Admin API (E.8) → compose restart. No Git needed. Takes effect immediately. |
| **Persistent config** (model selection, resource limits, Prometheus config) | Edit file → commit → push → deploy | Create branch → commit → PR → merge → deploy | Git-based (see below). Takes effect after deploy. |
| **Emergency** (service down, OOM) | Restart via `docker compose restart` | Call `/api/admin/restart/{service}` | Admin API. Immediate. No Git. |

**Git-based workflow (persistent changes only):**

1. Agent reads current config from `/api/admin/config`.
2. Agent modifies the relevant value (e.g., `whisper_model: medium`).
3. Agent creates a **Git commit** on a branch (e.g., `agent/gpu-oom-recovery-2026-04-03`).
4. Agent opens a **GitHub PR** with the change, tagged `agent-remediation`.
5. For `safety: auto` changes, PR is **auto-merged** if CI passes.
6. For `safety: approve` changes, PR waits for human review.
7. **After merge, deployment depends on the deploy model (F.3):**
   - **Manual deploy (F.3, option 1):** Agent posts to Slack: "PR merged. Run
     `./deploy/scripts/deploy.sh` to apply." Human runs it.
   - **Watchtower (F.3, option 2):** Watchtower auto-pulls the new image. Agent monitors
     health endpoints to confirm rollout.
   - **GitHub Actions CD (F.3, option 3):** Merge triggers the CD workflow automatically.
     Agent monitors the workflow run status.

**Key principle:** The agent does not invent a parallel deploy path. It uses the same
mechanisms the team already has. If the team uses `deploy.sh`, the agent tells someone to
run `deploy.sh`. If the team uses Watchtower, the agent lets Watchtower do its job and
monitors the result.

**Runtime vs persistent — when to use which:**

- **Runtime** (admin API): Quick fixes that don't need to survive a full redeploy. If the
  deploy rebuilds from Git, runtime-only changes are lost. Use for: emergency restarts,
  temporary concurrency adjustments, provider failover.
- **Persistent** (Git): Changes that should be the new default going forward. Use for:
  model downgrades, resource limit adjustments, new alert rules.

**Config files the agent can modify (persistent path):**

| File | What | Examples of agent changes |
| --- | --- | --- |
| `deploy/.env` | Runtime environment | `WHISPER_MODEL=medium`, `WORKER_CONCURRENCY=2` |
| `deploy/docker-compose.prod.yml` | Service definitions | Scale replicas, adjust resource limits |
| `deploy/prometheus/prometheus.yml` | Scrape config | Add/remove scrape targets |
| `deploy/runbooks/*.yml` | Runbook definitions | Tune thresholds based on observed patterns |

**What the agent must NEVER modify:**

- Application source code (`src/`)
- Test code (`tests/`)
- Pipeline logic
- Database schema or migrations
- Secrets (API keys, passwords)

#### E.9.6 Incident lifecycle (with AI agent)

```text
1. DETECT    Alert fires (Grafana → webhook → agent)
             ↓
2. OBSERVE   Agent queries logs (Loki), metrics (Prometheus),
             health endpoints (E.6), job states (E.8)
             ↓
3. DIAGNOSE  Agent matches to runbook; LLM reasons over collected data
             ↓
4. DECIDE    Known runbook + auto safety → execute
             Known runbook + approve safety → propose (PR/Slack)
             Unknown pattern → escalate with diagnostic bundle
             ↓
5. ACT       Execute via admin API (E.8) or Git commit (E.9.5)
             ↓
6. VERIFY    Re-check health endpoints; confirm alert resolves
             ↓
7. REPORT    Post incident summary:
             - What triggered the alert
             - What the agent observed
             - What action it took (or proposed)
             - Current system state
             - Whether the issue is resolved
             ↓
8. LEARN     Log full decision chain to `agent-actions` stream
             (enables periodic human review of agent judgment)
```

#### E.9.7 Implementation approach

The AI agent is a **lightweight service** — not a complex ML system:

| Component | Technology | Notes |
| --- | --- | --- |
| **Webhook receiver** | Python (FastAPI or Flask) | Receives Grafana/Alertmanager webhooks |
| **LLM reasoning** | OpenAI API / local Ollama | Processes logs + metrics into diagnosis; model choice depends on deployment budget |
| **Runbook engine** | YAML loader + decision tree | Matches alerts to runbooks; no ML needed for matching |
| **Admin API client** | Python `httpx` | Calls platform admin API (E.8) |
| **Git client** | `pygit2` or subprocess `git` | Creates branches, commits, PRs via GitHub API |
| **State** | SQLite or Postgres table | Tracks active incidents, remediation history, rate limits |

**Container:** `agent` service in Compose, ~1 CPU, 512 MB RAM. Subscribes to alert webhooks.
No GPU needed (uses API-based LLM or small local model for reasoning).

**Phasing:**

| Phase | Capability | Agent behavior |
| --- | --- | --- |
| **E4a: Shadow mode** | Receives alerts, diagnoses, proposes — but executes nothing | Build trust; human reviews agent proposals |
| **E4b: Auto-restart** | Auto-remediates container restarts and job retries only | Lowest-risk auto actions |
| **E4c: Config changes** | Auto-switches models/providers via admin API; PRs for persistent config | Wider remediation scope |
| **E4d: Full agent** | All runbook actions; learns patterns from incident history | Production on-call agent |

### E.10 Observability graduation path

Aligned with the unified graduation table in D.7. Each deployment tier has a minimum
observability tier that must be in place before graduating.

| Stage | What to add | Topology tier (D.7) | Cost / effort |
| --- | --- | --- | --- |
| **v0: CLI** | `--json-logs` + per-run `metrics.json` files (exists today) | v0: CLI | Zero — already done |
| **v1: Grafana + health** | Add `grafana` container; import dashboards; per-service health endpoints (E.6); Docker healthchecks | v1: Simple Compose | 1 container, ~512 MB, 2–3 days |
| **v2: PLG stack + alerting** | Add Prometheus + Loki; `/metrics` endpoints; Loki Docker log driver; alert rules (E.7); notification channels | v2: Split workers | 3 containers, ~3 GB, 2–3 days |
| **v3: Control plane** | Admin API routes (E.8); queue/worker/job management; auth (A.12 stage 2+) | v2: Split workers | Application code, 3–5 days |
| **v4: AI agent (shadow)** | Webhook receiver + runbook engine + LLM reasoning; observe-only | v2–v3 | 1 new container, 3–5 days |
| **v5: AI agent (active)** | Auto-remediation for safe actions; runtime + Git-based config changes (E.9.5) | v3: Multi-GPU/SaaS | Runbook library, 2–3 days |

**Invariant:** E.v(N) must be operational before D.v(N) deploys. Health checks (E.v1)
before any Compose deployment. PLG + alerting (E.v2) before splitting workers.

---

## Part F. Deployment lifecycle and configuration management

**Depends on:** Part B (Compose topology), Part D (container specs), Part E (observability).
**Motivated by:** The project has Docker images, CI workflows, and a Compose file — but no
defined path from "code merged on GitHub" to "running in production with the new version." This
part closes that loop: build → release → deploy → configure → restart → rollback.

### F.1 What exists today (audit)

| Capability | Status | Where |
| --- | --- | --- |
| **Docker image build** | CI builds and tests Docker image on PR/push | `.github/workflows/docker.yml`, Makefile `docker-*` targets |
| **Docker Compose** | Two compose files for local/dev use | `docker-compose.yml`, `docker-compose.llm-only.yml` |
| **Process management** | Supervisor support in container for restart/log | `docker/entrypoint.sh`, `docker/supervisord.conf` |
| **Docs deployment** | MkDocs to GitHub Pages on main push | `.github/workflows/docs.yml` |
| **Image registry** | Not configured — no push to GHCR/DockerHub | Gap |
| **Deployment automation** | Not implemented — no CD pipeline | Gap |
| **Configuration management** | Config file + env vars; no secrets management | Gap |
| **Rollback** | Not defined | Gap |
| **Infrastructure-as-code** | Not implemented — no Terraform/Ansible | Gap |

### F.2 Deployment pipeline (target)

```text
Developer → Push to branch → PR
                ↓
         GitHub Actions CI
         ├── Lint + type check
         ├── Unit + integration + E2E tests
         ├── Docker build + docker-test
         ├── UI build + UI tests (Vitest + Playwright)
         └── Security scan (Snyk, CodeQL)
                ↓
         Merge to main
                ↓
         Build & push Docker image → GitHub Container Registry (GHCR)
         Tag: git SHA + semver (e.g. ghcr.io/chipi/podcast_scraper:v2.7.0)
                ↓
         Deploy (one of):
         ├── [A] Compose: SSH + docker compose pull + up -d (Watchtower or manual)
         ├── [B] K8s: kubectl apply / Helm upgrade / ArgoCD sync
         └── [C] PaaS: fly deploy / render deploy / cloud run deploy
```

### F.3 Compose deployment (recommended v1)

For a single-host deployment (home server, VPS, Hetzner dedicated), Docker Compose is the
right tool. Here is the full lifecycle:

#### F.3.1 Image registry

Push images to **GitHub Container Registry (GHCR)** on every merge to `main`:

```yaml
# .github/workflows/docker.yml (addition)
- name: Push to GHCR
  if: github.ref == 'refs/heads/main'
  run: |
    echo ${{ secrets.GITHUB_TOKEN }} | docker login ghcr.io -u ${{ github.actor }} --password-stdin
    docker tag podcast_scraper:latest ghcr.io/${{ github.repository }}:${{ github.sha }}
    docker tag podcast_scraper:latest ghcr.io/${{ github.repository }}:latest
    docker push ghcr.io/${{ github.repository }}:${{ github.sha }}
    docker push ghcr.io/${{ github.repository }}:latest
```

Semantic versioning tags (`v2.7.0`) are applied via GitHub Releases.

#### F.3.2 Production Compose file

The production `docker-compose.prod.yml` extends the dev compose with production concerns:

```text
deploy/
├── docker-compose.prod.yml        # Production Compose (extends base)
├── .env.example                   # Environment variable template
├── caddy/
│   └── Caddyfile                  # Reverse proxy config (auto-HTTPS if public)
├── grafana/
│   ├── provisioning/
│   │   ├── datasources.yml        # Prometheus + Loki auto-configured
│   │   └── dashboards.yml         # Dashboard auto-import
│   └── dashboards/
│       ├── pipeline-overview.json
│       ├── gpu-health.json
│       ├── episode-lifecycle.json
│       ├── api-viewer.json
│       └── provider-costs.json
├── prometheus/
│   └── prometheus.yml             # Scrape config (api, workers, postgres, redis, node)
├── loki/
│   └── loki-config.yml            # Loki config (local storage, retention)
└── scripts/
    ├── deploy.sh                  # Pull + up -d + health check + notify
    ├── rollback.sh                # Roll back to previous image tag
    ├── backup-db.sh               # Postgres pg_dump to backup volume
    └── restore-db.sh              # Restore from backup
```

#### F.3.3 Deploy script (`deploy/scripts/deploy.sh`)

Automated deployment for a Compose host:

```bash
#!/bin/bash
set -euo pipefail

COMPOSE_FILE="deploy/docker-compose.prod.yml"
IMAGE_TAG="${1:-latest}"

# Pull new images
docker compose -f "$COMPOSE_FILE" pull

# Save current image SHAs for rollback
docker compose -f "$COMPOSE_FILE" images --format json > /tmp/pre-deploy-images.json

# Rolling restart: infrastructure first, then workers, then API
docker compose -f "$COMPOSE_FILE" up -d postgres redis
sleep 5  # wait for DB/Redis ready

docker compose -f "$COMPOSE_FILE" up -d worker-gpu worker-ml worker-io scheduler
sleep 10  # wait for workers to register with queue

docker compose -f "$COMPOSE_FILE" up -d api
sleep 5

# Health check
if curl -sf http://localhost:8100/api/health > /dev/null; then
  echo "Deploy successful: $IMAGE_TAG"
else
  echo "Health check failed — rolling back"
  exec deploy/scripts/rollback.sh
fi
```

#### F.3.4 Automated deployment options

| Approach | How | Complexity |
| --- | --- | --- |
| **Manual** | SSH → `deploy/scripts/deploy.sh` | Simplest; operator-initiated |
| **Watchtower** | Container that polls GHCR and auto-updates | Low; add `watchtower` to Compose |
| **GitHub Actions CD** | SSH action after image push; runs `deploy.sh` | Medium; needs SSH key in secrets |
| **Webhook** | Lightweight receiver on host; triggered by GitHub webhook on release | Medium; self-hosted webhook listener |

**Recommendation:** Start with **manual** (`deploy.sh`). Graduate to **GitHub Actions CD**
when deploying more than once per week.

### F.4 Kubernetes deployment (when to consider)

| Factor | Compose | Kubernetes |
| --- | --- | --- |
| **Hosts** | Single host | Multi-node cluster |
| **Scaling** | Manual (`docker compose up --scale worker-gpu=2`) | Auto-scaling (HPA on queue depth) |
| **GPU scheduling** | `deploy.resources.reservations.devices` in Compose | NVIDIA device plugin + `nvidia.com/gpu` resource |
| **Rolling updates** | Manual ordering in `deploy.sh` | Built-in deployment strategy |
| **Health checks** | Compose `healthcheck` directive | Readiness + liveness probes |
| **Secrets** | `.env` file or Docker secrets | K8s Secrets, Sealed Secrets, or Vault |
| **Observability** | PLG stack in same Compose | kube-prometheus-stack (Helm chart) |
| **Complexity** | Low — one host, one `docker compose up` | High — cluster management, networking, storage classes |
| **Cost** | €0 (self-managed host) | Managed K8s: €50–150/mo (EKS/GKE/AKS control plane) + nodes |

**When to graduate from Compose to K8s:**

- **Multi-node:** > 1 physical host (GPUs on separate machines).
- **Auto-scaling:** Queue depth triggers automatic worker scaling.
- **HA:** API must survive a single node failure.
- **Team size:** > 2 operators need standard deployment tooling (Helm, ArgoCD).

**Recommendation:** Stay on **Compose until > 50 feeds or multi-node.** The Part D.5 topology
runs on Compose with full resource limits, health checks, and restart policies. K8s adds
value only when you need what Compose can't do (auto-scaling, multi-node GPU scheduling).

#### F.4.1 K8s resource mapping (when ready)

If/when graduating to K8s, the Compose topology maps cleanly:

| Compose service | K8s resource | Notes |
| --- | --- | --- |
| `api` | Deployment + Service + Ingress | HPA on CPU/request count |
| `worker-gpu` | Deployment (GPU node pool) | `nvidia.com/gpu: 1` resource request; `replicas: 1` per GPU |
| `worker-ml` | Deployment (CPU node pool) | HPA on queue depth |
| `worker-io` | Deployment (CPU node pool) | HPA on queue depth |
| `scheduler` | Deployment (1 replica) | Leader election if HA |
| `postgres` | StatefulSet or managed DB (RDS/Cloud SQL) | Managed DB preferred for production |
| `redis` | StatefulSet or managed Redis (ElastiCache) | Managed Redis preferred for production |
| `caddy` | Ingress controller (nginx-ingress, Traefik) | K8s handles ingress natively |
| `prometheus + loki + grafana` | kube-prometheus-stack Helm chart | Replaces self-hosted PLG |

**Helm chart structure** (if packaging for K8s):

```text
charts/podcast-scraper/
├── Chart.yaml
├── values.yaml                  # Default values (image, replicas, resources, env)
├── templates/
│   ├── api-deployment.yaml
│   ├── api-service.yaml
│   ├── api-ingress.yaml
│   ├── worker-gpu-deployment.yaml
│   ├── worker-ml-deployment.yaml
│   ├── worker-io-deployment.yaml
│   ├── scheduler-deployment.yaml
│   ├── configmap.yaml           # Non-secret config
│   ├── secret.yaml              # API keys, DB password
│   └── _helpers.tpl
└── values-production.yaml       # Production overrides
```

### F.5 Configuration management

#### F.5.1 Configuration layers

The system needs three layers of configuration, from most to least dynamic:

| Layer | What | Where | Change frequency |
| --- | --- | --- | --- |
| **Infrastructure** | Host specs, network, storage volumes, GPU allocation | Compose file or K8s manifests | Rarely (hardware change) |
| **Application** | Provider keys, model selection, queue concurrency, feature flags | `.env` file → container env vars | Occasionally (per release or tuning) |
| **Runtime** | Pipeline config per job (`Config` model) | Postgres catalog or API request | Per job / per feed |

#### F.5.2 Secrets management

| Secret | Current | Target |
| --- | --- | --- |
| **Provider API keys** (OpenAI, Gemini, etc.) | Env vars on host | `.env` file (Compose) or K8s Secrets |
| **Postgres password** | Hardcoded in compose or env | Docker secrets (Compose) or K8s Secrets |
| **Redis password** | Often unset (local) | Docker secrets or K8s Secrets |
| **GHCR token** | GitHub Actions secret | CI-only; not on host |
| **Grafana admin password** | Default `admin` | `.env` or Docker secrets |

**Principle:** No secrets in Git. Ever. `.env.example` documents the shape; `.env` is
`.gitignore`-ed. For K8s, use Sealed Secrets or external secret operator (Vault, AWS SM).

#### F.5.3 Environment variable contract

Define a clear contract for all environment variables the system reads:

```bash
# Infrastructure
COMPOSE_PROJECT_NAME=podcast-scraper
DATA_DIR=/data                      # Artifact root (shared volume)
MODEL_CACHE_DIR=/models             # HuggingFace / Whisper model cache
FAISS_INDEX_DIR=/data/index         # Vector index directory

# Application
PODCAST_SCRAPER_CONFIG=/app/config.yaml  # Default pipeline config
LOG_LEVEL=INFO                           # INFO / DEBUG / WARNING
JSON_LOGS=true                           # Structured JSON logging

# Server (RFC-062)
SERVER_PORT=8100
ENABLE_VIEWER=true
ENABLE_PLATFORM=false

# Workers
QUEUES=transcribe                   # Queue subscription (per worker)
WORKER_CONCURRENCY=1                # Jobs processed concurrently
GPU_DEVICE=0                        # CUDA device index

# Provider API keys
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=...
MISTRAL_API_KEY=...

# Database
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=podcast_scraper
POSTGRES_USER=podcast_scraper
POSTGRES_PASSWORD=...               # Secret — not in Git

# Redis
REDIS_URL=redis://redis:6379/0
```

### F.6 Service management and restarts

#### F.6.1 Compose restart policies

```yaml
services:
  api:
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8100/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s

  worker-gpu:
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  postgres:
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U podcast_scraper"]
      interval: 10s
      timeout: 5s
      retries: 5
```

**Supervisor (inside container):** The existing `docker/supervisord.conf` handles process
restart within a container. For Compose with `restart: unless-stopped`, supervisor is
redundant for single-process containers. Keep supervisor only if running multiple processes
in one container (not recommended for Part D.5 topology).

#### F.6.2 Graceful shutdown

Workers must handle `SIGTERM` gracefully:

1. Stop accepting new jobs from the queue.
2. Finish the current job (or checkpoint and re-enqueue).
3. Flush metrics and logs.
4. Exit cleanly.

This is critical for `worker-gpu` — a Whisper transcription may take 10+ minutes. Docker's
default `stop_grace_period` is 10 seconds; set it to **120 seconds** for GPU workers:

```yaml
  worker-gpu:
    stop_grace_period: 120s
```

### F.7 Rollback strategy

| Scenario | Rollback method | Recovery time |
| --- | --- | --- |
| **Bad image** | `deploy/scripts/rollback.sh` — restore previous image tag, `docker compose up -d` | ~2 minutes |
| **DB migration failure** | Reverse migration script; restore from `backup-db.sh` output | ~5–10 minutes |
| **Config error** | Edit `.env`, `docker compose up -d` (restarts affected services) | ~1 minute |
| **Data corruption** | Restore Postgres from backup; re-process affected episodes | Variable |

**`deploy/scripts/rollback.sh`** reads `/tmp/pre-deploy-images.json` (saved by `deploy.sh`)
and pins all services to the previous image digest.

### F.8 Backup strategy

| What | How | Frequency | Retention |
| --- | --- | --- | --- |
| **Postgres** | `pg_dump` via `backup-db.sh` to backup volume or object storage | Daily | 30 days |
| **FAISS index** | Copy index directory to backup | Daily | 7 days (rebuildable from artifacts) |
| **Artifacts** (gi.json, kg.json, transcripts) | Rsync to backup volume or object storage | Daily (incremental) | Indefinite |
| **Config** | Git (`.env.example` template) + encrypted backup of `.env` | On change | Indefinite |
| **Grafana dashboards** | Git (`deploy/grafana/dashboards/`) | On change | Indefinite |

### F.9 Deployment file tree (complete)

```text
deploy/
├── docker-compose.prod.yml
├── .env.example
├── caddy/
│   └── Caddyfile
├── grafana/
│   ├── provisioning/
│   │   ├── datasources.yml
│   │   └── dashboards.yml
│   └── dashboards/
│       ├── pipeline-overview.json
│       ├── gpu-health.json
│       ├── episode-lifecycle.json
│       ├── api-viewer.json
│       ├── provider-costs.json
│       └── corpus-growth.json
├── prometheus/
│   └── prometheus.yml
├── loki/
│   └── loki-config.yml
├── scripts/
│   ├── deploy.sh
│   ├── rollback.sh
│   ├── backup-db.sh
│   ├── restore-db.sh
│   └── health-check.sh
└── k8s/                           # Future — only when graduating to K8s
    └── charts/
        └── podcast-scraper/
            ├── Chart.yaml
            ├── values.yaml
            └── templates/
```

### F.10 Deployment graduation path

Aligned with the unified graduation table in D.7.

| Stage | Topology tier (D.7) | Observability tier (E.10) | Deployment method | CI/CD | Config management | Auth (A.12) |
| --- | --- | --- | --- | --- | --- | --- |
| **v0: Dev** | v0: CLI | v0: CLI | `make serve` / `docker compose up` | `make ci-fast` locally | `.env` file | None |
| **v1: Single host** | v1: Simple Compose | v1: Grafana + health | `deploy.sh` via SSH | CI builds + pushes image → manual deploy | `.env` + Docker secrets | API key |
| **v2: Automated** | v2: Split workers | v2: PLG + alerting | GitHub Actions CD → SSH → `deploy.sh` | Full CI → auto-deploy on main | `.env` + Watchtower or webhook | API key |
| **v3: K8s / SaaS** | v3: Multi-GPU/SaaS | v4–v5: AI agent | ArgoCD / Flux syncing from Git | GitOps — merge triggers deploy | Helm values + K8s Secrets + Sealed Secrets | JWT + multi-tenant |

---

## Related documents (unified)

### Architecture & guides

- [Architecture](ARCHITECTURE.md) — one pipeline, one `Config`, module map
- [Docker Service Guide](../guides/DOCKER_SERVICE_GUIDE.md) — current one-shot service mode
- [CI/CD Overview](../ci/index.md) — workflows, metrics, quality trends
- [Non-Functional Requirements](NON_FUNCTIONAL_REQUIREMENTS.md) — observability, performance

### Product requirements (PRDs)

- [PRD-005: Episode summarization](../prd/PRD-005-episode-summarization.md)
- [PRD-007: AI quality & experiment platform](../prd/PRD-007-ai-quality-experiment-platform.md)
- [PRD-017: Grounded Insight Layer](../prd/PRD-017-grounded-insight-layer.md)
- [PRD-018: Database projection](../prd/PRD-018-database-projection-gil-kg.md)
- [PRD-019: Knowledge Graph Layer](../prd/PRD-019-knowledge-graph-layer.md)
- [PRD-020: Audio-based speaker diarization](../prd/PRD-020-audio-speaker-diarization.md)
- [PRD-021: Semantic corpus search](../prd/PRD-021-semantic-corpus-search.md)
- [PRD-015: Engineering governance & productivity](../prd/PRD-015-engineering-governance-productivity.md)
- [PRD-016: Operational observability & pipeline intelligence](../prd/PRD-016-operational-observability-pipeline-intelligence.md)

### Technical designs (RFCs)

- [RFC-025: Test metrics and health tracking](../rfc/RFC-025-test-metrics-and-health-tracking.md)
- [RFC-026: Metrics consumption and dashboards](../rfc/RFC-026-metrics-consumption-and-dashboards.md)
- [RFC-027: Pipeline metrics improvements](../rfc/RFC-027-pipeline-metrics-improvements.md)
- [RFC-028: ML model preloading and caching](../rfc/RFC-028-ml-model-preloading-and-caching.md)
- [RFC-040: Audio preprocessing pipeline](../rfc/RFC-040-audio-preprocessing-pipeline.md)
- [RFC-041: ML benchmarking framework](../rfc/RFC-041-podcast-ml-benchmarking-framework.md)
- [RFC-042: Hybrid summarization pipeline](../rfc/RFC-042-hybrid-summarization-pipeline.md)
- [RFC-044: Model registry](../rfc/RFC-044-model-registry.md)
- [RFC-049: GIL core](../rfc/RFC-049-grounded-insight-layer-core.md)
- [RFC-050: GIL use cases](../rfc/RFC-050-grounded-insight-layer-use-cases.md)
- [RFC-051: Database projection](../rfc/RFC-051-database-projection-gil-kg.md)
- [RFC-053: Adaptive summarization routing](../rfc/RFC-053-adaptive-summarization-routing.md)
- [RFC-055: KG core](../rfc/RFC-055-knowledge-graph-layer-core.md)
- [RFC-056: KG use cases](../rfc/RFC-056-knowledge-graph-layer-use-cases.md)
- [RFC-057: AutoResearch optimization loop](../rfc/RFC-057-autoresearch-optimization-loop.md)
- [RFC-058: Audio speaker diarization](../rfc/RFC-058-audio-speaker-diarization.md)
- [RFC-059: Speaker detection refactor](../rfc/RFC-059-speaker-detection-refactor-test-audio.md)
- [RFC-060: Diarization-aware commercial cleaning](../rfc/RFC-060-diarization-aware-commercial-cleaning.md)
- [RFC-061: Semantic corpus search](../rfc/RFC-061-semantic-corpus-search.md)
- [RFC-062: GI/KG Viewer v2 & server architecture](../rfc/RFC-062-gi-kg-viewer-v2.md)
- [RFC-039: Development workflow with git worktrees and CI](../rfc/RFC-039-development-workflow-worktrees-ci.md)
- [RFC-043: Automated metrics alerts](../rfc/RFC-043-automated-metrics-alerts.md)

### Architecture decisions (ADRs)

- [ADR-005: Lazy ML dependency loading](../adr/ADR-005-lazy-ml-dependency-loading.md)
- [ADR-007: Universal episode identity](../adr/ADR-007-universal-episode-identity.md)
- [ADR-036: Standardized pre-provider audio stage](../adr/ADR-036-standardized-pre-provider-audio-stage.md)
- [ADR-046: MPS exclusive mode for Apple Silicon](../adr/ADR-046-mps-exclusive-mode-apple-silicon.md)
- [ADR-054: Relational Postgres projection](../adr/ADR-054-relational-postgres-projection-for-gil-and-kg.md)
- [ADR-060: VectorStore protocol with backend abstraction](../adr/ADR-060-vectorstore-protocol-with-backend-abstraction.md)
- [ADR-061: FAISS Phase 1](../adr/ADR-061-faiss-phase-1-with-post-filter-metadata.md)
- [ADR-062: Sentence-boundary transcript chunking](../adr/ADR-062-sentence-boundary-transcript-chunking.md)
- [ADR-063: Transparent semantic upgrade for gi explore](../adr/ADR-063-transparent-semantic-upgrade-for-gi-explore.md)
- [ADR-064: Canonical server layer with feature-flagged routes](../adr/ADR-064-canonical-server-layer-with-feature-flagged-routes.md)
- [ADR-065: Vue 3 + Vite + Cytoscape frontend stack](../adr/ADR-065-vue3-vite-cytoscape-frontend-stack.md)
- [ADR-066: Playwright for UI E2E testing](../adr/ADR-066-playwright-for-ui-e2e-testing.md)

### UX specifications

- [UXS-001: GI/KG Viewer](../uxs/UXS-001-gi-kg-viewer.md) — visual and token contract

### Guides

- [Grounded Insights Guide](../guides/GROUNDED_INSIGHTS_GUIDE.md)
- [Knowledge Graph Guide](../guides/KNOWLEDGE_GRAPH_GUIDE.md)
- [ML Provider Reference](../guides/ML_PROVIDER_REFERENCE.md)

---

## Promotion (unified)

When hardening: split into **PRD(s)** (product, tenancy, digest, hardware, ops) and **RFC(s)**
(schema, worker protocol, Compose reference, job model, digest contract, hardware reference
architecture, observability stack, deployment pipeline). This file remains **WIP** until then.

**Candidate RFCs to seed from this megasketch:**

| Candidate | Source section | Scope |
| --- | --- | --- |
| RFC: Multi-tenant data model | Part A (A.3–A.6, A.12) | Schema, migrations, RLS, tenant lifecycle, auth evolution |
| RFC: Worker orchestration & job model | Part B (B.7–B.8, B.9.1, B.14, B.15, B.17) + Part D (D.5–D.6) | Two-tier queue design (simple/distributed), arq, job payload, state machine, DLQ, CLI entry points, file locking, testing strategy |
| RFC: Compose reference architecture | Part B (B.4, B.9.2) + Part D (D.5, D.11) + Part F (F.3, F.9) | docker-compose.yml, image strategy, secrets, volumes, networking, deploy scripts |
| RFC: Hardware reference architecture | Part D (D.1–D.4, D.10–D.16) | Minimum specs, model memory budget, graduation path, concrete configs, TCO analysis |
| RFC: Database migrations | Part B (B.16) | Alembic setup, migration workflow, container startup order, rollback |
| RFC: Corpus digest contract | Part C (C.3–C.9) | Digest artifact schema, time-scoped aggregation, pipeline integration, viewer routes |
| RFC: Observability stack | Part E (E.2–E.6) | PLG stack, metrics endpoints, dashboards, alerting rules |
| RFC: Correlation ID & distributed tracing | Part E (E.4, E.7) | Request → queue → worker trace; structured log fields |
| RFC: Control plane & admin API | Part E (E.7, E.8) | Admin routes, queue management, worker management |
| RFC: CD pipeline & deployment automation | Part F (F.2–F.4) | Image registry, deploy scripts, rollback, GitOps |
| RFC: Secrets & configuration management | Part F (F.5) | Environment contract, secrets strategy, config layers |
| RFC: Per-service health checks & degradation model | Part E (E.6) | Health endpoint contract, degradation levels, Docker healthchecks |
| RFC: AI agent on-call (actionable observability) | Part E (E.9) | Agent architecture, runbook engine, safety guardrails, human-mirrored deploy workflow |
