# Platform, corpus, and service megasketch (WIP)

**Status:** Single **merged** architecture / product notebook — **not** a PRD or RFC. Combines
prior WIPs on **multi-tenant platform**, **Docker / API / workers / queues**, and **corpus
digest / weekly rollup** so one file can seed **multiple RFCs** later.

**Audience:** PRD/RFC authors and implementers planning service mode, tenancy, deployment, and
downstream digest features.

**How to use this doc**

- **Part A** — Catalog, subscriptions, reuse, Postgres tenancy, CLI vs platform constraints.
- **Part B** — Compose, API vs worker, Redis, named queues, Whisper/`heavy`, RFC checklist.
- **Part C** — “Many podcasts, one brain”: weekly digest gaps and sequencing **on top of** core +
  projection.

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
even when **v1** runs a **single** tenant — avoid a painful “add `tenant_id` everywhere”
migration later.

### A.2 Product constraints (CLI + optional platform)

These are **non-negotiable directions** for how the platform relates to the existing tool:

1. **CLI stays first-class** — Easy runs from **config** (YAML/JSON): scripts, CI, ad-hoc use,
   outputs on disk **without** running a server. Most users can stay on this path forever.

2. **Service / platform mode is optional** — Long-lived process for **continuous** pull,
   multi-feed orchestration, Postgres projection, optional **UI/API**. Headless single-user

   platform and multi-tenant UI are **graduations** of the same service layer, not a separate
   product fork.

3. **One pipeline core, multiple shells** — Execution stays **`run_pipeline` / `service.run`**
   (one `Config` in, artifacts out). Platform code **schedules**, **dedupes**, **materializes**

   `Config` per job, handles **cursors** and **projection** — it **must not** duplicate or
   diverge from core pipeline logic.

4. **Avoid diverging config dialects** — Prefer **one `Config` model** and shared validation;
   the platform adds **catalog, subscriptions, tenants** in the DB and **builds** a `Config`

   (or equivalent dict) for each worker invocation instead of maintaining a parallel
   “platform-only” config schema that drifts from the CLI.

**Summary:** Library + **CLI** (simple path) + optional **service** (daemon / API / UI) on the
same engine — not “CLI tool vs platform” as two implementations.

### A.3 Separate three concepts

| Concept | Meaning |
| --- | --- |
| **Catalog** | Global **directory of feeds** (and normalized show metadata) the **platform** knows. “What exists / what we can process,” not “Alice’s list” alone. |
| **Subscription** | **Tenant** ↔ **catalog entity** (e.g. feed). “I want this show in my library.” |
| **Entitlement / visibility** | What the tenant may **read**: episodes, GI, KG, summaries. Often: subscription + optional admin grants or feature flags. |

**Ingestion** should be driven by **catalog + policy** (what is globally enabled, what has
subscribers), not by “run this user’s private config file” as the only source of truth.

### A.4 Process once, serve many (reuse)

Expensive steps (download, transcribe, summarize, GI, KG) run **once** per logical episode +
**pipeline fingerprint**, not once per user.

- **Episode identity:** Canonical feed id + episode `guid` (ADR-007).
- **Pipeline fingerprint:** Versions / hashes of models, providers, prompts, schemas that
  affect outputs (summaries, `gi.json`, KG artifact).

**Rule:** At most **one canonical artifact set** per `(episode_key, pipeline_fingerprint)`.
If tenant B subscribes to the same feed+episode, they **attach** to existing rows — no second
Whisper run for the same fingerprint.

**Workers** consume a **deduped queue** of “work needed,” not independent full pipeline runs
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
- **Reads:** “My library” = join subscriptions → episodes → canonical projections; enforce RLS
  or API-layer checks.

Avoid **schema-per-tenant** early unless compliance requires it.

### A.7 Service shape (logical components)

1. **API + UI** — Admin: catalog. User: subscriptions, entitled reads.
2. **Scheduler / worker** — Feeds with subscribers (or globally enabled); enqueue work;
  **cursors**.

3. **Projection / indexer** — Files or object storage → Postgres (RFC-051); optional search
   later.

**Pipeline core:** `run_pipeline`-class logic = **one unit of work** (config in, artifacts
out). Platform adds **queue, dedup, paths, post-run projection**.

### A.8 GI / KG / summaries under reuse

- **Store once** per `(episode, pipeline_fingerprint)` in canonical storage + projections.
- **Tenant view:** Filter by subscription / entitlements.
- **Overlays:** Tenant-scoped tables for bookmarks, labels.

### A.9 Phased delivery (platform)

| Phase | Deliver |
| --- | --- |
| **A** | `tenant_id`, `tenants`, subscriptions, catalog; single tenant row. |
| **B** | Long-lived worker + cursors; feeds from catalog ∩ subscriptions. |
| **C** | Projection to Postgres (RFC-051); API reads DB + blob pointers. |
| **D** | UI + auth (even one login). |
| **E** | Second tenant + RLS; quotas / billing later. |

Defer: billing, orgs, per-tenant pipeline overrides, legal review — but **name** shared-corpus /
ToS risks in a threat model.

### A.10 Risks / watch (platform)

- **Fingerprinting** wrong → wrong shared artifacts.
- **Ops surface:** Postgres, workers, object storage, auth, UI — scope consciously.
- **Legal:** Multi-user shared corpus → takedown, copyright, PII in logs.

### A.11 Relation to current repo

- **Today:** One `Config`, one `rss`, `run_pipeline` → filesystem; `service.run` one-shot.
  **CLI** = default mental model.

- **Platform:** Catalog + subscriptions **when** operator chooses platform mode; workers replace
  “cron per feed”; Postgres = state + projections; reuse via **episode key + fingerprint**.

---

## Part B. Deployment, API, workers, queues, Docker

**Depends on:** Part A (especially **A.2** product constraints and **A.7** service shape).

### B.1 Control plane vs data plane

| Layer | Responsibility | Typical runtime |
| --- | --- | --- |
| **API** | Auth, tenant context, **config** (catalog, subscriptions), **read** APIs for
  consumption (summaries, GI, KG) from **Postgres** + blob pointers — **not** Whisper per
  request. | Gunicorn + **Uvicorn** workers (ASGI) for FastAPI/Starlette. |
| **Worker(s)** | Pull from queue (or DB), **build `Config`**, **`run_pipeline`**, canonical
  files, **projection** to Postgres. | Long-lived process(es). |

**DB** = Postgres (metadata, projections, optional jobs). **Files** (or object store) first, then
**project** (RFC-051).

### B.2 End-to-end data flow (target)

```text
User/UI → API → Postgres (catalog, subscriptions, entitlements)
                    ↓
              enqueue job (Redis or Postgres `jobs` table)
                    ↓
Worker(s) → run_pipeline(cfg) → canonical files (+ optional object store)
                    ↓
              projection (inline or `projection` queue) → Postgres (RFC-051)
                    ↓
User/UI → API → read Postgres (+ signed URLs to blobs if needed)
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

**Add:** `redis` (RQ/Celery/arq), `minio` (S3-compatible blobs), `ui` (static SPA or behind
Caddy).

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
| **One worker service, many queues** | Same image; `ingest`, `heavy`, `projection` in one
  binary — **start here**. |
| **Multiple Compose services, same image** | Different `command` / queue subscription — when
  **contention** measured. |
| **Multiple images** | CUDA vs slim — when **dependencies** diverge. |

**Specialization does not fix:** shared code bugs, Postgres/Redis down, same bad deploy to all
pools.

**It helps:** GPU vs CPU isolation; **backlog** isolation across queues.

### B.7 Named queues vs pipeline *steps*

**Default v1:** Queues separate **job categories**, not every **internal** `run_pipeline` step.
One job can still run **download → Whisper → summarize → …** **sequentially**; Whisper stays a
**middle** step **in code**.

**Advanced:** Chain jobs per episode with **durable state** (`pending` / `transcribed` / …) —
only when retry/isolation demands it.

**Illustrative queues**

| Queue | Payload | Concurrency |
| --- | --- | --- |
| **`ingest`** | Poll RSS, enqueue `heavy` | Higher (I/O). |
| **`heavy`** | Full `run_pipeline` incl. local Whisper | **Low** — ~**1** per GPU. |
| **`projection`** | Files → Postgres only | Medium (CPU/DB). |

**First integration:** `heavy` job → **`run_pipeline(cfg)`**. Add `ingest` / `projection` when
backlog/retry needs justify.

### B.8 Local Whisper as bottleneck

- Put **full episode** jobs on **`heavy`**; **concurrency 1** (or 2 if VRAM allows) per GPU.
- Other queues handle RSS/projection so they **don’t** sit behind a Whisper pile on a **single**
  FIFO.

**This does not** add a queue **inside** one episode unless you adopt **B.7 advanced**.

### B.9 API + worker separation — timing

**Recommend early:** separate Compose services, **same image**, different `command` — security
boundary, independent scale, safer deploy cadence.

### B.10 Postgres (deployment recap)

Target **vanilla PostgreSQL** for SaaS; extensions as needed. **Canonical files + projection**
(RFC-051). Avoid **SQLite in prod** if you want one dialect (see separate DB discussions).

### B.11 Summary table (deployment)

| Topic | **Recommendation (v1)** |
| --- | --- |
| Orchestration | **Compose** |
| Queue backend | **Redis** with RQ/Celery/arq (or Postgres `jobs` minimal) |
| Workers | **One service**, **multiple queues** |
| Job size | **One job = full `run_pipeline`** first |
| Whisper | **`heavy` queue + low concurrency** |
| Pipeline vs queue | **Job boundary only** initially |

### B.12 Numbered recommendations (RFC authors)

1. **`run_pipeline`** = primary worker unit v1; materialize `Config` per job.
2. Queues: **`ingest`** (optional) + **`heavy`** early; **`projection`** when DB must not wait
   on Whisper backlog.

3. **`heavy` concurrency** ≈ **1** per GPU for local Whisper.
4. No **mid-pipeline** queue splits until metrics/retry require it.
5. Compose: **`postgres` + `api` + `worker` + `caddy`**; add **`redis`** with queue library.
6. RFC: job payload schema, idempotency (`episode_key` + `pipeline_fingerprint`), dead-letter.
7. **CI:** integration tests on **Postgres** (+ Redis if used) for platform paths.

### B.13 RFC checklist (deployment / orchestration)

- Job model JSON schema; dedup; queue names; worker ↔ queue mapping; concurrency per queue; GPU
  notes; retries, DLQ, visibility; projection inline vs async; Compose reference; secrets;

  observability (correlation id API → worker).

---

## Part C. Corpus digest and weekly rollup

**Depends on:** Stable core (transcripts, metadata, summaries, optional GI/KG) and **queryable**
corpus (PRD-018 / RFC-051 or documented file patterns).

### C.1 Problem sketch

User with **10–50 podcasts** wants to **navigate** recent arrivals, **consume** quickly, **dig
deep** selectively, and answer **“what happened last week across my library?”** without
duplicate noise and with **trust** when claims matter.

Today: **per-episode** artifacts only; no **first-class time-scoped cross-feed digest** contract.

### C.2 Layering (summaries / KG / GI)

| Layer | Role | Home |
| --- | --- | --- |
| **Summaries** | Fast **consumption** | PRD-005, metadata |
| **KG** | **Navigation** across episodes | PRD-019, RFC-055/056 |
| **GIL** | **Value and trust** | PRD-017, RFC-049/050 |

Digest features should assume these layers are **stable and versioned** before hard rollups.

### C.3 Gaps (product opportunities, not promises)

1. **Time-scoped aggregation** — Window (week, last N days) across corpus; one digest artifact or
   view.

2. **Cross-feed inbox** — “New since last run”; per-feed watermarks; backlog vs delta.
3. **Story clustering / dedup** — Same story across shows.
4. **Ranking / time budgets** — e.g. “30 minutes this week.”
5. **Change detection** — “What’s new on topic X **this week**” vs cumulative KG.
6. **Digest output contract** — Versioned JSON or doc (themes, GI-backed bullets, episode links).
7. **Presentation** — HTML/email/Obsidian out of core unless scoped; **contract** first.
8. **Personalization** — Watchlists; **user config** + KG.

### C.4 Sequencing (when core is stable)

1. Stable **published** dates + episode identity (ADR-007); summaries + optional GI/KG.
2. **Queryable corpus** (RFC-051 or file patterns).
3. **Digest v0** — Time filter + sorted episodes + summary lead + link; no clustering.
4. **Digest v1** — KG rollups, GI-highlighted bullets, dedup iteration.

### C.5 Non-goals (early digest)

- Replacing listening / primary sources for high-stakes decisions.
- Full **recommender** or social graph.
- **Merging** GI into KG artifacts — join at digest/query layer if needed.

---

## Related documents (unified)

- [Architecture](../ARCHITECTURE.md) — one pipeline, one `Config`
- [ADR-007: Universal episode identity](../adr/ADR-007-universal-episode-identity.md)
- [PRD-018: Database projection](../prd/PRD-018-database-projection-gil-kg.md) / RFC-051
- [PRD-005](../prd/PRD-005-episode-summarization.md), [PRD-017](../prd/PRD-017-grounded-insight-layer.md), [PRD-019](../prd/PRD-019-knowledge-graph-layer.md)
- [Grounded Insights Guide](../guides/GROUNDED_INSIGHTS_GUIDE.md), [Knowledge Graph Guide](../guides/KNOWLEDGE_GRAPH_GUIDE.md)
- [Docker Service Guide](../guides/DOCKER_SERVICE_GUIDE.md) — current one-shot service mode

---

## Promotion (unified)

When hardening: split into **PRD(s)** (product, tenancy, digest) and **RFC(s)** (schema, worker
protocol, Compose reference, job model, digest contract). This file remains **WIP** until then.
