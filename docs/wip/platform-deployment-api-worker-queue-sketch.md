# Platform deployment sketch — API, workers, queues, Docker (WIP)

**Status:** Exhaustive architecture notes for a future **RFC** (or RFC family) on SaaS /
self-hosted platform mode. **Not** a committed RFC. Supplements
[multi-tenant-platform-sketch.md](multi-tenant-platform-sketch.md) (catalog, tenants, CLI
constraints).

**Audience:** Authors of implementation RFCs for service mode, Compose/K8s layout, job
orchestration, and pipeline integration.

**Scope:** How to run **API + workers** under Docker, what **Redis/queues** buy you, how
**Whisper** fits **without** rewriting `run_pipeline` into micro-steps on day one, and
**explicit recommendations** vs **options** to defer.

---

## 1. Goals of this document

- Capture **deployment** and **orchestration** decisions discussed for a **platform** on top
  of the existing **CLI-first** tool (`run_pipeline`, `Config`, filesystem truth).

- Distinguish **options** (Compose vs K8s, Redis vs Postgres queue, one vs many worker
  containers) from **recommended defaults** for **v1**.

- Explain **named queues** and **Whisper** clearly enough that a later RFC can reference this
  file without re-deriving the reasoning.

- List **what an RFC should specify** when implementation starts.

---

## 2. Product constraints (inherited)

These are **fixed** for platform design; full rationale is in
[multi-tenant-platform-sketch.md § Product constraints](multi-tenant-platform-sketch.md#product-constraints-cli--optional-platform).

- **CLI remains first-class** — config-driven runs without a server.
- **Service/platform is optional** — same core engine.
- **One pipeline core** — `run_pipeline` / `service.run`; platform **wraps**, does not fork
  pipeline logic.

- **Avoid diverging config dialects** — platform state in DB **materializes** `Config` per job.

This document adds **how** the platform process is **split** (API vs worker) and **scheduled**
(queues, Docker).

---

## 3. Control plane vs data plane

| Layer | Responsibility | Typical runtime |
| --- | --- | --- |
| **API (web)** | Auth, tenant context, **config** (catalog, subscriptions, settings), **read**
  APIs for consumption (summaries, GI, KG, search) from **Postgres** and blob pointers —
  **not** running Whisper per HTTP request. | Gunicorn + **Uvicorn** workers (ASGI) if using
  FastAPI/Starlette; or equivalent. |
| **Worker(s)** | **Pull** work from a queue (or DB poll), **build `Config`**, call
  **`run_pipeline`** (or future stage entrypoints), write **canonical files**, run
  **projection** into Postgres. | Long-lived process(es); **no** interactive HTTP server. |

**Conclusion:** API = **thin, fast, read-heavy + control**; worker = **heavy, async, write
path**. The user’s mental model (“API for UI access/config/consumption; workers get data,
process, load into DB”) is **correct**. **DB** here means **Postgres** (metadata, projections,
optional job state); **canonical artifacts** often live on **disk or object storage** first,
then **project** (RFC-051).

---

## 4. End-to-end data flow (target)

```text
User/UI → API → Postgres (catalog, subscriptions, entitlements)
                    ↓
              enqueue job (optional Redis or Postgres `jobs` table)
                    ↓
Worker(s) → run_pipeline(cfg) → canonical files (+ optional object store)
                    ↓
              projection job / step → Postgres (RFC-051 tables)
                    ↓
User/UI → API → read Postgres (+ signed URLs to blobs if needed)
```

**Option:** Projection can run **inline** at the end of the same worker job as
`run_pipeline`, or as a **separate** queued job on a `projection` queue (see §9–10).

---

## 5. Docker Compose vs Kubernetes vs PaaS

### 5.1 Options

| Approach | When it fits |
| --- | --- |
| **Docker Compose** | Single host or small VPS; **SaaS v1**; you accept brief restarts on deploy;
  **recommended starting point**. |
| **Kubernetes** | Multi-node, heavy autoscaling, many replicas, team owns clusters — **defer**
  until requirements force it. |
| **Managed PaaS** (Fly, Railway, Render, Cloud Run, ECS Fargate) | Containers without
  operating K8s yourself — valid **middle path**. |

### 5.2 Conclusion

**Compose is enough** for early SaaS self-host. **K8s is not required** to be “production
serious.” Revisit when scale or HA requirements exceed one well-sized machine (or use **managed**
container platforms first).

---

## 6. Docker Compose — service catalog

### 6.1 Minimal production-shaped stack

| Service | Image / role | Notes |
| --- | --- | --- |
| **`postgres`** | `postgres:16` (or 15 LTS) | Catalog, subscriptions, cursors, RFC-051
  projections, sessions. **Source of truth** for relational data. |
| **`api`** | **Your** image | Gunicorn/Uvicorn; env: `DATABASE_URL`, secrets, **no** GPU
  required if API stays thin. |
| **`worker`** | **Same** image as `api`, **different** `command` | Queue consumer;
  may need **GPU** / ML volumes if Whisper local. |
| **`caddy`** (or Traefik) | `caddy:2` | TLS termination, reverse proxy to `api`. |

**Volumes:** Postgres data; **bind or named volume** for **artifact root** (transcripts,
`gi.json`, etc.) shared read/write between worker and (if needed) projection-only tasks.

### 6.2 Common additions

| Service | Role |
| --- | --- |
| **`redis`** | Backend for **RQ / Celery / arq** job queue (§7). |
| **`minio`** | S3-compatible **object storage** if blobs leave local disk. |
| **`ui`** | Static SPA behind Caddy or `nginx:alpine`, or served from `api`. |

### 6.3 What to avoid early

- **Kubernetes** (unless you already operate it).
- **ClickHouse / OpenSearch** until query load justifies ops.
- **One container per podcast** — wrong granularity; use **jobs** and **queues**.

---

## 7. Redis and the job queue

### 7.1 Role of Redis

Redis is typically the **broker** for a Python job library (**RQ**, **Celery**, **arq**):

- **Decouples** the API from long work: API **enqueues** a small payload, returns **202** +
  job id.

- Stores **pending / started / failed** job metadata (exact semantics depend on library).
- Enables **multiple worker processes** to **compete** on the same queue.
- Supports **retries**, **visibility**, and (with care) **dedup keys**.

Redis is **not** the system of record for **feeds, episodes, users** — **Postgres** is.

### 7.2 Alternatives to Redis

| Alternative | Tradeoff |
| --- | --- |
| **Postgres `jobs` table** + `SELECT … FOR UPDATE SKIP LOCKED` | One fewer moving part;
  good for **low** throughput; you implement retry/backoff. |
| **In-process queue** | Only for **single** worker process; no horizontal scale. |

### 7.3 Conclusion

- **Redis recommended** when you want **standard** queue semantics and **multiple workers**
  without hand-rolling a job store.

- **Postgres-only queue** acceptable for **very early** or strict minimalism.

---

## 8. Worker pools: one type vs many types

### 8.1 Options

| Model | Description |
| --- | --- |
| **Single worker service** | One Compose service, N replicas; consumes one or **many** queue
  names in one binary (`ingest`, `heavy`, `projection`). |
| **Multiple Compose services, same image** | e.g. `worker-heavy` with
  `command: … --queues=heavy`, `worker-light` with `--queues=ingest,projection`. |
| **Multiple images** | e.g. **CUDA** image for Whisper vs **slim** image for projection —
  when **dependencies** truly diverge. |

### 8.2 Failure isolation — what specialization actually buys

**Specialized workers do *not* eliminate:**

- Bugs in **shared** library code used by all workers.
- **Postgres / Redis / disk** outages affecting every pool.
- A **bad deploy** of the **same** image to all pools.

**They do help with:**

- **Resource isolation** — GPU OOM / long Whisper jobs vs cheap DB upserts.
- **Queue backlog isolation** — transcription backlog does not block **other** queues if
  workers listen to **different** queues.

- **Deploy risk** — only if you **split images** or deploy **independently**.

### 8.3 Conclusion and recommendation

1. **Start:** **one** `worker` service, **one** codebase image, **multiple named queues** inside
   the worker (§9).

2. **Next:** split into **two** Compose services (**same image**, different `command` / env for
   which queues to consume) when metrics show **contention**.

3. **Later:** **separate images** only for **CUDA vs non-CUDA** (or similar dependency split).

This sequences the decision: **design queue boundaries early**, **split processes when
measured**, not as a day-zero mandatory **N-container** topology.

---

## 9. Named queues — semantics and pipeline relationship

### 9.1 What “named queues” means

Physically: separate lists/streams in the broker (e.g. Redis lists/streams per name). Jobs are
**routed** to a queue by **job type** or **explicit queue** argument. Workers subscribe to one
or more queue names.

### 9.2 Queues vs pipeline *steps* (critical distinction)

**Default (recommended v1):**

- **Queues separate *jobs* and *categories of work***, not necessarily **every internal step**
  of `run_pipeline`.

- A single job may still run **download → Whisper → summarize → GI → …** **sequentially**
  inside one handler — **Whisper remains a middle step in code**.

**Advanced (optional later):**

- **Chain jobs** per episode: e.g. `transcribe` job → enqueue `post_transcript` job. Requires
  **durable episode state** (`pending` / `transcribed` / `projected`) and **refactor** to

  callable stages or idempotent resume.

### 9.3 Suggested queue names (illustrative)

| Queue | Typical payload | Consumer concurrency |
| --- | --- | --- |
| **`ingest`** | Poll RSS, discover new episode ids, enqueue **`heavy`** jobs | **Higher**
  (I/O bound). |
| **`heavy`** | **Full** `run_pipeline` for an episode (or feed batch) including **local
  Whisper** | **Low** — often **1** concurrent job per GPU (§10). |
| **`projection`** | Upsert files → Postgres only (episode already processed on disk) | **Medium**
  — CPU/DB bound, no GPU. |

**API** enqueues **`ingest`** or **`heavy`** depending on product flow; **`projection`** may be
triggered at end of **`heavy`** (same job) or as a **follow-up** job for isolation.

### 9.4 Conclusion

- You **do not** need to “evolve pipeline code to use queues **between** every step” to get
  value.

- **First integration:** worker pulls **`heavy`** job → calls **existing** `run_pipeline(cfg)`.
- **Second integration:** add **`ingest`** and **`projection`** queues when **backlog** or
  **retry** needs justify it.

---

## 10. Local Whisper as the dominant bottleneck

### 10.1 Problem

If all job types share **one** FIFO queue and **high** worker concurrency, **many** episodes
queue behind Whisper; cheap work (RSS poll, projection) **stalls** behind transcription.

### 10.2 Recommended mitigation (without splitting pipeline internally)

- Route **full episode processing** jobs to **`heavy`**.
- Set **worker concurrency for `heavy`** to **1** (or **2** only if VRAM and model size allow).
- Run **other workers** (or same binary with different queue subscription) on **`ingest`** /
  **`projection`** with **higher** concurrency.

**Important:** This does **not** insert a queue **between** “before Whisper” and “after Whisper”
**inside** one episode — the **entire** episode job sits on **`heavy`** and runs **sequentially**.

### 10.3 When to split Whisper into its own job (advanced)

Only if you need:

- **Projection** or **summarization** for **other** episodes to proceed **while** Whisper runs on
  this one **without** blocking the same queue, **and**

- **Retry** transcription **without** re-running downstream, **or** retry downstream **without**
  re-transcribing.

Then implement **multi-job** episode state (§9.2 advanced).

---

## 11. API + worker separation — timing

**Recommendation:** Separate **API** and **worker** **early** (different Compose services,
same image, different `command`).

**Rationale:**

- Clear **security** boundary (API not holding GPU credentials if you ever split).
- **Independent scale** (API replicas vs worker replicas).
- **Deploy** API without touching long-running jobs (with care for shared migrations).

---

## 12. Postgres as primary database (recap)

- **Self-hosted SaaS:** target **vanilla PostgreSQL**; add **extensions** (e.g. Timescale,
  `pg_trgm`) when needed — see prior discussions; **no SQLite** in prod path if avoiding

  dual-dialect cost.

- **Canonical files** + **Postgres projection** (RFC-051) remains the intended split: DB is
  **query/consumption** layer; files remain **auditable** source for pipeline outputs.

---

## 13. Summary table — decisions

| Topic | Option A | Option B | **Recommendation (v1)** |
| --- | --- | --- | --- |
| Orchestration | Compose | K8s | **Compose** |
| Queue backend | Redis | Postgres `jobs` | **Redis** when using RQ/Celery/arq |
| Worker topology | One service | Many specialized | **One service**, **multiple queues** |
| Job granularity | One job = `run_pipeline` | Stage-chained jobs | **One job = full pipeline** first |
| Whisper | Shared queue with all work | Isolated `heavy` queue + low concurrency | **`heavy` + concurrency 1/GPU** |
| Pipeline refactor | Queues between every step | Queues only at job boundary | **Job boundary only** initially |

---

## 14. Numbered recommendations (for RFC authors)

1. **Keep `run_pipeline` as the primary execution unit** for worker v1; enqueue **episode- or
   feed-scoped** jobs that **materialize `Config`** and call it.

2. **Introduce at least two queue names** early: **`ingest`** (optional) and **`heavy`**; add
   **`projection`** when DB upserts should not wait behind Whisper backlog.

3. **Bind `heavy` concurrency** to **GPU capacity** (default **1** per GPU for local Whisper).
4. **Do not** require **mid-pipeline queue boundaries** until **metrics or retry semantics**
   demand them.

5. **Compose** with **`postgres` + `api` + `worker` + `caddy`**; add **`redis`** when adopting a
   queue library.

6. **Document** job payload schema, idempotency keys (`episode_key` + `pipeline_fingerprint`
   per multi-tenant sketch), and **dead-letter** behavior in the RFC.

7. **CI:** run integration tests against **Postgres** (and Redis if used), not SQLite, for
   platform code paths.

---

## 15. What a future RFC should specify

- **Job model:** JSON schema for enqueue payload; **idempotency** and **dedup** rules.
- **Queue names** and **which worker processes** consume which queues (env / CLI).
- **Concurrency** defaults per queue; **GPU** scheduling notes.
- **Failure:** retries, exponential backoff, **dead-letter** queue, visibility timeout.
- **Projection trigger:** inline after `run_pipeline` vs **async** `projection` job.
- **Compose** reference file (services, volumes, healthchecks, **depends_on**).
- **Secrets:** rotation, `DATABASE_URL`, Redis URL, object storage credentials.
- **Observability:** structured logs, job metrics, correlation id from API → worker.

---

## Related documents

- [Multi-tenant platform sketch](multi-tenant-platform-sketch.md) — catalog, subscriptions,
  reuse, CLI constraints.

- [Corpus digest & weekly rollup (WIP)](corpus-digest-weekly-rollup-product-idea.md)
- [Architecture](../ARCHITECTURE.md) — one pipeline, one `Config`
- [PRD-018 / RFC-051](../prd/PRD-018-database-projection-gil-kg.md) — Postgres projection
- [Docker Service Guide](../guides/DOCKER_SERVICE_GUIDE.md) — current **one-shot** service mode

---

## Promotion

When implementation is scoped: split into **RFC(s)** e.g. **“Platform job orchestration”**,
**“Docker Compose reference deployment”**, or a single **“Service mode v2”** RFC with appendices.
Until then, **WIP** only.
