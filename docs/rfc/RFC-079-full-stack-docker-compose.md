# RFC-079: Full-Stack Docker Compose Topology

- **Status**: Draft
- **Authors**: Marko
- **Created**: 2026-04-22
- **Domain**: Infrastructure / DevOps
- **Related RFCs**:
  - `docs/rfc/RFC-077-viewer-feeds-and-serve-pipeline-jobs.md` (jobs API, operator config)
  - `docs/rfc/RFC-078-ephemeral-acceptance-smoke-test.md` (consumes this topology for CI smoke tests)
- **Tracking**:
  - Stack implementation (Phase 1): [GitHub #659](https://github.com/chipi/podcast_scraper/issues/659)
  - Jobs API ↔ Docker pipeline execution (Phase 2): [GitHub #660](https://github.com/chipi/podcast_scraper/issues/660)
  - **#659** and **#660** both implement **this RFC (RFC-079)**. [RFC-078](RFC-078-ephemeral-acceptance-smoke-test.md) **documents** smoke acceptance on top of this stack; **GitHub issues for that smoke work are not opened yet** (RFCs are not the task tracker).
- **Implementation (Phase 1 / #659):** `compose/docker-compose.stack.yml`, `docker/viewer/` (Nginx + Vue build),
  `docker/api/` (FastAPI serve), Makefile `stack-*` targets, `config/examples/docker-stack.example.yaml`,
  and **Full stack (RFC-079)** in `docs/guides/DOCKER_SERVICE_GUIDE.md`. Acceptance: build, `up -d`,
  `curl …/api/health` via Nginx, `compose run pipeline` with `CONFIG_FILE` + `output_dir: /app/output`.

## Abstract

Define a production-grade Docker Compose topology that packages the full podcast\_scraper
stack into three containers: **Nginx** (Vue static build + reverse proxy), **API** (FastAPI
backend), and **Pipeline** (one-shot ML pipeline runner). All three share a single data volume
for corpus output. This gives us a single `docker compose up` that runs the viewer and API,
and `docker compose run pipeline` to execute ingestion — matching the production deployment
model from day one.

This RFC is the **total** solution in **two phases**. **Phase 1** (primary deliverable, #659)
adds the compose topology, images, Makefile targets, and documentation. **`POST /api/jobs`**
is *not* rewired in Phase 1: it keeps the existing subprocess behavior (see **Jobs API and
pipeline execution** below). **Phase 2** (#660) chooses and implements how "run job" uses the
**`pipeline` container** (or an equivalent executor) under Docker so the viewer jobs flow
matches the split-image architecture without duplicating ML inside `api` unless that is an
explicit product decision. **Native** laptop / venv subprocess jobs stay supported unchanged
(see §Native vs Docker).

## Problem Statement

Today the stack runs as three loosely coupled processes on the developer's machine:

1. `make serve-api` — FastAPI on port 8000, reads corpus output from a local directory
2. `make serve-ui` — Vite dev server on port 5173, proxies `/api/*` to FastAPI
3. Pipeline CLI — `python -m podcast_scraper.service` or via Makefile, writes to output dir

This works for local dev but has no path to deployment:

- The viewer has no production build-and-serve story (only Vite dev server)
- The existing [`docker/pipeline/Dockerfile`](../docker/pipeline/Dockerfile) packages only the pipeline runner; the API server and viewer
  are not containerized
- The existing `compose/docker-compose.yml` defines only a `podcast_scraper` service (pipeline);
  there is no compose service for the API or viewer
- Without a compose topology, RFC-078 (ephemeral smoke test) cannot spin up the full stack
  in CI

**Use Cases:**

1. **Local production-like environment**: developer runs `docker compose up` and gets the
   full viewer + API on `localhost:80`, backed by real corpus data
2. **CI smoke test** (RFC-078): GitHub Actions builds the compose images, runs the pipeline
   against fixtures, starts the server, and runs Playwright assertions
3. **Future prod deployment**: the same compose file (with prod overrides) deploys to a VPS
   via `docker compose pull && docker compose up -d`

## Goals

1. **Three-container topology**: Nginx, API, Pipeline — each with a clear single responsibility
2. **Shared data volume**: pipeline writes, API reads, Nginx never touches data
3. **One `docker compose up`** starts the viewer (Nginx) + API; pipeline is invoked separately
4. **Production-grade Nginx**: serves pre-built Vue SPA, reverse-proxies `/api/*` to FastAPI,
   handles static caching headers, gzip, and SPA fallback routing
5. **Reuse existing pipeline Dockerfile** ([`docker/pipeline/Dockerfile`](../docker/pipeline/Dockerfile)) with minimal changes
6. **New Dockerfiles** under `docker/api/` and `docker/viewer/` for API (`.[server]` + semantic-search deps; not full `.[ml]`) and Nginx (multi-stage Vue build)
7. **Health checks** on all long-running containers
8. **Compose profiles** for dev-override and CI-override use cases — **executable backlog: [GitHub #659](https://github.com/chipi/podcast_scraper/issues/659)** (decision / optional `compose/docker-compose.dev.yml`).
9. **Document and track** the gap between viewer-triggered jobs and the `pipeline` service,
   closed by **[GitHub #660](https://github.com/chipi/podcast_scraper/issues/660)** once the stack exists

## Constraints and Assumptions

**Constraints:**

- No orchestration beyond Docker Compose (no Kubernetes, no Swarm)
- No external services (no DB, no Redis, no message queue) — all state is file-based
- Pipeline is one-shot (runs and exits); it is NOT a long-running service
- Single-host deployment (one machine, one person)
- ML model loading makes the pipeline image large (3-4 GB); the API image must stay slim

**Assumptions:**

- The API server imports search/FAISS routes at startup: **`pip install -e '.[server]'` alone is
  not sufficient** for `create_app`. The stack **API image** installs `.[server]` plus NumPy,
  `faiss-cpu`, and `sentence-transformers` (CPU torch). Full Whisper/spaCy/llama-cpp remains on
  the **pipeline** image. See `docker/api/Dockerfile` and `docs/guides/DOCKER_SERVICE_GUIDE.md`.
- The Vue viewer builds successfully with `npm run build` (produces `dist/`)
- The FastAPI server already mounts `StaticFiles` from `web/gi-kg-viewer/dist` when
  available, but in this topology Nginx handles static serving and the API runs
  with `--no-static`
- A future phase will add a database; the volume-based design accommodates that migration

## Design and Implementation

### Container Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│  docker compose up                                              │
│                                                                 │
│  ┌──────────────┐        ┌──────────────┐                       │
│  │    nginx     │ :80    │     api      │ :8000 (internal)      │
│  │              │───────>│              │                        │
│  │  Vue dist/   │ /api/* │  FastAPI     │                        │
│  │  SPA fallback│        │  --no-static │                        │
│  └──────────────┘        └──────┬───────┘                       │
│                                 │ reads                         │
│                          ┌──────┴───────┐                       │
│                          │  corpus_data │  (named volume)       │
│                          └──────┬───────┘                       │
│                                 │ writes                        │
│  ┌──────────────┐               │                               │
│  │   pipeline   │───────────────┘                               │
│  │  (one-shot)  │                                               │
│  │  docker      │                                               │
│  │  compose run │                                               │
│  └──────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
```

### 1. Nginx Container (Viewer)

Multi-stage build: Node stage builds the Vue app, Nginx stage serves it.

**`docker/viewer/Dockerfile`:**

```dockerfile
# ── Build stage ──────────────────────────────────────────────
FROM node:22-alpine AS builder

WORKDIR /build
COPY web/gi-kg-viewer/package.json web/gi-kg-viewer/package-lock.json ./
RUN npm ci
COPY web/gi-kg-viewer/ ./
RUN npm run build

# ── Serve stage ──────────────────────────────────────────────
FROM nginx:1.27-alpine

COPY docker/viewer/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /build/dist /usr/share/nginx/html

HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
    CMD wget -qO- http://localhost:80/ || exit 1

EXPOSE 80
```

**`docker/viewer/nginx.conf`:**

```nginx
upstream api {
    server api:8000;
}

server {
    listen 80;
    server_name _;

    root /usr/share/nginx/html;
    index index.html;

    # ── API reverse proxy ────────────────────────────────────
    location /api/ {
        proxy_pass         http://api;
        proxy_set_header   Host              $host;
        proxy_set_header   X-Real-IP         $remote_addr;
        proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
    }

    # ── Static assets (cache-busted by Vite hash) ───────────
    location /assets/ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # ── SPA fallback ────────────────────────────────────────
    location / {
        try_files $uri $uri/ /index.html;
    }

    # ── Gzip ────────────────────────────────────────────────
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml;
    gzip_min_length 256;
}
```

### 2. API Container

Slim image — no ML dependencies, no model preloading. Runs the FastAPI server with
`--no-static` (Nginx handles static files).

**`docker/api/Dockerfile`:**

```dockerfile
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install only server dependencies (no ML)
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir '.[server]'

# Non-root user
RUN useradd -m -u 1000 -s /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

ENTRYPOINT ["python", "-m", "podcast_scraper.cli", "serve"]
CMD ["--host", "0.0.0.0", "--port", "8000", "--no-static", "--output-dir", "/data/output"]
```

The `--output-dir` points at the shared volume mount. Additional flags
(`--enable-feeds-api`, `--enable-jobs-api`, etc.) are passed via compose
`command:` or environment variables.

### 3. Pipeline Container

Reuses [`docker/pipeline/Dockerfile`](../docker/pipeline/Dockerfile). The compose service definition
sets the one-shot behavior.

### 4. Compose File

**`compose/docker-compose.stack.yml`** (shipped; does not replace the existing
`compose/docker-compose.yml` which remains for standalone pipeline use). Paths below match the repo: **build `context: ..`** is the **repository root** (compose file lives under `compose/`).

```yaml
# Schematic — see repo file for provider env blocks and exact flags.

services:
  viewer:
    build:
      context: ..
      dockerfile: docker/viewer/Dockerfile
    ports:
      - "${VIEWER_PORT:-8080}:80"
    depends_on:
      api:
        condition: service_healthy

  api:
    build:
      context: ..
      dockerfile: docker/api/Dockerfile
    expose:
      - "8000"
    volumes:
      - corpus_data:/app/output
      - ${CONFIG_FILE:-../config.yaml}:/app/config.yaml:ro
    environment:
      PODCAST_SCRAPER_CONFIG: /app/config.yaml

  pipeline:
    profiles: [pipeline]
    build:
      context: ..
      dockerfile: docker/pipeline/Dockerfile
    volumes:
      - corpus_data:/app/output
      - ${CONFIG_FILE:-../config.yaml}:/app/config.yaml:ro

volumes:
  corpus_data: {}
```

**Docker job mode (#660):** optional second file [`compose/docker-compose.jobs-docker.yml`](https://github.com/chipi/podcast_scraper/blob/main/compose/docker-compose.jobs-docker.yml) merges extra **`api`** volumes + env (`docker.sock`, `PODCAST_PIPELINE_EXEC_MODE`, `PODCAST_DOCKER_PROJECT_DIR`). See `docs/guides/DOCKER_SERVICE_GUIDE.md` § Full stack.

**Usage patterns:**

```bash
# Start viewer + API (long-running)
docker compose -f compose/docker-compose.stack.yml up -d

# Run pipeline (one-shot, writes to shared volume)
docker compose -f compose/docker-compose.stack.yml run --rm pipeline

# View logs
docker compose -f compose/docker-compose.stack.yml logs -f api

# Tear down
docker compose -f compose/docker-compose.stack.yml down
```

The `pipeline` service is under a compose profile so `docker compose up` does not
start it. It is only invoked explicitly with `docker compose run`.

### 5. Compose Override for CI/Smoke (RFC-078)

**`compose/docker-compose.smoke.yml`** (override layered on top of `compose/docker-compose.stack.yml`):

```yaml
# Sketch — see repo ``compose/docker-compose.smoke.yml`` for the canonical overlay.

services:
  api:
    volumes:
      - smoke_data:/app/output

  pipeline:
    volumes:
      - smoke_data:/app/output
      - ./tests/fixtures/rss:/app/fixtures/rss:ro
      - ./config/ci/smoke-config.yaml:/app/config.yaml:ro
    profiles: []

  viewer:
    ports:
      - "8090:80"

volumes:
  smoke_data:
```

### 6. File Layout

```text
docker/
  viewer/
    Dockerfile          # Multi-stage: Node build + Nginx serve (under docker/viewer/)
    nginx.conf          # Reverse proxy + SPA fallback
  api/
    Dockerfile          # Slim FastAPI server (no ML) — under docker/api/
compose/
  docker-compose.stack.yml    # Full-stack topology (viewer + api + pipeline)
  docker-compose.smoke.yml    # RFC-078 smoke overlay (local + CI)
  docker-compose.yml          # Standalone pipeline compose
  docker-compose.llm-only.yml # LLM-only variant
docker/pipeline/Dockerfile    # Pipeline / ML / LLM runner
```

### 7. Makefile Targets

```makefile
# Full-stack compose
stack-up:
	docker compose -f compose/docker-compose.stack.yml up -d

stack-down:
	docker compose -f compose/docker-compose.stack.yml down

stack-build:
	docker compose -f compose/docker-compose.stack.yml build

stack-run-pipeline:
	docker compose -f compose/docker-compose.stack.yml run --rm pipeline

stack-logs:
	docker compose -f compose/docker-compose.stack.yml logs -f
```

## Key Decisions

1. **Nginx in a separate container (not FastAPI serving static files)**
   - **Decision**: Nginx serves the Vue SPA and proxies `/api/*` to FastAPI
   - **Rationale**: Production-grade from day one. Exposes configuration complexity early
     (reverse proxy, caching, SPA fallback) rather than discovering it during a future
     migration. Negligible ongoing maintenance after initial setup.

2. **Pipeline as one-shot, not long-running**
   - **Decision**: Pipeline runs via `docker compose run --rm`, exits when done
   - **Rationale**: ML models consume 2-4 GB RAM. Keeping them loaded 24/7 wastes resources
     on a single-host deployment. Cold start (model loading ~30-60s) is acceptable for a
     batch process that runs at most a few times per day. Scheduled execution is handled by
     host cron or a lightweight cron container, not an internal scheduler.

3. **Separate compose file (`compose/docker-compose.stack.yml`), not modifying existing**
   - **Decision**: New file alongside existing `compose/docker-compose.yml`
   - **Rationale**: The existing `compose/docker-compose.yml` and `compose/docker-compose.llm-only.yml` are
     used for standalone pipeline runs and Docker CI tests. Modifying them would break those
     workflows. The stack compose is a superset that adds viewer + API.

4. **API container: `.[server]` + semantic stack, not full `.[ml]`** (implemented in #659)
   - **Decision**: Install `.[server]`, CPU `torch`, `numpy`, `faiss-cpu`, and `sentence-transformers`
     so `create_app` and `/api/search` work against an on-disk index. Omit Whisper/spaCy/llama-cpp
     from `api` to avoid duplicating the pipeline image.
   - **Rationale**: Smaller than fat-ML `api`, while satisfying import-time router dependencies.
   - **Caveat**: **In-process** `POST /api/jobs` still uses `sys.executable` inside `api` (RFC-077);
     jobs that need the full ML CLI may fail until [GitHub #660](https://github.com/chipi/podcast_scraper/issues/660)
     delegates execution to the `pipeline` container. Use `docker compose … run pipeline` for ops runs.

5. **Shared named volume, not bind mount**
   - **Decision**: `corpus_data` named volume shared between pipeline and API (**both read-write**).
   - **Rationale**: Named volumes are managed by Docker, survive container restarts, and
     avoid host path permissions issues. Read-write on `api` avoids breaking index rebuild / lock
     files; for dev override, a bind mount can be layered via a `docker-compose.override.yml`.

## Jobs API and pipeline execution (total solution, phased)

RFC-077 defines **`POST /api/jobs`**: enqueue a pipeline job, persist registry rows under the
corpus (e.g. `.viewer/jobs/`), stream logs, cancel, reconcile. The **spawn** path today is
implemented in `src/podcast_scraper/server/pipeline_jobs.py`:

1. **`build_pipeline_argv`** builds argv:
   `[sys.executable, "-m", "podcast_scraper.cli", "--output-dir", <corpus>, … "--config", <operator.yaml>, …]`.
2. **`spawn_pipeline_subprocess`** (unless tests set `app.state.jobs_subprocess_factory`) runs
   **`asyncio.create_subprocess_exec(*argv, …, cwd=<corpus>)`**.

So when an operator clicks **Run job** in the viewer, the pipeline runs as a **child process
of the uvicorn process**, using **the same Python interpreter** as the API server. On a
developer machine (`make serve-api` with a full venv), that matches expectations.

### Native vs Docker — two supported workflows {:#native-vs-docker}

The product keeps **both** execution stories:

| Workflow | Typical entry | Job spawn | Operator YAML |
|----------|---------------|-----------|-----------------|
| **Native (default today)** | `make serve-api`, laptop venv, tests | **Subprocess** of `serve` — [RFC-077](RFC-077-viewer-feeds-and-serve-pipeline-jobs.md) argv + `create_subprocess_exec` | **Unchanged** from shipped RFC-077: no Docker-only keys required for PUT or for jobs. |
| **Docker stack** | `docker compose -f compose/docker-compose.stack.yml up` + viewer | **#660** (shipped): delegate to **`pipeline`** / **`pipeline-llm`** via factory + host Docker socket (RFC Option **B**) | **Docker path only:** **`pipeline_install_extras: ml \| llm`** in **`viewer_operator.yaml`** (aligns with **`docker/pipeline/Dockerfile`** **`INSTALL_EXTRAS`**) **required** when the server is configured to spawn jobs into containers — omission → **400** with a clear message. Gate on **`PODCAST_PIPELINE_EXEC_MODE=docker`**. Optional **`PODCAST_DOCKER_COMPOSE_FILES`** lists compose files passed to `docker compose -f` (default `compose/docker-compose.stack.yml` only); merge **`compose/docker-compose.jobs-docker.yml`** at **`up`** time so **`api`** gets the socket and env. |

**Principle:** never require Docker-only metadata for operators who only ever run **native** subprocess jobs; validate **`pipeline_install_extras`** (and profile↔tier checks) only on the **Docker enqueue/spawn** path.

### Phase 1 (this RFC, #659): compose capability without changing job spawn semantics

Delivering **`compose/docker-compose.stack.yml`** adds a **parallel, ops-first** execution path:

- **`docker compose run --rm pipeline`** runs the existing pipeline image against the shared
  volume (cron, manual ops, RFC-078 smoke, deploy scripts).

**Phase 1 does not** by itself:

- Mount the Docker socket into `api`
- Set `app.state.jobs_subprocess_factory` for compose
- Replace subprocess spawn with HTTP to a worker

Therefore **after Phase 1**, behavior is:

| Where the API runs | What "Run job" does |
|--------------------|---------------------|
| Laptop (`make serve-api`, full venv) | Same as today — subprocess `cli` on the host |
| Docker Compose (`api` container) | Subprocess **`cli` inside `api`** — requires that image to carry **everything** `build_pipeline_argv` needs, **or** jobs will fail until Phase 2 |

The dedicated **`pipeline` service** image is the right place for ML; **`api`** should stay
thin **once** Phase 2 delegates execution.

### Phase 2 (#660): close the loop (pick one strategy)

Implement and document **one** primary strategy for containerized deployments (dev/prod
matrix is allowed but must be explicit):

| Option | Description | Trade-off |
|--------|-------------|-----------|
| **A — Fat `api` image** | Install full `cli` + ML extras in `api` so the existing subprocess path works unchanged. | Large `api` image; duplicates pipeline image purpose; simple mentally. |
| **B — `jobs_subprocess_factory`** | On compose startup, register a factory that runs `docker compose run` / `docker run` for the pipeline image, wiring stdout to the job log path. | Requires Docker-in-Docker or **host socket** mount; security and path discipline must be documented. |
| **C — Worker service** | Add a small worker (or `pipeline` in worker mode) that claims jobs from disk/queue and runs one job per process; `api` stays slim. | More moving parts; clean separation. |

**Recommendation:** prefer **B or C** for production-like compose; use **A** only as a
deliberate short-term bridge if time-to-green matters more than image size.

**Tracking:** [GitHub #660](https://github.com/chipi/podcast_scraper/issues/660).

#### #660 implementation checklist (paste / track in the issue)

1. **Exec mode:** **`PODCAST_PIPELINE_EXEC_MODE`** selects **native** (unset / not `docker`: current subprocess) vs **`docker`** (factory / `docker compose run`). Native path: **no** new mandatory operator keys.
2. **Image selection:** when mode is **docker**, operator YAML used for the job must include **`pipeline_install_extras`** ∈ `{ ml, llm }` — maps to **`INSTALL_EXTRAS`** / the correct compose **service** (e.g. `pipeline` vs `pipeline-llm` once the LLM tier exists). **No** silent default to `ml` on omission.
3. **Profile↔tier:** optional script or test gate (see gap matrix / `make verify`) so packaged profiles do not declare capabilities the chosen image lacks.
4. **Secrets:** keys only via `.env` / CI secrets — never committed; compose uses `${VAR:-}` pass-through (see `DOCKER_SERVICE_GUIDE` + §Secrets in internal plan).
5. **Docs:** update this RFC’s rollout row, [DOCKER_SERVICE_GUIDE.md](../guides/DOCKER_SERVICE_GUIDE.md), and [RFC-077](RFC-077-viewer-feeds-and-serve-pipeline-jobs.md) compose extension after merge.
6. **Optional follow-ups:** skim [§Optional follow-ups](#optional-follow-ups) — close or defer items explicitly in the PR (e.g. new `viewer_operator.docker.example.yaml`).

## Alternatives Considered

1. **FastAPI serves static files (single container)**
   - **Description**: Use the existing `StaticFiles` mount in `create_app()` to serve the
     Vue build directly from the API container
   - **Pros**: Simpler (one fewer container, no Nginx config)
   - **Cons**: No caching headers, no gzip, no SPA fallback without custom middleware,
     mixes concerns, harder to scale later
   - **Why Rejected**: User decision — prefer production-grade from day one

2. **Pipeline as long-running service with internal scheduler**
   - **Description**: Pipeline container stays up, runs ingestion on a cron schedule
   - **Pros**: Models stay warm, no cold start
   - **Cons**: 2-4 GB idle RAM, memory leak risk, more complex process management
   - **Why Rejected**: Resource waste on single-host; one-shot with host cron is simpler

## Testing Strategy

**Validation of compose topology:**

- `docker compose -f compose/docker-compose.stack.yml build` succeeds (CI)
- `docker compose -f compose/docker-compose.stack.yml up -d` starts viewer + API
- `/api/health` returns 200 from the Nginx port (proves proxy works)
- `http://localhost/` serves the Vue SPA (proves static serving works)
- `docker compose run --rm pipeline --help` prints service help (entrypoint forwards `--help`)

**Integration with RFC-078:**

- The smoke test workflow layers `compose/docker-compose.smoke.yml` on top
- Pipeline runs against fixture feeds, writes to `smoke_data` volume
- API reads the output, Playwright tests the viewer through Nginx

**Where it is tracked:** Stack contracts and `compose/docker-compose.stack.yml` are **[#659](https://github.com/chipi/podcast_scraper/issues/659)** (handoff checklist there). Smoke workflow + Playwright are **RFC-078** and must be opened as **their own GitHub issues** when you start tracking that work — not as orphan bullets only in this RFC.

## Rollout

1. **Phase 1a** — **[#659](https://github.com/chipi/podcast_scraper/issues/659):** Create `docker/viewer/Dockerfile`, `docker/viewer/nginx.conf`,
   `docker/api/Dockerfile`, `compose/docker-compose.stack.yml`. Validate locally with
   `stack-up` and manual browser test; validate `docker compose run` for `pipeline`.
2. **Jobs / Docker** — **[#660](https://github.com/chipi/podcast_scraper/issues/660):** Jobs API ↔ Docker — implement chosen option (fat `api`, factory + Docker,
   or worker); document **native vs compose** operator behavior (see §Native vs Docker).
3. **RFC-078 smoke:** Add `compose/docker-compose.smoke.yml` and wire into CI smoke workflow (orthogonal to #660; can land in parallel). **Not tracked in this RFC as orphan work** — open **GitHub issues** for RFC-078 execution; **[#659](https://github.com/chipi/podcast_scraper/issues/659)** only carries the stack handoff checklist.
4. **Phase 3** — **[#659](https://github.com/chipi/podcast_scraper/issues/659):** Shipped starter
   [`compose/docker-compose.prod.yml`](https://github.com/chipi/podcast_scraper/blob/main/compose/docker-compose.prod.yml) (restart policies + commented VPS / external volume hints). Operators fork or extend for real prod.

**Success Criteria:**

1. `docker compose -f compose/docker-compose.stack.yml up -d` starts viewer + API in under 60s
   (pre-built images)
2. `http://localhost:${VIEWER_PORT:-8080}/` serves the Vue viewer; `/api/health` returns 200 via Nginx
3. `docker compose run --rm pipeline` completes ingestion and API serves the new data
   without restart

## Pipeline image tiers and profile compatibility {:#pipeline-image-tiers}

The pipeline service in `compose/docker-compose.stack.yml` reuses `docker/pipeline/Dockerfile` with two
build args: **`STACK_PIPELINE_INSTALL_EXTRAS`** and **`STACK_PIPELINE_PRELOAD_ML`**. These
produce two **image tiers** (same binary, different dependency surface):

| Tier | `INSTALL_EXTRAS` | `PRELOAD_ML_MODELS` | Approx size | What it can run |
|------|-----------------|---------------------|-------------|-----------------|
| **ML** (default) | `ml` | `true` (or `false` for faster dev builds) | 3-4 GB | Any profile: local Whisper, spaCy NER, transformers summarization, FAISS index build, SummLlama, **plus** cloud LLM calls |
| **LLM** (`pyproject` **`.[llm]`**) | `llm` | N/A (skipped) | ~1–1.5 GB (target) | Cloud/API-heavy profiles (e.g. **`cloud_thin`**) with **no** local torch/spaCy/FAISS stack; pairs with **`pipeline_install_extras: llm`** on the Docker job path |
| **Core / minimal** | `""` | N/A (skipped) | smallest | Bare pipeline without optional groups — dev-only or legacy; prefer **`llm`** tier once implemented for API-only profiles |

### Profile → minimum image tier

| Profile | Transcription | NER | Summary | GI/KG source | Vector search | **Minimum tier** |
|---------|--------------|-----|---------|--------------|---------------|-----------------|
| `config/profiles/airgapped.yaml` | Whisper (local) | spaCy trf (local) | SummLlama (local) | summary_bullets | FAISS (local) | **ML** |
| `config/profiles/local.yaml` | Whisper (local) | spaCy trf (local) | Ollama (local daemon) | provider | FAISS (local) | **ML** |
| `config/profiles/cloud_balanced.yaml` | OpenAI whisper-1 (API) | spaCy trf (local) | Gemini (API) | provider | FAISS (local) | **ML** |
| `config/profiles/cloud_quality.yaml` | OpenAI whisper-1 (API) | spaCy trf (local) | Anthropic (API) | provider | FAISS (local) | **ML** |
| `config/profiles/cloud_thin.yaml` | Cloud API only | Cloud API only | Cloud API only | provider | `false` | **LLM** (`INSTALL_EXTRAS=llm`) |

**Key insight:** today's "cloud" profiles (`cloud_balanced`, `cloud_quality`) still require
**spaCy** for NER and **FAISS** for vector indexing, so they need the **ML** tier. The
**`llm`** install tier (and **`cloud_thin`**) spans **stack compose / images / validator** (**[#659](https://github.com/chipi/podcast_scraper/issues/659)** for what is not already merged) and **Docker job validation** (**[#660](https://github.com/chipi/podcast_scraper/issues/660)** for `pipeline_install_extras: llm` path).

### Recommended compose usage

```bash
# Default: ML tier (works with any profile)
make stack-build
CONFIG_FILE=$PWD/config/profiles/cloud_balanced.yaml make stack-run-pipeline

# Dev: faster build for API-only profiles — today often `STACK_PIPELINE_INSTALL_EXTRAS=""`
# Once the `llm` tier lands, prefer: STACK_PIPELINE_INSTALL_EXTRAS=llm
STACK_PIPELINE_INSTALL_EXTRAS="" STACK_PIPELINE_PRELOAD_ML=false make stack-build
CONFIG_FILE=$PWD/my-llm-only-profile.yaml make stack-run-pipeline
```

## Optional follow-ups (deferred) {:#optional-follow-ups}

Each item below is **also** listed on **[#659](https://github.com/chipi/podcast_scraper/issues/659)** or **[#660](https://github.com/chipi/podcast_scraper/issues/660)** so nothing lives only here.

1. **RFC index** — [index.md](index.md) **Open RFCs (detail)** table now includes one-line rows for **RFC-078** and **RFC-079** (smoke vs stack, issue pointers).
2. **Example operator YAML** — **Done:** [`config/examples/viewer_operator.example.yaml`](https://github.com/chipi/podcast_scraper/blob/main/config/examples/viewer_operator.example.yaml) stays **native-default** (no `pipeline_install_extras`); Docker path documented in [`config/examples/viewer_operator.docker.example.yaml`](https://github.com/chipi/podcast_scraper/blob/main/config/examples/viewer_operator.docker.example.yaml). **#660** closure tracks guide/RFC cross-checks only.
3. **Automated coverage** — **#660** includes unit tests for the Docker factory helpers, integration tests for Docker-mode validation (with a fake subprocess factory), **`make verify-stack-profiles`** in CI when `config/profiles/**`, `compose/**`, the tier validator script, or **`python-app.yml`** changes, and **manual** stack + `jobs-docker` acceptance recorded on **#660**. Merge-blocking **real** `docker compose run` inside GitHub-hosted runners is **out of scope** for **#660** (open a **new** issue if required).

## Operational contracts (quick reference)

| Topic | Source of truth |
| ----- | ---------------- |
| Stack compose + images | [`compose/docker-compose.stack.yml`](https://github.com/chipi/podcast_scraper/blob/main/compose/docker-compose.stack.yml), [`docker/pipeline/Dockerfile`](https://github.com/chipi/podcast_scraper/blob/main/docker/pipeline/Dockerfile), [`docker/api/Dockerfile`](https://github.com/chipi/podcast_scraper/blob/main/docker/api/Dockerfile), [`docker/viewer/Dockerfile`](https://github.com/chipi/podcast_scraper/blob/main/docker/viewer/Dockerfile) |
| Makefile targets | `Makefile` (`stack-*`, `smoke-*`, `verify-stack-profiles`) |
| Secrets / `.env` | [`DOCKER_SERVICE_GUIDE.md` § Full stack → Secrets (stack)](../guides/DOCKER_SERVICE_GUIDE.md#secrets-stack) |
| Native vs Docker jobs | [§Native vs Docker](#native-vs-docker); [#660](https://github.com/chipi/podcast_scraper/issues/660) |
| Profile ↔ image tier | `scripts/tools/validate_profile_docker_tier.py`; RFC-079 [§Pipeline image tiers](#pipeline-image-tiers) |
| Ephemeral CI smoke | [RFC-078](RFC-078-ephemeral-acceptance-smoke-test.md), `compose/docker-compose.smoke.yml`, `make smoke-*` |
| Prod-style merge (restart, VPS notes) | [`compose/docker-compose.prod.yml`](https://github.com/chipi/podcast_scraper/blob/main/compose/docker-compose.prod.yml) + `DOCKER_SERVICE_GUIDE` § RFC-079 backlog |

## Open Questions

1. **VITE\_\* build-time env vars**: Resolved — the SPA uses relative `/api/` paths;
   Nginx proxies to `api:8000`. No `VITE_API_BASE_URL` needed. The only reference to
   `127.0.0.1:8000` is in `vite.config.ts` (dev proxy), which is not used at build time.
2. **Hot reload for dev**: **Resolved (Won’t ship)** — no `compose/docker-compose.dev.yml`.
   Use `make serve-api` / `make serve-ui` / `make serve` for hot reload; Compose targets CI/prod-like runs.
   See `docs/guides/DOCKER_SERVICE_GUIDE.md` § **RFC-079 backlog** → *Compose “dev override”*.
3. **Config file mounting**: Resolved — `CONFIG_FILE` env var (default `./config.yaml`)
   is bind-mounted at `/app/config.yaml` in both `api` and `pipeline`. The example config
   `config/examples/docker-stack.example.yaml` ships with `output_dir: /app/output`.
4. **API read-only volume race**: **Documented** — many small writes use atomic temp+replace;
   full corpus runs remain multi-file. **Operational rule:** finish `pipeline` before serving new
   data from `api` on the same volume (matches RFC-078 ordering). See `DOCKER_SERVICE_GUIDE.md`
   § **RFC-079 backlog** → *Concurrent pipeline writes and API reads*.
5. **FAISS index reload**: **Resolved** — `/api/search` loads `FaissVectorStore` from disk per request,
   so a **completed** pipeline write set is visible on the **next** search without restarting `api`.
   Use **`POST /api/index/rebuild`** for an explicit rebuild. See `DOCKER_SERVICE_GUIDE.md` § **RFC-079 backlog** → *FAISS / vector index*.
6. **Jobs in compose**: Default remains subprocess in `api`. **`PODCAST_PIPELINE_EXEC_MODE=docker`**
   (with Docker socket + `PODCAST_DOCKER_PROJECT_DIR` + operator **`pipeline_install_extras`**) runs jobs via `docker compose run` into
   **`pipeline`** / **`pipeline-llm`**; see `podcast_scraper.server.pipeline_docker_factory` and **DOCKER_SERVICE_GUIDE** § Full stack. **Shipped** under **[#660](https://github.com/chipi/podcast_scraper/issues/660)**.
7. **LLM pipeline tier vs cloud profiles**: Shipped **`cloud_thin.yaml`** + **`INSTALL_EXTRAS=llm`**
   (`pipeline-llm` service) pairs thin cloud-only runs with the **LLM** image tier. **`cloud_balanced`**
   / **`cloud_quality`** remain **ML** tier (spaCy + FAISS).
