# RFC-078: Ephemeral Acceptance Smoke Test (Post-Build Full-Stack Validation)

- **Status**: Draft
- **Authors**: Marko
- **Created**: 2026-04-21
- **Domain**: Infrastructure / DevOps
- **Materialization:** **GitHub issues** carry executable work. This RFC is **design only** — it does not replace an issue backlog. If a bullet has **no GitHub issue**, it is not yet materialized as trackable work.
- **Related RFCs**:
  - `docs/rfc/RFC-079-full-stack-docker-compose.md` (prerequisite — defines the compose topology)
  - `docs/rfc/RFC-077-viewer-feeds-and-serve-pipeline-jobs.md` (jobs API used by viewer)
- **GitHub (stack prerequisite — RFC-079):** [**#659**](https://github.com/chipi/podcast_scraper/issues/659) (Phase 1 — compose topology, images, `stack-*`), [**#660**](https://github.com/chipi/podcast_scraper/issues/660) (Phase 2 — `POST /api/jobs` ↔ Docker pipeline). Details: [RFC-079 §Tracking](RFC-079-full-stack-docker-compose.md).
- **GitHub (smoke acceptance — this design):** **No issue opened yet** for remaining smoke CI work (`workflow_run`, merge policy, BuildKit cache backends, threshold tuning). Open issues and link them here when execution is tracked.

## Abstract

**Target end state:** after every green `main` CI build, a dedicated job exercises the
full-stack Docker Compose topology (RFC-079): fixture feeds → pipeline → shared volume →
FastAPI → Nginx → Playwright, plus **log** and **artifact quality** gates. The environment
is ephemeral — nothing persists after the job.

**Today:** the [`smoke-test.yml`](https://github.com/chipi/podcast_scraper/blob/main/.github/workflows/smoke-test.yml) workflow runs on
**path-filtered pushes to `main`** and **`workflow_dispatch`**, runs `make smoke-*` (build,
pipeline, **assert-logs**, **export-corpus**, **assert-artifacts**, stack up, Playwright),
then tears down. **`workflow_run`** after `python-app`, merge policy, and BuildKit **GHA cache**
backends are **design options** documented below — they are **not** in-flight work until there
is a **GitHub issue** for them. **RFC-079** is the prerequisite: the full-stack Compose topology
is delivered via **#659** and **#660**, both of which implement **RFC-079** (the stack RFC, not this smoke design doc).

**Architecture Alignment:** This RFC depends on the compose topology defined in RFC-079.
It does not define new containers or Dockerfiles — it consumes the existing stack with a
smoke overlay (`compose/docker-compose.smoke.yml`) used **locally** (`make smoke-*`) and in CI.

## Problem Statement

The project has mature CI (lint, unit, integration, e2e, acceptance fixtures, Playwright
viewer tests) but every test tier runs in isolation:

- **Unit/integration tests** run Python code directly, no running services
- **Acceptance fixtures** (`test-acceptance-fixtures`) run the full pipeline but do not
  start the API server or viewer — they inspect static output artifacts
- **Viewer Playwright tests** (`viewer-e2e`) run the Vue dev server with mock API
  responses — they never see real pipeline output

No test in the current CI pipeline exercises the complete path:
**feed XML → pipeline ingestion → corpus output → FastAPI reads output → Nginx serves
viewer → user sees real data in the browser**.

This gap means wiring regressions (schema drift between pipeline output and API readers,
broken FAISS index loading, missing viewer routes for new data types) can reach `main`
undetected and only surface during manual testing or prod deployment.

**Use Cases:**

1. **Schema drift detection**: Pipeline changes the shape of `gi.json`; API deserializer
   breaks; unit tests pass because they use stale fixtures; this RFC catches it
2. **FAISS index wiring**: Pipeline produces a new index format; API's `/api/search`
   returns 500; no existing test covers this path with real index data
3. **Viewer data flow**: Pipeline adds a new field to insights; viewer component reads it;
   Playwright confirms it renders

## Goals

1. Validate the full data path: fixture feed → pipeline → API → viewer → browser
2. **Design target:** required status on every green `main` build (optional `workflow_run` + merge policy) — **needs GitHub issue(s)** before it is tracked execution work
3. Three assertion layers: artifact quality (GIL/KG metrics on exported corpus), log cleanliness (`make smoke-assert-logs`), Playwright e2e
4. Ephemeral — zero state persists after the job
5. Reuse the RFC-079 compose topology with a CI overlay (`compose/docker-compose.smoke.yml`)
6. Reuse existing test infrastructure where possible (acceptance assertions, viewer e2e
   selectors from `E2E_SURFACE_MAP.md`)
7. Complete in under 30 minutes (target: 15-20 minutes for the smoke job itself)

## Constraints and Assumptions

**Constraints:**

- GitHub Actions `ubuntu-latest` runners: 7 GB RAM, 2 CPUs, ~14 GB free disk after OS
- ML model preloading in the pipeline image is the main disk/time bottleneck
- The smoke job must not require secrets (no API keys) — fixture feeds are local, all
  ML inference uses local models
- `workflow_run` (only the default branch) is optional for PR coverage; the **current**
  workflow uses **path-filtered `push` to `main`** instead. Extending triggers is **design only**
  until captured in a **GitHub issue**.

**Assumptions:**

- **RFC-079** is in place enough to run smoke: **Phase 1 (#659)** provides `compose/docker-compose.stack.yml`,
  `docker/viewer/`, `docker/api/`, Makefile `stack-*` targets, example config. **Phase 2 (#660)**
  completes RFC-079’s Docker job path but is not required for “pipeline one-shot + API + viewer” smoke.
  The smoke overlay (`compose/docker-compose.smoke.yml`) layers on top.
- The pipeline supports local file paths as feed sources (or fixture feeds are served
  via a local HTTP server in the compose network)
- The FastAPI server hot-reloads or auto-detects new corpus data when the pipeline writes
  to the shared volume (or the job restarts the API after pipeline completes)
- Playwright smoke tests use **`SMOKE_VIEWER_PORT`** (default **8090** in `compose/docker-compose.smoke.yml` + CI)
- The pipeline image **must be ML tier** (`INSTALL_EXTRAS=ml`) for offline fixture
  execution (no API keys in CI). See §Pipeline image tier for CI smoke.

## Design and Implementation

### Relationship to Existing CI

The smoke test does **not** replace any existing CI job. It adds a new layer on top:

| CI Job | What It Validates | Runs On |
|---|---|---|
| `lint` | Code formatting, types, imports | Every push |
| `test-unit` | Python logic, no services | Every push |
| `test-integration` | Python with ML models, no services | Push to main |
| `test-acceptance-fixtures` | Full pipeline, fixture feeds, artifact quality | Push to main |
| `viewer-e2e` | Vue app with mock API data | Every push |
| **`smoke-test`** (implements **design** in this RFC) | **Full stack: pipeline + API + viewer with real data** | **Push to `main` (path filters) + `workflow_dispatch`** |

The smoke test is the only job that exercises the served stack with real pipeline output.

### Trigger

**Implemented (`.github/workflows/smoke-test.yml`):** `push` to `main` when `compose/**`,
`docker/**`, `config/ci/**`, `tests/smoke/**`, `Makefile`, `pyproject.toml`, or
`docker/pipeline/**` change, plus **`workflow_dispatch`**. The job enables **BuildKit**
(`DOCKER_BUILDKIT`, `COMPOSE_DOCKER_CLI_BUILD`), sets up **Buildx**, runs **layer 2**
(`make smoke-assert-logs`) and **layer 1** (`make smoke-export-corpus` →
`make smoke-assert-artifacts`) before bringing viewer + API up for Playwright.

**Blueprint (no GitHub issue yet):** chain smoke after **`python-app`** with `workflow_run` so
every green main build is followed by smoke (even when paths above did not change), add
`cache-from: type=gha` for compose builds, and decide **merge-blocking** vs post-merge
signal.

```yaml
# Blueprint only — not the active trigger today
on:
  workflow_run:
    workflows: ["Python application"]
    types: [completed]
    branches: [main]
```

### Pipeline Execution

The smoke job builds the compose stack and runs the pipeline as a one-shot service:

```yaml
- name: Build all images
  run: |
    docker compose -f compose/docker-compose.stack.yml -f compose/docker-compose.smoke.yml build

- name: Run pipeline against fixture feeds
  run: |
    docker compose -f compose/docker-compose.stack.yml -f compose/docker-compose.smoke.yml \
      run --rm pipeline \
      2>&1 | tee pipeline.log
```

The `compose/docker-compose.smoke.yml` overlay (layered on top of `compose/docker-compose.stack.yml` from
RFC-079 / #659) points the pipeline at committed fixture feeds and writes output to the
ephemeral `smoke_data` volume. Sketch:

```yaml
# compose/docker-compose.smoke.yml — CI overlay on compose/docker-compose.stack.yml.
# Paths use /app/output to match the implemented stack volume mount.

services:
  api:
    volumes:
      - smoke_data:/app/output
    environment:
      - PODCAST_SCRAPER_CONFIG=/app/config.yaml

  pipeline:
    volumes:
      - smoke_data:/app/output
      - ../tests/fixtures/rss:/app/fixtures/rss:ro
      - ../config/ci/smoke-config.yaml:/app/config.yaml:ro
    environment:
      - PODCAST_SCRAPER_CONFIG=/app/config.yaml
    profiles: []  # Remove profile gate so build/run includes pipeline

  viewer:
    ports:
      - "8080:80"

volumes:
  smoke_data:
    # Ephemeral — destroyed on `down -v`
```

### Mock Feed Fixtures

Static RSS/Atom XML files committed to the repo. The pipeline processes these instead of
fetching from the internet, giving deterministic output.

```text
tests/fixtures/rss/   # canonical path mounted at /app/fixtures/rss in smoke compose
  p01_fast.xml          # default single-feed smoke (see config/ci/smoke-config.yaml)
  …                     # richer multi-feed set — file a GitHub issue before implementing
```

The smoke config (`config/ci/smoke-config.yaml`) points at `p01_fast.xml` today and sets
conservative limits (`max_episodes: 1`, `tiny.en` Whisper). Expand feeds here when the
smoke job needs broader entity/search coverage.

Whether fixtures are served via `file://` paths (volume-mounted into the pipeline
container) or via a lightweight HTTP server in the compose network depends on pipeline
support — see Open Questions.

### Assertion Layers

Three layers run in sequence. Failure at any layer stops the job.

#### Layer 1: Artifact Quality Assertions

Runs after the pipeline completes, before the server starts. Validates the pipeline
produced structurally correct output in the shared volume.

These assertions are the same checks currently enforced by `test-acceptance-fixtures`
and the GIL/KG quality metrics scripts in CI, adapted to run against the smoke output:

- At least N insight nodes were produced (N derived from fixture episode count)
- All insight nodes have required fields (`id`, `body`, `episode_ref`, `grounding`)
- No insight body is empty or below a minimum token length
- KG artifact exists with at least M entity nodes
- Grounding contracts are intact (each insight references a valid episode source)
- Corpus run summary exists and reports no errors

Implementation: call the existing `scripts/tools/gil_quality_metrics.py` and
`scripts/tools/kg_quality_metrics.py` against the smoke output directory, with
`--enforce` and appropriate thresholds.

```yaml
- name: Assert artifact quality
  run: |
    export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
    # GIL quality gate
    python scripts/tools/gil_quality_metrics.py \
      /path/to/smoke_output \
      --enforce --strict-schema --fail-on-errors \
      --min-extraction-coverage 0.8 \
      --min-grounded-insight-rate 0.8 \
      --min-avg-insights 1

    # KG quality gate
    python scripts/tools/kg_quality_metrics.py \
      /path/to/smoke_output \
      --enforce --strict-schema --fail-on-errors \
      --min-artifacts 1 \
      --min-avg-nodes 1 \
      --min-extraction-coverage 0.8
```

#### Layer 2: Log Assertions

Check pipeline output for silent failures — runs that exit 0 but logged errors or
warnings indicating degraded quality.

```yaml
- name: Assert pipeline logs clean
  run: |
    # No ERROR/CRITICAL lines in pipeline output
    if grep -qiE "^(ERROR|CRITICAL)" pipeline.log; then
      echo "Pipeline log contains errors:"
      grep -iE "^(ERROR|CRITICAL)" pipeline.log
      exit 1
    fi

    # Pipeline emitted a completion event
    if ! grep -q "corpus_run_complete\|pipeline_run_complete\|Run complete" pipeline.log; then
      echo "No completion marker in log — pipeline may have exited early"
      exit 1
    fi

    echo "Logs clean"
```

The completion marker regex should match whatever the pipeline currently emits at the end
of a successful run. Verify against actual log output and adjust.

#### Layer 3: Playwright E2E (Smoke Level)

After layers 1-2 pass, start the viewer + API and run a minimal Playwright suite that
confirms data flowed all the way through to the browser.

```yaml
- name: Start viewer and API
  run: |
    docker compose -f compose/docker-compose.stack.yml -f compose/docker-compose.smoke.yml \
      up -d viewer api

- name: Wait for stack healthy
  run: |
    for i in $(seq 1 30); do
      if curl -sf http://localhost:8080/api/health; then
        echo "Stack healthy"
        break
      fi
      echo "Waiting... ($i/30)"
      sleep 2
    done
    curl -f http://localhost:8080/api/health
```

The Playwright smoke suite is intentionally minimal — it confirms data is visible, not
exhaustive UI coverage. The existing `viewer-e2e` tests cover UI behavior with mocks;
these tests cover the data integration.

**`tests/smoke/viewer-smoke.spec.ts`:**

```typescript
import { test, expect } from '@playwright/test';

const BASE = process.env.VIEWER_URL ?? 'http://localhost:8080';

test('app loads without JS errors', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto(BASE);
  await page.waitForLoadState('networkidle');
  expect(errors).toHaveLength(0);
});

test('health endpoint returns OK through Nginx', async ({ request }) => {
  const resp = await request.get(`${BASE}/api/health`);
  expect(resp.ok()).toBeTruthy();
  const body = await resp.json();
  expect(body.status).toBe('ok');
});

test('graph canvas renders nodes from real data', async ({ page }) => {
  await page.goto(BASE);
  await page.waitForSelector('[data-testid="graph-canvas"]', { timeout: 15000 });
  // Verify Cytoscape rendered at least one node
  const nodeCount = await page.evaluate(() => {
    const cy = (window as any).__cy;
    return cy ? cy.nodes().length : 0;
  });
  expect(nodeCount).toBeGreaterThan(0);
});

test('insight panel shows real insights', async ({ page }) => {
  await page.goto(BASE);
  // Use selectors from E2E_SURFACE_MAP.md where available
  const cards = page.locator('[data-testid="insight-card"]');
  await expect(cards.first()).toBeVisible({ timeout: 15000 });
});

test('API returns search results for known fixture term', async ({ request }) => {
  // Fixture feeds contain known entities — search for one
  const resp = await request.get(`${BASE}/api/search?q=podcast`);
  expect(resp.ok()).toBeTruthy();
  const body = await resp.json();
  expect(body.results?.length).toBeGreaterThan(0);
});
```

### Full Workflow

**`.github/workflows/smoke-test.yml`:**

```yaml
name: Smoke Test

on:
  workflow_run:
    workflows: ["Python application"]
    types: [completed]
    branches: [main]

jobs:
  smoke-test:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    if: ${{ github.event.workflow_run.conclusion == 'success' }}

    steps:
      - name: Checkout
        uses: actions/checkout@v6

      - name: Free disk space
        run: |
          sudo rm -rf /usr/share/dotnet /usr/local/lib/android /opt/ghc
          docker image prune -af || true
          sudo apt-get clean && sudo rm -rf /var/lib/apt/lists/*
          df -h

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v4

      - name: Set up Python (for assertion scripts)
        uses: actions/setup-python@v6
        with:
          python-version: "3.11.8"

      - name: Install assertion dependencies
        run: pip install -e '.[dev]'

      - name: Set up Node.js (for Playwright)
        uses: actions/setup-node@v6
        with:
          node-version: "22"

      - name: Install Playwright
        run: |
          cd tests/smoke
          npm ci
          npx playwright install --with-deps chromium

      # ── Build ──────────────────────────────────────────────
      - name: Build compose images
        run: |
          docker compose -f compose/docker-compose.stack.yml \
                         -f compose/docker-compose.smoke.yml build

      # ── Pipeline run ───────────────────────────────────────
      - name: Run pipeline against fixtures
        run: |
          docker compose -f compose/docker-compose.stack.yml \
                         -f compose/docker-compose.smoke.yml \
            run --rm pipeline 2>&1 | tee pipeline.log

      # ── Layer 1: Artifact quality ─────────────────────────
      - name: Assert artifact quality
        run: |
          export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
          # Extract smoke output from volume
          SMOKE_DIR=$(docker volume inspect smoke_data --format '{{ .Mountpoint }}') || true
          # Fallback: copy from container if direct mount path isn't accessible
          if [ ! -d "$SMOKE_DIR" ]; then
            mkdir -p /tmp/smoke_output
            docker compose -f compose/docker-compose.stack.yml \
                           -f compose/docker-compose.smoke.yml \
              cp api:/data/output /tmp/smoke_output
            SMOKE_DIR=/tmp/smoke_output
          fi

          python scripts/tools/gil_quality_metrics.py "$SMOKE_DIR" \
            --enforce --strict-schema --fail-on-errors \
            --min-extraction-coverage 0.8 \
            --min-grounded-insight-rate 0.8 \
            --min-avg-insights 1

          python scripts/tools/kg_quality_metrics.py "$SMOKE_DIR" \
            --enforce --strict-schema --fail-on-errors \
            --min-artifacts 1 \
            --min-avg-nodes 1 \
            --min-extraction-coverage 0.8

      # ── Layer 2: Log assertions ────────────────────────────
      - name: Assert pipeline logs clean
        run: |
          if grep -qiE "^(ERROR|CRITICAL)" pipeline.log; then
            echo "Pipeline log contains errors:"
            grep -iE "^(ERROR|CRITICAL)" pipeline.log
            exit 1
          fi
          grep -q "corpus_run_complete\|pipeline_run_complete\|Run complete" \
            pipeline.log || \
            (echo "No completion marker in log" && exit 1)
          echo "Logs clean"

      # ── Start server ───────────────────────────────────────
      - name: Start viewer and API
        run: |
          docker compose -f compose/docker-compose.stack.yml \
                         -f compose/docker-compose.smoke.yml \
            up -d viewer api

      - name: Wait for stack healthy
        run: |
          for i in $(seq 1 30); do
            curl -sf http://localhost:8080/api/health && break
            echo "Waiting... ($i/30)"
            sleep 2
          done
          curl -f http://localhost:8080/api/health

      # ── Layer 3: Playwright smoke ──────────────────────────
      - name: Run Playwright smoke suite
        env:
          VIEWER_URL: http://localhost:8080
        run: cd tests/smoke && npx playwright test

      - name: Upload Playwright report on failure
        if: failure()
        uses: actions/upload-artifact@v7
        with:
          name: smoke-playwright-report
          path: tests/smoke/playwright-report/

      # ── Teardown ───────────────────────────────────────────
      - name: Tear down
        if: always()
        run: |
          docker compose -f compose/docker-compose.stack.yml \
                         -f compose/docker-compose.smoke.yml down -v
```

## Key Decisions

1. **Post-merge signal, not merge gate**
   - **Decision**: The smoke test runs after merge to `main`, not on PRs
   - **Rationale**: `workflow_run` only fires on the default branch. Making it a PR gate
     would require building and caching images on every PR, which is expensive. The
     existing `test-acceptance-fixtures` + `viewer-e2e` jobs provide pre-merge coverage.
     The smoke test catches integration issues that slip through.

2. **Reuse existing quality scripts, not new assertion code**
   - **Decision**: Layer 1 calls `gil_quality_metrics.py` and `kg_quality_metrics.py`
   - **Rationale**: These scripts are already maintained, enforced in CI, and understand
     the current artifact schema. Writing parallel assertion code would drift.

3. **Minimal Playwright suite, not full viewer coverage**
   - **Decision**: 5-6 smoke tests, not the full `viewer-e2e` suite
   - **Rationale**: The smoke suite validates data integration (real data visible in
     browser), not UI behavior (which `viewer-e2e` covers with mocks). Fewer tests
     mean faster feedback and less maintenance.

4. **Build images on the runner, not pull from registry**
   - **Decision**: `docker compose build` on the runner
   - **Rationale**: No GHCR push exists yet (that comes with prod deploy). Building
     locally avoids registry setup as a prerequisite. BuildKit layer caching in GH Actions
     (`cache-from: type=gha`) keeps rebuild times reasonable.

## Alternatives Considered

1. **Run natively on the runner (no Docker)**
   - **Description**: `pip install -e '.[ml,server]'` + `npm run build` + run everything
     as bare processes
   - **Pros**: Simpler, no Docker build time, reuses existing CI patterns
   - **Cons**: Doesn't validate the Docker topology, can't catch container wiring issues,
     doesn't match the prod deployment model
   - **Why Rejected**: The whole point is to validate the containerized stack

2. **Run the full `viewer-e2e` suite instead of a dedicated smoke suite**
   - **Description**: Point the existing Playwright specs at the smoke stack
   - **Pros**: No new test code to maintain
   - **Cons**: The existing specs use mock API responses via `msw` or fixtures; they would
     need significant refactoring to work against a real API. Also slower.
   - **Why Rejected**: Smoke and viewer-e2e serve different purposes (data integration vs
     UI behavior). A small dedicated smoke suite is cleaner.

## Testing Strategy

The smoke test **is** the test. It validates itself by running in CI. Manual verification
during development:

1. `make smoke-build`
2. `make smoke-run-pipeline` → `make smoke-assert-logs` → `make smoke-export-corpus` → `make smoke-assert-artifacts`
3. `make smoke-up` → `make smoke-test-playwright`

## Rollout

1. ~~**Phase 0** (prerequisite): RFC-079 — compose topology, Dockerfiles~~ **Phase 1 done**
   ([#659](https://github.com/chipi/podcast_scraper/issues/659)); **#660** is **RFC-079 Phase 2**
   (Docker job execution), not RFC-078.
2. **Phase 1**: ~~Compose smoke overlay, fixtures, smoke config, Playwright under `tests/smoke/`, Makefile helpers~~ **Done in tree**; further CI workflow upgrades (`workflow_run`, caches, merge gate) beyond today’s `smoke-test.yml` need **new GitHub issues** — this RFC does not track them.
3. **Phase 2**: **In progress** — `smoke-test.yml` on `main`; monitor flakiness, tune timeouts / **`smoke-assert-artifacts`** thresholds.
4. **Phase 3** (future): Cache images to reduce build time; consider running on PRs.

**Success Criteria:**

1. **Design target:** smoke runs on every green `main` build (`workflow_run`) — **GitHub issue TBD**; **today:** path-filtered pushes complete in under 30 minutes
2. All three assertion layers pass on a healthy codebase (logs + GIL/KG + Playwright)
3. The test catches at least one real integration issue within the first month that
   existing CI would have missed

## Relationship to Other RFCs and Issues

**RFC-079** is the **prerequisite** for **RFC-078**. GitHub **#659** and **#660** are both
**implementations of RFC-079** (Phase 1 and Phase 2); neither issue “is” RFC-078.

1. **RFC-079: Full-Stack Docker Compose Topology** — Nginx + API + Pipeline, shared volume,
   `stack-*` Makefile targets. **#659** closes Phase 1 (compose topology, images, docs).
   **#660** closes Phase 2 (`POST /api/jobs` → **`pipeline`** under Docker; factory / socket /
   worker; **native** subprocess jobs stay default — see [RFC-079 §Native vs Docker](RFC-079-full-stack-docker-compose.md#native-vs-docker)
   and [RFC-077](RFC-077-viewer-feeds-and-serve-pipeline-jobs.md)). Provides `compose/docker-compose.stack.yml`
   as the base file.
2. **RFC-078: Ephemeral Acceptance Smoke Test** (this document — **design only**) — describes how
   to consume that topology via `compose/docker-compose.smoke.yml`, assertion layers, and
   `smoke-test.yml`. **Remaining smoke CI polish has no GitHub issue yet** (see **GitHub (smoke acceptance)** above).
3. **Future: GHCR image push** — CI/release workflow tags and pushes images so VPS can
   `docker compose pull`. Not needed for smoke (builds locally on the runner).
4. **Future: Prod Deploy RFC** — prod compose overrides (TLS, external volumes, resource limits),
   tag-triggered deployment, manual approval gate.

Together, the **design** in RFC-079 and RFC-078 describes a deployment pipeline story:
compose topology → ephemeral validation → (future) gated prod deploy — **materialized** in GitHub issues and code, not by the markdown alone.

## Work split: Smoke acceptance (GitHub TBD) vs GitHub #660 (Docker job execution) {:#rfc078-vs-660}

| Track | Owns | Does **not** own |
| ----- | ---- | ---------------- |
| **Smoke acceptance** (design in this RFC; **GitHub issues TBD**) | Smoke compose overlay, `config/ci/smoke-config.yaml`, RSS fixtures under `tests/fixtures/rss/`, `tests/smoke/` Playwright, Makefile `smoke-*`, **`smoke-test.yml`** (log + artifact gates + Playwright), `smoke-export-corpus` | Changing how `POST /api/jobs` spawns work (subprocess vs Docker factory), `PODCAST_PIPELINE_EXEC_MODE`, host socket mounts, or API image Docker CLI packaging |
| **#660** ([RFC-079 Phase 2](RFC-079-full-stack-docker-compose.md)) | Docker pipeline job factory (`pipeline_docker_factory.py`), compose file list env defaults, operator `pipeline_install_extras`, API container requirements for **Docker** execution mode | Fixture feeds, smoke Playwright suite, **`workflow_run`** orchestration policy, GIL/KG **threshold tuning** (product/QA — **design** here; **GitHub** when filed) |

**Design backlog not yet in a GitHub issue** (not #660): `workflow_run` after green `python-app`, **BuildKit `cache-from: type=gha`** for compose (Buildx is already enabled in `smoke-test.yml`), merge-blocking vs post-merge.

## Pipeline image tier for CI smoke {:#pipeline-image-tier}

The smoke test **must use the ML pipeline image** (`INSTALL_EXTRAS=ml`) because the
airgapped/fixture smoke config needs local Whisper, spaCy, transformers, and FAISS for
deterministic offline execution (no API keys in CI). See **RFC-079 §Pipeline image tiers
and profile compatibility** for the full profile → image mapping.

**CI disk budget implication:** the ML image is 3-4 GB; after `Free disk space`, the
runner has ~14 GB. BuildKit layer cache (`cache-from: type=gha`) keeps rebuilds
incremental, but the first build of a new cache key will be slow (~15 min). Set
`PRELOAD_ML_MODELS=false` for faster CI builds (model download happens at pipeline
runtime from volume-mounted fixtures / Hugging Face cache).

## Dependency on RFC-079 Phase 1 (#659) — resolved {:#dependency-rfc079}

RFC-079 Phase 1 has been implemented:

- `compose/docker-compose.stack.yml`: viewer (Nginx) + api (FastAPI) + pipeline (profile `pipeline`).
- `docker/pipeline/Dockerfile` (pipeline / ML / LLM runner).
- `docker/viewer/Dockerfile` (multi-stage Node 22 + `nginx:alpine`) + `docker/viewer/nginx.conf`.
- `docker/api/Dockerfile` (`.[server]` + CPU torch / FAISS / sentence-transformers).
- `docker/api/entrypoint.sh` (ensures `/app/output` exists before serve).
- `config/examples/docker-stack.example.yaml` (`output_dir: /app/output`, `dry_run: true`).
- Makefile: `stack-build`, `stack-up`, `stack-down`, `stack-logs`, `stack-run-pipeline`.
- Local acceptance passed: `make stack-build`, `stack-up`, `curl /api/health` → 200 via Nginx.

**As implemented in tree** (descriptive; **GitHub issues** still own prioritization and closure — on top of the RFC-079 compose topology):

1. ~~`compose/docker-compose.smoke.yml`~~ — **Done** (smoke volume, fixture mounts, `pipeline` / `pipeline-llm` profiles cleared for one-shot runs).
2. ~~`config/ci/smoke-config.yaml`~~ — **Done** (minimal airgapped smoke; explicit `vector_search` / `vector_backend` for doc clarity).
3. ~~`tests/smoke/` Playwright + `package.json`~~ — **Done** (`stack-viewer.spec.ts`; see `tests/smoke/README.md`).
4. ~~Makefile `smoke-*`~~ — **Done**; **added** `smoke-assert-logs`, `smoke-export-corpus`
   (volume → `.smoke-corpus/`), and `smoke-assert-artifacts` (defaults **`SMOKE_CORPUS_ROOT=$(PWD)/.smoke-corpus`**).
5. **No GitHub issue yet — CI design backlog:** `workflow_run` after `python-app`, **`cache-from: type=gha`** for
   compose image layers, merge gate policy. **`smoke-test.yml`** already runs Buildx + BuildKit
   and layers 1–2 + Playwright on path-filtered `main` pushes.

## Open Questions

1. **Feed fixture delivery**: Does the pipeline support `file://` paths or local
   file paths as feed sources? If not, a tiny HTTP server container (or `python -m
   http.server` sidecar) would need to serve the fixture XML within the compose network.
2. **Runner disk budget**: The ML-enabled pipeline image is 3-4 GB. After freeing disk
   space, the runner has ~14 GB. Build cache, Node modules, Playwright browsers, and
   the image layers need to fit. Measure actual usage on a test run.
3. **API data reload**: After the pipeline writes new data, does the API server need a
   restart to pick it up? Or does it read from disk on each request? If restart is needed,
   the workflow must start the API **after** the pipeline completes (current design does
   this, but document the requirement).
4. **Playwright selector stability**: The smoke tests reference `data-testid` attributes.
   These must exist in the viewer components. Cross-reference with `E2E_SURFACE_MAP.md`
   and add any missing attributes as a prerequisite task.
5. **Completion log marker**: Layer 2 assumes the pipeline logs contain
   `corpus_run_complete` or similar. Verify the actual log event name and update.
6. **FAISS index in smoke**: The smoke profile must set `vector_search: true` and
   `vector_backend: faiss` so the pipeline builds an index for `/api/search` tests. The
   ML image tier includes `faiss-cpu`; the LLM-only tier does not.
