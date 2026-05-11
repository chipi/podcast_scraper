# ADR-084: Full-Stack Docker Compose Topology (API, Viewer, Pipeline)

- **Status**: Accepted
- **Date**: 2026-05-08
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-079](../rfc/RFC-079-full-stack-docker-compose.md)

## Context & Problem Statement

The stack historically ran as separate laptop processes (`make serve-api`, Vite dev server, ad-hoc
pipeline). That hid production gaps: no Nginx-served Vue build, no single compose story for CI or
VPS deploys, and no shared volume contract between pipeline writers and API readers.

## Decision

We adopt a **three-service Docker Compose** reference topology:

1. **`viewer`** — Nginx serves the production Vue build and reverse-proxies **`/api/*`** to the API.
2. **`api`** — FastAPI **`podcast serve`** reads the corpus from a **bind-mounted or named volume**
   at **`/app/output`** (same layout as local runs), typically **`--no-static`** because Nginx owns
   static assets.
3. **`pipeline`** — **One-shot** **`docker compose run`** (compose **profile** so it does not start
   on `up`); full ML image reuses **`docker/pipeline/Dockerfile`**; writes into the **same**
   **`corpus_data`** volume as the API.

**Single source compose file:** **`compose/docker-compose.stack.yml`** (does not replace
**`compose/docker-compose.yml`**, which remains for standalone pipeline workflows).

**Optional Docker job execution:** When **`PODCAST_PIPELINE_EXEC_MODE=docker`**, the API’s job path
delegates to **`docker compose run`** against **`pipeline`** / **`pipeline-llm`** via a factory and
host Docker socket (**merge file** **`compose/docker-compose.jobs-docker.yml`** — see
**`docs/guides/DOCKER_SERVICE_GUIDE.md`**). **Native subprocess** jobs remain the default for local
laptop development.

## Rationale

- **Prod parity** — VPS deploys use the same image set and volume semantics as CI and local
  **`make stack-test-*`**.
- **Clear boundaries** — Nginx never reads corpus files; API + pipeline share data; viewer is static
  with reverse-proxy only.
- **No Kubernetes** — Explicit non-goal at this scale; Compose is the orchestration ceiling.

## Alternatives Considered

1. **Single mega-container** — Rejected; couples API ML deps with pipeline image bloat and slows
   iteration.
2. **API serves Vue static in-process everywhere** — Rejected for this topology; Nginx matches
   TLS and caching expectations on tailnet / prod.
3. **Swarm or Nomad** — Rejected for same footprint reasons as Kubernetes.

## Consequences

- **Positive**: One **`docker compose up`** story for demos, CI, and operator docs.
- **Negative**: Large **`pipeline`** image pulls; disk and CI time must be budgeted.
- **Neutral**: **`compose/docker-compose.prod.yml`** and host **`deploy.sh`** layer env-specific
   flags without changing the three-service shape.

## Implementation Notes

- **Paths**: `compose/docker-compose.stack.yml`, `docker/api/Dockerfile`, `docker/viewer/Dockerfile`,
  `docker/pipeline/Dockerfile`, `compose/docker-compose.jobs-docker.yml`
- **Docs**: `docs/guides/DOCKER_SERVICE_GUIDE.md`

## References

- [RFC-079: Full-stack Docker Compose topology](../rfc/RFC-079-full-stack-docker-compose.md)
- [ADR-085: Ephemeral stack-test gate](ADR-085-ephemeral-stack-test-integration-gate.md)
- [ADR-082: GitOps app deploy](ADR-082-gitops-app-deploy-via-stack-test-and-gha.md) (promotion context)
