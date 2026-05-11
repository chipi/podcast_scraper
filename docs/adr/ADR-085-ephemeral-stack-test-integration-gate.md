# ADR-085: Ephemeral Stack-Test Integration Gate on Main

- **Status**: Accepted
- **Date**: 2026-05-08
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-078](../rfc/RFC-078-ephemeral-acceptance-smoke-test.md), [RFC-079](../rfc/RFC-079-full-stack-docker-compose.md)

## Context & Problem Statement

Unit, integration, and viewer Playwright tests each cover slices of the system. None guaranteed the
full path **fixture feed → pipeline → on-disk corpus → real FastAPI → Nginx → browser**, so schema
drift between pipeline output and API readers could reach **`main`** undetected.

## Decision

We run an **ephemeral full-stack** validation on **`main`** (path-filtered) and via
**`workflow_dispatch`**, using:

- **`compose/docker-compose.stack-test.yml`** as an **overlay** on
  **`compose/docker-compose.stack.yml`** ([ADR-084](ADR-084-full-stack-docker-compose-topology.md)).
- **`make stack-test-*`** targets to build, seed, run pipeline steps, assert logs and artifact
  quality, bring the stack up, and run **Playwright** under `tests/stack-test/`.
- **`.github/workflows/stack-test.yml`** (workflow name **Stack test**) as the CI orchestration.

The environment is **always torn down**; no persistent CI corpus.

This gate is **distinct** from script-based README acceptance (**[ADR-021](ADR-021-acceptance-test-tier-as-final-ci-gate.md)**):
stack-test proves the **Dockerized** integration path; acceptance proves documented CLI/config
flows.

## Rationale

- **End-to-end signal** — Catches wiring failures (FAISS load, route registration, viewer data flow)
  that mocked or Python-only tests miss.
- **No secrets in CI smoke** — Fixture feeds and local ML profiles keep the job forkable for
  contributors.
- **Composable with `make ci`** — Full local **`make ci`** ends with **`stack-test-ml-ci`** as a
  final gate ([`Makefile` at repo root](https://github.com/chipi/podcast_scraper/blob/main/Makefile) targets **`ci`** / **`_ci_body`**).

## Alternatives Considered

1. **Only increase pytest integration coverage** — Insufficient; does not run real Nginx or
   containerized API with the same entrypoints as prod.
2. **Always-on staging environment as the only gate** — Cost and flake surface; ephemeral compose
   keeps feedback on every relevant **`main`** push.
3. **Merge stack-test into viewer-e2e with mocks** — Rejected; mocks hide real pipeline output shape.

## Consequences

- **Positive**: Regressions in the hot path surface in GitHub Actions logs with compose diagnostics.
- **Negative**: Runtime and disk cost on CI; must keep profile **`airgapped_thin`** (or equivalent)
  bounded.
- **Neutral**: Further triggers (**`workflow_run`**, merge policy, BuildKit cache) remain design
  options tracked in RFC-078 and issues, not required for Phase 1 acceptance of this ADR.

## Implementation Notes

- **Workflow**: `.github/workflows/stack-test.yml`
- **Compose**: `compose/docker-compose.stack-test.yml`
- **Tests**: `tests/stack-test/`, `web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md` (automation contract)

## References

- [RFC-078: Ephemeral acceptance smoke test](../rfc/RFC-078-ephemeral-acceptance-smoke-test.md)
- [ADR-084: Full-stack Compose topology](ADR-084-full-stack-docker-compose-topology.md)
- [ADR-021: Acceptance test tier](ADR-021-acceptance-test-tier-as-final-ci-gate.md)
