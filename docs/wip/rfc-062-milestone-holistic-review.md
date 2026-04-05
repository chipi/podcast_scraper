# RFC-062 / viewer v2 — holistic milestone review

**Date:** 2026-04-04 (rev 2)
**Purpose:** Cross-check milestone-driven work (M1–M7) against repo reality; flag gaps,
stale docs, and options to simplify.

## Summary

Implementation has **substantially landed** (`src/podcast_scraper/server/`, `web/gi-kg-viewer/`,
Playwright, `serve` CLI, unit tests under `tests/unit/podcast_scraper/server/`). Several
**planning and RFC artifacts were stale** (still describing "not started" or wrong tooling)
and have since been corrected.

Most **Definition-of-Done items from RFC-062** are now fulfilled: SERVER_GUIDE created,
ARCHITECTURE updated, integration tests added (`tests/integration/test_server_api.py`),
CI installs `[server]` extra so FastAPI tests run. The remaining open item is
**Vitest / `make test-ui`** (frontend unit tests for TS logic).

---

## 1. Stale meta-documents — Resolved

| Artifact | Issue | Resolution |
| -------- | ----- | ---------- |
| `docs/wip/rfc-status-audit.md` | RFC-062 row was **Not Started** | Updated to **Completed (M1-M7)** |
| `docs/wip/adr-status-audit.md` | ADR-065 row tied to "no `.vue`" | Updated to **Implemented** |
| `docs/ROADMAP.md` | RFC-062 **Not Started** | Updated to **Implemented (M1-M7)** |
| RFC-062 body | Chromium/port/path mismatches | Corrected to Firefox, port 5174, actual config path |
| RFC-062 Test Execution | Referenced `make test-ui` (Vitest) | Table updated; Vitest deferred |

---

## 2. RFC-062 Definition of Done — status

From RFC-062 "Documentation Deliverables" and test organization:

| Deliverable | Status (2026-04-04) |
| ----------- | ------------------- |
| `docs/guides/SERVER_GUIDE.md` | **Created** — architecture, endpoints, testing, platform evolution. |
| `docs/architecture/ARCHITECTURE.md` server section | **Updated** — links to Server Guide; server/FastAPI section present. |
| `tests/integration/test_server_api.py` | **Created** — wired `create_app` + real filesystem; `@pytest.mark.integration`. CI installs `[server]`. |
| Vitest + `make test-ui` | **Implemented** — 7 test files, 105 tests covering all `src/utils/*.ts`. CI job `viewer-unit`. |

---

## 3. ~~Dual viewer + dual serve paths~~ — Resolved

v1 (`web/gi-kg-viz/` + `scripts/gi_kg_viz_server.py` + `make serve-gi-kg-viz`) has been
**removed**. Only v2 remains: `cli serve` + built `dist/` + `make serve` / Vite proxy.
Docs now reference a single viewer path.

---

## 4. Milestone artifacts in code — status

| Item | Notes |
| ---- | ----- |
| `server/routes/platform/*.py` | Consolidated into `platform/__init__.py` with a detailed docstring explaining planned endpoints and how to add routes. Individual stub files removed. |
| `create_app()` | `enable_platform: bool = False` parameter added — currently a no-op; stubs exist but no routers implemented yet. |
| Makefile `test-ui-e2e` comment | Corrected from "chromium" to "firefox". |

---

## 5. Testing strategy consistency

**Strengths:** Playwright documented in Testing Strategy + guides; `viewer-e2e` CI job; pytest
`test_viewer_*.py` for API contracts; `test_server_api.py` for wired integration.

**Remaining gaps:**

- Playwright tests currently emphasize **Vite-only** flows; **full stack** (FastAPI + built SPA)
  scenarios are fewer — intentional for speed, but **document the split** ("what CI proves vs
  what manual `serve` proves").
- ~~**Vitest** for pure TS logic (parsers, stores) would reduce reliance on slow browser tests.~~ Done — 7 test files, 105 tests.

---

## 6. Suggested optimization order (remaining)

1. ~~Refresh rfc-status-audit, adr-status-audit, ROADMAP, RFC-062 stale snippets~~ Done.
2. ~~Add SERVER_GUIDE.md~~ Done.
3. ~~Update ARCHITECTURE.md with server section~~ Done.
4. ~~Decide Vitest: add minimal `make test-ui`~~ Done — Vitest installed, 105 tests, `viewer-unit` CI job.
5. ~~`tests/integration/test_server_api.py` with `TestClient` and tmp output dir~~ Done.
6. ~~Remove `web/gi-kg-viz/` + `gi_kg_viz_server.py` per RFC Phase 4~~ Done.

---

## References

- [RFC-062](../rfc/RFC-062-gi-kg-viewer-v2.md)
- [ADR-066](../adr/ADR-066-playwright-for-ui-e2e-testing.md)
- [Testing Strategy](../architecture/TESTING_STRATEGY.md)
- [Server Guide](../guides/SERVER_GUIDE.md)
