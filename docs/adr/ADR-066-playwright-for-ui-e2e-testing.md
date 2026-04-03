# ADR-066: Playwright for UI End-to-End Testing

- **Status**: Accepted
- **Date**: 2026-04-03
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-062](../rfc/RFC-062-gi-kg-viewer-v2.md)
- **Related ADRs**: [ADR-020](ADR-019-standardized-test-pyramid.md)

## Context & Problem Statement

The viewer v2 (RFC-062) introduces a Vue 3 SPA with Cytoscape.js graph rendering,
search integration, and dashboard views. The existing test pyramid (ADR-020) covers
Python unit/integration/E2E but has no browser-based testing layer. Graph rendering,
search-to-graph focus, filter toggles, and dark mode are visual behaviors that cannot
be tested without a real browser. The project needs a UI E2E framework that is CI
friendly, lightweight, and extensible to future platform views.

## Decision

We adopt **Playwright** as the browser E2E test framework:

1. **Test runner**: Playwright Test with TypeScript.
2. **Browser**: Headless Chromium by default; Firefox and WebKit for extended runs.
3. **Test location**: `web/gi-kg-viewer/e2e/tests/` alongside the frontend source.
4. **Test fixtures**: Sample `gi.json`, `kg.json`, and pre-built FAISS index in
   `e2e/fixtures/` for deterministic, self-contained tests.
5. **Server management**: Playwright `webServer` config starts the FastAPI backend
   and Vite dev server automatically. Tests run against `localhost:8100`.
6. **CI integration**: `make test-ui-e2e` runs Playwright in headless mode. Part of
   `make ci` (full suite), not `make ci-fast`.
7. **Failure artifacts**: Screenshots on failure, trace files for debugging. Retained
   in CI artifacts.

## Rationale

- **Official Vue recommendation**: Vue docs recommend Playwright for E2E testing.
- **Headless by default**: Runs in CI without display server. No Electron dependency
  (unlike Cypress).
- **Multi-browser**: Chromium, Firefox, WebKit in one framework. Important for
  detecting rendering differences in Cytoscape.js.
- **Built-in web assertions**: `expect(page.getByRole('button')).toBeVisible()` —
  no custom wait logic.
- **Lightweight**: ~50 MB install vs Cypress ~300 MB. Faster CI setup.
- **Test isolation**: Each test gets a fresh browser context. No cross-test state
  leakage.
- **Extensible**: Platform views (#50, #347) add tests in the same structure
  (`e2e/tests/platform/`).

## Alternatives Considered

1. **Cypress**: Rejected; Electron-based (~300 MB), no native multi-browser,
   `cy.visit()` model less ergonomic for SPA navigation, heavier CI footprint.
2. **Puppeteer**: Rejected; Chromium-only, lower-level API, no built-in test runner
   or assertions.
3. **Selenium + WebDriver**: Rejected; heavy setup, slower, less ergonomic for modern
   SPA testing.
4. **No E2E tests (manual testing only)**: Rejected; graph rendering, search-to-graph
   focus, and dark mode are regression-prone and need automated coverage.

## Consequences

- **Positive**: Automated browser regression testing. CI catches visual and functional
  breaks. Test structure grows with frontend (viewer now, platform later).
  Extends the test pyramid (ADR-020) with a UI layer.
- **Negative**: Adds Node.js tooling to CI. Playwright install downloads browser
  binaries (~50 MB). E2E tests are slower than unit tests (~30s per test).
- **Neutral**: Requires test fixtures (sample artifacts, small FAISS index) that must
  be maintained alongside the test suite.

## Implementation Notes

- **Test dir**: `web/gi-kg-viewer/e2e/`
- **Config**: `web/gi-kg-viewer/e2e/playwright.config.ts`
- **Makefile**: `make test-ui-e2e` (headless), `make test-ui-e2e-ui` (interactive)
- **Fixtures**: `e2e/fixtures/sample-gi.json`, `sample-kg.json`, `sample-index/`
- **CI**: Runs in `make ci` (full suite). Screenshots/traces uploaded as artifacts on
  failure.
- **Pattern**: Each spec file maps to a component group (graph, search, dashboard,
  offline)

## References

- [ADR-019: Standardized Test Pyramid](ADR-019-standardized-test-pyramid.md) — extended
  with UI E2E layer
- [RFC-062: GI/KG Viewer v2 — Testing Strategy](../rfc/RFC-062-gi-kg-viewer-v2.md)
- [Playwright Documentation](https://playwright.dev/)
