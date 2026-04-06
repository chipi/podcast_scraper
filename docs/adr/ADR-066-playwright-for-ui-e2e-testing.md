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
2. **Browser**: **Firefox** (Desktop profile in `playwright.config.ts`); other engines optional.
3. **Test location**: `web/gi-kg-viewer/e2e/*.spec.ts` alongside the frontend source.
4. **Test fixtures**: Deterministic data via `e2e/fixtures.ts`, helpers, and route mocks
   in specs (see repository).
5. **Server management**: Playwright `webServer` starts **Vite** on **127.0.0.1:5174**
   (see `playwright.config.ts`). Offline-focused specs do not require the Python API.
6. **CI integration**: `make test-ui-e2e` runs Playwright headless. GitHub Actions job
   **`viewer-e2e`**; not collected by pytest / `make test`.
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
- **Extensible**: Platform views (#50, #347) can add specs under the same `e2e/` tree.

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
- **Config**: `web/gi-kg-viewer/playwright.config.ts`
- **Makefile**: `make test-ui-e2e` (headless). Interactive: `cd web/gi-kg-viewer && npx playwright test --ui`
- **CI**: Workflow job **`viewer-e2e`**. Screenshots/traces as Playwright HTML report / artifacts on failure.
- **Pattern**: Each spec file maps to a component group (graph, search, dashboard,
  offline)

## Related: Vitest for TS unit tests

This ADR covers **browser E2E** only. Pure TypeScript utility logic (parsing, merge,
metrics, formatting) is tested by **Vitest** (`make test-ui`, `npm run test:unit`),
which runs without a browser in ~150 ms. See
[Testing Guide — Browser E2E](../guides/TESTING_GUIDE.md#browser-e2e-gi-kg-viewer-v2)
for the full viewer test matrix.

## References

- [ADR-019: Standardized Test Pyramid](ADR-019-standardized-test-pyramid.md) — extended
  with UI E2E layer
- [RFC-062: GI/KG Viewer v2 — Testing Strategy](../rfc/RFC-062-gi-kg-viewer-v2.md)
- [Playwright Documentation](https://playwright.dev/)
