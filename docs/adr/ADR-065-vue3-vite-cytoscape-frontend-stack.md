# ADR-065: Vue 3 + Vite + Cytoscape.js Frontend Stack

- **Status**: Accepted
- **Date**: 2026-04-03
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-062](../rfc/RFC-062-gi-kg-viewer-v2.md)
- **Related Issues**: [#445](https://github.com/chipi/podcast_scraper/issues/445), [#489](https://github.com/chipi/podcast_scraper/issues/489)

## Context & Problem Statement

The viewer v1 (#445) used vanilla JavaScript with CDN-loaded vis-network and
Cytoscape.js in parallel pages. The v2 rebuild (RFC-062) needs a proper frontend
framework, a single graph engine, and modern build tooling. These technology choices
affect all current and future UI work — the viewer now, and platform UI views
(#50, #347) later.

## Decision

We adopt the following frontend stack for all UI work:

1. **Vue 3** with Composition API and `<script setup>` SFCs.
2. **Vite** as build tool (dev server with HMR, production static build).
3. **Cytoscape.js** as the sole graph engine (vis-network dropped).
4. **Pinia** for state management (replaces `window.GiKgViz` globals).
5. **Tailwind CSS** for styling (replaces custom 600+ line stylesheet).
6. **TypeScript** for type safety across components, stores, and API client.
7. **Chart.js 4** (via vue-chartjs) for distribution and stats charts.
8. **vue-router** for view navigation (Explore, Dashboard, Detail, future platform).

## Rationale

**Vue 3 over React:**

- Lighter bundle (~16 KB vs ~42 KB gzipped). Payload is dominated by Cytoscape +
  Chart.js; framework overhead matters.
- Natural reactivity for filter/search/graph state without `useCallback`/`useMemo`.
- Single-file components (template + script + style) are cleaner for a small codebase.
- Composition API provides the same composable/hooks pattern.

**Cytoscape.js over vis-network:**

- Richer programmatic API for computed styles, filtering, and batch operations.
  Critical for search-result-to-node highlighting.
- Compound nodes for feed clustering and semantic grouping.
- Extension ecosystem: cola/dagre layouts, popper tooltips, cxtmenu context menus.
- WebGL escape hatch (`cytoscape-canvas`) for large corpora.
- Better TypeScript support and community momentum.

**Tailwind over component library:**

- The v1 `styles.css` needs rewriting regardless. A component library (PrimeVue,
  Vuetify) adds weight and fights data-visualization UI. Tailwind gives dark mode,
  responsive layout, and consistent spacing with full design control.

## Alternatives Considered

1. **React + Next.js**: Rejected; heavier bundle, SSR unnecessary for a local tool, no
   SEO requirements. More boilerplate for state management.
2. **Svelte + SvelteKit**: Rejected; smallest bundle but smaller ecosystem for
   Cytoscape/Chart.js integration. Fewer community examples for this use case.
3. **Keep vis-network alongside Cytoscape**: Rejected; the v1 comparison served its
   purpose. Doubles graph code for no ongoing value.
4. **PrimeVue or Vuetify component library**: Rejected for v2.6; adds weight and
   opinion. Can be added for platform CRUD views in v2.7 if needed.

## Consequences

- **Positive**: Modern, type-safe frontend with fast dev experience (Vite HMR). Single
  graph engine halves graph code. Scales cleanly from viewer to platform UI.
- **Negative**: Requires npm/Node.js in the dev environment (new for a Python project).
  Build step needed for production.
- **Neutral**: Full rebuild of frontend (v1 code not reused). Justified by the
  architectural gap between vanilla JS and Vue 3 component model.

## Implementation Notes

- **Frontend dir**: `web/gi-kg-viewer/`
- **Build**: `cd web/gi-kg-viewer && npm run build` (Vite production build to `dist/`)
- **Dev**: `make serve-ui` (Vite dev server with proxy to FastAPI on port 5173)
- **Unit tests**: `make test-ui` (Vitest — `src/utils/*.test.ts`)
- **Browser E2E**: `make test-ui-e2e` (Playwright, Firefox, Vite on 5174)
- **Graph component**: `src/components/graph/GraphCanvas.vue` wrapping Cytoscape.js
- **Stores**: `src/stores/` — artifacts, search, explore, graphNavigation, graphFilters, shell (Pinia)

## References

- [RFC-062: GI/KG Viewer v2](../rfc/RFC-062-gi-kg-viewer-v2.md)
- [#445: Viewer v1](https://github.com/chipi/podcast_scraper/issues/445) — prior art
- [#489: Viewer v2 implementation](https://github.com/chipi/podcast_scraper/issues/489)
