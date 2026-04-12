# ADR-076: Streamlit for Operator Run Comparison and Performance Views

- **Status**: Accepted
- **Date**: 2026-04-11
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-047](../rfc/RFC-047-run-comparison-visual-tool.md),
  [RFC-066](../rfc/RFC-066-run-compare-performance-tab.md)
- **Related PRDs**: [PRD-007](../prd/PRD-007-ai-quality-experiment-platform.md),
  [PRD-016](../prd/PRD-016-operational-observability-pipeline-intelligence.md)

## Context & Problem Statement

[ADR-065](ADR-065-vue3-vite-cytoscape-frontend-stack.md) and [RFC-062](../rfc/RFC-062-gi-kg-viewer-v2.md)
standardize **Vue 3 + Vite** for the **GI/KG viewer** served by FastAPI. Separately,
[RFC-047](../rfc/RFC-047-run-comparison-visual-tool.md) introduced a **Streamlit** app over
`data/eval/` artifacts for **ML run comparison**. [RFC-066](../rfc/RFC-066-run-compare-performance-tab.md)
extended that app with a **Performance** page joining eval runs and **frozen YAML profiles**
([ADR-075](ADR-075-frozen-yaml-performance-profiles-for-release-baselines.md)).

Without an explicit decision, contributors might duplicate run-compare or performance charts inside
the Vue app (splitting maintenance, auth, and data loading) or deprecate Streamlit prematurely.

## Decision

1. **Streamlit remains the home for operator-facing eval tooling**: **`tools/run_compare/`** —
   quality comparisons, diagnostics, and the **Performance** page — stay on **Streamlit + Plotly**
   (optional **`[compare]`** extra), not in `web/gi-kg-viewer/`.
2. **Vue viewer scope**: The SPA focuses on **corpus exploration** (graph, search, library, digest,
   dashboard) against a resolved corpus root and `/api/*` — not on batch eval directory workflows.
3. **Join semantics**: When UI needs both eval metrics and frozen profiles, **release tag** is the
   primary join key ([RFC-066](../rfc/RFC-066-run-compare-performance-tab.md)); implementation
   stays in `tools/run_compare/`.
4. **Optional extra**: Keeping Streamlit behind **`[compare]`** preserves lean installs for users
   who never open eval tools ([RFC-047](../rfc/RFC-047-run-comparison-visual-tool.md)).

## Rationale

- **Different data roots**: Eval runs live under `data/eval/`; the viewer consumes **live corpus**
  roots — merging them in one SPA would couple unrelated release cycles.
- **Velocity**: Streamlit is fast for internal Plotly dashboards; the viewer stack optimizes for
  Cytoscape, Pinia, and Playwright E2E.
- **Clear ownership**: ML operators use `make run-compare`; corpus operators use `podcast serve` +
  viewer.

## Alternatives Considered

1. **Rebuild run compare in Vue + FastAPI**: Rejected; large duplicate of charts, file scanners,
   and session state; slower iteration for eval workflows.
2. **Single “mega” Streamlit for viewer + eval**: Rejected; loses Cytoscape-first UX, typed API
   contracts, and ADR-064 server architecture.
3. **Jupyter-only notebooks for comparison**: Rejected for onboarding; Streamlit gives one command
   and shared README entrypoint.

## Consequences

- **Positive**: Stable split of stacks; RFC-047/066 remain authoritative for Streamlit behavior.
- **Negative**: Two UI stacks to maintain (Python extras vs Node); acceptable given distinct users.
- **Neutral**: Links from docs may point operators to both `make run-compare` and `make serve`.

## Implementation Notes

- **Module**: `tools/run_compare/` (`app.py`, `data.py`, README).
- **Install**: `pip install -e ".[compare]"`; run `make run-compare` or
  `streamlit run tools/run_compare/app.py`.

## References

- [RFC-047: Run comparison visual tool](../rfc/RFC-047-run-comparison-visual-tool.md)
- [RFC-066: Run compare — Performance tab](../rfc/RFC-066-run-compare-performance-tab.md)
- [ADR-064: Canonical server layer](ADR-064-canonical-server-layer-with-feature-flagged-routes.md)
- [ADR-065: Vue 3 + Vite + Cytoscape](ADR-065-vue3-vite-cytoscape-frontend-stack.md)
- [ADR-075: Frozen YAML performance profiles](ADR-075-frozen-yaml-performance-profiles-for-release-baselines.md)
