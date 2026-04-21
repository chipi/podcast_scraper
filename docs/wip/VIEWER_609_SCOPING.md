# Viewer UI #609 — Scoping

Scoping doc for "Viewer UI: insight cluster integration + explore
expansion features" (#609). Captures the breakdown into shippable slices
so the work can be picked up in small PRs.

## State of the viewer (inventory)

- `web/gi-kg-viewer/src/components/explore/ExplorePanel.vue` — exists.
- `web/gi-kg-viewer/src/components/search/{SearchPanel,ResultCard,…}.vue` — exists.
- Playwright specs touching clusters: `sibling-merge-cluster-mocks.spec.ts`,
  `explore-supporting-quotes-mocks.spec.ts` — some cluster UI already
  lives in the graph / explore surface.
- No grep hits for `insight_cluster`, `explore-quotes`, or `topic-insights`
  in `web/gi-kg-viewer/src/` → the five #609 features are **all new UI work**.

## Backend status (what the UI can rely on)

- `insight_clusters.json` artifact — **shipped**.
- `topic_clusters.json` artifact — **shipped**.
- `gi.json` multi-quote per insight — **shipped** (#600).
- Explore expansion CLI (`gi explore --expand-clusters`, `gi explore-quotes`,
  `gi topic-insights`, `gi explore --sort evidence-density`) — **shipped** (#601).
- Post-reingestion validation harness — **ready** (this PR's scaffold).

All data is file-based; **no new API endpoints needed**. The viewer
already fetches artifacts by relative path from the mounted corpus.

## Slice plan (5 features → 5 PRs)

Each slice is ~0.5 day of TS/Vue + Playwright. Ordered by user value:

### Slice 1 — Cluster-expanded search results (highest value)

**User value.** When a search matches an insight that belongs to a cluster,
the user sees evidence from *other* episodes — cross-episode corroboration.

**Changes.**

- `SearchPanel.vue` / `ResultCard.vue`: read `insight_clusters.json`;
  badge matching results with cluster membership + member count.
- New expandable "Cross-episode evidence" section in the detail panel
  showing quotes from cluster-member insights.
- E2E spec: `search-cluster-expansion-mocks.spec.ts`.
- Update `e2e/E2E_SURFACE_MAP.md` with new cluster-badge data-testid.

### Slice 2 — Corpus Insights tab (cluster browse)

**User value.** Top-down view of what the corpus *claims*, sorted by
evidence strength.

**Changes.**

- New `src/components/corpus-insights/CorpusInsightsPanel.vue` (mirrors
  existing tab structure).
- New tab in the main shell alongside Topics / Entities.
- Sort toggle: member count vs evidence density.
- Expand to show member insights + quotes.
- E2E spec: `corpus-insights-tab-mocks.spec.ts`.
- Update `UXS-001-gi-kg-viewer.md` with tab tour.

### Slice 3 — Quote-level search (mode toggle)

**User value.** Find verbatim quotes, not just insight claims.

**Changes.**

- `SearchPanel.vue`: mode toggle "Insights | Quotes".
- Quote mode reads FAISS `doc_type: "quote"` (already indexed).
- `ResultCard.vue`: quote variant with speaker + episode context.
- E2E: extend `search-*.spec.ts` with quote-mode assertions.

### Slice 4 — Topic × Insight matrix

**User value.** "What is claimed about this topic?" — surfaces the ABOUT
edges on a topic node.

**Changes.**

- Graph node detail panel: when a Topic node is activated, query
  `insight_clusters.json` for clusters whose canonical insight has an
  ABOUT edge to this topic.
- Show cluster list + canonical insight per cluster.
- E2E spec: `topic-insights-mocks.spec.ts`.

### Slice 5 — Evidence density visual weight

**User value.** High-density clusters visually pop in the graph.

**Changes.**

- Graph node styling (Cytoscape): dynamic border width / node size based
  on `evidence_density` field.
- Confidence badge component.
- E2E: assert density-weighted styling on representative fixture.

## Risks & gotchas

- Playwright specs already have a cluster file
  (`sibling-merge-cluster-mocks.spec.ts`) — **reuse fixture patterns**,
  do not diverge.
- `docs/uxs/UXS-001-gi-kg-viewer.md` is load-bearing for visual contract —
  update on any visible change per CLAUDE.md GI/KG rule.
- `docs/guides/AGENT_BROWSER_LOOP_GUIDE.md` *symmetry rule*: if you
  reproduce a bug in the browser, validate the fix in the browser too,
  not just Playwright.
- `make test-ui` + `make test-ui-e2e` must pass before merge.

## Prerequisites

All backend work is **done**. Slice 1 can start immediately.

## References

- Backend PRs: #611 (cluster ingestion), #611 (multi-quote),
  #603 (explore expansion CLI), #646 (mega-bundle — produces the
  insight-cluster inputs these features consume).
- UXS-001-gi-kg-viewer.md — viewer experience contract.
- E2E_SURFACE_MAP.md — Playwright automation contract.
