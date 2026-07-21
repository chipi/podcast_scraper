# GI / KG viewer — E2E surface map

This document is the **Playwright automation contract** for the GI/KG viewer (`web/gi-kg-viewer`).
It lists surfaces, entry paths, owning specs, and selectors tests rely on. Contributors and agents
also use it when **debugging** viewer issues or driving the UI via tools that consume the **accessibility
tree** (Playwright, Playwright MCP, Chrome DevTools MCP snapshots): it records expected roles, labels,
and disambiguation patterns, not only test selectors. It complements
shell IA [VIEWER_IA.md](../../../docs/uxs/VIEWER_IA.md), [UXS-001](../../../docs/uxs/UXS-001-gi-kg-viewer.md) (shared design system),
and feature UXS files ([UXS-002](../../../docs/uxs/UXS-002-corpus-digest.md) Digest,
[UXS-003](../../../docs/uxs/UXS-003-corpus-library.md) Library,
[UXS-004](../../../docs/uxs/UXS-004-graph-exploration.md) Graph,
[UXS-005](../../../docs/uxs/UXS-005-semantic-search.md) Search,
[UXS-006](../../../docs/uxs/UXS-006-dashboard.md) Dashboard) -- and
[RFC-062](../../../docs/rfc/RFC-062-gi-kg-viewer-v2.md) (technical design); it does not replace them.

**Related:** [ADR-066](../../../docs/adr/ADR-066-playwright-for-ui-e2e-testing.md). Tracked in
[GitHub #509](https://github.com/chipi/podcast_scraper/issues/509).

## Graph navigation — incremental work and structural debt (automation contract)

**Related issues**: [#748](https://github.com/chipi/podcast_scraper/issues/748),
[#749](https://github.com/chipi/podcast_scraper/issues/749).

**Scenario list / acceptance language**: [INCREMENTAL_LOADING_TEST_CRITERIA.md](INCREMENTAL_LOADING_TEST_CRITERIA.md).

**Broader multi-entry E2E backlog**: [GitHub #750](https://github.com/chipi/podcast_scraper/issues/750).

Incremental fixes in Digest, Library, Search, Dashboard, and graph-internal expansion have
landed in scattered places (`DigestView.vue`, `LibraryView.vue`, `NodeDetail.vue`,
`artifacts` store, `graphNavigation`, `GraphCanvas.vue`). They can make one path work and
then regress another because the **underlying architecture is not “one pipeline”**.

### What is structurally wrong (not “one more if” in GraphCanvas)

1. **Two graphs, loosely coupled**
   - **Logical graph**: merged `filteredArtifact` / parsed nodes in Pinia + graph store
     composables.
   - **Rendered graph**: Cytoscape `core` with its own element collection and layout timing.
   - Code often assumes “if the artifact has a node id, `core.$id(id)` is non-empty”. That
     is false after incremental merges until (and sometimes after) `redraw` / `finishLayoutPass`
     completes. Any handoff that resolves an id from artifacts but animates the camera on
     `core` without a **sync barrier** can race.

2. **Many entry points, no single “navigation intent” API**
   - Library **G**, Digest pills / **Open graph**, Search **Show on graph**, Dashboard topic
     landscape, neighbourhood **Open in graph**, graph-internal **Load**, etc. each set
     overlapping state: `subject` (episode meta, optional `episodeId`, optional
     `graphConnectionsCyId`), `nav.pendingFocusNodeId`, `artifacts` selection / append,
     `lastLoadSource`, territory / auto-load watchers, zoom anchor, pending recenter timers.
   - Without one function (or store action) that means **“user asked to show cyId X (or
     episode Y) on the graph after data is on the canvas”**, every new surface re-implements
     corners and ordering.

3. **Episode identity is three-tier and easy to mismatch**
   - Corpus **`metadata_relative_path`** (Library / API).
   - Logical **`episode_id`** (UUID) matching `__unified_ep__:UUID` and
     `logicalEpisodeIdFromGraphNodeId`.
   - Artifact **Episode** rows that may **not** include `metadata_relative_path` in
     `properties` (only `podcast_id`, `title`, etc.), so metadata-only lookup returns null
     even when the episode is in the merged artifact.
   - Handlers that only pass metadata path break until something else sets UUID (e.g. detail
     fetch) and until the canvas actually contains that node.

4. **Incremental append vs full redraw is policy, not physics**
   - The `filteredArtifact` watcher can **return early** (“incremental append, no pending
     focus, not external”) and skip `scheduleRedraw()`. Then the merged artifact grows but
     Cytoscape never receives new elements: symptoms like `COLL_EMPTY` for a resolved id, or
     “first Library G works, second does not”. Load-source flags (`digest-external`,
     `library-external`, `graph-internal`) are band-aids on this policy split.

5. **Camera / selection are separate concerns bolted together**
   - Selection, dimming, zoom anchor, `animateCameraToFocusedNode`, `pendingRecenter`, and
     layout completion all interact. Fixing “pan” in one path without a single post-layout
     hook (“data visible + layout stopped + viewport size known → center target”) repeats
     fragile timing work.

### What we should change (directional, for a future refactor)

Prefer **one orchestration layer** (name illustrative: `navigateGraphToSubject` or Pinia
action on `graphNavigation` + `artifacts`) that **every** UI entry point calls:

1. **Normalize target**: episode → `{ metadataPath, episodeId?, preferredCyId? }` or topic →
   `{ raw graph id }`, etc.
2. **Ensure artifact set** includes the slice (append or replace) with an explicit **intent**:
   external handoff vs internal expand (drives merge / cap / auto-load).
3. **Await or subscribe to “canvas has element”** for the resolved cy id (or register a
   pending focus that the single reducer applies once, not six watchers).
4. **Then** apply selection, dimming, and camera in one place (after layout stop + resize),
   with tests locking the contract.

Until that exists, E2E must assume **regressions are multi-path**: a fix for Digest does not
prove Library or Dashboard.

**Deeper code walk (entry points, stores, failure modes, refactor order):**
[docs/wip/GRAPH_NAVIGATION_HANDOFF_ANALYSIS.md](../../../docs/wip/GRAPH_NAVIGATION_HANDOFF_ANALYSIS.md).

### How agents and humans should validate (this repo’s contract)

- **Playwright** (CI): `make test-ui-e2e` / `make ci-ui-fast`; specs under `e2e/*.spec.ts`;
   `baseURL` for the harness is **5174** (see Runtime table below). Use for repeatable
   regressions once scenarios are encoded.
- **Chrome + DevTools MCP** (user-facing bug reports, visual/timing): mandatory for
   viewer bugs per `docs/guides/AGENT_BROWSER_LOOP_GUIDE.md` — reproduce and re-check the
   **same** clicks after a change; optionally use `window.__GIKG_CY_DEV__` /
   `cy.nodes(':selected')` / Pinia in `evaluate_script` as in `.cursor/rules/agent-browser-ui-fixes.mdc`.
- **Do not** treat “one manual path + unit tests” as proof for cross-surface graph handoff.

### E2E matrix (minimum bar before claiming “graph handoff fixed”)

For each row: from **Entry** perform the user action, land on **Graph**, then assert
**Selection** (focused episode/topic id or class contract), **Rail** (subject matches
click), and **Viewport** (focused node in visible canvas or documented zoom policy). Order
rows in random and repeated order in automation when stable.

| Entry | Action (example) | Notes |
| ----- | ---------------- | ----- |
| Library | **G** / open in graph on row A, then row B, repeat | Catches incremental redraw / `lastLoadSource` / stale `episodeId` |
| Library → Shows (UXS-015) | Shows mode → open show A → episode → graph; back → show B → episode → graph | Same `metadata_relative_path` handoff as Library rows; catches show-detail → graph stale-focus across two shows (`shows-library.spec.ts` mocked; `stack-shows-library.spec.ts` served) |
| Show rail signals (UXS-015 Phase 2) | Open a show → `show-rail-signals` band → click a `show-rail-topic` / `show-rail-person` chip → node view (same rail, ‹ Back to show) | `GET /api/corpus/feed-signals` counts Topic/Person across episode KGs. Band render covered by `shows-library.spec.ts` (mocked); chip → `focusTopic`/`focusPerson` + Back covered by `ShowRailPanel.mount.test.ts`. Downstream graph-node handoff reuses the shared `focusGraphNode` path (Digest/Graph rows above). |
| Digest | Topic pill / cluster / open graph | Catches `pendingFocus` vs same-episode topic changes |
| Search | **Show on graph** | Catches search handoff + focus id |
| Dashboard | Intelligence → topic / cluster → graph | Catches tab + async load |
| Graph | Neighbourhood load / expand | Catches `graph-internal` vs external |

Extend this table in [INCREMENTAL_LOADING_TEST_CRITERIA.md](INCREMENTAL_LOADING_TEST_CRITERIA.md)
as scenarios harden; link new Playwright specs here under **Surfaces and owning specs**.

## Runtime

| Item | Value |
| ---- | ----- |
| Config | [playwright.config.ts](../playwright.config.ts) |
| `baseURL` | `http://127.0.0.1:5174` |
| Dev server | Vite via Playwright `webServer` (dedicated port; avoids clashing with `npm run dev` on 5173) |
| Browser | Firefox (single project) |
| Specs | `e2e/*.spec.ts`, shared [fixtures.ts](fixtures.ts), [helpers.ts](helpers.ts) |

### DR drill — live stack (not this `baseURL`)

The orchestrator **`drill-exercise`** (after **`drill-e2e`**) runs **`tests/stack-test/stack-viewer.spec.ts`** against the deployed drill host over Tailscale (**HTTPS** MagicDNS, `STACK_TEST_BASE_URL`, see [playwright.config.ts](../../../tests/stack-test/playwright.config.ts)). That suite asserts SPA + nginx + API + **`/app/output`** corpus; it is the **live** browser check, separate from the mocked specs in this directory.

## Updated testids — chip-bar refactor (#658 / #669 / #670 / #673 / #674)

The graph + library + digest filter UIs were rebuilt as chip bars. New
testids replace several legacy ones; existing rows below still reference
the old ids in places where the dense one-line entries haven't been
rewritten yet. Treat this section as the source of truth for the new
chip surfaces; legacy callouts will be migrated as Playwright specs
that touch them are updated.

**Graph filter chip bar (#658)** — replaces the Types row +
``graph-toolbar-more-filters`` ⚙ popover:

| Element | testid |
| ---- | ----- |
| Bar container | ``graph-filter-bar`` |
| Types chip button / popover | ``graph-chip-types`` / ``graph-popover-types`` |
| Feed chip button / popover | ``graph-chip-feed`` / ``graph-popover-feed`` (+ ``graph-feed-filter-panel`` / ``graph-feed-filter-search`` / ``graph-feed-filter-list``) |
| Sources chip button / popover (rendered only when ``kind === 'both'``) | ``graph-chip-sources`` / ``graph-popover-sources`` |
| Edges chip button / popover | ``graph-chip-edges`` / ``graph-popover-edges`` |
| Degree chip button / popover | ``graph-chip-degree`` / ``graph-popover-degree`` |
| Reset all | ``graph-chip-reset-all`` |
| Types reset (inside the Types popover) | ``graph-types-reset`` (kept) |

Removed: ``graph-toolbar-types``, ``graph-toolbar-more-filters``,
``graph-filters-popover``, ``graph-filters-sources``,
``graph-filters-edges``, ``graph-filters-degree``.

**Library filter chip bar (#669)** — replaces the
``CollapsibleSection`` filter form:

| Element | testid |
| ---- | ----- |
| Bar container | ``library-filter-bar`` |
| Feed chip button / popover | ``library-chip-feed`` / ``library-popover-feed`` (+ ``library-feed-filter-panel`` / ``library-feed-filter-search`` / ``library-feed-filter-list``) |
| Date chip button / popover | ``library-chip-date`` / ``library-popover-date`` |
| Clustered toggle chip | ``library-chip-clustered`` |
| Reset | ``library-chip-reset`` |
| Title persistent input | ``library-filter-title`` |
| Summary persistent input | ``library-filter-summary`` |

Removed: ``library-feed-list-scroll``, ``library-feed-filter-search``
(panel-internal id collides — see new ``library-feed-filter-search``
inside the popover panel), ``library-topic-cluster-toggle``.

**Digest date chip (#670)** — replaces the date label + raw text input

- four preset buttons:

| Element | testid |
| ---- | ----- |
| Date chip button / popover | ``digest-chip-date`` / ``digest-popover-date`` |

**Shared chip primitives** used by all surfaces:

- Composable: ``composables/useFilterChipPopover.ts``
- Shared feed panel: ``components/shared/CorpusFeedFilterPanel.vue``
  (``corpus-feed-filter-panel`` / ``corpus-feed-filter-search`` /
  ``corpus-feed-filter-list`` / ``corpus-feed-filter-all`` /
  ``corpus-feed-filter-clear``; each consumer chip overrides the test
  ids via props so callsites can target the right surface).
- Shared date chip: ``components/shared/DateChip.vue`` (caller passes
  ``chip-testid`` / ``popover-testid`` props).

**Shell collapse (#673) + keyboard (#674)**:

| Element | testid |
| ---- | ----- |
| Left panel collapse toggle | ``left-panel-collapse-toggle`` |
| Left panel collapsed strip | ``left-panel-collapsed-strip`` |

Removed: ``left-rail-edge-toggle`` (renamed).

Right rail collapse state now persists to ``localStorage`` key
``ps_right_panel_open`` (mirrors the existing ``ps_left_panel_open``).

Main-tab keyboard shortcuts (#674 item 5): ``1`` Digest / ``2`` Library
/ ``3`` Graph / ``4`` Dashboard, gated by the same
``isEditableTarget`` check as the existing ``/`` shortcut. Search v3 §S2
reserved slot ``3`` for the new Search main tab and shifted Graph to
``4`` and Dashboard to ``5``; §S3 added ``Cmd-K`` / ``Ctrl-K`` alongside
``/`` for the shell-wide command palette.

**Search v3 shell-level testids (§S2 / §S3 / §S4-shell — PRD-045, RFC-107)**:

| Element | testid |
| ---- | ----- |
| Search main tab workspace (region) | ``search-workspace`` |
| Left rail Saved / Recent aside (§S4-shell) | ``left-panel-saved-queries`` |
| Left rail Saved list + honest empty | ``left-panel-saved-list`` / ``left-panel-saved-empty`` |
| Left rail Recent list + honest empty | ``left-panel-recent-list`` / ``left-panel-recent-empty`` |
| Command palette dialog + input | ``command-palette`` / ``command-palette-input`` |
| Palette recent / saved sections | ``command-palette-recent-list`` / ``command-palette-recent-empty`` / ``command-palette-saved-empty`` |
| Palette per-hit actions | ``command-palette-action-open-workspace`` / ``command-palette-action-pin-rail`` / ``command-palette-action-show-graph`` |

Retired in §S4-shell: ``workspace-sidebar`` / ``workspace-sidebar-saved-empty`` /
``workspace-sidebar-recent-empty`` / ``workspace-sidebar-recent-list``
(the workspace sidebar was folded into the left rail; the new
``left-panel-*`` testids above are the source of truth). The compact
SearchPanel launcher retired in the same pivot — ``#search-q`` now
exists ONLY inside the Search main tab, so tests that fill it must
switch to Search first via
``mainViewsNav(page).getByRole('button', { name: 'Search' }).click()``.
Row-click on a ResultCard opens the Episode subject panel; the
standalone ``L`` / ``S`` per-hit buttons were retired at the same time,
so ``getByRole('button', { name: 'Open episode in subject panel' })``
now matches the article itself (``role="button" tabindex="0"``), not a
child chip.

**Search v3 result-set operator bar (§S4a + §S4b — #1234, RFC-107 §7.4)**:

Renders above the hit cards inside the Search workspace once a query
returns ≥ 1 result. Four chips; two client-only (Timeline, On-graph),
two server-side via ``/api/search?operator=…`` (Cluster, Consensus).

| Element | testid |
| ---- | ----- |
| Bar container (region) | ``result-set-operator-bar`` |
| Chip: Cluster | ``operator-chip-cluster`` |
| Chip: Timeline | ``operator-chip-timeline`` |
| Chip: On graph (label suffixes ``(N)``) | ``operator-chip-graph`` |
| Chip: Consensus | ``operator-chip-consensus`` |
| Operator error banner | ``operator-error`` |
| Timeline panel | ``operator-timeline-panel`` |
| Timeline undated tally line | ``operator-timeline-undated`` |
| Cluster panel container | ``operator-cluster-panel`` |
| Cluster loading / empty / list | ``operator-cluster-loading`` / ``operator-cluster-empty`` / ``operator-cluster-list`` |
| Consensus panel container | ``operator-consensus-panel`` |
| Consensus loading / empty / list | ``operator-consensus-loading`` / ``operator-consensus-empty`` / ``operator-consensus-list`` |

Chip behaviour:

- **Timeline** — pure client bucket-by-YYYY-MM histogram; toggles the
  ``operator-timeline-panel`` on/off without any fetch;
  ``aria-pressed`` reflects state. When any hit has a missing / invalid
  ``metadata.publish_date`` the ``operator-timeline-undated`` line shows
  ``N hits without a publish date not shown.``.
- **On graph** — de-dupes hit ids preferring ``metadata.source_id``
  (topic / entity) over ``metadata.episode_id``, then App switches
  ``mainTab`` to ``'graph'`` and applies the yellow-ring highlight set +
  ``requestFitAfterLoad``. Disabled state ``operator-chip-graph``
  ``disabled=true`` with label ``On graph (no ids)`` when no hit
  resolves to a graph id (all hits are aux with no source_id and no
  episode_id).
- **Cluster** — first click fires ``GET /api/search?…&operator=cluster``
  with ``top_k × 3`` over-fetch (cap 100 per RFC-107 §7.4); response
  ``clusters[]`` renders as ``operator-cluster-list li`` rows with
  ``label`` + ``size`` + a ``cluster_kind`` badge. Ungrouped bucket
  labelled ``Other`` in the badge. Second click on the chip toggles the
  panel off WITHOUT re-fetching (cached in the store).
- **Consensus** — same fetch pattern with ``operator=consensus``;
  server reads ``enrichments/topic_consensus.json`` (ADR-108, precision
  ~0.91 on prod-v2) and filters pairs to topics surfaced in the hit
  page. Empty state ``operator-consensus-empty`` renders when the file
  is missing or no pairs match.

Owning spec: ``search-operator-bar.spec.ts`` (mocks ``/api/search``
with a route callback that branches on the ``operator`` query param).

**Search v3 enriched-answer hero (§S5 — #1235, RFC-107 §S5, UXS-016 +
UXS-008)**:

Sits directly above ``result-set-operator-bar`` on the Search main tab.
Renders an aggregated summary of the shipped QueryEnricher chain
(RFC-088 chunk 5); today the chain only decorates hits with
``metadata.query_enrichments.related_topics``, so the hero surfaces the
top related topics by summed similarity across the hit page. Shape is
open — future ``synthesized_answer`` fields plug in without a schema
change.

| Element | testid |
| ---- | ----- |
| Enriched-answer chip on filter bar | ``search-chip-enriched`` |
| Hero container (region) | ``enriched-answer-hero`` |
| Loading skeleton | ``enriched-answer-skeleton`` |
| Error alert (server ``enrichment_error``) | ``enriched-answer-error`` |
| Topic chip list | ``enriched-answer-topics`` |
| Per-topic chip | ``enriched-answer-topic-<topic_id>`` |
| Overflow count (>6 topics) | ``enriched-answer-overflow`` |

State machine (UXS-008):

- **Hidden** — enrichment effectively off (chip disabled + no explicit
  opt-in) OR no hits carry ``query_enrichments.related_topics`` AND
  no ``enrichment_error`` AND not loading. Section renders nothing.
- **Skeleton** — enrichment on AND ``search.loading === true`` AND no
  content yet AND no error. Placeholder rows so layout doesn't jump.
- **Error** — server sent ``enrichment_error`` (non-fatal). Muted
  ``role="alert"`` line: the QueryEnricher chain failed but vector
  hits above are still valid.
- **Rendered** — one or more hits carried decorations. Chips are
  clickable — click routes through ``subject.focusTopic`` to the
  Topic subject rail (same handoff other Topic entry points use).
  Chip title tooltip shows ``hitCount / total`` and summed similarity
  score for auditability.

Chip semantics:

- Tri-state field ``search.filters.enrichResults``:
  ``null`` = auto (mirrors ``shell.enrichedSearchAvailable``);
  ``true`` / ``false`` = explicit user choice.
- ``search-chip-enriched`` is disabled with tooltip "Enrichment not
  configured on this server" when
  ``!shell.enrichedSearchAvailable``.
- Toggle flips the effective state — first click when auto-on writes
  ``false``; first click when auto-off writes ``true``.

Store integration: ``runSearch`` resolves the tri-state at fire time
and sends ``enrich_results=true`` on the URL when effectively on; the
server's ``enrichment_error`` field surfaces via
``search.enrichmentCallFailed``.

Owning spec: ``search-enriched-hero.spec.ts`` (mocks ``/api/search``
with a route callback that branches on the ``enrich_results`` query
param and only decorates hits when the flag is on).

## Surfaces and owning specs

| Surface | Intent (short) | Typical entry | Spec files |
| ------- | -------------- | ------------- | ---------- |
| **Graph shell** | Graph: **top of card** — **`data-testid="graph-status-line"`** (counts only when a full merged graph is loaded: **`graph-status-lens-label`**, **`graph-status-episode-count`**, **`graph-status-node-count`**, **`graph-status-component-count`**, optional **`(capped)`**); on that strip **right**, **`data-testid="graph-gesture-overlay-reopen"`** (**Gestures**); above optional **`data-testid="graph-search-highlight-chip"`**; **Types** row (`data-testid="graph-toolbar-types"`): type checkboxes + **all** / **none** + swatch counts (includes **Topic cluster** when corpus topic-cluster overlay applies); **`data-testid="graph-types-reset"`** when node-type defaults are overridden; **`data-testid="graph-toolbar-more-filters"`** **⚙** opens **`data-testid="graph-filters-popover"`** with **`graph-filters-sources`** / **`graph-filters-edges`** / **`graph-filters-degree`** (merged **GI**/**KG**, **Hide ungrounded**, edge-type toggles, degree buckets + **Clear degree filter**); **`data-testid="graph-bottom-bar"`** persistent bar under the canvas (`aria-expanded`, collapse persisted as **`ps_graph_bottom_bar_collapsed`** in `localStorage`, **Alt+B** toggles) — **centre** (full graph only): **`data-testid="graph-bottom-bar-centre"`** wraps **`data-testid="graph-status-line-controls"`** (**`graph-status-lens-selector`**, **`graph-status-since-input`**, optional **`data-testid="graph-status-reset"`** after RFC-076 cross-episode **expand**); **left**: **`data-testid="graph-minimap-toggle"`** (⊞), **`data-testid="graph-relayout"`**, **`data-testid="graph-layout-cycle"`** (cycles **cose → breadthfirst → circle → grid** and re-layouts); **right**: `role="toolbar"` **Graph fit, zoom, and export** — **`graph-zoom-fit`**, **`graph-zoom-out`**, **`graph-zoom-in`**, **`graph-zoom-reset`**, optional **`graph-gesture-overlay-reopen`** (**Gestures** when there is no full-graph stats strip; reopens overlay without clearing **`ps_graph_hints_seen`**), **`graph-export-png`** (**PNG**, `aria-label` **Export PNG**), **`data-testid="graph-bottom-bar-toggle"`** (**⌄** collapse); collapsed strip **`data-testid="graph-bottom-bar-expand"`** (**⌃**); **gesture discovery overlay** — `data-testid="graph-gesture-overlay"` over **`.graph-canvas`** on first non-empty graph load per browser until dismissed (`data-testid="graph-gesture-overlay-dismiss"` **Got it**, backdrop outside the card, **Escape** when focus is inside the overlay); cross-episode **teal** / **blue** ring semantics and gestures (RFC-076) live in UXS-004 + `docs/wip/GRAPH-GESTURE-OVERLAY.md`; **minimap** **`data-testid="graph-minimap"`** lower-left + **`data-testid="graph-minimap-close"`** (**×**); canvas **`.graph-layout-controls`** and **`.graph-zoom-controls`** overlays removed. **Selected node:** **Re-layout** and degree filter keep the selection anchored; **wheel** and **Graph fit, zoom, and export** toolbar adjust pan so the selected node stays at the same on-screen position (incremental zoom steps); **Fit** reframes all visible elements and resets that anchor; wheel zoom-out does not auto-center while a node is selected. With a healthy API corpus path, the client may **GET /api/corpus/topic-clusters** after loading artifacts; when **`search/topic_clusters.json`** exists, **TopicCluster** compound parents group matching **Topic** nodes (Cytoscape `parent`). | After artifact load (healthy API: **Corpus path** + first **Graph** tab visit auto-loads a **graph lens** slice + cap; or manual **List** → **Load into graph**; offline: file picker) | `offline-graph.spec.ts`, `graph-gesture-overlay.spec.ts`, `export-png.spec.ts`, `search-to-graph-mocks.spec.ts` (post-load), `keyboard-shortcuts.spec.ts` (**Esc** test only — after `loadGraphViaFilePicker`) |
| **Shell — semantic search** | **`/`** focuses `#search-q` in the left **query** column (expands the left rail when collapsed) | `goto('/')` with **`**/api/health`** fulfilled (200 JSON) — **no** graph load | `keyboard-shortcuts.spec.ts` (first test) |
| **Dashboard** | **`data-testid="briefing-card"`** above **`role="tablist"`** **`aria-label="Dashboard tabs"`** — **Coverage** (default), **Intelligence**, **Pipeline**. **Coverage**: charts/tables + **`data-testid="index-status-card"`** (status-only — last rebuilt / vectors / **⚠ Rebuild recommended** / last error; **`index-status-manage`** **Manage in Configuration →** opens the Configuration dialog at its **Index** section (**`sources-dialog-index-panel`**) where index facts + rebuilds live. The rebuild *action* no longer fires from the dashboard card — config consolidation). **Intelligence**: snapshot, **`data-testid="topic-briefing-cards"`** (PRD-033 FR6.1 #888 — retrieval-grounded **`topic-briefing-card`** per top topic: top segment + score + episode count, mapped topics add **`topic-briefing-card-cross-show`** from `cross_show_synthesis`; **`topic-briefing-card-link`** → **Topic Entity View** rail via `focusTopic`), **`data-testid="topic-clusters-status-block"`**, topic landscape (**cluster cards → Graph + graph node detail rail**, TopicCluster compound when `tc:…` id present), top voices, stubs. **`data-testid="query-activity-chart"`** (PRD-033 FR6.2 — daily **search-activity** bar chart from `GET /api/corpus/query-activity`, the append-only log `/api/search` writes; shows only when ≥1 search is logged. Honest scope: *search volume over time*, not query-by-topic — queries aren't topic-tagged). **Pipeline**: sub-tabs **`dashboard-pipeline-subtab-jobs`** / **`dashboard-pipeline-subtab-job-history`** / **`dashboard-pipeline-subtab-history`** — **`data-testid="pipeline-jobs-card"`** (active jobs when **`jobs_api`** in **`GET /api/health`**; per-row **View log** **`pipeline-job-log-link`** opens the in-app **`pipeline-job-log-dialog`** instead of a new-tab download, #695); **Job history** **`pipeline-job-history-strip`** — two-column **`pipeline-job-history-layout`**, filter **`pipeline-job-history-filter`**, scrollable listbox **`pipeline-job-history-select`**, rows **`pipeline-job-history-row`**, cap hint **`pipeline-job-history-picker-hint`**, summary **`pipeline-job-history-summary-line`**, **View log** **`pipeline-job-history-log-link`** (opens **`pipeline-job-log-dialog`**); **Run history** **`pipeline-run-history-strip`** (filter **`pipeline-run-history-filter`**, list **`pipeline-run-history-select`**); timing charts. **Corpus artifacts** (**List** / **All** / **None** / **Load into graph**) live in the status bar dialog **`data-testid="artifact-list-dialog"`** (not on the Dashboard tab body). | **Dashboard** main tab (`openCorpusDataWorkspace` → waits **`briefing-card`**) | `dashboard.spec.ts`, `dashboard-index-rebuild-mocks.spec.ts`, `dashboard-pipeline-jobs-mocks.spec.ts`, `pipeline-job-log-viewer.spec.ts`, `search-to-graph-mocks.spec.ts`, `corpus-hints.spec.ts`, `graph-expansion-mocks.spec.ts` |
| **Status bar (offline + corpus path)** | Footer **`data-testid="app-status-bar"`**: corpus path **`data-testid="status-bar-corpus-path"`**, hidden file input **`data-testid="status-bar-local-file-input"`**, **`aria-label` Choose corpus files** (**Files**), **`data-testid="status-bar-list-artifacts"`** **List** (healthy API + path → fetch artifacts + open **`artifact-list-dialog`**), **Configuration** (status button) when **`GET /api/health`** sets **`feeds_api`** and/or **`operator_config_api`** (**`data-testid="status-bar-sources-trigger"`** — dialog title **Configuration**; default tab **Feeds** when feeds API is on, else **Job Configuration** only); **`jobs_api`** adds **Pipeline jobs API** = **Yes** in the Health dialog and enables the **Dashboard → Pipeline** **`pipeline-jobs-card`** (mock **`GET /api/jobs`**); **Configuration** dialog **`data-testid="status-bar-sources-dialog"`** (widened modal, **left sub-nav rail** `<nav aria-label="Configuration sections">` — was a top tab-strip) — sections **`sources-dialog-tab-feeds`** (**Feeds**) / **`sources-dialog-tab-operator`** (**Job Configuration**) / **`sources-dialog-tab-scheduled`** (**Scheduled**) / **`sources-dialog-tab-index`** (**Index** — index facts + rebuild) / **`sources-dialog-tab-health`** (**Health**) (each section loads only its API); **Scheduled** = **`scheduled-jobs-section`** (#709 — `GET /api/scheduled-jobs`: rows `scheduled-jobs-row-{idx}`, `scheduled-jobs-toggle-{idx}` enable/disable → operator-YAML line-rewrite + `PUT /api/operator-config`, `scheduled-jobs-next-{idx}` relative next-run / `—` when disabled, `scheduled-jobs-invalid-cron` badge; spec `scheduled-jobs-mocks.spec.ts`), **Feeds** tab: sub-tabs **`sources-dialog-feeds-panel-list`** (**Manage**) / **`sources-dialog-feeds-panel-json`** (**Raw JSON**); **Manage**: **`sources-dialog-feeds-list`** / **`sources-dialog-feeds-row-*`** (per-row **Edit** / **`sources-dialog-feeds-row-configure-{idx}`** Configure / **Delete**), **`sources-dialog-feeds-add-url`** + **`sources-dialog-feeds-add-btn`** (**Add feed** → **`PUT /api/feeds`**); **Configure** opens the per-feed override drill-in **`feed-override-editor`** (#694 — `feed-override-max-episodes` / `feed-override-order` / `feed-override-offset` / `feed-override-since` / `feed-override-until`; collapsible **Advanced** with structured inputs **`feed-override-adv-{key}`** (retry / delay / circuit-breaker / conditional-GET / episode-retry / `user_agent`) + a raw-JSON **`feed-override-extras`** for unknown keys; `feed-override-save`, `feed-override-back`; spec `feed-overrides-mocks.spec.ts`); **Raw JSON**: **`sources-dialog-feeds-textarea`** + **`sources-dialog-feeds-apply-json`** (**Apply JSON** → **`PUT`**); **Job Configuration** tab is one page with two sub-tabs (default **Profile**): **`sources-dialog-operator-subtab-profile`** — **`sources-dialog-profile-select`** (packaged **`profile:`**, **None** = unset; server may seed **`profile: cloud_balanced`** when the operator file is missing/empty) + a **what-it-brings** viewer **`sources-dialog-profile-content`** (parsed key settings **`sources-dialog-profile-settings`** + full YAML in a `details`), bodies from **`GET /api/operator-config/profiles`**; and **`sources-dialog-operator-subtab-config`** — **`sources-dialog-operator-textarea`** (YAML overrides) + live **`cron-schedule-preview`** (#709 — validates `scheduled_jobs` crons as you type). One shared **`sources-dialog-save-overrides`** (**Save**) persists profile + overrides; **Health** dialog (**`data-testid="status-bar-health-trigger"`**, **Retry health**, offline **Choose files…** inside dialog), optional **Index** (**`data-testid="status-bar-rebuild-indicator"`**) | Always visible | `status-bar-feeds-operator-mocks.spec.ts`; `dashboard-pipeline-jobs-mocks.spec.ts`; `loadGraphViaFilePicker` consumers; keyboard **Esc** offline graph |
| **Offline GI/KG file load** | Abort **`/api/health`**, **Graph** tab, set files on **`status-bar-local-file-input`** | `loadGraphViaFilePicker` in [helpers.ts](helpers.ts) | `offline-graph.spec.ts`, `graph-label-disc.spec.ts` (Cytoscape label bbox vs disc), `dashboard.spec.ts`, `export-png.spec.ts`, `keyboard-shortcuts.spec.ts` (**Esc** test) |
| **API panel — mocked corpus graph** | Healthy API + corpus path → after first **Graph** tab visit, auto `GET /api/artifacts` (each item needs **`publish_date`**) + fetch selected GI/KG (merged graph under graph lens + cap); manual **List** / **Load into graph** still available; **`beforeEach`** also mocks **`GET /api/corpus/topic-clusters`** (topic-clusters payload v2) for Types-row / overlay coverage | `goto('/')` + mocks in `beforeEach` | `search-to-graph-mocks.spec.ts` |
| **Transcript viewer (Quote rail)** | **View transcript** loads `GET /api/corpus/text-file` (+ optional `.segments.json`); **Close** dismisses `dialog` | Mocked `**/api/corpus/text-file**` + search **Show on graph** on a **quote** hit | `transcript-viewer-dialog.spec.ts` |
| **Corpus hints** | Banner when `GET /api/artifacts` returns `hints` | Mocked artifacts + **List** | `corpus-hints.spec.ts` |
| **Corpus version warning** | When **`GET /api/health`** returns non-empty **`corpus_version_warning`**, alert banner **`data-testid="corpus-version-warning-banner"`** renders above **`data-testid="app-status-bar"`** (viewer passes **`path=`** query when status-bar corpus path is set) | Mocked health with warning on `goto('/')` | `corpus-version-warning.spec.ts` |
| **Corpus Digest** | Panel **`h2` Digest** (`#digest-main-heading`) **left** + **`?`** **About Digest** **`HelpTip`** (topic bands → **Graph** / **Search topic** / **Recent** / CIL **topic pills** / **Library**). **Recent** row **click** → **Episode** subject rail without leaving **Digest**; **topic band** **title**, **hit row** **click**, and **Recent** **CIL pills** (`digest-recent-cil-pills`, **`topic_id`**) → **Graph** when the digest row includes CIL topics (focus digest **`graph_topic_id`** / **`topic_id`** when that **`topic:`** node exists in the merged slice, else episode). **No** summary-bullet pill row on **Recent**. **Digest** ↔ **Library** retains rail selection when the episode is still in the digest payload and still listed under Library for current filters (see UXS). **Recent** list rows use **`bg-overlay`** when selected in the **Episode** subject rail (same cue as **Library**); **Recent** rows share **Library**-style list chrome (no bordered **`bg-elevated`** card wrapper). Toolbar: first row — **`h2` Digest** + **`?`** **About Digest** **left**, **Published on or after** + presets **right**; second row when loaded — rolling **range** (human-readable UTC via `<time datetime="ISO">` · **N** **episodes** / **1 episode**). `GET /api/corpus/digest` + **`GET /api/corpus/feeds`**; **`region` `Topic bands`** wraps the topic **grid** without an outer **max-height** scrollbar: **first three** bands by default, then **`data-testid="digest-topic-bands-show-more"`** when more exist; **first** band **elevated** + **`border-primary/20`**; topic titles **`text-sm`** (**bold** on first band, **semibold** on others); **`region` `Recent episodes`** (accessible name **`Recent episodes, N items`** / **`1 item`**): **`h2` Recent** (`#digest-recent-heading`, **`text-sm`** semibold) + tabular **`(N)`** count (same **N** as the rolling-window line · **N episodes**) + **`?`** (**About the Recent digest list**); episode row **meta** (**right** of title): **feed**, **publish date**, **E#**, **duration** on one **`text-[10px]`** baseline-wrap line; tight gap before recap. **Topic band** cards: compact padding; each **hit row** is **`role="button"`** (same **`episode_title`, `feed`** pattern as **Recent**), **click** → **Episode** subject rail + **Graph** as above; **grid** left column **`h-9`** **cover** only (**`w-9`**), right column full-width **title**; recap **`col-span-2`** (**semantic match** label when **`score`** present — **Strong** / **Good** / **Weak**; raw score on that label’s **`title`**); row **`title`**: publish date, **E#**, duration, feed, description, RSS (no duplicate vector block). **Search topic** opens Search with the topic **`query`** and **Since (date)** set only when the shared corpus lens has a valid **YYYY-MM-DD** (omitted for all time). **Truncated** feed + hover on **Recent** rows as above. **episode rows** — `h-9` cover, recap **2-line clamp** when unselected (**`line-clamp-2`**), **full** when selected; optional **`success`** **recency dot** before titles when publish day is within rolling **24h** (local midnight parse); **CIL pills** (**`kg`** cluster tint on Digest Recent) → **Graph** when present; no **GI/KG** chips | Default main tab on `goto('/')` | `digest.spec.ts` (mocked health + digest + **`feeds`** + optional `**/api/artifacts/metadata/...gi.json**`) |
| **Corpus Library** | **Filters** collapsible + **Feed** column: **`region` `Feeds`**; feed rows live in **`data-testid="library-feed-list-scroll"`** with **`max-height`** ~**two** feed rows and vertical scroll for the rest. When **`feeds.length > 15`**, **`data-testid="library-feed-filter-search"`** filters client-side. With no feed selection the episode list is **all** feeds; **Clear feed filter** (beside the **Feed** label) is always visible (**disabled** until a feed row is selected). On **`lg`**, filters use a **two-column** layout (**~60% / 40%**): **left** = **date + presets** on one row (horizontal scroll if needed), then **Title** / **Summary** filter inputs on a shared **`grid`**; **Clear all filters** and **Apply**; **right** = **Feed** list. **`?`** **About Library filters** sits **immediately after** the **Filters** title. **Below** **Filters** (always visible): **`data-testid="library-topic-cluster-toggle"`** (**Clustered episodes only**) — **`GET /api/corpus/episodes`** **`topic_cluster_only=true`** when checked (same server semantics as before). Row **`title`** hover adds **RSS** + **description** when `GET /api/corpus/feeds` includes them. **Episodes** heading + **?** **HelpTip**; **Episodes** list — **cursor pagination** + **Load more** + **scroll-to-load**; per-row **meta** like Digest **Recent**; recap **2-line clamp** when unselected, **full** when selected; optional **recency dot**; **no** topic chips on list rows. **Episode** subject rail unchanged for detail chrome. **No** embedded **24h digest** strip — use **Digest** tab for discovery. | **Library** tab + corpus path; mock corpus + optional `index/stats` / `similar` | `library.spec.ts` |
| **Theme tokens** | `--ps-canvas` matches asserted dark/light hex in [theme.spec.ts](theme.spec.ts) | `goto('/')` + `emulateMedia` and/or `localStorage` | `theme.spec.ts` |
| **Search v3 result-set operator bar** | ``result-set-operator-bar`` above the hit cards on Search main tab. Chips: Cluster (S4b, server ``operator=cluster`` top_k×3 over-fetch) / Timeline (S4a, client YYYY-MM histogram over ``metadata.publish_date``) / On graph (S4a, App-level ``activateGraphTab`` + yellow-ring highlight set) / Consensus (S4b, server ``operator=consensus`` reads ``enrichments/topic_consensus.json``). Panels: ``operator-{cluster,timeline,consensus}-panel`` with ``-loading`` / ``-empty`` / ``-list`` states. See the **Search v3 result-set operator bar** section above for the full testid contract. | `goto('/')` + Search tab + submit query → bar renders once response has ≥ 1 hit | `search-operator-bar.spec.ts` |
| **Search v3 enriched-answer hero** | ``enriched-answer-hero`` above the operator bar on Search main tab. Renders the shipped QueryEnricher chain output (RFC-088 chunk 5) — today that's ``related_topics`` per hit, aggregated + ranked by summed similarity and surfaced as clickable Topic chips. Chip on filter bar: ``search-chip-enriched`` (tri-state; auto-adopts ``shell.enrichedSearchAvailable``). Hero states: hidden / ``enriched-answer-skeleton`` / ``enriched-answer-error`` / ``enriched-answer-topics`` list (with ``enriched-answer-overflow`` when >6 topics). See the **Search v3 enriched-answer hero** section above. | `goto('/')` + Search tab + Enriched chip on + submit query with enrichment-decorated hits | `search-enriched-hero.spec.ts` |

### Offline graph load (shared helper)

[`loadGraphViaFilePicker`](helpers.ts):

1. Route `**/api/health` → **abort** (`failed`).
2. `goto('/')`, wait for heading **Podcast Intelligence Platform** (`SHELL_HEADING_RE` in [helpers.ts](helpers.ts)).
3. Click **Graph** (default app tab is **Digest**; graph canvas lives on **Graph**).
4. Set files on **`data-testid="status-bar-local-file-input"`** using [GI_SAMPLE_FIXTURE](fixtures.ts) (`e2e/fixtures/ci_sample.gi.json`).
5. Wait for **Fit** visible.

[`dismissGraphGestureOverlayIfPresent`](helpers.ts) — optional after graph is visible when a spec must click **`.graph-canvas`** or send pointer events to Cytoscape (first-run **`graph-gesture-overlay`** sits above the canvas). Gesture overlay specs omit this on purpose.

### Graph Cytoscape (automation hooks)

- **Main** graph (container inside **`.graph-canvas`**): Cytoscape assigns **`graph-label-tier-none`**, **`graph-label-tier-short`**, or **`graph-label-tier-full`** on nodes from **zoom** (`syncGraphLabelTierClasses` in `cyGraphLabelTier.ts`, called from **`GraphCanvas.vue`**). Selection dim uses classes **`graph-dimmed`**, **`graph-focused`**, **`graph-neighbour`**, **`graph-edge-dimmed`**, **`graph-edge-neighbour`** (no `data-testid` on graph elements). **`graph-label-disc.spec.ts`** checks label bounding boxes vs node discs using dev-only **`window.__GIKG_CY_DEV__`** when needed.
- **Episode subject rail** minimap (**`data-testid="graph-neighborhood-mini"`**): second Cytoscape instance; same stylesheet **compact** profile, **`prefers-reduced-motion`** from **`prefersReducedMotionQuery()`**, and **`syncGraphLabelTierClasses`** after **`layoutstop`** + **fit** so tier-dependent label rules apply at that zoom.

### Mocked API corpus path → graph → search (`search-to-graph-mocks.spec.ts`)

- Fulfill `**/api/health`, `**/api/artifacts?**`, per-file `GET /api/artifacts/{relPath}?**`, and `**/api/search?**` as in that spec. Artifact list JSON must include **`mtime_utc`** and **`publish_date`** on each item (server schema; #507 + graph initial load).
- If a spec drives **Dashboard** index actions, also fulfill `**/api/corpus/coverage?**`, `**/api/corpus/stats?**`, `**/api/corpus/digest**`, `**/api/corpus/runs/summary?**`, `**/api/corpus/persons/top?**` (shared helper **`setupCorpusDashboardDataRoutes`** in [dashboardApiMocks.ts](dashboardApiMocks.ts)), `**/api/index/stats?**` (200 JSON with `available`, `reindex_recommended`, `rebuild_in_progress`, etc.), and when testing rebuild `**/api/index/rebuild?**` POST → **202**.
- Fill corpus placeholder (triggers auto **List** + load of all GI/KG when health is ok); open **Graph** tab, wait for **Fit** (no manual **List** / **Load into graph** required for the happy path).
- Fill `#search-q`, **Search** in the **Semantic search** `section` (`locator('section').filter({ has: heading Semantic search }).getByRole('button', { name: 'Search', exact: true })` — avoids clashing with the collapsed-left **Search** shortcut button), wait for stub result text; optional **Topic cluster:** line above hit body text when the search stub includes **`metadata.topic_cluster`** on a **`kg_topic`** row (topic cluster metadata); optional **`getByRole('button', { name: 'Search result insights' })`** → **`dialog` Search result insights** with **`region` Doc types** / **`region` Publish month**, then **Close**; then **`getByRole('button', { name: 'Show on graph' })`**, assert **Fit** still visible and `.graph-canvas` visible.
- On the **Graph** node rail (**`embed-in-rail`**), **`graph-connections-section`** and neighbor-row **`graph-connection-open-library`** / **`graph-connection-focus-graph`** / **`graph-connection-prefill-search`** are on the **Neighbourhood** tab only (**`data-testid="node-detail-rail-tab-neighbourhood"`**); **Details** is the default after focus. Specs that assert those controls must activate **Neighbourhood** first (see **`search-to-graph-mocks.spec.ts`**).

### Index rebuild mocks (`dashboard-index-rebuild-mocks.spec.ts`)

- Fulfill `**/api/health` (200), `**/api/index/stats**` (200 JSON with `available: true`, `rebuild_in_progress: false`, minimal `stats`), `**/api/artifacts?**`, and **`setupCorpusDashboardDataRoutes`** (coverage, stats, digest, runs, persons, feeds).
- Fulfill `**/api/index/rebuild**` on **POST** with **202** + `IndexRebuildAccepted` JSON.
- `goto('/')` → set corpus path → **Dashboard** (briefing visible) → **`index-status-card`** is status-only; click **`index-status-manage`** → opens **`status-bar-sources-dialog`** at **`sources-dialog-index-panel`** → **`index-dialog-update`** / **`index-dialog-full-rebuild`** → `waitForRequest` on POST; incremental must not set `rebuild=true`; full rebuild must include `rebuild=true` in query.

### Keyboard shortcuts (`keyboard-shortcuts.spec.ts`)

Two isolated setups:

1. **Search focus:** Fulfill `**/api/health` → `goto('/')` → wait for `#search-q` enabled → click `body` (blur) → `keyboard.press('/')` → expect `#search-q` focused.
2. **Esc on graph:** `loadGraphViaFilePicker` → `dismissGraphGestureOverlayIfPresent` (so the click reaches **`.graph-canvas`**) → click `.graph-canvas` → `keyboard.press('Escape')` → expect **Fit** visible.

### Theme tokens (`theme.spec.ts`)

- **Dark:** `emulateMedia({ colorScheme: 'dark' })`, `goto('/')`, expect `--ps-canvas` trimmed lowercase **`#111418`**.
- **Light:** `emulateMedia({ colorScheme: 'light' })`, init script sets `localStorage` **`gi-kg-viewer-theme`** = **`light`**, `goto('/')`, expect **`#f6f7f9`**.

## Stable selectors and hooks (contract)

Prefer updating this section when Playwright assertions change.

**Digest / Library hooks asserted in Playwright** (`digest.spec.ts`, `library.spec.ts`):

- Topic-band hit **`button`** **`aria-label`** ends with **`, Strong match`** (or **Good** / **Weak**) when the mocked hit includes a vector **`score`** (semantic tier is part of the accessible name).
- **`digest-topic-bands-show-more`** — fourth and further topic bands stay hidden until **Show N more topics** is clicked (`digest.spec.ts` overrides **`GET /api/corpus/digest`**).
- Recency dot — **`role="img"`** with **`aria-label`** **Published …** when **`publish_date`** is **today** in local **`YYYY-MM-DD`** (`digest.spec.ts` digest override).
- **`library-feed-filter-search`** — visible when **`GET /api/corpus/feeds`** returns more than **15** feeds; typing filters client-side by display title (`library.spec.ts` overrides the feeds route).

**Roles / accessible names**

- `heading` **Podcast Intelligence Platform** (`SHELL_HEADING_RE` in [helpers.ts](helpers.ts)) — shell ready (v2 in child span).
- `navigation` **Main views** — header tabs (**Digest**, **Library**, **Graph**, **Dashboard**); scope tab clicks here so Playwright does not match **Load into graph** or unrelated **Library** substrings (`mainViewsNav` in [helpers.ts](helpers.ts)).
- **Left query column** — **Semantic search** is the only surface (Search v3 §S1 merged Explore into Search; RETIRED test IDs: `left-panel-slide-host`, `left-panel-enter-explore`, `left-panel-back-search`, `left-panel-explore-footer`). Explore's five filters live as chips on **`search-filter-bar`** — see the new IDs `search-chip-topic-contains`, `search-chip-speaker-contains`, `search-chip-min-confidence`, `search-chip-grounded-only` alongside the original `search-chip-since`, `search-chip-topk`, `search-chip-doctypes`, `search-chip-more`. No left **API · Data** tab; corpus artifact list opens from the status bar **List** button → **`artifact-list-dialog`**; helper **`openCorpusDataWorkspace`** in [helpers.ts](helpers.ts) switches to **Dashboard** and waits **`briefing-card`**).
- **Explore (GI) — RETIRED (Search v3 §S1).** The dedicated Explore panel was merged into the Search compact launcher: chip IDs `explore-advanced-open`, `explore-advanced-dialog`, `explore-filtered-submit`, `explore-clear-output`, `explore-insight-text-highlight`, `explore-top-speaker-link` and the region `Active explore filters` no longer exist. The four filters that were dialog-only (Grounded only, Limit / Top-k, Min confidence, Sort) plus the two on-card (Topic contains, Speaker contains) are all chips on the search filter bar. Topic + Min confidence are client-side over the returned top-K (accuracy caveat inside each chip); Speaker + Grounded pass through to `/api/search` server-side.
- `button` **Digest**, **Library**, **Graph**, **Dashboard**, **Fit**, **Re-layout**, **Export PNG**, **100%**, **List**, **Load into graph**, **Files** (status bar offline picker). Semantic **Search** submit: scope under `section` **Semantic search** + `exact: true`, or use `#semantic-search-form`, so it does not match unrelated **Search** labels.
- `toolbar` **Graph fit, zoom, and export** (or **fit, zoom, gestures, and export** when there is **no** full merged graph strip) — **`data-testid="graph-bottom-bar"`** right zone (below canvas); contains **Fit** (primary), **−** / **+** / **100%** (resets **`zoom(1)`** then **centers** `:visible` elements), optional **Gestures** (`data-testid="graph-gesture-overlay-reopen"`) when the stats strip is absent, **PNG** export button (`data-testid="graph-export-png"`, `aria-label` **Export PNG**), **⌄** collapse (`data-testid="graph-bottom-bar-toggle"`).
- `button` **Zoom out**, **Zoom in** — `aria-label` on **−** / **+** (visible label is the glyph only; inside that toolbar).
- **Graph tab + Episode subject rail** — Switching the main tab to **Graph** while **`region` `Episode`** is open (Library or Digest selection) applies the same **library highlight** (search-hit ring) and **pending focus** as **Open in graph**: the **Episode** node whose ``metadata_relative_path`` matches the rail is centered with a modest zoom when it exists in the **merged filtered** graph (no-op if the episode is not in the current load or **Episode** type is filtered off). **Open in graph** itself loads episode GI/KG into the artifact store first; the graph canvas **filtered-artifact** reset must **not** call **`subject.clearSubject()`** while the subject is **episode** or **graph-node**, or the rail would lose context before the Graph tab switch (regression: felt like the app stopped responding). Neighbor-row **`L`** opens the **Episode** subject rail (same as semantic search **L**); main tab stays on **Graph** when already there.
- **Graph layout + degree** — **`data-testid="graph-relayout"`**, **`data-testid="graph-layout-cycle"`**, and degree buckets live in **`graph-filters-popover`** / **`graph-filters-degree`** (not a canvas `region`). **Double-tap** a node (without Shift): **Episode** nodes open the same **Episode** subject rail as Library when metadata path resolves (node properties, loaded ``.gi.json``/``.kg.json`` stem → ``.metadata.json``, or corpus episode list by ``episode_id``): **Details** is corpus body + similar episodes; on the **Graph** main tab with a graph center id on the rail, **`role="tablist"`** — **`data-testid="episode-detail-rail-tab-details"`** vs **`data-testid="episode-detail-rail-tab-neighbourhood"`** — and **`region` `Graph neighborhood and connections`** (**`data-testid="graph-connections-section"`**) lives on **Neighbourhood** only (neighbor list uses parent scroll like graph-node rail). That strip includes a read-only **Local neighborhood** Cytoscape preview (**`data-testid="graph-neighborhood-mini"`**) — **1-hop ego** subgraph around the selected node (same as ``filterArtifactEgoOneHop``), then **Connections** with per-row actions: **`L`** (**`data-testid="graph-connection-open-library"`**, **Episode** neighbors only — opens the **Episode** subject rail when metadata path resolves), **`G`** (**`data-testid="graph-connection-focus-graph"`** — focus on graph), **`S`** (**`data-testid="graph-connection-prefill-search"`** — prefill **Semantic search** with that node’s primary text, truncated). Rows use **`data-connection-node-id`** for the neighbor graph id. Tab resets to **Details** when the episode path or graph center id changes; Digest/Library episode rail has **no** tablist when the connections strip does not apply. **focus** neighbor via **`G`** + switch to **Graph**. Other node types open **graph node detail** in that rail (type-colored avatar tile + **Connections**). **Graph node detail** heading uses **three** lines max (``line-clamp-3``); the **`<h3>`** has **no** native ``title`` (full text is not in a hover tooltip). **Quote** nodes add a first body section **Full quote** (**`data-testid="node-detail-full-quote"`**): title row with **`button` Copy** on the right (**`data-testid="node-detail-full-quote-copy"`**, clipboard; **Copied** / **Copy failed** feedback) and the **full** passage below (``whitespace-pre-wrap``, ``select-text``). **Quote**-style nodes show **Where this appears in the episode** (speaker, **View transcript** → in-app **`dialog` Transcript** (`data-testid="transcript-viewer-dialog"`; highlight `data-testid="transcript-viewer-highlight"`; optional **Timeline** `details` with `data-testid="transcript-viewer-timeline"` when `.segments.json` exists), **Open transcript in new tab** → ``GET /api/corpus/text-file``, passage position, audio timing; when timestamps are **0 / 0**, visible copy includes substring **Audio timing not specified** and points to the Development Guide / GitHub **543**) and friendlier property labels elsewhere. Optional line **Also in other graph layers:** (RFC-072) when sibling ``.bridge.json`` was loaded with the corpus selection (auto-fetched next to the selected ``.gi.json`` when the file exists). **Graph node id** / **Episode id**: **`E`** chip uses graph legend **Episode** hex (**`searchResultActionStyles`**, blue fill + white glyph, native ``title`` tooltip); **`?`** **`Graph node diagnostics`** is **below** **`E`** in a vertical stack (right of the title) on a high-contrast neutral chip (near-black glyph, light fill; dark theme inverted); property list omits redundant **`episode_id`** on **Insight** / **Quote** (and other non-**Episode** nodes). No inline ``g:…`` id. **Subject rail** header: **`data-testid="subject-rail-close"`** (**×**, **`aria-label` Close subject panel**). **Single tap** on the canvas background updates graph selection only; **onetap** on a node opens episode or graph-node **subject** detail in the **right** column. **Tap** empty canvas clears selection and **`subject`** (empty **`data-testid="subject-rail-empty"`** until a new selection).
- `button` **Toggle minimap** — **`data-testid="graph-minimap-toggle"`** (⊞) in **`graph-bottom-bar`** left zone; **`data-testid="graph-minimap-close"`** (**×**) on the minimap frame. Minimap sits **bottom-left** inside the graph canvas host (inset panel ~10.5rem × 7.5rem, capped by viewport fraction), not a floating viewport-fixed tile (`aria-label` **Graph minimap** on container when shown).
- `button` **Clear degree filter** — `aria-label` **Clear degree filter** (visible label often **Clear** only); shown when a degree bucket filter is active.
- `button` matching degree bucket label, e.g. **`0 (3)`** — `aria-pressed` reflects active filter; bucket ids **`0`**, **`1`**, **`2-5`**, **`6-10`**, **`11+`**.
- `button` **all** (under **Edges** in **`graph-filters-popover`**) — re-enables all edge-type toggles (substring-safe: scope under popover or graph chrome).
- `button` **all** / **none** (under **Types** on the graph card, **`graph-toolbar-types`**) — node-type visibility; scope under graph chrome to avoid clashing with **Edges** in the popover.
- Visible text **Sources** (inside **`graph-filters-popover`**) — merged **GI** / **KG** checkboxes, **Hide ungrounded** when applicable; **⚙** shows a **warning** dot when popover filters (sources/edges/degree) are non-default (type toggles use **`graph-types-reset`** on the Types row).
- `data-testid="graph-gesture-overlay"` / `graph-gesture-overlay-dismiss` / `graph-gesture-overlay-reopen` — first-load gesture card over **`.graph-canvas`**; **`graph-gesture-overlay-reopen`** sits **right** on the stats strip when a full graph is loaded, else in the bottom bar (see **Graph shell** table row); **Types** + **⚙** on **`graph-toolbar-types`**.
- **Shift+double-click graph ego** — The canvas uses a **1-hop** edge neighborhood around the focus node, then **merges** every **TopicCluster** that intersects that slice (when **`GET /api/corpus/topic-clusters`** returned JSON): compound, in-merge member topics, and **one hop** from each member (same idea as **Cluster neighborhood** on the minimap). Union avoids “broken” cluster views when siblings are not direct edge neighbors of the focus node. Topics that only exist in **other episodes** still require those artifacts in the corpus selection. Playwright coverage for “ego shrinks node count” uses **`topic:ci-policy`** (large **Quote** labels can sit on the rendered node center and steal **dbltap** hits under COSE).
- Search (#671 chip bar): **`#semantic-search-form`** — **`#search-q`** (no visible **Query** label; `aria-label` **Search query**); chip bar **region Search filters** (**`data-testid="search-filter-bar"`**) below the textarea, with chips **Since** (`search-chip-since` / `search-popover-since`, shared **`DateChip`**), **Top‑k** (`search-chip-topk` / `search-popover-topk`, default 10; popover input **`search-popover-topk-input`** + **`search-popover-topk-reset`** when non-default), **Doc types** (`search-chip-doctypes` / `search-popover-doctypes`, empty selection = all), **More** (`search-chip-more`, opens **dialog Advanced search** — **Grounded insights only**, **Feed** (`#search-advanced-feed`, substring on catalog `feed_id` for `GET /api/search`; **Library prefill** shows **catalog display title** when known from `GET /api/corpus/feeds`, with native **`title`** hint for the id until the user edits), **Speaker**, **Embedding model**, **Merge duplicate KG surfaces**; **Close** dismisses; the **More** chip label switches to **More: N** when any of those advanced fields is non-default — replaces the legacy **region Active advanced filters** read-only summary block). **Search** / **Clear** below (submit uses `form="semantic-search-form"`). When results exist: optional visible substring **`Lift:`** (ratio **linked / transcript** when `lift_stats.transcript_hits_returned` > 0 from `GET /api/search`); per **transcript** hit card, optional collapsible **region Lifted GI insight** (`aria-label` **Lifted GI insight**) when the hit JSON includes **`lifted`**; **button Search result insights** opens **dialog Search result insights** (`h2` + hit count + dominant-type insight line); **region Doc types** and **region Publish month** (side-by-side on wide viewports); **region Episodes** / **Feeds** / **Similarity scores** / **Terms**; **Close** dismisses.
- Explore (#671 chip bar): **region Explore filters** (**`data-testid="explore-filter-bar"`**) — **Topic** (`explore-chip-topic` / `explore-popover-topic`, substring), **Speaker** (`explore-chip-speaker` / `explore-popover-speaker`, substring), **More** (`explore-chip-more`, opens **dialog Advanced explore** — **Grounded only**, **Strict schema**, **Limit**, **Sort**, **Min confidence**; the chip label switches to **More: N** when any field deviates from the API default — replaces the legacy **region Active explore filters** read-only summary block). Buttons **Explore** / **Clear** below.
- `button` **Refresh** — legacy **Dashboard** data cards (if reintroduced elsewhere); **Update index** (**`index-dialog-update`**) / **Full rebuild** (**`index-dialog-full-rebuild`**) live **only** in the Configuration dialog's **Index** section (**`sources-dialog-tab-index`** → **`sources-dialog-index-panel`**, #507) — the single rebuild-action home, alongside the index facts (`sources-dialog-index-stats`: vectors / embedding model / size; feeds shown by **display name** via `/api/corpus/feeds`), **Documents by type** proportional bars (`sources-dialog-index-doctypes`), a **Documents by month** multi-series timeseries (`index-timeseries`) — an **Episodes** line (corpus total, from `/api/corpus/coverage` `by_month`) plus one line per indexed **document type** (from `/api/index/timeseries`, `doc_type` × publish month — transcript / insight / quote / summary / kg_entity / kg_topic), each toggleable (`index-series-toggle-episodes`, `index-series-toggle-doc:{doc_type}`) with a shared month-range filter (`index-date-from` / `index-date-to`); all by publish month since the index records no per-doc write time — and the reindex-reason banner (`sources-dialog-index-banner`). Dashboard **`index-status-card`** is status-only and routes there via **`index-status-manage`**; the **`status-bar-rebuild-indicator`** bolt and the Briefing **Open index controls** action also open Configuration → Index.
- `button` **Choose corpus files** — status bar (**Files**); **`Choose files…`** also appears inside the **Health** dialog when offline.
- `checkbox` name matching `/ci_sample\.gi\.json/`.
- `tablist` **Dashboard tabs** — **Coverage** / **Intelligence** / **Pipeline** (`tab`, `aria-selected`; default **Coverage**).
- Placeholder **`/path/to/output`** — corpus root field on the **status bar** (same placeholder as legacy tests; **`data-testid="status-bar-corpus-path"`**).
- Visible text **Corpus path hint**, **Unified semantic index** (substring) — corpus hints spec.
- Visible text **Summary insight (stub)** (non-exact) — mocked search result card.
- Visible text **Reindex recommended**, **Search / corpus note**, **Background index job running**, **Last rebuild error** (substring) — **Dashboard** **Vector index** health / rebuild (#507); exact copy may evolve.

**IDs and DOM hooks (semi-stable)**

- `#search-q` — semantic search query; fill + mocked search; **Enter** submits (same as **Search**; **Shift+Enter** newline); **`/`** shortcut: switch to **Search** mode, then focus (after `body` click to blur).
- **`data-testid="semantic-search-results-scroll"`** — scroll host for errors + hit cards (**second** row of a **`grid-rows-[auto_minmax(0,1fr)]`** shell so height is bounded); **`overflow-y-auto`**, **`scrollbar-gutter: stable`**, thin **`scrollbar-width`** + WebKit thumb styling.
- `#semantic-search-form` — wraps **`#search-q`** + **Since (date)** + **`#search-top-k`**; **Search** submit button may be a sibling with `form="semantic-search-form"`.
- **PRD-033 FR1 (#884) results chrome + cards:**
  - **`data-testid="search-query-type"`** (FR1.4) — intent indicator chip in the results header, **`Intent: <label>`** (Entity lookup / Raw evidence / Temporal tracking / Cross-show synthesis / Semantic); present only when the response carries `query_type`.
  - **`data-testid="search-evidence-toggle"`** (FR1.3) — `role="group"` segmented control with buttons **`search-evidence-insight`** / **`search-evidence-segment`** / **`search-evidence-both`** (labels **Insights** / **Transcript** / **Both**); `aria-pressed` marks the active tier; filters the rendered cards by `source_tier` client-side (Both = no filter).
  - **`data-testid="search-result-tier"`** (FR1.1) — per-card tier badge: **Insight** (`insight`) / **Transcript** (`segment`) / **Reference** (`aux`), derived from the server `source_tier`.
  - **`data-testid="search-result-compound"`** (FR1.1) — **`+ insight`** badge on a `segment`-tier card that carries a lifted insight (the card already renders both the segment text and the linked insight = one compound result).
  - **`data-testid="search-result-lifted-speaker-link"`** / **`search-result-lifted-topic-link"`** (FR1.5) — clickable lifted speaker / topic names on a compound card → Person / Topic Detail rail (canonical `person:` / `topic:` ids from `lifted`).
  - **`data-testid="search-result-entity-link"`** (FR1.5) — **Open Person/Topic panel →** on a `kg_entity` / `kg_topic` hit, linking `metadata.source_id` to its Detail rail. (Supporting-quote speaker links keep their existing **`search-result-speaker-link`**.)
- **Advanced search** modal — checkbox **Merge duplicate KG surfaces (kg_entity / kg_topic)** (default checked); when unchecked, search requests include **`dedupe_kg_surfaces=false`**.
- **`data-testid="node-detail-view-transcript"`** — **View transcript** `button` on **Quote** node detail (compact row; no duplicate “open in new tab” link — raw file link lives in the dialog header).
- **`data-testid="node-detail-quote-speaker-unavailable"`** — optional muted **`GI_QUOTE_SPEAKER_UNAVAILABLE_HINT`** on **Quote** node detail, **to the right of** **View transcript** in the same row when **`href`** resolves; if there is only a **`transcript_ref`** (no corpus path / API), same copy **below** the ref line (#541). Asserted in **`transcript-viewer-dialog.spec.ts`** (mock quote on graph).
- **`data-testid="node-detail-full-quote"`** / **`data-testid="node-detail-full-quote-copy"`** — **Quote**: full passage is the expanding header **h3** title; **`C`** square chip (same footprint as **E** / **`?`**) under **`?`**, native tooltip / **`aria-label`** **Copy title** (then **Copied to clipboard** / **Copy failed; try again** briefly).
- **`data-testid="node-detail-full-topic"`** / **`data-testid="node-detail-full-topic-copy"`** — **Topic**: same pattern.
- **`data-testid="node-detail-full-insight"`** / **`data-testid="node-detail-full-insight-copy"`** — **Insight**: same pattern.
- **`data-testid="node-detail-full-person-entity"`** / **`data-testid="node-detail-full-person-entity-copy"`** — **Person** / **Speaker** / **Entity**: same pattern.
- **`data-testid="node-detail-person-entity-role"`** — optional **In this graph:** line (counts of **SPOKEN_BY** quotes and **SPOKE_IN** episode links in the loaded slice) when at least one is non-zero.
- **`data-testid="node-detail-person-entity-aliases"`** — optional **Aliases:** line when **`properties.aliases`** is a non-empty string array (same pattern as topic aliases).
- **`data-testid="graph-node-detail-rail"`** — **`GraphNodeRailPanel`** (**region** **Graph node:** …) inside **`SubjectRail`**: visible whenever **`subject.kind === 'graph-node'`** (main tab may be **Digest** / **Library** / **Graph** / **Dashboard**). **`NodeDetail`** with **`embed-in-rail`** adds **`role="tablist"`** — **`data-testid="node-detail-rail-tab-details"`** (**Details**, default) vs **`data-testid="node-detail-rail-tab-neighbourhood"`** (**Neighbourhood**). **`graph-connections-section`** (minimap + connections list) is on **Neighbourhood** only (`v-if` when that tab is active). Tab resets to **Details** when the focused graph node id changes. **`data-testid="node-detail-rail-neighbourhood-unavailable"`** — hint on **Neighbourhood** when the center node is missing from the merged slice (same condition as hidden **`graph-connections-section`**).
- **#672 Topic / Entity node view (embedded in NodeDetail rail)** — **`data-testid="topic-entity-view"`** folded **`embedded`** into NodeDetail's **Details** rail tab when a Topic / non-person Entity is focused (**`subject.focusTopic`** / **`focusEntity`** → **`focusGraphNode`**, **`subject.kind === 'graph-node'`**). NodeDetail owns the header — the rail (**`graph-node-detail-rail`**) titles an off-slice **`topic:`** / **`entity:`** id **Topic** / **Entity** via **`inferredKindFromId`** — so TEV's own **`topic-entity-view-kind`** / **`topic-entity-view-name`** header and **`topic-entity-view-go-graph`** / **`topic-entity-view-prefill-search`** action buttons are hidden (**`v-if="!embedded"`**). The standalone **`topic-entity-view-stats`** / **`topic-entity-view-mentions`** / **`topic-entity-view-empty`** sections are retired — the corpus-wide mentions timeline now lives in NodeDetail's **Timeline** rail tab (**`node-detail-rail-tab-timeline`** → **`node-detail-inline-timeline`**). Embedded TEV surfaces: optional **`topic-entity-view-aliases`** + **`topic-entity-view-description`**; **#1055 related topics** **`tev-related-topics`** (**`tev-related-topic-chip`**, topics that *share insights*, sits under NodeDetail's *Theme* which is topics *discussed together*); and the **PRD-033 FR4.2 (#886) relational sections** (async, skeleton-first, loaded on subject change): **`tev-cross-show`** — *Across shows* (cross_show_synthesis), rows **`tev-cross-show-row`** in **`tev-cross-show-list`** + **`tev-cross-show-open`**, loading **`tev-cross-show-loading`**; **`tev-voices`** — *Key voices* (who_said), rows **`tev-voice-row`** in **`tev-voices-list`**, each with a **`tev-voice-link`** opening that person's node view (`subject.focusPerson`); **`tev-entities`** — *Entities involved* (`entities_in_topic`, ranked by mention frequency), **`tev-entity-chip`** chips in **`tev-entities-list`** (+ **`tev-entity-shows`**) each opening that entity's Person/Topic rail (`focusPerson`/`focusEntity`). Entry points: Dashboard topic-cluster chip / NodeDetail related-topic rows / any **`@go-graph`** with a **`topic:`** id → **`subject.focusTopic`** (the legacy Digest topic-title click was removed in the V2 change).
- **Post node-view fold (CURRENT surface)** — the standalone Person rail is retired. **`subject.focusPerson`** now opens the generic **`focusGraphNode`** → **`NodeDetail`** rail (**`subject.kind === 'graph-node'`**), and **`PersonLandingView`** renders **`embedded`** inside it. NodeDetail (not PLV) owns the header + tabs, so PLV's own **`person-landing-view-name`** header and internal tablist (**`person-landing-tab-profile`** / **`person-landing-tab-position-tracker`**) are hidden (**`v-if="!embedded"`**). Current contract: the rail header (**`graph-node-detail-rail`**) titles an off-slice **`person:`** id **"Person"** (via NodeDetail **`inferredKindFromId`**); rail tabs **`node-detail-rail-tab-details`** (**Details**, default → **`person-landing-view`** + **`person-landing-panel-profile`**) and **`node-detail-rail-tab-position-tracker`** (**Positions** → the dedicated positions instance **`person-landing-positions-view`** / **`person-landing-panel-positions`**). Inside the Positions tab, the **By topic** lens (default) carries **`person-landing-insights-voiced`** (topic → Position Tracker entry via **`person-landing-insights-voiced-topic-button`**), and the **All positions** lens (**`person-landing-positions-lens-all`**) carries **`person-landing-stated`**. Picking a topic auto-switches the rail to the Positions tab (NodeDetail watches **`positionTrackerTopicId`**). Off-slice render requires a **`person:`**-prefixed id (the GI speaker_id convention).
- **Person node view — Details tab** (**`person-landing-panel-profile`**, PLV **`view='profile'`**) surfaces the aggregate person view per PRD-029 / UXS-010: **`person-landing-episode-count`** line (SPOKE_IN tally), **`person-landing-organizations`** chip strip (**`person-landing-organization-chip`** per co-mentioned **Organization** via **MENTIONS_PERSON ∩ MENTIONS_ORG**), **`person-landing-role-embedded`** Host / Guest / Mention badge (`data-role`), **Appears in shows** **`person-landing-shows`** (**`person-landing-show-chip`**), the **`person-landing-connections`** relational block — **`person-landing-topics`** (**`person-landing-topic-chip`** + **`person-landing-topics-toggle`** / **`person-landing-topics-empty`**) and **`person-landing-co-speakers`** (**`person-landing-co-speaker-chip`** + **`person-landing-co-speakers-toggle`** / **`person-landing-co-speakers-empty`**) — optional **`person-landing-aliases`** + **`person-landing-description`**, and **Episodes appeared in** **`person-landing-episodes-appeared`** (rows **`person-landing-episodes-appeared-row`** with **`person-landing-episodes-appeared-date`** / **`person-landing-episodes-appeared-date-unknown`**). NOTE: **Topics discussed** **`person-landing-ranked-topics`** (rows **`person-landing-ranked-topic-row`** / **`person-landing-ranked-topic-button`** / **`person-landing-ranked-topic-count`**) renders ONLY in the retired **`view='full'`** standalone — NOT in the embedded Details tab; the embedded topic → Position Tracker entry point is **Insights voiced** in the Positions tab (below).
- **Person node view — Positions tab** (**`person-landing-positions-view`** / **`person-landing-panel-positions`**, PLV **`view='positions'`**) — before a topic is picked, a **`person-landing-positions-lens`** segmented toggle switches two lenses: **By topic** (**`person-landing-positions-lens-by-topic`**, default) → **`person-landing-positions-by-topic`** → **Insights voiced** **`person-landing-insights-voiced`** grouped by Topic (per-group **`person-landing-insights-voiced-group`** `data-topic-id`, header **`person-landing-insights-voiced-topic-button`** + **`person-landing-insights-voiced-topic-count`** that calls `selectTopicForPositionTracker`, a **`person-landing-insights-voiced-toggle`** Show/Hide revealing **`person-landing-insights-voiced-rows`** with **`person-landing-insights-voiced-row`** + **`person-landing-insights-voiced-row-type`** + **`person-landing-insights-voiced-row-text`**); and **All positions** (**`person-landing-positions-lens-all`**) → **`person-landing-positions-all`** → *Stated positions* (PRD-033 FR4.1 — **`person-landing-stated`** / **`person-landing-stated-row`**), *Across the corpus* (**`person-landing-corpus`** / **`person-landing-corpus-list`** / **`person-landing-corpus-row`** / overflow), and in-graph **Attributed quotes** (**`person-landing-positions`**, capped at 50; **`person-landing-positions-pager`**; **`person-landing-positions-empty`** when none). Picking a topic (via **`person-landing-insights-voiced-topic-button`**, or anywhere that sets `positionTrackerTopicId`) renders the **`<PositionTrackerPanel>`** (**`position-tracker-panel`**, PRD-028 / #1049) in this same Positions tab — NodeDetail auto-switches to it. Three UXS-009 states: (1) **`position-tracker-no-topic`** when `subject.positionTrackerTopicId` is null; (2) **`position-tracker-empty`** when a Topic is selected but the loaded slice yields zero rows; (3) **`position-tracker-arc`** with rows — header **`position-tracker-topic-name`** + **`position-tracker-clear-topic`**, multi-select filter strip **`position-tracker-filters`** with one **`position-tracker-filter-<type>`** per insight_type (claim / recommendation / observation / question / unknown), **`position-tracker-filter-empty`** when the filter narrows to zero, and the timeline **`position-tracker-timeline`** of **`position-tracker-row`** cards (per-row **`position-tracker-row-date`** / **`position-tracker-row-date-unknown`**, **`position-tracker-row-type`**, **`position-tracker-row-text`**, plus **`position-tracker-row-quotes`** / **`position-tracker-row-quote`**). Rows join `MENTIONS_PERSON ∩ ABOUT` over the loaded slice, sorted by `Episode.publish_date` then `Insight.position_hint` (`personTopicPositionArc` in `utils/parsing.ts`). Entry points into the whole surface: **Explore Top speakers** rollup → **`explore-top-speaker-link`** → **`subject.focusPerson(speaker_id)`**; **Search result supporting-quote speaker** → **`search-result-speaker-link`** → same (both pass a **`person:`**-prefixed id).
- **`data-testid="episode-related-insights"`** (PRD-033 FR4.3, #886) — *Related insights* region in the **Episode** rail Details panel (below **Similar episodes**); rows **`episode-related-insights-row`** in **`episode-related-insights-list`**, loading **`episode-related-insights-loading`**. From `GET /api/relational/episode-insights?episode=<episode_id>` (the topic/entity siblings of the episode's own insights), async + StaleGeneration-gated. *(FR4.3 transcript-highlighting is a separate deferred feature.)*
- **`data-testid="node-detail-kind-row"`** — first line inside **Details** (when applicable): graph **Type** pill (e.g. **PERSON**, **ENTITY**) plus optional **`entity_kind`**-style label (**person**, **organization**, …); not under the rail **h3**. In-rail **Person** / **Speaker** / **Entity** always get the type pill here; other types follow the non-rail rule (type pill when the overlay would have shown it) plus **`entity_kind`** when surfaced.
- **`data-testid="node-detail-person-entity-prefill-search"`** / **`data-testid="node-detail-person-entity-explore-filter"`** — **Prefill semantic search** and short-label second button (**Speaker filter** / **Topic filter**): people / non-org entities / **Speaker** → **Speaker contains** + clear topic; organization **Entity** → **Topic contains** + clear speaker. Full intent in **`aria-label`**. Disabled when API unhealthy. **`?`** HelpTip explains GI **Person** vs KG **Entity** and the two Explore paths.
- **`data-testid="node-detail-open-topic-profile"`** / **`data-testid="node-detail-open-person-profile"`** (PRD-033 FR5.3, #887) — **Open full Topic/Person/Entity profile →** button at the top of a topic / person-entity node's Details. Opens the populated FR4 Detail rail (`subject.focusTopic` → **Topic Entity View** = `topic-entity-view`; `subject.focusPerson` / `focusEntity` → **Person Landing** = `person-landing-view`), so a graph node is never a dead-end. NodeDetail (neighbourhood / expand / connections) is unchanged — this augments, it does not replace.
- **Graph FR5.1 node emphasis (#887)** — while a search context is active (`activeSearchContext`), nodes whose episode is relevant get the **`context-relevant`** class (size bump + `primary` ring), distinct from the yellow **`search-hit`** ring. Applied/cleared by `GraphCanvas.applyContextEmphasis` on the search-context `byEpisode` watch; no testid (visual class — asserted via the stylesheet unit test `node.context-relevant`).
- **`data-testid="node-detail-insight-details-tip"`** — under **Full insight**, a text-style **HelpTip** trigger (underlined muted label): **`Grounded`**, **`Not grounded`**, or **`Extraction details`** when **`properties.grounded`** is absent but other tip content exists (same string for **`aria-label`**). Tooltip body **`data-testid="node-detail-insight-details-tooltip-body"`** holds the **Grounding** explainer, optional **Other fields** (**Type** / **Position** / **Confidence**), and **Lineage** (model, prompt, optional **`extraction.extracted_at`**, artifact name). Asserted in **`search-to-graph-mocks.spec.ts`** (opens via **`Grounded`** button when the fixture insight is grounded).
- **`data-testid="node-detail-insight-prefill-search"`** / **`data-testid="node-detail-insight-explore-filters"`** — **Prefill semantic search** (truncated insight text) and **Set Explore filters** (clears topic/speaker, sets **Grounded only** + optional **Min confidence** in the store — use **`explore-advanced-open`** to open **Advanced explore** and inspect); disabled when API unhealthy. Asserted in **`search-to-graph-mocks.spec.ts`**.
- **`data-testid="node-detail-insight-related-topics"`** — **Related topics** region (Library **Similar episodes**–style border); list host **`data-testid="node-detail-insight-related-topics-list"`** (no inner max-height or scroll; rail body scrolls). Per-row **`data-testid="node-detail-insight-related-topic-row"`** full-width row **click** focuses that **Topic** on the graph (same as neighbor **G**). Asserted in **`search-to-graph-mocks.spec.ts`**.
- **`data-testid="node-detail-insight-supporting-quotes"`** — **Supporting quotes** (**SUPPORTED_BY** out-edges), sorted by **`char_start`** then **`timestamp_start_ms`**, per-row **G**; **`data-testid="node-detail-insight-view-transcript-all-quotes"`** **Transcript (all quotes)** opens the in-app transcript when every quote shares one **`transcript_ref`** and has finite **`char_start`** / **`char_end`** (multi-span highlights: several **`data-testid="transcript-viewer-highlight"`** marks); **`data-testid="node-detail-insight-supporting-quotes-toggle-expand`** when more than five quotes (**Show all N** / **Show fewer quotes**). Asserted in **`search-to-graph-mocks.spec.ts`**.
- **`data-testid="node-detail-topic-aliases"`** — optional **Aliases:** line when the GI topic node has non-empty **`properties.aliases`**.
- **`data-testid="node-detail-topic-cluster-context"`** — optional **Topic cluster:** line when **`GET /api/corpus/topic-clusters`** returned clustering JSON and this **Topic** is a member but the full cluster panel is **not** shown (legacy path); **`?`** explains corpus clustering vs graph selection. When JSON is loaded, **compound** and **member Topic** selections use the same **Member topics** block instead (**`node-detail-topic-cluster-members`**).
- **`data-testid="node-detail-topic-cluster-members"`** — when **`GET /api/corpus/topic-clusters`** matches the selection (**TopicCluster** compound **or** a **Topic** that is a cluster member): header **TC** avatar + **Topic cluster** badge line (non-rail); **Member topics** list with per-row **Focus** and optional **Load** (catalog-assisted load when `episode_ids` exist for an unloaded member), plus **Hide topics on graph** / **Show topics on graph**. Merged **Cluster neighborhood** minimap + **Connections** / **Via:** copy lives under graph rail **Neighbourhood** tab (**`graph-connections-section`**), not in this member block alone.
- **`data-testid="node-detail-cluster-member-load`** — per-row **Load** when the member topic is **not** in the merge but **`topic_clusters.json`** lists **`members[].episode_ids`** — resolves GI/KG via catalog (same cap as sibling auto-load) and appends to the graph selection; **`data-testid="node-detail-cluster-member-load-message`** — short status line after a load attempt.
- **`data-testid="node-detail-cluster-timeline-unavailable"`** — optional warning when no member topic ids were resolved from JSON or graph-parented Topics (cannot open merged cluster timeline).
- **`data-testid="node-detail-inline-timeline"`** — **inline** timeline section inside **Details** (no timeline modal launch in graph node details). Heading is **Topic timeline** for regular topics and **Cluster timeline** for TopicCluster/cluster-member context. Body states: **`node-detail-inline-timeline-loading`**, **`node-detail-inline-timeline-error`**, **`node-detail-inline-timeline-empty`**, **`node-detail-inline-timeline-results`**. Data comes from **`GET /api/topics/{topic_id}/timeline`** (single topic) or merged **`POST /api/topics/timeline`** (cluster ids). In this flow, tests should assert **`topic-timeline-dialog`** and **`node-detail-topic-timeline`** are absent.
- **`data-testid="graph-tab-panel"`** — root wrapper for the **Graph** main-tab column (sibling merge banner, graph expansion status strip, **`GraphCanvas`**, empty-state copy). Cached with **`keep-alive`** at **`App.vue`** when leaving the **Graph** tab so a loaded graph is not torn down; **`GraphCanvas`** still unmounts when **`displayArtifact`** is false while **Graph** is selected.
- **`data-testid="graph-sibling-merge-line"`** — optional inline line on the **Graph** tab after cluster sibling auto-load (success copy: **`+N new · M in cluster · cap C`**, optional **`· K miss(es)`** when the catalog had no row for some ids). Mocked flows: **`sibling-merge-cluster-mocks.spec.ts`**.
- **`data-testid="sibling-merge-error-banner"`** / **`data-testid="sibling-merge-error-dismiss"`** — top **`role="alert"`** strip when catalog resolve fails; **Dismiss** clears the message.
- **`data-testid="graph-expansion-truncation-line"`** / **`data-testid="graph-expansion-truncation-dismiss"`** — optional strip on the **Graph** tab when `POST /api/corpus/node-episodes` returns **`truncated`**, empty matches, errors, or corpus-path hints; **Dismiss** clears the line. Mocked flows: **`graph-expansion-mocks.spec.ts`** (two quick click cycles on **`.graph-canvas`** at the node to trigger Cytoscape **`dbltap`** on **`topic:ci-policy`** in the patched CI GI fixture).
- **`data-testid="node-detail-topic-prefill-search"`** — **Prefill semantic search** on **Topic** graph node detail (below full topic / aliases; focuses **`#search-q`** in the left query column and fills query; **subject** graph node stays open on the right).
- **`data-testid="node-detail-topic-explore-filter"`** — **Set Explore topic filter** on **Topic** graph node detail (switches left column to **Explore** mode with **Topic contains** filled; user runs explore; **subject** graph node stays open).
- **`data-testid="topic-timeline-dialog"`** — legacy topic timeline modal component. It is not the contract surface for graph node details anymore; transcript modal coverage remains separate under **`transcript-viewer-dialog`**.
- **`data-testid="supporting-quote-speaker-unavailable"`** — muted line under expanded **supporting quotes** on **Search** result cards and **Explore** insight rows when the quote has text but no speaker (same #541 contract). Asserted in **`search-to-graph-mocks.spec.ts`** (Search) and **`explore-supporting-quotes-mocks.spec.ts`** (Explore).
- **`data-testid="search-lifted-quote-speaker-unavailable"`** — muted line in the **Lifted GI insight** block on **Search** transcript hits when **`lifted.quote`** has at least one finite **`timestamp_*_ms`** but **`lifted.speaker`** has no usable display label (#541). Asserted in **`search-to-graph-mocks.spec.ts`**.
- **`data-testid="transcript-viewer-dialog"`** — In-app transcript **`dialog`** (`h2` **Transcript**); header stacks subtitle, **Audio** (**`data-testid="transcript-viewer-audio"`** when corpus **`media/`** resolves for the transcript path — native **`<audio controls>`**; optional seek to **`timestamp_start_ms`** on open), **Passage** (**`data-testid="transcript-viewer-char-range"`** when GI char range exists — single range label or **N character spans (supporting quotes)** for insight multi-quote opens), **`data-testid="transcript-viewer-open-raw"`**, and a short approximate-highlight note; body has **`data-testid="transcript-viewer-body"`**, optional one or more **`data-testid="transcript-viewer-highlight"`** marks, optional **`data-testid="transcript-viewer-timeline"`** (inside **Timeline** `details`).
- **`data-testid="pipeline-job-log-dialog"`** — In-app pipeline job log viewer (#695), built on the shared **`AppDialog`** (close via **`pipeline-job-log-close`**, Esc, or backdrop). Opened from any pipeline surface (`pipeline-job-log-link`, `pipeline-job-history-log-link`, `pipeline-job-explore-full-log-link`, `pipeline-job-explore-metrics-full-log-fallback`) via the `pipelineJobLog` store. Header has tail-size selector **`pipeline-job-log-tail-size`** (16/64/256 KB, default 64), **`pipeline-job-log-refresh`**, **`pipeline-job-log-copy`** (clipboard; **Copied** / **Copy failed**). In-log find bar: **`pipeline-job-log-search`** (highlights matches in **`pipeline-job-log-body`**), **`pipeline-job-log-search-count`** (`active/total`), **`pipeline-job-log-search-prev`** / **`pipeline-job-log-search-next`** (Enter / Shift+Enter step). Body **`pipeline-job-log-body`** (monospace tail, auto-scrolls to bottom); optional **`pipeline-job-log-truncated-hint`** when `truncated`; **`pipeline-job-log-error`** on fetch failure; footer keeps **Download full log** **`pipeline-job-log-download`**. Auto-refreshes (3→10s adaptive) while live, polling **both** the tail and `GET /api/jobs` status so it stops on the real running→terminal transition. Spec: `pipeline-job-log-viewer.spec.ts`.
- **`data-testid="cron-schedule-preview"`** — Live preview under the Job Configuration YAML editor (#709): per `scheduled_jobs` entry rows **`cron-schedule-preview-row-{idx}`** (name + cron + next-run, or **`cron-schedule-preview-invalid-{idx}`** for a bad cron), with a **`cron-schedule-preview-invalid-summary`** count. Client-side via `cron-parser`. Spec: `cron-preview-mocks.spec.ts`.
- **Shared `AppDialog`** — every modal (`status-bar-sources-dialog`, `artifact-list-dialog`, `transcript-viewer-dialog`, `pipeline-job-log-dialog`) is built on it: native `<dialog>` with Esc/backdrop/Close, focus trap, `<h2>` title. Each sets a distinct close testid (`sources-dialog-close`, `artifact-list-close`, `transcript-viewer-close`, `pipeline-job-log-close`); the Close button is also reachable via `getByRole('button', { name: 'Close' })` scoped to the dialog. (The former standalone Vector-index dialog was folded into the Configuration **Index** section.)
- **`data-testid="digest-root"`** — Corpus Digest main column; visible on **Digest** tab (default).
- **`h2` Digest** (`#digest-main-heading`) — panel title **left**; **`?`** **About Digest** — **`HelpTip`** body matches **Episodes** / **Recent** heading tips (**`button-aria-label` About Digest**).
- **`data-testid="digest-toolbar-filters"`** — **Published on or after** label, **`#digest-filter-since`**, and **All time** / **7d** / **30d** / **90d** presets in a **`flex-wrap`** row **below** the title row (full width of the digest column so narrow layouts do not squeeze title + filters on one line).
- **Digest toolbar** — **stacked**: row 1 — **Digest** + **`?`**; row 2 — **`digest-toolbar-filters`** (wraps like Library date row); when digest loaded, a further muted **`text-[10px]`** line: rolling window + episode count.
- **`region` Topic bands** — topic **grid** (**`minmax(min(100%,12rem),1fr)`** auto-fit tracks) without outer **max-height** scroll (**first three** bands + optional **`digest-topic-bands-show-more`**); per-topic **`section`** cards; **first** band **elevated** + **`border-primary/20`**; topic title **`text-sm`** **bold** on first band, **semibold** on others. **Band order is now ranked by retrieval signal** (PRD-033 FR3.1 #885 — best hit score lifted by hit density + distinct-show coverage; config order is the stable tiebreak), so the **first three** are the strongest, not config-order.
- **PRD-033 FR3 (#885) digest topic surfaces:**
  - **`data-testid="digest-band-topic-link"`** (FR3.3) — a topic band whose label maps to a KG topic node (`graph_topic_id` present) renders the label as a **button**; click → **Topic Entity View** Detail rail (`subject.focusTopic`; assert **`data-testid="topic-entity-view"`**). Editorial-only bands (no mapping) stay a static `span` (the V2 non-clickable-headline guard still holds for them).
  - **`data-testid="digest-topic-hit-feed-link"`** (FR3.3) — the feed/show name on a topic-band hit row, a **button** scoping the **Library** to that feed (`feed_id`; same handoff as the Recent **`digest-feed-name-link`**). `aria-label` **`Show only episodes from <feed>`**.
  - **`data-testid="digest-cross-show-toggle"`** (FR3.2) — **Across shows** / **Hide cross-show** disclosure under each mapped band; `aria-expanded`; lazy-loads `GET /api/relational/cross-show?topic=<graph_topic_id>` on first open.
  - **`data-testid="digest-cross-show-band"`** (FR3.2) — the expanded panel; holds **`data-testid="digest-cross-show-row"`** rows, one per distinct show (resolved show label + top insight text), or a muted "No cross-show coverage yet" empty state.
- `region` **Recent episodes** — Digest **Recent** list (below topic bands); **`h2` Recent (`#digest-recent-heading`)** **`text-sm`** semibold + visible **`(N)`**; **`aria-label`** **`Recent episodes, N items`** (or **`1 item`**); **`?`** **About the Recent digest list** explains diversification vs topic bands; focusable rows use **`data-digest-recent-row`** — **Arrow up/down**, **Home**, **End** move focus and update the **Episode** subject rail like a click.
- `button` matching **`Open top hit for topic`** … **`in graph`** — Digest topic band title → **Graph** tab + load + focus (top semantic hit).
- Topic band **hit rows** — **`role="button"`**, accessible name **Open graph and episode details:** then **`episode_title`, `feed`**, then **`, Strong match`** / **`, Good match`** / **`, Weak match`** when **`score`** is present (same tier as the visible label; raw score stays on that label’s native **`title`** only); **click** → **Episode** subject rail + **Graph** (digest topic focus when GI/KG exist on disk), **Digest** tab leaves for **Graph** (no separate **Graph** control on the row); **`grid`** **`h-9`** cover only (**`w-9`**), title + optional **semantic match** label row, recap **`col-span-2`** (**2-line clamp** unless selected); native row **`title`** includes publish date, **E#**, duration, feed hints (not the numeric similarity); optional **recency dot** (**`role="img"`**, **`aria-label`** / **`title`**).
- **`data-testid="library-root"`** — Corpus Library main column; visible on **Library** tab.
- **`data-testid="library-row-why"`** (PRD-033 FR2.1) — "Why this episode" relevance snippet on an episode row; rendered **only when a search/filter context is active** (after a semantic search runs) and the row's `episode_id` matched a hit. Leads with bold **Why this episode:** then the top-scoring segment/insight text. Absent (count 0) by default. The same active context also re-orders rows by hybrid relevance (FR2.2): matched episodes float above non-matches in stable fetch order.
- **`data-testid="library-row-scope-show"`** (PRD-033 FR2.3) — the feed/show name on an episode row, now a **button**; click scopes the Library to that show by setting the feed filter (fires `GET /api/corpus/episodes` with **`feed_id=<id>`**). `aria-label` **`Show only episodes from <feed>`**; hover `title` keeps the catalog feed-name. No-op for ungrouped episodes (empty feed id).
- **`data-testid="library-similar"`** — Similar-episodes region inside the **Episode** subject rail; **`?`** **`About similar episodes`** — tooltip with embedding explanation + **`query_used`** when present.
- **`data-testid="podcast-cover"`** — Optional artwork tile (Library feeds/episodes/detail, Digest cards/topic rows, similar list); shows  placeholder when no URL or image error.
- `region` **Feeds**, **Episodes** — Library main column; **Episodes** panel **`h2`** shows **`Episodes`** + **`(N)`** / **`(N+)`**; list **`region`** **`aria-label`** describes count and whether more pages exist; **Episodes** rows **`data-library-episode-row`** — **Arrow up/down**, **Home**, **End** move focus and update the **Episode** subject rail like a click. **`region` `Episode`** — **shell right column** (**`SubjectRail`**) when an episode is selected or opened from Digest; **Search** / **Explore** live in the **left** query column and do not replace this column. Chrome order: thin **Episode** label row, then **`data-testid="episode-detail-rail-body"`** — hero row **PodcastCover** (``4.5rem`` tile, **left**) + text column (**right**): **h3** episode title (**`node-detail-primary-title`**, same as graph detail), **E** / diagnostics **`?`** / **`C`** (**`data-testid="episode-detail-header-title-copy"`**, **Copy title** → **Copied to clipboard** / **Copy failed; try again**), then feed/date meta; on **Graph** with a graph center id, **`episode-detail-rail-tab-*`** tablist sits **under** that hero and **above** the **Details** scroll (summary, **Open in graph**, **Similar episodes**); **Neighbourhood** holds **`graph-connections-section`**. In Playwright, use **`exact: true`** for **`role="region", name: "Episode"`** so it does not substring-match **Episodes** (list region names start with **Episodes,**).
- `button` **Filters** (collapsible title): **`?` About Library filters** (narrow-down blurb) **immediately after** the title; **All time** / **7d** / **30d** / **90d** on the date row; **Title** + **Summary** inputs on a **shared-width** grid column, **Clear all filters** / **Apply** in the third column; **Clear feed filter** (always; **disabled** until a feed is selected); optional **`library-feed-filter-search`** when many feeds; **`?`** next to **Episodes** heading — **About the Library episode list** (`HelpTip`); **Load more**, **Open in graph**, **Prefill semantic search** — Library. **Below** **Filters**: **`library-topic-cluster-toggle`** (**Clustered episodes only**).
- Collapsed **left** rail: vertical **Search** only (**`LeftPanel`**: expands to **Search** mode + focuses **`#search-q`**). **Explore** is opened from the expanded column (**Explore corpus →**) or graph hand-offs. Collapsed **right** rail: vertical **Details** (**`data-testid="rail-collapsed-subject"`**) expands the **SubjectRail** column only.
- **Strict names (avoid substring clashes):** Digest **Recent** episode row uses **`episode_title`, `feed`** (same pattern as Library episode rows), e.g. **`Digest Episode Alpha, Mock Feed Show`**. Topic band hit rows add the **Open graph and episode details:** prefix to that pair and append **`, Strong match`** (etc.) when **`score`** is mocked; use an **exact** or **full-string** match when both rows exist so **Recent** is not chosen by substring. Library feed row vs episode row: prefer **`Mock Show, feed id f1, 1 episodes`** vs **`Mock Episode Title, Mock Feed Show`** (episode name includes show), not a bare `/Mock Show/` regex.
- **Episode** subject rail — meta block: **feed** on the **first** line (**full width**, **wrap**, same **native `title`** hover as list rows: RSS / feed id / description); **publish date**, **E#**, duration on the **second** line below, **left**-aligned (list-scale `muted`); CIL digest topic pills (**`data-testid="episode-detail-cil-pills"`**) when present, **no** separate **Canonical topics** heading; heading **Key points** (`h4`) when bullets follow summary title/text, with **`border-t`** separator; hero chip stack beside the title: **`E`**, diagnostics **`?`**, **`C`** (**`episode-detail-header-title-copy`**) same pattern as graph node **Copy title**; **`?`** opens **role="tooltip"** with troubleshooting rows (**Feed in vector index**, metadata/GI/KG paths, ids, index stats when loaded) — Library Phase 3 (RFC-067).
- **Prefill semantic search** (Library) / **Search topic** (Digest) open Search with query (and feed / **Since (date)** for Digest from **`window_start_utc`**) already filled; no separate handoff banner. Library prefill uses the same field order as **Similar episodes** (`build_similarity_query`) with **client-side length caps** (long recap in `summary.title` or one giant bullet does not fill the query box).
- **`data-testid="library-similar-empty"`** — Similar-episodes ran successfully but returned no peer rows.
- `locator('body')` — click at a small offset to move focus away from the search field before **`/`** ([keyboard-shortcuts.spec.ts](keyboard-shortcuts.spec.ts)).
- `button` **Show on graph** — search result row (**G**, GI token); mocked search → graph flow ([search-to-graph-mocks.spec.ts](search-to-graph-mocks.spec.ts)). **`Open episode in subject panel`** (**L**) when hit metadata includes **`source_metadata_relative_path`**, corpus path is set, health is OK, and the row is **not** a merged KG surface (**`kg_surface_match_count` ≤ 1** or absent). **`Episode summary in right panel`** (**S**) — same gating as **L**; opens the **Episode** subject rail without switching main tab to **Library** (focus/hit navigation does **not** auto-open episode mode). **E** hidden under the same merged condition; **G** remains with an explanatory tooltip.
- `.graph-canvas` — graph container; click before **Esc** test; visibility after mocked search path; changing the class requires updating specs and this map.
- **`data-testid="graph-bottom-bar"`** — replaces canvas **`.graph-zoom-controls`** / **`.graph-layout-controls`**; use testids above for **Fit**, zoom, optional **Gestures** (when no full-graph stats strip), **`graph-export-png`** (**PNG**, `aria-label` **Export PNG**), **Re-layout**, **layout cycle**, and bar collapse/expand.

**Network routes (tests differ by intent)**

- **`**/api/health` — abort** (`failed`) — `loadGraphViaFilePicker` (offline file picker path).
- **`**/api/health` — fulfill** 200 + `{ status: 'ok', corpus_library_api: true, corpus_digest_api: true }` (extra health booleans optional; mocks may omit **Artifacts** / **search** / … flags → UI treats as **Yes**). Used by `keyboard-shortcuts`, `corpus-hints`, `search-to-graph-mocks`, `dashboard-index-rebuild-mocks`, `library.spec.ts`, `digest.spec.ts` (**Digest** needs **`corpus_digest_api`** or upgrade message; **Library** no longer calls digest).
- Library tab mocks (`library.spec.ts`): `**/api/corpus/feeds`, `**/api/corpus/episodes`, `**/api/corpus/episodes/detail`, `**/api/index/stats`, `**/api/corpus/episodes/similar` (when testing Phase 3 UI). One spec **unroutes** `**/api/corpus/episodes**` and re-fulfills so it can assert the next Library **`GET /api/corpus/episodes`** after checking **Clustered episodes only** (**`library-topic-cluster-toggle`**) includes **`topic_cluster_only=true`**. Another spec **unroutes** **`**/api/corpus/feeds**`** with **16** feeds to assert **`library-feed-filter-search`**.
- **Dashboard** corpus charts (when online + corpus path): `**/api/corpus/stats?**`, `**/api/corpus/runs/summary?**`, `**/api/corpus/feeds?**` (for manifest bar labels; 404 OK if ignored), optional `**/api/corpus/documents/manifest?**` (404 OK); `**/api/artifacts?**` for GI+KG mtime timeline.
- **`**/api/index/stats`** (optional) — Dashboard index metrics + staleness + `rebuild_in_progress` / `rebuild_last_error` (#507). **`**/api/index/rebuild`** POST (optional) — background index job; respond **202** for happy path tests.
- **Viewer-only API paths** (`status-bar-feeds-operator-mocks.spec.ts`, `dashboard-pipeline-jobs-mocks.spec.ts`): use a **pathname predicate** (exact `/api/feeds`, `/api/operator-config`, `/api/jobs`, `/api/health`, …) instead of a loose glob like `**/api/feeds*`, or Vite may load `/src/api/feedsApi.ts` and receive JSON instead of the TS module (blank app / timeout).

**Theme**

- `localStorage` key **`gi-kg-viewer-theme`** — set to **`light`** in light test (`addInitScript` before `goto`).
- CSS variable **`--ps-canvas`** on `document.documentElement` — asserted **`#111418`** (dark) / **`#f6f7f9`** (light); keep in sync with [theme.spec.ts](theme.spec.ts) and UXS-001.

## Maintenance

Use this order for **viewer UX** work (humans and agents); details also live in
[E2E Testing Guide — When you change viewer UX](../../../docs/guides/E2E_TESTING_GUIDE.md#when-you-change-viewer-ux-required-workflow).

1. **This file (`E2E_SURFACE_MAP.md`)** — When **user-visible labels**, **routes**, **E2E entry flows**,
   or **selectors** in specs change, update the map in the **same PR** (usually **before** or
   alongside test edits so the contract stays obvious).
2. **Playwright** — Update `e2e/*.spec.ts`, `helpers.ts`, or `fixtures.ts`; run **`make test-ui-e2e`**.
3. **UXS** — Update [VIEWER_IA.md](../../../docs/uxs/VIEWER_IA.md) when **shell information architecture**
   changes (regions, axes, persistence, clearing). Update the relevant feature UXS file
   ([UXS-002](../../../docs/uxs/UXS-002-corpus-digest.md) through
   [UXS-006](../../../docs/uxs/UXS-006-dashboard.md)) when the **visual or experience contract**
   (layout, density, documented patterns) changes; update
   [UXS-001](../../../docs/uxs/UXS-001-gi-kg-viewer.md) when **shared** tokens or design
   system primitives change. See [UXS index](../../../docs/uxs/index.md) for the full list.

- **Reviewers:** if a PR changes Playwright selectors or primary control labels, confirm
  `E2E_SURFACE_MAP.md` was updated when applicable.
- If copy is expected to churn often, consider adding `data-testid` in Vue and documenting it
  here (follow-up; not required for v1).

## Commands

From repo root: `make test-ui-e2e`. In package: `npm run test:e2e` (under `web/gi-kg-viewer`).
