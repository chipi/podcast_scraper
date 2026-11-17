# Viewer graph — implementation specification

> **Single place** for Cytoscape load, lens window + episode cap, status line,
> default node types, stylesheet, gesture overlay, and graph-focus entry points.
> **User-visible** behaviour and test IDs: [UXS-004](../uxs/UXS-004-graph-exploration.md).
>
> **Where it renders (2026-11-17).** This is the spec of the public Workbench graph
> (`web/workbench/`, desktop only). The nodes a public install draws come from KG
> artifacts. Grounded-insight (GI) rendering is drawn by a downstream build; the UXS docs'
> "Where it renders" sections in `docs/uxs/` say which parts. `web/workbench/src/extensions/index.ts`
> registers nothing in this build, so the extension-fed behaviour below (`graphTypeTraits`,
> `episodeScoreBonuses`) is empty here. The Explore surface is retired; its search lives in Search.

## Graph initial load

**Status:** Implemented. Written as a design in April 2026; sections 3, 4, 5, 6 and 9 below describe
the shipped behaviour, and where the original design differed the section says so.
**Repo:** `chipi/podcast_scraper`
**Target area:** `web/workbench/src/`
**Related docs:** UXS-001, UXS-004, RFC-062, RFC-076
**Scope:** Load strategy, graph time lens, default node type visibility,
status line. No Cytoscape stylesheet changes (those are in the Graph
Styling spec).

---

### Initial load — 1. Problem

Before this work the graph had no defined default load strategy. In online mode
with a corpus path set, loading all GI/KG artifacts produces a hairball
for any non-trivial corpus. There was no time lens on the Graph tab, no
episode cap, and no visual indication of what is loaded or how to expand.

Simultaneously, the shared `corpusLens` (Digest + Library) is the only
time filter in the app. If Graph naively inherits it, a user who set
Library to "all time" to browse the full catalog would trigger a full
corpus graph load — defeating any load cap.

---

### 2. Direction

Three connected decisions:

1. **graphLens** — Graph owns its own time lens, seeded from `corpusLens`
   at first open but diverging independently thereafter.

2. **Episode cap** — Graph loads at most N episodes (default 22) from
   the `graphLens` window. RFC-076 progressive expand is the mechanism

   for loading more.

3. **Default node type visibility** — High-noise, low-signal node types can
   be hidden by default. A build declares which (see section 6); this build
   declares none.

These three together produce a first graph load that is readable,
fast, and coherent with the rest of the viewer.

---

### 3. graphLens

#### Concept

The graph time lens is owned exclusively by the Graph tab. It is
**not** `corpusLens` and does not share state with Digest or Library. It lives
in the `graphExplorer` store (`web/workbench/src/stores/graphExplorer.ts`) as
`sinceYmd`, a local `YYYY-MM-DD` lower bound, where the empty string means all
time.

#### Initialisation

On first Graph tab open per session, `seedFromCorpusLensIfNeeded()` sets
`sinceYmd`:

- If `corpusLens` has a date set (the user set Library to "30d"), Graph
  seeds from that date.
- Otherwise it uses `seedDefaultSinceYmd()`: **last 7 days**. The build-time
  variable `VITE_DEFAULT_GRAPH_LENS_DAYS` overrides it: a positive integer N is
  the last N days, `0` or empty is all time. The stack-test compose overlay builds
  with `0` because its fixtures carry static 2025 dates.

After the seed, the lens is independent. Changing `corpusLens` in
Digest/Library does not update it; changing it in Graph does not affect
Digest/Library.

#### Values

| Control | Action on the store | `sinceYmd` |
| --- | --- | --- |
| `7d` / `30d` / `90d` | `setPresetDays(7 \| 30 \| 90)` | that many days ago |
| `All` | `setAllTime()` | `''` |
| Since date input | `setSinceYmd(value)` | the typed date |

#### Store

```typescript
// useGraphExplorerStore (stores/graphExplorer.ts)
sinceYmd: Ref<string>              // '' = all time
seeded: Ref<boolean>               // true once initialised
graphTabOpenedThisSession: Ref<boolean>
lastAutoLoadWasCapped: Ref<boolean>

seedFromCorpusLensIfNeeded(): void
setSinceYmd(next: string): void
setPresetDays(days: 7 | 30 | 90): void
setAllTime(): void
resetSinceYmdToInitialCorpusSeed(): void   // the Graph "Reset" button
resetGraphLensForNewCorpus(): void         // called on corpus path change (App.vue)
```

On corpus path change `resetGraphLensForNewCorpus()` clears the seeded flag
so the next Graph tab open re-seeds from the new corpus context.

---

### 4. Episode Cap

#### Rule

When loading GI/KG artifacts for the graph, select at most **N episodes**
from the candidate pool defined by `graphLens`.

**Candidate pool selection (implemented in `web/workbench/src/utils/graphEpisodeSelection.ts`):**

1. Build one **episode stem** per GI/KG/bridge basename group; use listing `publish_date` (YYYY-MM-DD) per stem.
2. **Filter** by `graphLens`: dated window = `publish_date >= sinceYmd`; **all time** = all stems.
3. **Score** each stem: `recency_weight + registered bonuses`. Recency is linear from **0.2** (oldest in the dated pool, or older than the trailing **90**-day window from the newest publish for **all time**) to **1.0** at the newest. Bonuses come from entries registered on the `episodeScoreBonuses` extension point (`web/workbench/src/extensions/index.ts`); this build registers none, so the score is the recency term.
4. Sort by **score descending**, then **publish date descending**, then stem id (stable tie-break).
5. Take the **top N** stems; load all GI/KG/bridge paths for those stems only.

The cap controls load time and canvas density, not node count directly.

#### Default cap value

**N = 22** (`GRAPH_DEFAULT_EPISODE_CAP` in `graphEpisodeSelection.ts`). It was 15 in the
original design; the constant's comment in that file records why it is tuned for a retina
laptop's first paint and why it must not rise toward the full corpus without interaction
mitigations.

#### Artifact load implementation

In the artifact loading flow (`stores/artifacts.ts`, which passes `GRAPH_DEFAULT_EPISODE_CAP`):

1. `GET /api/artifacts` returns all available artifact paths (with `publish_date` per row).
2. `selectRelPathsForGraphLoad` runs the **score + cap** selection above, and only the chosen paths are loaded into the merged graph. A local-file load uses `selectParsedArtifactsForGraphLoad` with the same scoring.

`GRAPH_SCORE_GI_DENSITY_MAX` exists as a documented tunable for a future per-episode GI density term; it is not applied.

---

### 5. Graph Status Line

The status line has two parts, both from `GraphStatusLine.vue`. The **summary** row
(`variant="summary"`) sits above the canvas and shows counts. The **controls** row (lens
selector, Since input and Reset) sits in the **bottom bar** centre zone (`GraphBottomBar`).
Both update reactively.

#### Content

```text

Examples:

Showing all time · 22 episodes (capped) · 934 nodes · 3 components

```

- Node count: use `k` suffix for counts ≥ 1000 (e.g. `1.2k`)
- Episode count rounds down: show loaded count, not candidate pool size

#### Styling

- `text-[10px]`, `muted` token — unobtrusive, same density as toolbar
- The summary row is left-aligned text; the lens controls are centred in the bottom bar
- Same compact scale as toolbar rows

#### Lens selector

Compact inline control — same visual style as the Digest Window
presets but smaller. Options: **7d · 30d · 90d · All** and a **Since** date input.

Active option uses `primary` ring or underline (same pattern as
Digest preset active state). Selecting a new value:

1. Updates the graph lens (`sinceYmd`)
2. Re-runs artifact load with new window
3. Clears current graph (replaces, does not append)
4. RFC-076 expansion state is reset

A **Reset** button (`graph-status-reset`, shown when the graph differs from its initial slice)
clears cross-episode expansions, restores the default time slice and fits the graph.

#### data-testid values

```text

data-testid="graph-status-line"               (summary row)
data-testid="graph-status-lens-label"
data-testid="graph-status-episode-count"
data-testid="graph-status-capped"             (only when the auto-load was capped)
data-testid="graph-status-node-count"
data-testid="graph-status-component-count"
data-testid="graph-status-line-controls"      (controls row)
data-testid="graph-status-lens-selector"
data-testid="graph-status-since-input"
data-testid="graph-status-reset"

```

`graph-status-episode-count`, `graph-status-node-count` and `graph-status-component-count` wrap **numeric text only** (node count may use the `k` suffix from formatting).

---

### 6. Default Node Type Visibility

#### Problem

The Types chip has per-type checkboxes. When every type is on, evidence-layer
and plumbing nodes (Quote, Speaker, Episode in the original design) create
visual noise without adding to the first-read value.

#### Mechanism

`defaultFilterState` (`utils/parsing.ts`) starts each node type on unless
`typeHiddenByDefault(type)` is true, which reads `defaultHidden` from the
entries registered on the `graphTypeTraits` extension point. A build that
wants Quote, Speaker and Episode off by default declares them there.

**This build registers no `graphTypeTraits`, so every node type present in the
loaded KG artifacts starts on.** The Types chip lets the user turn any type on
or off freely.

#### Visual indicator

When type visibility differs from the defaults, a **reset** control
(`data-testid="graph-types-reset"`, in `GraphTypesChip.vue`) appears; clicking it
calls `resetGraphTypeVisibilityDefaults()` and returns Types to the defaults.

---

### 7. Interaction with RFC-076 Progressive Expand

The episode cap and the graph lens define the **starting graph**. RFC-076
expand is the mechanism for growing it.

When the user double-clicks an eligible node:

- `POST /api/corpus/node-episodes` returns additional episode paths for
  that identity

- Those paths are appended to the merged graph regardless of the lens
  window — expand deliberately goes beyond the current lens

- The status line updates: episode count and node count increase
- The expanded node shows the blue ring (already in RFC-076 spec)
- The status line does NOT show "(capped)" for expanded content — the
  cap only applies to the initial auto-load

When the lens is changed (user selects a new window from the lens selector):

- All RFC-076 expanded content is cleared
- Fresh load from new window with cap applied
- This is expected — changing the lens is a deliberate context switch

Any other **full graph reload** from the artifacts store (Digest/Library handoff to graph, Dashboard **Load into graph**, corpus sync, etc.) uses the same rule: **`loadSelected()`** resets RFC-076 expansion by default. Only **append/remove** reload paths preserve expansion so expand/collapse stay consistent (see RFC-076 *Expansion reset vs full reload*).

---

### 8. Interaction with Gesture Overlay

The gesture overlay (separate spec) teaches the dbl-click expand gesture.
The status line complements it with persistent context — "22 episodes
loaded" tells the user the graph is a slice, not the whole corpus, and
that there's more to explore via expand.

The overlay's "Dbl-click → Load more episodes for this node" hint
directly addresses the affordance the status line implies. The two
features are designed to work together.

---

### 9. Where it lives

```text

web/workbench/src/stores/graphExplorer.ts
  — graph time lens (sinceYmd, seeded), seedDefaultSinceYmd

web/workbench/src/utils/graphEpisodeSelection.ts
  — score + cap selection, GRAPH_DEFAULT_EPISODE_CAP

web/workbench/src/stores/artifacts.ts
  — applies the selection before loading artifact paths

web/workbench/src/utils/parsing.ts
  — defaultFilterState / typeHiddenByDefault (node type defaults)

web/workbench/src/components/graph/GraphStatusLine.vue
  — summary row and lens controls row

web/workbench/src/components/graph/GraphBottomBar.vue
  — hosts the lens controls (centre zone); GraphCanvas.vue hosts the summary row

web/workbench/src/components/graph/chips/GraphTypesChip.vue
  — Types chip and the reset control

web/workbench/e2e/E2E_SURFACE_MAP.md
  — selectors for the surfaces above

```

User-visible behaviour and test IDs are documented in
[UXS-004](../uxs/UXS-004-graph-exploration.md); the cap and recency tunables in
[UXS-001](../uxs/UXS-001-workbench.md).

---

### 10. Checkpoints

**Checkpoint 1 — graph lens**

- `sinceYmd` seeds from `corpusLens` on first Graph tab open, resets on corpus
  path change

- Changing `corpusLens` in Digest/Library does not affect the graph lens
  after seed

**Checkpoint 2 — Episode cap**

- Graph loads at most 22 episodes from the lens window (scored selection, not pure recency)
- "(capped)" indicator appears when pool was truncated
- "all time" lens still caps at 22 **highest-scoring** episodes across the corpus (not the entire corpus)

**Checkpoint 3 — Status line**

- Status line shows correct episode count, node count, active lens
- Lens selector changes the lens and triggers reload
- Counts update after RFC-076 expand

**Checkpoint 4 — Default type visibility**

- With no `graphTypeTraits` registered, every type present starts on
- With `defaultHidden` declared, those types start off and the reset control restores them

---

### 11. Open Questions (Not Blocking)

1. **Cap value** — 22 is a tuned default (see the comment on
   `GRAPH_DEFAULT_EPISODE_CAP`). Revisit with a real corpus when canvas
   interaction cost changes.

2. **GI density term** — `GRAPH_SCORE_GI_DENSITY_MAX` is declared but unused
   until a per-episode GI count is available from the API.

---

## Graph handoff orchestrator

**Status:** Active — see [ADR-094](../adr/ADR-094-graph-handoff-orchestrator-fsm.md)
for the locked decisions and rationale.
**Repo:** `chipi/podcast_scraper`
**Target area:** `web/workbench/src/services/graphHandoffFsm.ts`,
`stores/graphHandoff.ts`, every entry-point component (Library / Digest /
Search / Dashboard / Episode panel / NodeDetail / GraphConnectionsSection /
SubjectRail / StatusBar / App-shell tab / GraphCanvas direct
interactions). The Explore surface this lists in the original audit is retired.
**Related docs:** [ADR-094](../adr/ADR-094-graph-handoff-orchestrator-fsm.md)
(canonical decisions), [VIEWER_ASYNC_STABILITY.md](VIEWER_ASYNC_STABILITY.md)
(adjacent async patterns),
`web/workbench/src/services/README.md`
(code-local FSM contract),
`web/workbench/e2e/HANDOFF_MATRIX.md`
(coverage contract).
**Scope:** Operational reference for adding new entry surfaces and debugging
handoff bugs. The decisions and rationale live in ADR-094; this section is
"how the orchestrator integrates with the rest of the graph viewer."

---

### Why the orchestrator exists

Pre-fix, navigating to graph from any of 13 entry surfaces (Library row,
Digest pill, Search "Show on graph", Dashboard topic landscape, Episode
panel, NodeDetail Load, GraphConnections neighbour, SubjectRail @go-graph,
StatusBar @go-graph, Explore focus, App-shell tab, plus canvas direct
interactions) mutated some combination of three Pinia stores (`subject`,
`graphNavigation`, `artifacts`) in slightly different orders. No single
owner enforced the invariant that `filteredArtifact` containing node N
implies `cy.core.$id(N).nonempty()` before focus / camera asserts. The
canonical user-visible bug was "second Library G does nothing" — a load-source
asymmetry let the `filteredArtifact` watcher early-return without scheduling
a redraw. See [ADR-094](../adr/ADR-094-graph-handoff-orchestrator-fsm.md) and
[RFC-085](../rfc/RFC-085-graph-handoff-orchestrator-retrospective.md) for the
audit and decisions (the original pre-fix audit at
the historical `GRAPH_NAVIGATION_HANDOFF_ANALYSIS.md` (removed after) was retired after the
orchestrator shipped).

The fix: a finite-state machine that owns the handoff lifecycle from
click to settled selection.

---

### Entry-point catalog (compressed)

Reference for each surface's FSM event + envelope shape. Full per-row test
coverage in
`web/workbench/e2e/HANDOFF_MATRIX.md`.

| ID | Surface | FSM event | source | loadSource | camera |
| --- | --- | --- | --- | --- | --- |
| L1 | Library row "Open in graph" | `handoffRequested` | `library` | `subject-external` | `center-on-target` |
| D1 | Digest recent topic pill (drawn by a downstream build, not the public `DigestView`) | `handoffRequested` | `digest` | `digest-external` | `center-on-target` |
| D2 | Digest topic-band hit row | `handoffRequested` | `digest` | `digest-external` | `center` (topicFocus / eid) |
| D3 | Digest topic-band title | `handoffRequested` | `digest` | `digest-external` | `fit` (multi-episode) |
| S1 | Search "Show on graph" | `handoffRequested` (dispatched directly from `SearchPanel.onFocusHit`) | `search` | `subject-external` | `center-on-target` |
| E1 | Episode panel "Open in graph" | `handoffRequested` | `episode-panel` | `subject-external` | `center-on-target` |
| O3 | NodeDetail Load + neighbour go-graph | `expansionRequested` (dispatched directly from `NodeDetail.focusTopicClusterMember` / `focusNeighborOnGraph`) | `node-detail` | `graph-internal` | `center-on-target` |
| O1/O2/O4–O6 | Dashboard / SubjectRail / StatusBar @go-graph (Explore retired) | `handoffRequested` (via `activateGraphTab(source: …)`) | (per surface) | `subject-external` | `center-on-target` |
| G1/G2 | Canvas single-tap (incl. onetap rail) | `canvasTapped` | `canvas-tap` | `graph-internal` | `center` (cyId) |
| G3 | Canvas double-tap expand | `expansionRequested` | `double-tap-expand` | `graph-internal` | `preserve` |
| G6 | Mini-map / GraphConnections neighbour | `canvasTapped` (with `suppressCamera: true`) | `minimap` | `graph-internal` | `preserve` |
| K1 | Escape (clear graph focus) | `focusCleared` | — | — | — |
| — | Corpus path change | `corpusReloaded` | — | — | — |
| — | First mount with saved preference | internal init `handoffRequested({ source: 'restore-preference' })` | `restore-preference` | `subject-external` | `center` |

The `loadSource` enum has 3 values (decision #2): `subject-external` for
non-graph surfaces opening a single subject, `digest-external` for Digest
paths, `graph-internal` for in-graph expansions and direct interactions
(Definition X — graph-internal = "expansion that preserves layout").

The `camera` strategy has 5 variants
(decision #11): `center` (explicit cyId + optional bbox includes),
`center-on-target` (resolve cyId during apply phase from envelope's
`cyId | metadataPath | episodeId`), `fit` (multi-node loads), `preserve`
(canvas tap, mini-map), `none` (explicit no-op).

---

### State-walking lifecycle

A typical Library "Open in graph" handoff walks all 8 states:

```text
idle / ready
   │
   │ handoffRequested(envelope)
   ▼
loading_fetch                         (territory auto-load HTTP starts)
   │
   │ advanceState (in loadEpisodeSliceForTerritoryStrip)
   ▼
loading_merge                         (artifacts.appendRelativeArtifacts)
   │
   │ filteredArtifact watcher → scheduleRedraw → runRelayout
   │ runRelayout: graphHandoff.advanceState('redrawing_full')
   ▼
redrawing_full                        (cy layout running)
   │
   │ lo.one('layoutstop'): graphHandoff.notifyLayoutStop()
   ▼
applying                              (finishLayoutPass: select + dim + camera)
   │
   │ end of finishLayoutPass: graphHandoff.recordApplied(cyId)
   ▼
ready                                 (stuck timer disarmed; lastResult = 'applied')
```

For canvas direct interactions (G1/G2/G3/G6), the FSM jumps `idle/ready →
applying → ready` without going through `loading_*` (no artifacts to
fetch, no layout to run for a tap). Because no later layout pass finishes
that handoff, the single-tap path in `GraphCanvas.vue` (`onetap`) calls
`graphHandoff.recordApplied(id)` itself right after it centres the camera;
without that the handoff sat in `applying` until the 15 s stuck-timeout and
raised the error strip.

For canvas double-tap expand (G3), the FSM walks `loading_fetch →
loading_merge → redrawing_incremental → applying → ready` (incremental
layout preserves positions of unchanged nodes).

---

### Generation-token check points (the bare-await contract)

Every async `await` inside orchestrator code (notably
`loadEpisodeSliceForTerritoryStrip` in
`web/workbench/src/components/graph/GraphCanvas.vue`)
**must** be paired with an `isStale(envelope.generation)` check on the
next non-empty line. Without it, an in-flight handoff can mutate UI state
for an envelope that has already been superseded by a newer click —
producing the exact "old episode wins after rapid clicks" race the FSM is
designed to prevent.

The 8 documented check points are listed in the
`web/workbench/src/services/graphHandoffFsm.ts` header.
Adding a new `await` to an orchestrator path requires a matching
`isStale()` check; the contract is currently review-only (no automated
ESLint rule yet).

---

### Adding a new entry surface (recipe)

1. **Build the envelope** with the right `kind` / `cyId | metadataPath |
   episodeId` / `source` / `loadSource` / `camera` (see catalog above for
   the right combination per surface type).
2. **Fire the right FSM event** at click time:
   - cross-surface handoff (most cases) → `graphHandoff.handoffRequested(envelope)`
   - canvas direct selection → `graphHandoff.canvasTapped(envelope)`
   - additive expansion (load more on existing graph) → `graphHandoff.expansionRequested(envelope)`
3. **Synchronously set the load-source** on `artifacts`
   (`artifacts.setLoadSource(envelope.loadSource)`). Decision #1: this
   prevents the canonical "second G does nothing" bug class.
4. **Add a row to**
   `web/workbench/e2e/HANDOFF_MATRIX.md`
   under the right section (cold-start / hot-state / repeat-click /
   cross-entry / concurrency / failure / lifecycle).
5. **Write a Playwright contract test** asserting the surface fires the
   right event with the right envelope payload (see existing examples in
   `web/workbench/e2e/handoff/contracts.spec.ts`). This mechanically
   enforces the migration so a future refactor can't silently undo it.

---

### Tab-switch during in-flight handoff — institutional contract

**The rule:** when `GraphCanvas.onActivated` fires and `graphHandoff.pending`
is non-null with FSM in any in-flight state (`loading_fetch` / `loading_bootstrap` /
`loading_merge` / `redrawing_*` / `applying`), the helper
`tryApplyPendingFsmEnvelopeFromTabReturn(cy)` must drive the FSM forward to
`applied` (when the target resolves in cy) or leave it for the stuck-timer
to surface as `failed`.

**Why this exists:** without the helper, an L1 / E1 click followed by an
immediate tab-switch leaves the FSM in `loading_fetch` with no path to
terminal — the user returns to the Graph tab to find an empty canvas + a
15-second-delayed "stuck-timeout" error strip. The natural redraw-driven
apply path (`finishLayoutPass` → `recordApplied`) only fires after a real
`layoutstop`, which may not happen if the load completed while the tab was
inactive.

**Don't regress:** if you refactor `GraphCanvas.onActivated`, preserve the
call to `tryApplyPendingFsmEnvelopeFromTabReturn` on every activation where
`graphHandoff.pending` is non-null. The Tier-3 P5.2 spec at
`web/workbench/e2e/validation/handoff-matrix-real-corpus.spec.ts` catches
removal of this hook end-to-end.

**Related fix in click handlers:** the L1 handler (`LibraryView.openEpisodeInGraph`)
also calls `artifacts.loadSelected({ preserveExpansion: true })` after
`appendRelativeArtifacts` to force a redraw when the target's artifacts are
already in the store (so the append is a no-op). Without this, two
in-rapid-succession handoffs targeting the same episode stuck-timeout.
The cross-entry sequences are exercised by Tier-2
`e2e/handoff-production/cross-entry.spec.ts` and Tier-3
`e2e/validation/handoff-matrix-real-corpus.spec.ts`.

---

### Debugging

- **Inspect the FSM in dev**: `window.__GIKG_FSM__` exposes
  `{ state, pending, generation, lastResult }` (read-only). Useful in DevTools
  console while reproducing a stuck handoff. The hooks (`__GIKG_FSM__`,
  `__GIKG_HANDOFF_STORE__`, `__GIKG_SUBJECT__`) exist in dev builds and in
  builds made with `VITE_E2E_HOOKS=true` (the stack-test overlay sets it);
  a normal production build strips them.
- **Inspect Cytoscape directly**: `window.__GIKG_CY_DEV__` (existing
  pattern, predates the FSM). Combine with `__GIKG_FSM__` to see whether
  the FSM thinks the handoff completed but cy.core doesn't have the
  expected node — that's a self-healing reconciliation case.
- **Read error-strip messages**: failed handoffs render
  `data-testid="handoff-error-strip"`, which reads "Could not open this in the graph: <reason>". Stuck-timeout failures
  carry `reason: "stuck-timeout after 15000ms"`; HTTP failures carry the
  `Error.message` of the originating fetch.
- **Self-healing warnings**: invariant violations log
  `[graphHandoff invariant] expected nodes missing from cy core: count=N
  sample=...` on every `layoutstop`. After a targeted reconciliation:
  `reconciled missing=N via targeted core.add (retry budget consumed for
  gen=G)`. After exhausted retry: `divergence accepted: missing=N retried=true`.
- **Umami telemetry**: `graph_handoff_started/applied/failed/stuck` events fire
  from `stores/graphHandoff.ts` via `lib/analytics.ts` `track()`. Use to track
  handoff success rates per source in production. (Migrated off PostHog Cloud →
  self-hosted Umami; see ADR-126 (not in this repository).)

---

### Matrix assertion layers (L0–L6)

The 41-row `web/workbench/e2e/HANDOFF_MATRIX.md` drives the 6-point
standard contract through six composable assertion layers. Each row asserts as deeply as its fixture allows; the layer profile
per row is recorded in the matrix doc's "Layers" column.

| Layer | What it asserts | Hook | Helper |
| --- | --- | --- | --- |
| **L0** | API mocks return expected shape | n/a | `setupHandoffMatrixMocks` |
| **L1** | FSM `ready` + `pending=null` + `lastResult.status=applied` + generation bumped | `__GIKG_FSM__` | `readFsmState` |
| **L2** | Exactly 1 cy node selected with id matching expected (modulo `g:`/`k:`/`__unified_ep__:` prefix) | `__GIKG_CY_DEV__` | `assertHandoffApplied` |
| **L3** | Camera zoom in `[0.2, 5]` AND target node `renderedPosition` within inner `cameraCenterTolerance × viewport` (default 0.35) box; no console errors; Episode panel title (when relevant) | `__GIKG_CY_DEV__` + `console.error` | `assertHandoffApplied` |
| **L4** | FSM event envelope shape: `type` + `source` + `kind` + `loadSource` + `camera.kind` | `__GIKG_FSM_EVENT_LOG__` | `assertFsmEventEnvelope` |
| **L5** | Subject store: `subject.kind` + matching id field reflects target | `__GIKG_SUBJECT__` | `assertHandoffApplied` |
| **L6** | Self-healing invariant: `viewWithEgo(focusNodeId)` ⊖ `core.nodes()` is empty after most recent `finishLayoutPass` | `__GIKG_FSM__.lastInvariant` | `assertHandoffApplied` |

**UI-driven rows** (Library / Digest pill / Search "Show on graph" /
Episode panel / NodeDetail Load / Dashboard topic-cluster chip / hot-state /
repeat / cross-entry / lifecycle tab-reconcile) assert **L0+L1+L2+L3+L5+L6**
through `assertHandoffApplied`. L6 is advisory (accepts `null` when no
layoutstop fired since the handoff — typical when the target was already
in the graph and no redraw was needed).

**Dev-hook-driven rows** (SubjectRail / StatusBar / Mini-map no-target
funnels, composite multi-envelope sequences, failure modes, lifecycle
restore) assert **L0+L1+L4** through `assertFsmEventEnvelope`. They
dispatch envelopes via `__GIKG_HANDOFF_STORE__` and can't reach the
"applied" outcome without a real graph change, so they pin the FSM-event
contract instead of the full outcome contract.

**Filter rows** (Section 8) assert the inverse: no `handoffRequested`
fires and FSM state stays quiescent — filters must not drive graph state.

The **camera-center check** (part of L3) addresses the GH #771 failure
class where `cy.zoom()` is fine but pan misaligned, leaving the user
looking at empty space or wrong nodes. Implementation polls
`node.renderedPosition()` until it stabilizes (2-consecutive identical
reads, cap 2.5 s — waits for `cy.animate` to settle since `recordApplied`
marks `ready` before animation completes), then asserts pixel distance
from viewport center. Skip via `skipCameraCenter: true` for envelopes
with `camera: 'fit'` / `'preserve'`.

The **L6 self-healing invariant** is stashed by `finishLayoutPass` on
every layoutstop (in production, not just dev) via
`graphHandoff.recordInvariant(missing, extra)`. The same predicate drives
production reconciliation: `< 20` missing nodes triggers a targeted
`core.add()`; subsequent layoutstop reruns the invariant; second violation
accepts divergence with a logged warning (single retry budget per
envelope generation).

---

## Graph visual styling

**Status:** Active — implementation tracks [GitHub #608](https://github.com/chipi/podcast_scraper/issues/608).
**Author:** Design session (Marko + Claude), April 2026
**Repo:** `chipi/podcast_scraper`
**Target area:** `web/workbench/src/utils/cyGraphStylesheet.ts`,
`cyGraphLabelTier.ts`, `cyCoseLayoutOptions.ts`, `parsing.ts`, `GraphCanvas.vue`,
`GraphNeighborhoodMiniMap.vue`
**Related docs:** UXS-001, UXS-004, RFC-062
**Scope:** Pure visual — Cytoscape stylesheet (`cyGraphStylesheet.ts`), shared
zoom-tier helper (`cyGraphLabelTier.ts`), COSE options (`cyCoseLayoutOptions.ts`),
node data enrichment (`parsing.ts` → `toCytoElements`), and **main graph wiring**
in `GraphCanvas.vue` (selection dim classes, zoom listener → label tiers,
post-layout Topic degree heat). No store logic changes, no API changes. Works
independently of the Initial Load spec.

---

### 1. Objective

> **Status of the values below.** They record the April 2026 design. `cyGraphStylesheet.ts`
> is the source of truth where they differ; graph-v3 retuned the node sizes (3.1) and the
> edge strokes (3.2), and 3.1 and 3.2 below carry the current values.

Transform the graph from a data dump into an **intelligence communication
surface**. The visual system should do cognitive work for the user:

- Important things attract the eye before any label is read
- Relationship types are readable at a glance from edge style alone
- The topology communicates corpus structure — topic islands, hubs, recency
- Selection focuses the view — noise recedes, signal comes forward

Styles and parsed node fields live in the files above; **interaction wiring**
(selection dim, zoom tier sync, degree heat batch) lives in **`GraphCanvas.vue`**
(and the Episode subject rail **`GraphNeighborhoodMiniMap.vue`** preview reuses the same
stylesheet + tier sync). No store changes, no API changes.

---

### 2. Phasing

**Phase 1 — Stylesheet (low risk, ship together):**
Node size hierarchy, edge visual language, selection dim, zoom-responsive
labels, TopicCluster compound fill, drop shadows, default edge arrows.
Pure stylesheet values — no data dependencies.

**Phase 2 — Computed properties (medium effort, ship after Phase 1):**
Recency tint, grounding confidence tint, degree heat. These require
node property enrichment at parse time in `parsing.ts`.

Phase 1 is complete and coherent without Phase 2. Phase 2 is additive.

---

### 3. Phase 1 — Stylesheet Changes

#### 3.1 Node size by tier

Nodes render at different base sizes by type. Size = cognitive importance.

| Node type | Diameter (main profile) |
| --- | --- |
| Topic | 40px |
| Episode | 22px high, 1.35 times wider (a card) |
| Entity_person, Entity_organization, Entity_object, Podcast | 12px (degree-heat scaling still applies) |
| Any other type | 18px (`NODE_DIAMETER_MAIN_PX` default) |

The compact profile scales these by 14/18 with a 12px floor. Sizes live in
`NODE_DIAMETER_MAIN_PX` and `NODE_ASPECT_W` in `cyGraphStylesheet.ts`; the original design
(Insight 44, Topic 40, TopicCluster 48, Person 34, Entity 26, Quote 22, Speaker and Episode 18)
is recorded in the comment above that table.

**TopicCluster sizing (implementation):** The compound uses **padding** (WIP §3.6)
and Cytoscape’s **auto compound** geometry so member topics lay out correctly.
The stylesheet **does not** force a fixed **48×48** compound width/height; treat
**48px** as the **visual weight / tier** of the cluster chrome, not a hard layout
box.

---

#### 3.2 Edge visual language

Each edge type gets a distinct visual treatment so relationship type
is readable without hovering.

| Edge type | Width (main) | Style | Colour | Arrow |
| --- | --- | --- | --- | --- |
| `RELATED_TO` | 0.85px | solid | `kg` token | none |
| `MENTIONS` | 0.85px | dotted | `muted` token | none |
| `HAS_MEMBER` | 2px, 60% line opacity | solid | `kg` token | none |
| `HAS_EPISODE` | 2.75px | solid | `primary` token | target arrowhead |
| other / `(unknown)` | 1.5px | solid | `muted` token | target arrowhead |

This stylesheet declares no rule for the GI edge types of the original design (`HAS_INSIGHT`,
`ABOUT`, `SUPPORTED_BY`, `SPOKE_IN`); they take the generic `edge` rule above.

**Opacity:** the default edge opacity is low and depends on the theme (`graphEdgeOpacity`),
neighbourhood edges of the selection are at 90% (`graph-edge-neighbour`), and unrelated
edges while something is selected are at 20% (`graph-edge-dimmed`).

**Curve style:** `bezier` for all edges. Prevents straight lines
overlapping node borders at dense cluster areas.

**Label:** No edge labels by default. Edge type is communicated by
visual style alone. Labels add too much canvas noise.

Cytoscape edge data uses **`edgeType`** (see `toCytoElements` in `parsing.ts`), not `type`. The
rules are `edge[edgeType = "…"]` selectors, with an explicit `edge[edgeType = "(unknown)"]`
fallback and the invisible `_tc_cohesion` layout edge.

---

#### 3.3 Selection focus — dim unrelated nodes

When a node is selected, unrelated nodes dim to **40%** opacity (times
**`recencyWeight`** when present — see §4.1). The **focused** selected node uses
**full** opacity × recency. **1-hop neighbours** (closed neighbourhood) use
**~85%** opacity × recency in the shipped stylesheet so they read slightly softer
than the focus node while still clearly in the in-group (Cytoscape classes
**`graph-neighbour`**).

Stylesheet rules use prefixed classes on **nodes** and **edges** (e.g. `graph-dimmed`, `graph-focused`, `graph-neighbour`, `graph-edge-dimmed`, `graph-edge-neighbour`) so they do not clash with other graph classes.

Wire in `GraphCanvas.vue` on Cytoscape `select` and `unselect` events (and clear on destroy).

**Single selection:** The viewer interaction model keeps **at most one** selected
graph node at a time (tap handlers clear others). Dimming logic assumes a single
**focused** node; **multi-select is not supported** and is not a target scenario.

Example:

```typescript

cy.on('select', 'node', (e) => {
  const node = e.target
  cy.nodes().addClass('graph-dimmed')
  cy.edges().addClass('graph-edge-dimmed')
  node.addClass('graph-focused').removeClass('graph-dimmed')
  node.closedNeighborhood().nodes().addClass('graph-neighbour').removeClass('graph-dimmed')
  node.closedNeighborhood().edges().addClass('graph-edge-neighbour').removeClass('graph-edge-dimmed')
})

cy.on('unselect', 'node', () => {
  cy.nodes().removeClass('graph-dimmed graph-focused graph-neighbour')
  cy.edges().removeClass('graph-edge-dimmed graph-edge-neighbour')
})

```

Transition: add `transition-opacity 0.15s` via Cytoscape transition
properties so the dim/brighten animates smoothly rather than jumping.

`prefers-reduced-motion`: skip transition, apply instantly.

---

#### 3.4 Drop shadows on Tier 1 nodes

Insight and Topic nodes get a subtle drop shadow — creates visual lift,
signals importance tier without changing colour.

```typescript

{
  selector: 'node[type = "Insight"], node[type = "Topic"]',
  style: {
    'shadow-blur': 8,
    'shadow-color': 'var(--ps-border)',
    'shadow-offset-x': 0,
    'shadow-offset-y': 2,
    'shadow-opacity': 0.6
  }
}

```

Tier 2 (Person, TopicCluster): no shadow.
Tier 3/4: no shadow.

---

#### 3.5 Zoom-responsive labels

Labels show/hide and truncate based on zoom level. Cytoscape **does not** support CSS-like `min-zoom` / `max-zoom` stylesheet selectors on graph elements.

**Implementation:** `GraphCanvas.vue` listens to the existing Cytoscape **`zoom`** event and assigns **one** of three mutually exclusive class families on every node (e.g. `graph-label-tier-none`, `graph-label-tier-short`, `graph-label-tier-full`). The stylesheet maps those classes to `label` / `text-opacity` (Tier 1+2 use `data(shortLabel)` in the short tier; low tiers hide text in that band). Thresholds match the table below.

**shortLabel** — add to node data at parse time in `parsing.ts`:

```typescript

shortLabel: label.length > 18 ? label.slice(0, 16) + '…' : label

```

**Label placement (main graph today):** The default **merged graph** uses **side**
labels (`text-halign: center` + dynamic `text-margin-x` from
`cytoscapeSideLabelMarginXCallback` in `GraphCanvas.vue`’s `buildCyStyle`), with
canvas-tinted halo styles from the shared stylesheet. **`buildGiKgCyStylesheet`**
also supports **`above`** / **`below`** for previews or future shell changes.
UXS-004 describes the halo requirement; **default horizontal placement is side**,
not above the disc.

**Zoom thresholds:**

| Zoom | Visible labels |
| --- | --- |
| < 0.5 | None |
| 0.5 – 1.0 | Tier 1 + 2 only, shortLabel |
| > 1.0 | All visible nodes, full label |

**Label collision:** a label is drawn only when it does not overlap a label already kept.
`cyGraphLabelCollision.ts` keeps labels greedily by priority (the selected node, its neighbours,
then the most connected); every other overlapping label gets the `graph-label-collides` class,
which the stylesheet uses to hide its text while the node stays. Overlap is recomputed when the
label tier switches, when nodes move and when the selection moves the priorities.

---

#### 3.6 TopicCluster compound fill tint

TopicCluster compound parent nodes get a very faint `kg` token fill —
defines cluster territory visually without dominating the canvas.

```typescript

{
  selector: 'node[type = "TopicCluster"]',
  style: {
    'background-color': 'var(--ps-kg)',
    'background-opacity': 0.06,  // very faint — territory, not colour
    'border-style': 'dashed',
    'border-color': 'var(--ps-kg)',
    'border-opacity': 0.4,
    'border-width': 1.5,
    padding: '18px'  // space between compound border and member nodes
  }
}

```

The compound outline was already dashed per UXS-004 — this adds the
faint fill tint and ensures the padding gives member nodes breathing room.

---

#### 3.7 COSE layout parameter tuning

Tune spring strength by edge type to produce semantic gravity —
Insights cluster around their Topics, Quotes stay close to their
Insights without pulling them off cluster.

In the COSE layout config (wherever it is called in `GraphCanvas.vue`
or `graphExplorer.ts`):

```typescript

const layoutOptions = {
  name: 'cose',
  // ... existing options
  idealEdgeLength: (edge) => {
    switch (edge.data('edgeType')) {
      case 'HAS_INSIGHT': return 60   // episode–insight anchor (also scaled in compact profile)
      case 'ABOUT':      return 80   // Insight close to Topic
      case 'SUPPORTED_BY': return 40 // Quote close to Insight
      case 'RELATED_TO': return 120  // Topics spread laterally
      case 'SPOKE_IN':   return 100  // Person near but not inside cluster
      case 'MENTIONS':   return 150  // Loose reference — far
      default:           return 100
    }
  },
  edgeElasticity: (edge) => {
    switch (edge.data('edgeType')) {
      case 'HAS_INSIGHT': return 180
      case 'ABOUT':      return 200  // strong spring
      case 'SUPPORTED_BY': return 150
      case 'RELATED_TO': return 100
      case 'SPOKE_IN':   return 120
      case 'MENTIONS':   return 60   // weak spring
      default:           return 100
    }
  }
}

```

These values are starting points — visual testing with a real corpus
will require tuning. Add them to UXS-001 tunable parameters table as
Open (not Frozen) so they can be adjusted without a UXS revision.

---

### 4. Phase 2 — Computed Property Tints

Phase 2 adds three visual dimensions that require node property data
computed at parse time. All three enrich the node `data()` object in
`parsing.ts` and are then read by the stylesheet.

#### 4.1 Recency tint

Nodes from more recent episodes render at higher opacity / saturation.
Older nodes are muted.

**At parse time** (`parsing.ts`), for each node compute:

```typescript

// recencyWeight: 1.0 = current week, 0.4 = 90+ days ago
const daysSince = (Date.now() - episodePublishDate) / 86400000
const recencyWeight = Math.max(0.4, 1.0 - (daysSince / 90) * 0.6)
node.data.recencyWeight = recencyWeight

```

**`recencyWeight`** with **selection-dim** factors (dimmed **0.4×**,
neighbour **0.85×**, focused or default **1×**). Cytoscape could map
`opacity: 'data(recencyWeight)'` alone, but that would not compose with §3.3;
the callback keeps one combined path. **`recencyWeight`** is clamped to
**[0.4, 1]** in `parsing.ts` and the same floor is applied when reading the
field on the node so bad values cannot drop below **0.4**.

Requires episode publish date to be available on the parsed node when
recency should vary; otherwise **`recencyWeight` defaults to 1.0** (see
`toCytoElements`).

---

#### 4.2 Grounding confidence tint — Insight nodes only

Insight nodes with high confidence score render at full `gi` colour.
Low-confidence Insights are slightly desaturated — a subtle signal to
treat them with more scepticism.

**At parse time:**

```typescript

// confidence: 0.0 – 1.0 from GI schema
// confidenceOpacity: maps to fill opacity
const confidence = node.properties?.confidence ?? 0.7
node.data.confidenceOpacity = 0.5 + confidence * 0.5  // range 0.5–1.0

```

```typescript

{
  selector: 'node[type = "Insight"]',
  style: {
    'background-opacity': 'data(confidenceOpacity)'
  }
}

```

When `confidence` is absent (field not populated), default to 0.7
(neutral — neither highlights nor suppresses).

---

#### 4.3 Degree heat — Topic nodes only

High-degree Topic nodes (many connections) get a slightly warmer, more
prominent visual treatment — they are the conceptual hubs of the corpus.

**At parse time** (or post-layout in `GraphCanvas.vue` after Cytoscape
has computed degree):

```typescript

// degree: number of connected edges
// Cap normalisation to top of expected range
const degree = cy.$(`#${nodeId}`).degree()
const maxDegree = 30  // tune based on real corpus
const heat = Math.min(1.0, degree / maxDegree)
cy.$(`#${nodeId}`).data('degreeHeat', heat)

```

**In stylesheet — two effects:**

1. Border width scales with heat (thicker border = higher degree). **No border**
   when **`degreeHeat`** is **0** (isolates / low connectivity in the current slice)

   so the Topic disc does not compete with **RFC-076** rings or **search-hit**;
   ramp from **1px** once heat is positive, up to **4px** at heat **1.0** (main profile):

```typescript

{
  selector: 'node[type = "Topic"]',
  style: {
    'border-width': (ele) => {
      const h = Number(ele.data('degreeHeat'))
      if (!Number.isFinite(h) || h <= 0) return 0
      return 1 + Math.min(1, h) * 3  // 0, then ~1px–4px
    }
  }
}

```

```typescript

{
  selector: 'node[type = "Topic"][?highDegree]',  // flag set at parse
  style: {
    'shadow-blur': 12,
    'shadow-color': 'var(--ps-kg)',
    'shadow-opacity': 0.5
  }
}

```

**Note:** Degree heat should be computed post-layout (after COSE runs),
not at parse time, since degree depends on which nodes are currently
in the merged graph. Wire it as a post-layout callback in `GraphCanvas.vue`.

---

### 5. Minimap Contrast

The minimap (lower-left, already in spec) becomes more useful as a
navigation aid if Tier 1 nodes render with stronger contrast in it.

In Cytoscape, the minimap renders the same stylesheet. No special minimap
styling is needed — the size hierarchy and shadow from Phase 1 already
make Insight and Topic nodes more visible in the minimap than Quote and
Episode nodes. No additional change required.

However: ensure minimap background uses `canvas` token so the faint
TopicCluster compound fill tint is visible in the minimap. If minimap
background is currently hardcoded, update to use the CSS variable.

---

### 6. Light Mode Considerations

All colour values should use CSS variables (`var(--ps-gi)`,
`var(--ps-kg)`, etc.) not hardcoded hex. `cyGraphStylesheet.ts` likely
already does this for domain colours — extend to all new properties.

The `text-background-color` for label halos should use `var(--ps-canvas)`
so it matches the page background in both light and dark mode.

Drop shadow colours (`rgba(0,0,0,0.4)`) work in dark mode but are too
strong in light mode where the canvas is light. Use:

```typescript

'shadow-color': 'var(--ps-border)'  // adapts to mode

```

---

### 7. Files to Touch

#### Visual styling — modified files

```text

  — All Phase 1 and Phase 2 stylesheet rules

web/workbench/src/utils/parsing.ts
  — shortLabel generation (Phase 1)
  — recencyWeight computation (Phase 2)
  — confidenceOpacity computation (Phase 2)

web/workbench/src/components/graph/GraphCanvas.vue
  — Selection dim/focus class application (Phase 1)
  — Post-layout degree heat computation (Phase 2)
  — COSE layout parameter update (Phase 1)

web/workbench/e2e/E2E_SURFACE_MAP.md
  — Note stylesheet changes (no new selectors needed for E2E;
    these are visual-only changes)

```

#### Visual styling — UXS amendment (after implementation)

```text

  — Add section: ## Selection focus behaviour
  — Update Toolbar section: COSE parameters noted as tunable
  — Note Phase 1 vs Phase 2 separation

docs/uxs/UXS-001-workbench.md
  — Add COSE spring parameters to Tunable parameters table (Open)
  — Add degree heat maxDegree to Tunable parameters table (Open)
  — Add recency decay window (90 days) to Tunable parameters table (Open)

```

---

### 8. Tunable Parameters Added to UXS-001

| Parameter | Default | Status | Notes |
| --- | --- | --- | --- |
| COSE ABOUT edge ideal length | 80px | Open | Tune with real corpus |
| COSE MENTIONS edge ideal length | 150px | Open | Tune with real corpus |
| Recency decay window | 90 days | Open | Full decay at 90 days |
| Recency minimum opacity | 0.4 | Open | Floor for oldest nodes |
| Degree heat max degree | 30 | Open | Normalisation ceiling |
| Label zoom threshold (none) | 0.5 | Open | Below = no labels |
| Label zoom threshold (full) | 1.0 | Open | Above = full labels |
| Compound fill opacity | 0.06 | Open | Very faint territory tint |

---

### 9. Phase 1 Checkpoints

1. Node sizes reflect tier table — Insight largest, Episode/Speaker smallest
2. Edge types visually distinct — ABOUT solid green, SUPPORTED_BY dashed muted,
   SPOKE_IN solid primary with arrowhead

3. Selecting a node dims unrelated nodes to 40% opacity; deselect restores
4. Tier 1 nodes have drop shadow; Tier 3/4 have none
5. Labels hidden at zoom < 0.5; Tier 1/2 short labels at 0.5–1.0; full at > 1.0
6. TopicCluster compound has faint kg fill tint
7. Light mode and dark mode both render correctly (CSS variables used throughout)

### 10. Phase 2 Checkpoints

1. Nodes from last 7 days at full opacity; 90+ day nodes at ~40% opacity
2. High-confidence Insights brighter; low-confidence slightly faded
3. High-degree Topics have thicker border than low-degree Topics
4. Post-layout callback runs degree heat computation without blocking layout
5. All computed properties fall back gracefully when source data is absent

---

### 11. What This Does Not Change

- Node shapes (already defined in existing stylesheet)
- GI/KG domain token colours (gi green, kg purple) — Phase 2 modulates
  within these colours but does not change them

- RFC-076 ring colours (teal = expandable, blue = expanded) — untouched
- TopicCluster compound outline style (dashed) — Phase 1 only adds fill
- Any store logic, API calls, or component behaviour
- The gesture overlay (separate spec)
- The initial load strategy (separate spec)

---

## Graph gesture overlay

**Status:** Implemented in viewer (keep doc in sync with code when behaviour changes)
**Author:** Design session (Marko + Claude), April 2026
**Repo:** `chipi/podcast_scraper`
**Target area:** `web/workbench/src/components/graph/`
**Related docs:** UXS-004, RFC-062, [RFC-076](../rfc/RFC-076-progressive-graph-expansion.md) (progressive expand / ring semantics)
**Scope:** Small, self-contained. No store changes. No API changes.

---

### Gesture overlay — problem statement

The graph has 5 non-obvious interaction gestures and 2 visual ring cues that
were easy to miss when they only appeared in a single line of small
`text-[10px]` copy on the top toolbar row.

Current gesture inventory a user needs to learn:

| Gesture | Effect |
| --- | --- |
| Single click (`onetap`) | Open subject rail for that node |
| Shift + double-click | 1-hop ego / neighbourhood expand |
| Shift + drag | Box zoom / selection |
| Plain double-click (`dbltap`) | Progressive expand — loads more episodes for eligible nodes (RFC-076) |
| Second plain double-click | Collapse that expansion |

Visual ring cues:

| Ring colour | Meaning |
| --- | --- |
| Teal border | Node is expandable (has episodes not yet in the graph) |
| Blue border | Node is currently expanded |

None of these are in a menu. None are labelled on the canvas. A developer
opening the graph for the first time will discover single-click and pan/zoom
immediately (mouse-native), but will miss neighbourhood expand, progressive
expand, and the ring semantics entirely unless they open the gesture overlay
(**Gestures** / first visit) or stumble on them by accident.

---

### 2. Solution

A **one-time dismissible gesture hint overlay** that appears on the graph
canvas the **first time per browser profile** the user sees a non-empty merged
graph (`localStorage` key `ps_graph_hints_seen` unset). After dismissal the
flag is set to `1` and the overlay does not auto-open again until storage is
cleared.

**Dismiss:** **Got it**, **backdrop** click outside the card (`@click.self` on
the overlay root; the card uses `@click.stop` so row clicks do not dismiss),
and **Escape** as the primary keyboard dismiss. Do **not** treat arbitrary
keypresses or canvas drags as dismiss (avoids accidental close and matches
graph chrome focus rules).

This is a standard pattern for non-obvious gestures in data tools (Grafana
new feature hints, Figma gesture tours). The goal is one moment of teaching,
not persistent UI noise.

---

### 3. Visual Spec

#### Overlay container

- Positioned: `absolute inset-0` over the graph canvas host
  (`.graph-canvas` or the same `overflow-hidden` region as the minimap)

- Background: `rgba` of `canvas` token at ~60% opacity —
  `color-mix(in srgb, var(--ps-canvas) 65%, transparent)` when supported; otherwise a neutral translucent veil (see scoped `.ps-gesture-overlay-root` in `GraphGestureOverlay.vue`)

- `z-index`: above the main **`.graph-canvas`** Cytoscape layer; **below** the minimap panel when it is open (minimap uses a higher stacking order inside the same canvas host). **`GraphBottomBar`** is mounted **below** `canvasHost`, so it is **not** covered by this overlay and stays usable while the gesture card is open.

- The dimmed backdrop covers only the **canvas host** region (not the bottom bar)

#### Hint card

- Centered in the overlay: implementation uses **flex** on the backdrop root (`items-center justify-center`); an equivalent pattern is `absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2` on the card
- Background: `elevated` token
- Border: `border` token, `rounded-sm` (UXS-001 radius)
- Padding: `p-4`
- Max width: `20rem` — stays compact, doesn't fill the canvas
- Shadow: subtle (`shadow-md`) — gives depth against the dimmed canvas

#### Card header

- Text: **"Graph gestures"**
- Scale: `text-sm font-semibold`, `surface-foreground` token
- Bottom margin: `mb-3`

#### Gesture rows

Five rows, one per gesture. Layout per row:

```text

(icon | label | effect) — horizontal row; see bullets below

```

- Single click → `MousePointer` (the shipped overlay draws inline `<svg>` glyphs, not an icon package; these names describe the shapes)
- Shift + double-click → `Expand` or `Network`
- Shift + drag → `Scan` or `BoxSelect`
- Double-click → `PlusCircle`
- Second double-click → `MinusCircle`
- Gesture label: `text-xs font-mono`, `surface-foreground` — e.g.
  **"Click"**, **"Shift + dbl-click"**, **"Shift + drag"**,

  **"Dbl-click"**, **"Dbl-click again"**

- Effect: `text-xs`, `muted` — short plain language description

Row gap: `gap-2` between rows. Within each row: `gap-3` between icon,
label, effect.

Full row content:

| Icon | Label | Effect |
| --- | --- | --- |
| MousePointer | Click | Open node details |
| Network | Shift + dbl-click | Expand 1-hop neighbourhood |
| BoxSelect | Shift + drag | Box zoom / select |
| PlusCircle | Dbl-click | Load more episodes for this node |
| MinusCircle | Dbl-click again | Collapse loaded episodes |

#### Ring legend

Below the gesture rows, a `border-t border-border mt-3 pt-3` divider,
then two ring legend rows:

```json

[●  blue ring]   Episodes loaded (expanded)

```javascript

- Dot: 10px circle, inline SVG or `w-2.5 h-2.5 rounded-full`
- Teal dot: `border-2` with teal colour matching the RFC-076 expandable
  ring colour (use the same CSS var or value used in `cyGraphStylesheet.ts`)

- Blue dot: `border-2` with the RFC-076 expanded ring colour
- Label: `text-xs`, `muted`

#### Dismiss control

Below the ring legend, `mt-3`, right-aligned:

- Button: **"Got it"** — `text-xs`, `primary` token (same as secondary
  action buttons elsewhere), `px-3 py-1`

- `aria-label`: "Dismiss graph gesture hints"
- `data-testid`: `graph-gesture-overlay-dismiss`

#### Dismiss behaviour

Overlay dismisses when **any** of these occur:

- Clicking **Got it**
- Clicking the **dimmed backdrop** outside the hint card (not clicks on the Cytoscape layer beneath — the dimmed layer receives the hit; card uses `@click.stop`)
- **Escape**, when focus is inside the overlay (e.g. **Got it**) or when focus is on the graph canvas host after open — **not** when focus is in other graph chrome (**⚙** filters popover, **Since** in **`graph-status-line`**, **Fit** / zoom / **Gestures** in **`graph-bottom-bar`**, etc.) so users are not surprised while using those controls

On dismiss: set `localStorage` key `ps_graph_hints_seen` = `"1"`.
The overlay does not auto-open again; users can still use the optional **Gestures** reopen control (Section 7) without clearing `localStorage`.

#### Accessibility

- Overlay root: `data-testid="graph-gesture-overlay"` (backdrop + dim layer); **Escape** containment uses a Vue **template ref** on that root (`contains(activeElement)`), not `document.querySelector`, so multiple graph roots would not cross-wire (only one graph is mounted today).
- Card: `role="dialog"`, `aria-modal="true"`, `aria-labelledby` referencing the **Graph gestures** heading.
- Initial focus: **Got it** after open; on dismiss, return focus to the graph canvas host (`tabindex="-1"`) for keyboard continuity.

#### Motion

Default: hint card fades in over `150ms` (`opacity-0` → `opacity-100`),
using the same transition pattern as existing modals in the codebase.

`prefers-reduced-motion`: skip transition, render static at full opacity
immediately.

---

### 4. Trigger Condition

The overlay renders when **all** of:

1. `mainTab === 'graph'`
2. The merged graph has at least one node (i.e. artifacts are loaded —
   don't show on an empty canvas)

3. `localStorage.getItem('ps_graph_hints_seen')` is null or absent

Check `hasNodes` and `localStorage` when the component mounts and when
`hasNodes` becomes true. If the flag is already set, do not auto-open.

**Do not** auto-show the overlay:

- On the empty "no corpus loaded" graph state
- If the user has previously dismissed it (`localStorage` flag)
- When `hasNodes` is false (filters / ego view with zero nodes)

**Implementation note:** Today the **Graph** tab is the only place `GraphCanvas`
mounts: [`App.vue`](https://github.com/chipi/podcast_scraper/blob/main/web/workbench/src/App.vue) uses `v-if="mainTab === 'graph'"` on
`GraphTabPanel`. The overlay therefore does **not** require a separate `mainTab`
prop unless the shell is refactored (then pass e.g. `isGraphTabActive`).

---

### 5. Component

#### `GraphGestureOverlay.vue`

Location: `web/workbench/src/components/graph/GraphGestureOverlay.vue`

Props: **`{ hasNodes: boolean }` only** (parent derives from Cytoscape node count).

The parent (`GraphCanvas.vue`) passes `hasNodes` and renders the overlay inside
the same `absolute inset-0` canvas host as `.graph-canvas`, with `z-index`
above the Cytoscape layer and **below** minimap / zoom / layout chrome so
chrome stays usable while the overlay is open.

**Sketch (behaviour, not copy-paste):** `visible` is driven by `hasNodes` +
`localStorage` for auto-open; optional `reopen()` sets `visible` without
clearing `localStorage`. **Escape** handling is registered while `visible` and
ignores keys when focus is outside the overlay (graph chrome).

---

### 6. Toolbar Hint Row — Simplify

With the overlay handling first-time teaching, the **top** toolbar row is
**Types** + **⚙** filters only — **no** persistent Shift / double-click / ring prose.
The **search highlight** chip (when applicable) sits in the **bottom bar** centre
zone under the canvas (see [UXS-004 Graph chrome](../uxs/UXS-004-graph-exploration.md#graph-chrome-toolbar-bottom-bar-filters-popover)).

**Requirement:** Removing that copy **requires** either the optional **Gestures**
reopen control (Section 7) or another minimal affordance (e.g. a **?** entry) so
users who dismissed quickly are not stranded without a path back to the legend.

---

### 7. Re-open Affordance

A **Gestures** (or **?** / **Shortcuts**) control in the **graph bottom bar** right zone
(next to **Export PNG**) calls `reopen()` so the card appears **without** clearing
`localStorage` (manual open only; auto-open still suppressed after dismiss).

`data-testid`: `graph-gesture-overlay-reopen`

**Shipped:** persistent gesture prose was removed from the top toolbar in favour of this overlay; **Gestures** in **`GraphBottomBar`** is the deliberate path back to the overlay copy.

---

### 8. Files to Touch

#### Gesture overlay — new component

```text

web/workbench/src/components/graph/GraphGestureOverlay.vue
  — dismissible overlay card + backdrop; localStorage gate

```

#### Gesture overlay — modified files

```text

  — render GraphGestureOverlay inside canvas host
  — pass hasNodes computed prop
  — shorten or remove toolbar gesture hint text

web/workbench/e2e/E2E_SURFACE_MAP.md
  — add: graph-gesture-overlay, graph-gesture-overlay-dismiss

```

#### Gesture overlay — UXS amendment (after implementation)

```text

  — Add section: ## Gesture discovery overlay
  — Reference visual spec (tokens, layout, dismiss behaviour)
  — Note toolbar hint row simplified after overlay ships
  — Note localStorage key: ps_graph_hints_seen

```

---

### 9. E2E Contract

Add to `e2e/E2E_SURFACE_MAP.md`:

| Surface | Selector / role | Notes |
| --- | --- | --- |
| Overlay container | `data-testid="graph-gesture-overlay"` | Present on first graph load with nodes |
| Dismiss button | `data-testid="graph-gesture-overlay-dismiss"` | Click to dismiss |
| Re-open button | `data-testid="graph-gesture-overlay-reopen"` | **Gestures** in **`graph-bottom-bar`** right zone |

Playwright spec: `e2e/graph-gesture-overlay.spec.ts`

**Vitest:** The package ships `@vue/test-utils` (the Workbench has component mount tests); the overlay itself is locked by the Playwright spec above.

Test cases:

1. Overlay appears when graph has nodes and localStorage flag is absent
2. Clicking "Got it" dismisses overlay and sets localStorage flag
3. Clicking outside card (on overlay backdrop) dismisses overlay
4. Reloading page after dismiss: overlay does not appear
5. `hasNodes = false`: overlay does not appear
6. Re-open button (if implemented): overlay re-appears without resetting flag

---

### 10. What This Does Not Change

- Cytoscape event handlers (`onetap`, `dbltap`, Shift+drag) — unchanged
- RFC-076 progressive expansion logic — unchanged
- Ring colours in `cyGraphStylesheet.ts` — unchanged (overlay references
  them visually but does not own them)

- All other graph chrome (**layout cycle**, **Re-layout**, minimap, **Fit** / zoom / **Export PNG** in **`GraphBottomBar`**) — behaviour unchanged by the overlay; only placement vs legacy canvas overlays evolved per [UXS-004 Graph chrome](../uxs/UXS-004-graph-exploration.md#graph-chrome-toolbar-bottom-bar-filters-popover)
- Token system — uses only existing UXS-001 tokens

---

*Small, contained, one new component. No store changes, no API changes.*

---

## Graph focus entry points

**Status:** Draft checklist (not indexed). **Purpose:** align every “open graph” path with
the **node id vocabulary** the merged GI+KG graph actually contains (`topic:`, `tc:`,
episode-scoped slugs from bullets, search hit payloads, and so on).

### Phase 0 (landed)

- **Code:** `web/workbench/src/utils/cilGraphFocus.ts` — maps a `CilDigestTopicPill` +
  optional episode id to `graphNavigation.requestFocusNode` (primary `topic:…`, fallback

  episode id, optional `pendingFocusCameraIncludeRawIds` for `tc:…` when
  `in_topic_cluster`). No component in this repository imports it today: the CIL pills it maps
  (Digest Recent, Episode rail) are not drawn by the public Workbench (UXS-002, UXS-003 "Where it
  renders"), so it serves a downstream build and is covered here by its Vitest only.

- **Tests:** `web/workbench/src/utils/cilGraphFocus.test.ts` (Vitest).
- **Docs:** [Development Guide — Viewer v2](../guides/DEVELOPMENT_GUIDE.md#viewer-v2-rfc-062-489) (bullet **CIL pill to graph focus**).

### Why this exists

CIL and digest **pills** unify **bridge** `topic:` ids with optional topic clustering
**`topic_cluster_compound_id`** (`tc:`) for the viewer. The **cytoscape merge** still
contains multiple node kinds and legacy paths (for example bullet-derived `topic:` slugs).
If one surface passes the wrong token, focus can **miss** or highlight the **wrong**
vertex.

### Entry surfaces to audit (code + UX)

| Surface | Expected focus token | Notes |
| ------- | -------------------- | ----- |
| Digest — CIL chip | `topic:` and/or `tc:` from `cil_digest_topics[]` | Server builds pills; viewer must pass compound when clustered. |
| Digest — topic band / semantic row | Topic ids from digest API / GI load | Confirm same slugging as graph merge. |
| Search — open graph from hit | Hit payload (`lifted`, anchors, …) | Transcript lift vs insight hit may differ. |
| Library — Episode subject rail CIL | Same pill shape as digest detail | List rows omit pills; detail only. |
| Graph — double-tap cross-episode expand | Canonical `node_id` from graph | Already constrained by API contract. |
| Person / org drill-ins | `person:` / `org:` | Less overlap with `tc:` but worth a row in tests. |

### Suggested engineering outcomes (remaining)

1. ~~**Single helper** for CIL pills~~ — done for Digest + Episode subject rail (`cilGraphFocus.ts`).
2. **Extend or reuse** the same contract for other surfaces (topic band rows,
   any future chips) and optionally dedupe SearchPanel’s `topic_cluster` camera logic

   with one shared primitive.

3. **Playwright** assertion that clustered pill passes camera ids (optional: spy on
   store or assert zoom behaviour if stable in mocks).

4. Optional: **dev-only log** when focus id is not found in the loaded graph (guarded so
   production builds stay quiet).

### When to fold into UXS / Development Guide

After the audit is done, fold the **contract** into [UXS-002 Corpus
Digest](../uxs/UXS-002-corpus-digest.md) / [UXS-003 Corpus
Library](../uxs/UXS-003-corpus-library.md) or the [Development Guide — GI / KG
viewer](../guides/DEVELOPMENT_GUIDE.md#gi-kg-browser-viewer-local-prototype) and trim or archive this checklist.

### GitHub issue

Tracked as [#596](https://github.com/chipi/podcast_scraper/issues/596) (body template:
[GitHub #596](https://github.com/chipi/podcast_scraper/issues/596).

---

## Cluster sibling episode load

**Status:** Design (2026-04-15 iteration); not implemented in this build. There is no sibling cap,
no `VITE_CLUSTER_SIBLING_EPISODE_CAP` and no sibling-merge hook in `web/workbench/src`; what the
graph does today is the episode cap and RFC-076 expand described above.

### Clarification: what the earlier “options” meant

Three **UX triggers** were discussed for *when* to pull in sibling episodes:

| Option | Meaning |
| -------- | -------- |
| **Button only** | User clicks something like “Load cluster episodes” — nothing loads until they click. |
| **Confirm dialog** | After an action (e.g. Open in graph), a dialog: “Load N related episodes?” Confirm / Skip. |
| **Automatic with cap** | The app loads siblings without a click, but **stops at a max** (configurable) to avoid huge graphs by accident. |

### Locked product decision (this iteration)

- **Trigger:** **Automatic** in the **Graph** context: when a topic cluster applies and sibling episodes can be resolved from `topic_clusters.json` + catalog, **merge-load** their GI/KG into the current graph selection up to a **safety cap** on how many **additional** episodes to pull in per merge (not unlimited).
- **Safety cap — default `10`:** At most **10** sibling episodes are auto-merged per trigger (tune if needed). Order: deterministic (e.g. catalog sort, or first N unresolved ids) — specify in implementation.
- **Configurable “higher up”:** One central place (e.g. viewer env `VITE_CLUSTER_SIBLING_EPISODE_CAP` with default `10`, or a small `viewerConfig` module parsing `import.meta.env` + fallback). Document in Development / Polyglot guide and optional `.env.example` so power users can raise the cap without code edits.
- **Scope:** **Graph / artifact merge path only** for this phase — **no change** to **Digest** or **Library** behavior or layout (those can be revisited later if we want parity).
- **Transparency:** When the cap trims the list, UI should indicate **“Loaded N of M sibling episodes (cap …)”** or similar so users know more exist.

### Is this a “good” answer?

**Yes.** A default of **10** with a **configurable** cap balances full-cluster intent with predictable worst-case load. Raising the default or env for large monitors / fast machines is easy; lowering it protects laptops and huge corpora.

**Technically:** Same as before: `episode_ids` on cluster members, resolve to paths via catalog, **merge** (not replace) `selectedRelPaths`, apply cap when selecting which sibling paths to add.

### Implementation reminders (unchanged core)

1. **Store:** `appendRelativeArtifacts` / merge load — required because `loadRelativeArtifacts` replaces selection today.
2. **Server:** Resolve `episode_id` → `gi_relative_path` / `kg_relative_path` (catalog scan per request or cached server-side if needed).
3. **Client:** From loaded graph + `topicClustersDoc`, compute sibling episode ids → resolve → merge load **automatically** when conditions are met (define exact hook: e.g. after `loadSelected` completes when graph tab is active and cluster has unresolved siblings).

### Out of scope (this phase)

- Digest “open in graph” flows.
- Library episode detail / “Open in graph” (unless we later wire the same merge helper behind an explicit control there).

### See also (layout, separate workstream)

- [Graph layout — topic cluster](#graph-layout-topic-cluster) — tighter TopicCluster footprint on the main canvas; neighborhood minimap COSE/2D instead of breadthfirst strip.

---

## Graph layout — topic cluster {#graph-layout-topic-cluster}

**Status:** Cluster compaction tuned (section 1); minimap layout done (section 2).

### 1. Cluster compound: tighter footprint on main canvas

**Issue:** For 2-4 nodes inside a TopicCluster compound, the cluster **region** can occupy
~1/4 of the whole graph -- poor use of space and visually "pulls" the layout.

**Applied (cyCoseLayoutOptions.ts):**

- **Member repulsion** cut from 420k to 180k (main) / 57k to 24k (compact) -- members
  pack much tighter inside the compound.

- **Intra-cluster ideal edge length** cut from 58 to 36 (main) / 32 to 20 (compact) --
  connected members pull closer together.

- **Gravity** raised from 0.15 to 0.18 (main) / 0.28 to 0.32 (compact); the main value was
  later lowered to 0.12 (graph-v3) so communities drift apart, the compact value stays 0.32.

- **Nesting factor** raised from 1.38 to 1.52 -- cross-boundary edges stretch more
  relative to intra-cluster edges, keeping external nodes further from the compound

  while internals stay tight.

- **numIter** set to 2500 (explicit) -- ensures the simulation converges well at the
  lower repulsion values.

- Compound **padding** (`cyGraphStylesheet.ts`) left at 6px/3px -- already minimal; the
  issue was internal node spacing, not border padding.

### 2. Neighborhood minimap: 2D layout, not a single line

**Issue (resolved):** The local-neighborhood preview used `breadthfirst`, which laid
nodes out as a single row or column.

**Now:** `GraphNeighborhoodMiniMap.vue` runs the compact COSE option set
(`giKgCoseLayoutOptionsCompact()` from `cyCoseLayoutOptions.ts`) and keeps `fit` after
layout. The main canvas's force-directed layout is `fcose` (#967); the cycle order is in
`GRAPH_LAYOUT_CYCLE_ORDER` in `graphExplorer.ts`.

### 3. Relationship to sibling-episode auto-load

Independent: episode merge behavior is data volume; this doc is **pure
layout/presentation**.

### 4. UX / E2E

When layout strings or minimap behavior become E2E-visible, update
`web/workbench/e2e/E2E_SURFACE_MAP.md` and the relevant UXS if the visual contract
changes.

---
