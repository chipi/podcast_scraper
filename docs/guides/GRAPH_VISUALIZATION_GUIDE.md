# Graph Visualization Guide

**Audience:** anyone touching the graph — future you, another engineer, or an
LLM agent asked to change how a node looks / behaves. **Scope:** the
gi-kg-viewer graph canvas: every visual rule, every business decision, every
"why is it like that", plus the code path where it lives. Read the section
you care about; the sections are deliberately independent.

**What this is not:**

- Not the *design proposal*. The proposals live in `docs/rfc/RFC-080-graph-visualization-extensions.md`
  and older tier notes in `docs/wip/graph-v3/SUMMARY.md`.
- Not the *implementation spec*. That's `docs/architecture/VIEWER_GRAPH_SPEC.md` —
  handoff orchestrator, camera modes, matrix contracts, gesture overlay.
- Not the *knowledge-graph model*. `docs/guides/KNOWLEDGE_GRAPH_GUIDE.md`
  covers node/edge kinds at the data layer.

This guide is the "reference book" — what each dial does and why.

---

## Table of contents

1. [Mental model](#mental-model)
2. [Node encoding — every rule that touches a node](#node-encoding)
3. [Edge encoding — every rule that touches an edge](#edge-encoding)
4. [Layout](#layout)
5. [Interaction](#interaction)
6. [Filters](#filters)
7. [Lenses (the toggle catalog)](#lenses)
8. [Load modes (Everything vs. Top-down)](#load-modes)
9. [Legend (theme cluster hierarchy)](#legend)
10. [State stores](#state-stores)
11. [Enricher artifacts (data contract)](#enricher-artifacts)
12. [Testing model](#testing-model)
13. [Cross-references](#cross-references)

---

## Mental model

The graph is a **composable rendering** of a merged GI+KG artifact:

- **Base layer** — nodes and edges from the artifact, colored + shaped +
  sized by type and by structural properties (degree, confidence, recency).
- **Enricher overlays** — additional class-based visual signals derived from
  corpus-scope enrichers (theme clusters, temporal velocity, credibility,
  consensus, co-appearance). Each overlay is gated on the presence of its
  enricher artifact — if the corpus never ran that enricher, the overlay
  and its lens toggle are hidden.
- **Interaction layer** — selection, hover, focus, search, camera. These
  never mutate the artifact; they toggle CSS classes on the Cytoscape core.

The three layers **compose freely**. A Topic can carry, at the same time:
type color + degree size + theme-region tint + velocity halo + bridge ring +
hover dim. Cytoscape resolves by stylesheet source order; interaction states
(`.search-hit`, `:selected`, hover) win over lens overlays.

Every visual rule below documents its **gate** (when it fires), its **why**
(what problem it addresses), and its **code entry** (file:line).

---

## Node encoding

### 1. Type-based shape

**Rule:** each node's shape signals its type at a glance, before you can read
its label.

| Type | Shape | Rationale |
| ------ | ------- | ----------- |
| Podcast | hexagon | Top-of-hierarchy container. Hex reads as "structural aggregator". |
| Episode | round-rectangle | Card-shape carries a title well; renders 30×22 with `NODE_ASPECT_W=1.35`. |
| Entity_organization | round-rectangle (35×26) | Same card treatment as Episode — same reading habit. |
| Insight | diamond | Evidence unit; diamond signals "claim". |
| Quote | round-tag | Bottom of the evidence chain; tag shape recalls a caption. |
| Speaker | round-diamond | Rare; distinct from Insight so the two don't merge visually. |
| Person / Entity_person / Topic / TopicCluster | ellipse / roundrectangle-compound | The "people and things" fallback; keeps the graph readable when types are mixed at overview zoom. |

**Why:** shape variance is the highest-signal channel at low zoom, where
labels are illegible. Colour alone would collapse on colour-blind viewers.

**Code:** `web/gi-kg-viewer/src/utils/cyGraphStylesheet.ts` — `shapeByType`
map inside `buildGiKgCyStylesheet`. Compact-profile scaling handled in the
same function.

**Gate:** always on. No user override; shape *is* type.

**Live-verify status:** Speaker + Quote emit 0 in prod-v2; code-verified
only. Speaker/Quote shape live-verify pending on a corpus that emits them
(tracked in `docs/wip/graph-v3/SUMMARY.md` §Not done).

---

### 2. Type-based colour (brand palette)

**Rule:** each node type has a distinct hex from a coordinated brand palette.
The **same palette** feeds the graph, the Types chip, the CIL legend, the
topic pill on the reader — one source of truth.

**Why one palette:** UXS-014 says surface visuals must not drift. If Person is
amber `#c49a28` in the graph, it must be amber in the Types chip and in the
CIL topic list. Hand-editing hexes across three files caused a collision
before (`Entity_person` and `TopicCluster` were both `#9775fa`); the fix
routed everything through `utils/colors.ts::graphNodeTypeStyles`.

**Code:** `web/gi-kg-viewer/src/utils/colors.ts` — `graphNodeTypeStyles`
map. Collision guard test: `utils/colors.test.ts` — asserts every type has
a unique background AND that the Entity/Entity_person alias whitelist is
respected.

**Gate:** always on. No user override.

---

### 3. Size by connectivity (degree heat)

**Rule:** node width and height scale with graph degree — hubs render larger
than leaves. Range `0.7×` (isolate) to `1.5×` (top-connected). Aspect ratio
preserved: card-shaped types (Episode, Entity_organization) scale W and H
against per-type bases so the card shape doesn't distort.

**Why:** the analyst's first question is "what's central here?" Answer at a
glance instead of counting incident edges.

**Extends to:** Topic (always), Person, Entity_organization, Episode. Previously
Topic-only; expanded in Tier 3-J.

**Gate:** **default on** (Tier C flipped the default). Users can disable via
the "Size by connectivity" row in the Lenses chip.

**Store flag:** `useGraphLensesStore.nodeSizeByDegree` — persisted to
localStorage, mirrored to server via USERPREFS-1.

**Code:**

- `web/gi-kg-viewer/src/utils/cyGraphStylesheet.ts` — `mapData(degreeHeat, 0.7, 1.5)` selector, scoped to `[type][degreeHeat]` so per-element warnings don't fire on nodes without the field.
- `web/gi-kg-viewer/src/components/graph/GraphCanvas.vue` — `applyTopicDegreeHeat` populates `data.degreeHeat` post-layout.

**Caveat:** `topic-heat-high` class (bright text-outline for the top-N
connected Topics) is still Topic-only — existing selector contract kept
to avoid Person/Episode text-outline noise.

---

### 4. Zoom-gated tier reshape (plumbing dots)

**Rule:** below zoom `GRAPH_NODE_ZOOM_INSIGHT_MIN = 0.9`, "plumbing" types
render as compact dots without labels. Above `0.9`, they render at full
size with labels.

**Types affected:** Insight, Quote, Speaker, Person cameo (Persons with
degree = 1).

**Why:** Tier 6-1. On prod-v2 (833 nodes), the initial paint was
"drowning in evidence dots" — Insights and Quotes outnumbered Topics 10:1
and every one carried a label. Below overview zoom the labels are illegible
anyway; hiding them until the analyst zooms in is honest, not lossy.

**Gate:** always on. Not user-toggleable — this is a decluttering primitive,
not a lens.

**Code:** `web/gi-kg-viewer/src/utils/cyGraphLabelTier.ts` —
`syncGraphNodeVisibilityTierClasses`. Fires on every layout pass and zoom
change.

---

### 5. Zoom-gated visibility (hidden below zoom)

**Rule:** Insight and Quote nodes carry a `graph-node-hidden-below-zoom`
class when the current zoom is below `0.9`, hiding them entirely (not just
their labels). Re-appear on zoom-in without a re-layout.

**Why:** Tier 6-2. Even as dots, 100+ Insight nodes at overview zoom compete
for pixels with the actual hubs. Hiding them until they can be read leaves
the topology visible. This is the primary declutter action of tier 6.

**Gate:** always on. Not user-toggleable at the graph level. The **default-
hide Quotes filter** (§6) hides Quotes at all zoom levels by default; users
who re-enable Quotes via the Types chip still get zoom-gated Quotes.

**Code:** `web/gi-kg-viewer/src/utils/cyGraphLabelTier.ts` —
`syncGraphNodeVisibilityTierClasses`. Test coverage:
`utils/cyGraphLabelTier.test.ts`.

**Interaction with search:** search reveals hidden nodes even below zoom
(§Interaction).

---

### 6. Bridge ring (betweenness)

**Rule:** nodes with graph-level betweenness ≥ `0.05` on Topic / Podcast /
Person / Entity_organization types render with a **rose dashed border**
(`.graph-bridge` class). Episode and Insight are excluded — they're always
structural connectors by the shape of the graph, so a "bridge" ring on
every Episode carries no signal.

**Why:** hub-discovery at overview zoom. On prod-v2, 44 nodes qualify;
they're the ones that connect two theme regions together. The dashed
rose ring reads distinct from the solid coloured borders used by
credibility / velocity so lenses compose without conflict.

**Perf:** betweenness costs ~200ms on 833 nodes. Computed once per layout,
cached until the artifact changes.

**Gate:** `useGraphLensesStore.bridgeRing` — **default on**. User can
disable via the "Bridge nodes" row in Lenses.

**Code:**

- `web/gi-kg-viewer/src/components/graph/GraphCanvas.vue` — `applyBridgeNodeClass`.
- Stylesheet class `.graph-bridge` in `utils/cyGraphStylesheet.ts`.
- Hover tooltip surfaces the two theme regions the bridge connects
  (Tier 6-4; hover handler in `GraphCanvas.vue`).

---

### 7. Velocity halo (temporal_velocity enricher)

**Rule:** Topic and Person nodes get a coloured border reflecting their
6-month trend:

- **Green** — velocity ↑ (accelerating)
- **Red** — velocity ↓ (decelerating)
- **Amber** — velocity → (flat, within threshold band)

**Enricher:** `temporal_velocity` — computes `velocity_last_over_6mo`
per Topic and per Person. Thresholds shared with the reader's arc-tint
colour via `web/gi-kg-viewer/src/utils/trend.ts` — the graph and the arcs
tell the same story.

**Gate:** `useGraphLensesStore.velocityHalo`. Row hidden entirely when the
`temporal_velocity` corpus artifact is absent (probed on chip mount +
corpus-path change; the row is enricher-gated, not disabled — no dead
controls per operator direction).

**Code:**

- Lens overlay: `web/gi-kg-viewer/src/utils/cyGraphLensOverlays.ts` —
  `applyVelocityHaloClasses` / `clearVelocityHaloClasses`.
- Availability probe: `web/gi-kg-viewer/src/composables/useEnrichmentEnvelopeCache.ts`
  (Map-based cache; probed once per corpus path).

**Live count (prod-v2):** 164 up, 129 down.

---

### 8. Person credibility border (grounding_rate enricher)

**Rule:** Person nodes get a border keyed on their grounded-insight rate:

- Green solid, ≥ 0.7 (well-grounded)
- Amber, 0.4–0.7
- Red dashed, < 0.4

**Enricher:** `grounding_rate` — counts a Person's Insights that carry
supporting Quote evidence vs total, per corpus.

**Gate:** `useGraphLensesStore.personCredibility`. Enricher-gated.

**Composes with:** velocity halo. Velocity is a **halo** (outer glow class);
credibility is a **border colour**. Both are on Person; they never conflict
because they use different visual channels.

**Code:** `web/gi-kg-viewer/src/utils/cyGraphLensOverlays.ts` —
`applyPersonCredibilityClasses`.

**Live count (prod-v2):** 51 Persons at high credibility.

---

### 9. Person community underlay (guest_coappearance enricher)

**Rule:** Persons who repeatedly co-appear in the same episodes get grouped
into an **underlay tint** — soft coloured circle behind the Person node,
one colour per community.

**Enricher:** `guest_coappearance` v1.1.0+ — runs union-find over
Person↔Person co-appearance edges (≥ 2 shared episodes).

**Why underlay not border:** Person nodes already carry credibility on the
border and velocity on the halo. Adding a third border colour for community
membership would collide. Underlay reads as "these belong together" without
adding a fourth border layer.

**Gate:** `useGraphLensesStore.personCommunities`. Same availability as
`coGuestEdges` (both derive from the same enricher).

**Code:**

- `web/gi-kg-viewer/src/utils/cyGraphLensOverlays.ts` — `applyPersonCommunityClasses`.
- Enricher: `src/podcast_scraper/enrichment/enrichers/guest_coappearance.py`.

---

### 10. Theme region tint (topic_theme_clusters enricher)

**Rule:** every node gets an **underlay tint** matching its theme cluster
(`themeClusterId`). Topics are seeded directly from
`topic_theme_clusters.json`; Insight / Person / Org / Podcast inherit via
one-hop / two-hop propagation.

**Propagation model:**

- **Topic** ← direct membership (`member.topic_id` in the cluster's members).
- **Episode** ← via `member.episode_ids` from the artifact.
- **Insight** ← one hop from its parent Topic (`HAS_INSIGHT` edge).
- **Person / Entity_person** ← two hops via `SPOKE_IN → Episode ← ABOUT →
  Topic`.
- **Podcast** ← two hops via `Podcast ← HAS_EPISODE → Episode → ABOUT →
  Topic`.

**First-cluster-wins tagging** — a node reachable from multiple clusters
gets the first cluster's tint. Consistent across sessions because the seed
order is deterministic.

**Palette:** 8 pastel hues around HSL, sat 45% / light 65%, opacity 0.14.
Same `graph_compound_parent_id` maps to the same colour across sessions
via djb2 hash. Palette lives in
`web/gi-kg-viewer/src/utils/themeRegionPalette.ts` — imported by both the
legend and the stylesheet so hand-edits can't drift.

**Gate:** `useGraphLensesStore.themeClusterRegions` — **default off**.
Row hidden when `artifacts.themeClustersDoc === null` (enricher artifact
absent).

**Legacy compatibility:** the flag was renamed `communityColours →
themeClusterRegions` in Tier 4-V. `stores/graphLenses.ts` reads the legacy
key and migrates on first load — users who opted into the earlier flag
retain their opt-in.

**Code:**

- `web/gi-kg-viewer/src/components/graph/GraphCanvas.vue` — `applyThemeRegionClasses`.
- Propagation: `web/gi-kg-viewer/src/utils/topicClustersOverlay.ts`.
- Enricher: `src/podcast_scraper/enrichment/enrichers/topic_theme_clusters.py`.
- Legend: `web/gi-kg-viewer/src/components/graph/ThemeClusterLegend.vue`
  (see §Legend).

**Live count (prod-v2):** 161 nodes tagged across 4 hash-buckets.

---

## Edge encoding

### 1. Type-based colour + width tiers

Edges are ranked by structural importance. The tier decides both colour and
width. Wider + brighter edges read as "trust this line more".

| Tier | Types | Width | Rationale |
| ------ | ------- | ------- | ----------- |
| **Structural** | HAS_INSIGHT, SPOKE_IN, HAS_EPISODE | 2.75 | The graph's spine — hierarchy edges. Widest so the skeleton is legible even at overview zoom. |
| **Membership** | HAS_MEMBER | 2.0 | Compound → member. Slightly thinner than the spine so members don't overpower the container's boundary. |
| **Evidence** | SUPPORTED_BY | 0.75–0.85 | Insight ↔ Quote. Thin because they're numerous and their existence — not their weight — is what matters. |
| **Discovery** | RELATED_TO, MENTIONS | 0.75–0.85 | Discovered co-references. Thin so they don't compete with the spine. |
| **Descriptive** | ABOUT, SPOKEN_BY | mapData(confidence) | Width scales with confidence 0.25–1.0 (§Confidence). Confidence-carrying tier; not tier-flat. |
| **Aggregate** | ABOUT_AGG, SPOKE_IN_AGG | mapData(weight) | Roll-ups over Insight-scoped edges; width = weight (§Aggregated edges lens). |

**Why tiers, not one width:** the graph is a bipartite mesh. A single width
would either drown the spine (if narrow) or overpower Insights (if wide).
Tiers keep both readable.

**Compact profile:** widths scale proportionally under the compact-profile
render path. Aggregate `mapData(weight)` untouched (already scaled).

**Code:** `web/gi-kg-viewer/src/utils/cyGraphStylesheet.ts` — edge-type
selectors + `edgeWidthByType`. Tier assertions:
`utils/cyGraphStylesheet.test.ts`.

---

### 2. Confidence opacity (ABOUT, SPOKEN_BY)

**Rule:** for confidence-carrying edges, opacity maps to `data.confidence`
in the range `0.25` (min alpha) to `1.0` (opaque). Insights without a
confidence value fall through to the tier default.

**Why:** an ABOUT edge with 0.3 confidence and an ABOUT edge with 0.95
confidence tell different stories. Width could encode this too, but width
is already used by the tier system.

**Gate:** always on.

**Code:** stylesheet selectors in `utils/cyGraphStylesheet.ts`.

---

### 3. Temporal recency fade

**Rule:** edges from recent episodes render at opacity `1.0`; edges from
older episodes fade toward opacity `0.4` (floor) via `data.recencyWeight`.

**Why:** the recent conversation is what the analyst usually cares about.
Older mentions should be visible (they're the base rate) but shouldn't
compete with the new activity.

**Gate:** always on. Floor at 0.4 so nothing disappears.

**Code:** `mapData(recencyWeight)` selector in
`web/gi-kg-viewer/src/utils/cyGraphStylesheet.ts`. Recency computed at
artifact-merge time.

---

### 4. Default edge opacity (theme-aware)

**Rule:** base edge opacity is `0.3` on dark theme, `0.5` on light. Neighbor
edges (hovered/selected node's incident edges) are lifted to `0.9`. Dimmed
edges (not incident on selection) drop to `0.2`.

**Why the dark/light split:** on `#0a0d10` (dark canvas), 0.5 washes out
the base; on white, 0.3 disappears. Reading the current theme via
`data-theme` attribute lets the same graph render honestly on both.

**Code:** `utils/cyGraphStylesheet.ts` — `isLightThemeActive()` reads
`data-theme`, falls back to `matchMedia('(prefers-color-scheme)')`.

---

### 5. Aggregated edges (roll-up lens)

**Rule:** when the "Aggregated edges" lens is on, add Episode↔Topic
(`ABOUT_AGG`) and Episode↔Person (`SPOKE_IN_AGG`) edges rolling per-Insight
edges up. Width scales with the roll-up weight.

**Why:** at overview zoom, seeing 100 individual Insight→Topic ABOUT edges
tells you nothing. One thick Episode→Topic ABOUT_AGG edge says "this
episode is about this topic" at a glance.

**Gate:** `useGraphLensesStore.aggregatedEdges`. **Data-gated** (not
enricher-gated): the roll-up happens at runtime in `toCytoElements`. The
lens row is hidden when the artifact has no source `ABOUT` / `SPOKEN_BY`
edges to roll up — check runs against `artifacts.displayArtifact?.data?.edges`.

**Code:**

- Roll-up: `web/gi-kg-viewer/src/utils/cyGraphElements.ts` — `toCytoElements`.
- Chip data-gate: `web/gi-kg-viewer/src/components/graph/chips/GraphLensesChip.vue`
  — `aggregatedEdgesAvailable` computed.

---

### 6. Consensus edges (topic_consensus enricher)

**Rule:** when the "Consensus edges" lens is on, draw **green
unbundled-bezier arcs** between two Persons who corroborate on the same
Topic. Deduped per (pair, topic).

**Enricher:** `topic_consensus` — emits Person↔Person pairs with a Topic
label and an agreement score.

**Why unbundled bezier:** consensus edges cross a lot of the graph; bezier
routing keeps them from stacking on top of the ABOUT edges. Green because
"agreement".

**Gate:** `useGraphLensesStore.consensusEdges`. Enricher-gated.

**Code:** `web/gi-kg-viewer/src/utils/cyGraphLensOverlays.ts` —
`applyConsensusEdges`.

**Live count (prod-v2):** 5 edges — sparse by design; consensus is rare.

---

### 7. Co-guest edges (guest_coappearance enricher)

**Rule:** when "Co-guest edges" is on, draw **dotted amber arcs** between
Persons who share ≥ 2 episodes. Width scales with `episode_count`.

**Enricher:** `guest_coappearance` — emits Person↔Person pairs with a shared
episode count.

**Composes with:** person communities (§9). The lens is amber (edges); the
community overlay is a tint (nodes). Two views on the same signal.

**Gate:** `useGraphLensesStore.coGuestEdges`. Enricher-gated.

**Code:** `web/gi-kg-viewer/src/utils/cyGraphLensOverlays.ts` —
`applyCoGuestEdges`.

**Live count (prod-v2):** 1 edge — very sparse; kicks in on larger corpora.

---

## Layout

### fcose defaults (the everyday layout)

**Choice:** fcose (force-directed, coarse-to-fine). Chosen over cose in #967
for perf on 800+ node corpora — fcose settles in ~1.5s vs cose's 8s+.

**Key parameters** (in `web/gi-kg-viewer/src/utils/cyCoseLayoutOptions.ts`):

- `nodeRepulsion: 880_000` — base repulsion. Not tuned to the docs' 8000
  proposal (units differ from the current base).
- `idealEdgeLength: 90` (default), `120` for `RELATED_TO` (the one
  same-type edge).
- `gravity: 0.12` — Tier 3-M lowered from 0.18. Modest cluster-drift.
  Proposals to drop to 0.08 not applied — bipartite shape would strand
  orphan Insights.
- `randomize: false` — layouts are deterministic given the same artifact,
  so screenshots + fixtures are stable.

**LCP baseline:** 1561ms on prod-v2 (833 nodes, DPR-2, 1440×900) after
Tiers A/B/C. Trace at `docs/wip/graph-v3/traces/03-C-first-paint.json.gz`.

**Why not cose-bilkent / cola / dagre:** cose-bilkent settled slower on the
bipartite shape; cola scales worse past 300 nodes; dagre wants a DAG and
we're not.

---

### Timeline layout (RFC-080 V3)

**When:** the operator switches the graph to timeline mode via the Layout
chip. Nodes are placed on a horizontal time axis by `published_date`;
non-temporal nodes cluster below the axis.

**Implementation:** `timelineLayoutSpec` in the layout options module.

**Why coexist with fcose:** timeline surfaces the *temporal shape* — bursts,
gaps, host-vs-guest tempo. fcose surfaces the *topological shape*.
Different questions.

---

### TopicCluster compound layout

**Rule:** `TopicCluster` nodes are rendered as **compound containers** in
Cytoscape — Topic children live inside their parent's boundary. Tighter
footprint on the main canvas (Tier 3 topic-cluster refinement §graph-layout-topic-cluster in `VIEWER_GRAPH_SPEC.md`).

**Neighborhood minimap** uses a **2D layout** for its own render — not a
single-line stripe. Reason and details in `VIEWER_GRAPH_SPEC.md`
§graph-layout-topic-cluster.

---

## Interaction

### Selection dim

**Rule:** clicking a node highlights it and its 1-hop neighbourhood.
Non-neighbours dim to `.dimmed` opacity `0.2`. Selection wins over hover
(guard on `:selected` count).

**Code:** `applyGraphSelectionDimFromNode` / `clearGraphSelectionDim` in
`GraphCanvas.vue`.

---

### Hover dim (Tier D)

**Rule:** mouseover a node applies the same neighbour-lift + others-dim.
Mouseout clears it. Guarded by `:selected` count so selection dim takes
precedence.

**Perf:** single-frame class toggle; smoothed by an existing 120ms opacity
transition on nodes + edges. No layout cost.

**Code:** `cy.on('mouseover'/'mouseout', 'node', ...)` inside
`GraphCanvas.vue`.

---

### Handoff orchestrator

**Rule:** all "open X in graph" callers — Library, Digest, Search, Node
Detail — funnel through a single FSM in `stores/graphHandoff.ts`. States:
`idle → loading_fetch → loading_bootstrap → loading_merge →
redrawing_incremental|redrawing_full → applying → ready`. Timeout:
`STUCK_TIMEOUT_MS = 15000`.

**Why one FSM:** without it, tab-switch-during-handoff and rapid successive
handoffs raced (generation counters ensure a late fetch doesn't overwrite a
newer selection).

**Full spec:** `docs/architecture/VIEWER_GRAPH_SPEC.md` §Graph handoff
orchestrator. Entry-point catalog and recipe for adding a new entry surface
in that section.

**Tier 2 P2.5 regression:** when a prior handoff already loaded the target
artifacts, `appendRelativeArtifacts` short-circuits on `add.length === 0`,
so no redraw fires and the FSM stucks in `loading_fetch`. Fix (in
`GraphCanvas.vue`, ~L2093 & L4181): defer `handoffFailed` when the artifact
already has the pending focus target, and extend `FOCUS_RESOLVE_FRAME_BUDGET
= 120` frames (~2s at 60Hz) to survive parallel-worker CPU pressure.

---

### Focus + camera

**Modes** (in `handoffRequested.camera.kind`):

- `center-on-target` — pan so the target node is at viewport centre; zoom
  preserved.
- `fit-to-neighborhood` — zoom + pan to fit the 1-hop ego neighbourhood.
- (extensible via the recipe in `VIEWER_GRAPH_SPEC.md`).

**Focus polling:** `pollForFocusTarget` retries frame-by-frame up to
`FOCUS_RESOLVE_FRAME_BUDGET = 120` frames waiting for Cytoscape to resolve
the target id. If it misses, `handoffFailed` fires.

---

### Search reveals hidden

**Rule:** when a Search hit points at a node that's hidden by zoom-gating,
type-filter, or top-down collapse — the graph auto-un-hides it as part of
resolving the pending focus.

**Top-down variant:** if the target lives under a collapsed super-theme,
`maybeExpandTopDownForPendingFocus` (in `GraphCanvas.vue`) auto-expands the
super-theme so `tryApplyPendingFocus` succeeds on the next redraw.

**Why:** search's contract is "take me there". A hit that lands on an
invisible node is a broken contract.

---

## Filters

### Type filters

**Rule:** users toggle node types on/off via the Types chip in the graph
bottom bar. Edges to filtered-out types drop with them.

**Default-hidden types** (Tier 6-3): `DEFAULT_HIDDEN_TYPES = ['Quote',
'Speaker']` — the initial `allowedTypes` map starts with these off.

**Why the default:** Quotes and Speakers are evidence detail — useful when
the analyst is auditing a specific claim, noise at overview. The
"Everything visible" default was too busy.

**Reset semantics:** the "Reset" indicator lights up on *deviation from
default* — not on "anything unchecked". A user who ticks Quote *on* and
Speaker *off* sees the reset indicator because Speaker is at default but
Quote isn't. The `filtersActive` computed does the deviation math.

**Code:** `web/gi-kg-viewer/src/utils/parsing.ts::DEFAULT_HIDDEN_TYPES`
and `stores/graphFilters.ts`.

---

### Top-down mode filter override

**Rule:** in top-down load mode, `graphFilters.filteredArtifact` forces
`allowedTypes.SuperTheme = true` so the canvas isn't emptied by a stale
"SuperTheme unchecked" filter state.

**Code:** `web/gi-kg-viewer/src/stores/graphFilters.ts` — top-down routing
plus SuperTheme force-through. Test:
`web/gi-kg-viewer/src/stores/graphFilters.test.ts`.

---

## Lenses

The full catalog. Every lens is a boolean flag in `useGraphLensesStore`
persisted to localStorage and synced to the server via USERPREFS-1.

| Lens | Flag | Default | Gate | Type |
| ------ | ------ | --------- | ------ | ------ |
| Size by connectivity | `nodeSizeByDegree` | on | always visible | node width/height |
| Bridge nodes | `bridgeRing` | on | always visible | node border |
| Theme regions | `themeClusterRegions` | off | `topic_theme_clusters` enricher present | node underlay tint |
| Velocity halo | `velocityHalo` | off | `temporal_velocity` enricher present | node halo |
| Person credibility | `personCredibility` | off | `grounding_rate` enricher present | node border |
| Consensus edges | `consensusEdges` | off | `topic_consensus` enricher present | new edges |
| Co-guest edges | `coGuestEdges` | off | `guest_coappearance` enricher present | new edges |
| Person communities | `personCommunities` | off | `guest_coappearance` enricher present | node underlay tint |
| Aggregated edges | `aggregatedEdges` | off | artifact has ABOUT/SPOKEN_BY edges (data-gated) | new edges |

**Enricher-gated pattern:**

1. `useEnrichmentEnvelopeCache.fetchCachedCorpusEnvelope(root, enricherId)`
   probes availability on chip mount + on `shell.corpusPath` change. The
   fetch is Map-cached per root/enricher.
2. When the fetch returns `null`, the corresponding lens row is **hidden
   entirely** (not disabled). Operator direction: no dead controls.
3. If the enricher is present, the row shows in `GraphLensesChip.vue`.
4. Toggling the flag fires `refreshEnricherLensOverlays()` — a single
   watcher-driven refresh that re-fetches the envelope (via the cache) and
   re-applies the corresponding overlay function from
   `utils/cyGraphLensOverlays.ts`.

**Reset behaviour:** the "reset" link at the top of the chip calls
`lenses.resetToDefaults()` — sets defaults back to `nodeSizeByDegree=on`,
`bridgeRing=on`, everything else off.

**Test coverage:** `utils/cyGraphLensOverlays.test.ts` (21 tests) covers
apply/clear for all 8 lenses via a hand-rolled MockCore — null/happy
paths, threshold boundaries, dedupe rules, weight preservation.

---

## Load modes

### Everything (default)

The graph mounts the full artifact — every Topic, Insight, Person, Quote,
etc. Users navigate by zoom, filter, or search.

**Best for:** corpora ≤ ~1500 nodes on modern desktop; investigative
sessions where the analyst wants to see everything at once.

---

### Top-down (opt-in, USERPREFS-1)

**Mount:** synthetic slice with `6–8 SuperTheme` bubbles derived from
`topic_theme_clusters.json`'s `super_themes` field. No Insights, Persons,
or Topics rendered initially.

**Expand-on-tap:** tapping a SuperTheme node adds its children (Topics and
their propagated one-hop nodes) into the slice. Multiple SuperThemes can
be expanded at once; state in `graphTopDown.expandedSuperThemeIds`.

**Bridge-derived cross-super-theme edges** (Gap 3): if two SuperThemes have
bridge Topics between them (from the enricher's bridge output), the
top-down slice draws a cross-super-theme edge — the *reason* to expand.

**Clamp** (Gap 4): if the corpus has > 8 super_themes, the mount trims to
the 8 largest by member weight. Reason: 7±2 rule — legend as a browse
surface stops working above ~8 groups.

**Ego re-scope** (Tier 8-6): ego expansion uses the **full** `displayArtifact`,
not the top-down slice — walking a super-theme's ego neighbourhood surfaces
cross-episode structure the top-down mount would hide.

**Flag:** `useGraphLoadModeStore.mode` — `'everything'` (default) or
`'top-down'`. Persisted via localStorage + USERPREFS-1.

**When to use:** corpora > ~1500 nodes; onboarding a new analyst; giving a
presentation. Top-down is a "browse surface first, detail on demand"
approach.

**Code:**

- Slice derivation: `web/gi-kg-viewer/src/utils/topDownSlice.ts`.
- Store: `web/gi-kg-viewer/src/stores/graphTopDown.ts`,
  `stores/graphLoadMode.ts`.
- Search auto-expand: `GraphCanvas.vue::maybeExpandTopDownForPendingFocus`.
- Filter routing: `stores/graphFilters.ts`.

---

## Legend

The Theme Cluster Legend is a floating panel — one entry per theme cluster
in the current corpus, coloured by the theme-region palette hash. It lives
in `web/gi-kg-viewer/src/components/graph/ThemeClusterLegend.vue`.

### Hierarchical rollup (Tier 7-1)

**Rule:** when `topic_theme_clusters.json` schema is v1.1.0+ with
`super_theme_id` on each cluster, the legend renders as **two-level**:

- 6–8 super-themes as parent rows
- 100+ themes as expandable children under their parent

Backward-compat with pre-v1.1.0 corpora — falls back to flat listing when
`super_theme_id` is absent.

**Super-theme derivation** (enricher-side): `super_theme_id` computed via
cross-cluster average-linkage over co-occurrence lift. Bounded to [5, 8]
super-themes regardless of corpus size (7±2 legend rule).

---

### Search (Tier 7-2)

**Rule:** typeahead filters the legend tree in place. When any child
matches, its parent super-theme header stays visible so the hierarchy
context is preserved.

---

### Focus interaction (Tier 7-3)

**Rule:** clicking a legend entry writes its cluster id to
`graphThemeFocus.focusedThemeId` (single-focus bus). GraphCanvas watches
and dims all nodes outside the focused theme region.

**Clear:** second click on the same entry, or Escape.

**Why single focus:** multi-focus was tempting but interacts poorly with
the search-reveals-hidden and expand-on-tap flows in top-down mode. Single
focus is cleaner to reason about.

**Store:** `web/gi-kg-viewer/src/stores/graphThemeFocus.ts`.

---

## State stores

Full store map for the graph:

| Store | File | Responsibility |
| ------- | ------ | --------------- |
| `useGraphLensesStore` | `stores/graphLenses.ts` | 9 lens flags + resetToDefaults; localStorage + USERPREFS-1 |
| `useGraphFiltersStore` | `stores/graphFilters.ts` | `allowedTypes`, `filteredArtifact`, top-down routing, SuperTheme force-through |
| `useGraphLoadModeStore` | `stores/graphLoadMode.ts` | `'everything'` vs `'top-down'`; localStorage + USERPREFS-1 |
| `useGraphTopDownStore` | `stores/graphTopDown.ts` | `expandedSuperThemeIds`; drives top-down slice re-derivation |
| `useGraphThemeFocusStore` | `stores/graphThemeFocus.ts` | Single focus bus for legend interaction |
| `useGraphExplorerStore` | `stores/graphExplorer.ts` | Ego expansion, cross-episode re-scope |
| `useGraphHandoffStore` | `stores/graphHandoff.ts` | Handoff FSM (see `VIEWER_GRAPH_SPEC.md`) |
| `useArtifactsStore` | `stores/artifacts.ts` | Merged GI+KG artifact; `displayArtifact`, `topDownDisplayArtifact` computeds |

**USERPREFS-1** (`web/gi-kg-viewer/src/composables/useUserPreferences.ts` +
`api/userPreferencesApi.ts`): file-backed per-user preferences. Read on
mount, patched on each toggle, replaced on `resetToDefaults`. Backend at
`src/podcast_scraper/server/routes/app_user_preferences.py` — one JSON blob
per user, versioned via optimistic concurrency.

**localStorage-first**: every persisted flag reads localStorage synchronously
at store init, then reconciles with the server-side value on the async
watch. Users offline get their last-known preferences immediately.

---

## Enricher artifacts

Each visual overlay that depends on corpus data reads a specific enricher
output. Contracts:

| Artifact | Path (in corpus) | Consumed by | Fields we read |
| ---------- | ------------------ | ------------- | ---------------- |
| `topic_theme_clusters.json` | `<corpus>/enrichments/topic_theme_clusters.json` | Theme regions, hierarchical legend, top-down mount | `clusters[].graph_compound_parent_id`, `clusters[].canonical_label`, `clusters[].members[].topic_id`, `clusters[].members[].episode_ids`, `super_themes[]` (v1.1.0+), `clusters[].super_theme_id` (v1.1.0+) |
| `temporal_velocity.json` | `<corpus>/enrichments/temporal_velocity.json` | Velocity halo lens | `topics[].topic_id`, `topics[].velocity_last_over_6mo`, same for `persons[]` |
| `grounding_rate.json` | `<corpus>/enrichments/grounding_rate.json` | Person credibility lens | `persons[].person_id`, `persons[].grounded_rate` |
| `topic_consensus.json` | `<corpus>/enrichments/topic_consensus.json` | Consensus edges lens | `pairs[].person_a`, `pairs[].person_b`, `pairs[].topic_id`, `pairs[].agreement` |
| `guest_coappearance.json` | `<corpus>/enrichments/guest_coappearance.json` | Co-guest edges lens + Person communities lens | `pairs[].person_a`, `pairs[].person_b`, `pairs[].episode_count`, `communities[]` (v1.1.0+) |
| `insight_sentiment.json` (per-episode sidecar) | `<corpus>/metadata/enrichments/{stem}.insight_sentiment.json` | Reader arc-tint (**not** a graph lens — see below) | `sentiments[].insight_id`, `sentiments[].compound`, `sentiments[].label` |

**Enricher-gate contract:** `GET /api/corpus/{enricher_id}` returns `404`
when the artifact isn't present; `useArtifactsStore` (or the on-demand
`fetchCachedCorpusEnvelope` for lens gates) surfaces that as `null`; every
row that depends on the missing artifact is filtered out at the chip level.

**Insight sentiment — descoped from graph lens:** the sentiment enricher
ships per-episode sidecars that the read-time CIL layer already joins into
arc responses (`cil_queries._attach_sentiment`). Sentiment surfaces as an
**arc-tint on the player conversation-arc + position-arc** panels — not a
graph-canvas lens. A per-Insight tint on a graph node would fire only in
the fine-grained view where a per-Topic sentiment histogram is the
actually-useful signal, so it would ship as a new corpus-scope enricher
(`insight_sentiment_corpus`), not a viewer patch. See
`docs/wip/graph-v3/SUMMARY.md` §Not done for the closure rationale.

---

## Testing model

### Unit (vitest)

Each stylesheet, lens overlay, and store has a `.test.ts` sibling. Key
files:

- `utils/cyGraphStylesheet.test.ts` — every edge tier, node shape,
  degree-heat selector, palette hex.
- `utils/cyGraphLensOverlays.test.ts` — 21 tests, all 8 apply/clear
  overlay functions via MockCore.
- `utils/topicClustersOverlay.test.ts` — episode_ids tagging, one-hop /
  two-hop propagation, first-cluster-wins, type allowlist.
- `stores/graphLenses.test.ts` — every flag, resetToDefaults, USERPREFS-1
  round-trip, legacy-key migration.
- `stores/graphFilters.test.ts` — top-down routing, SuperTheme
  force-through, filtersActive computed.
- `stores/graphLoadMode.test.ts`, `stores/graphTopDown.test.ts`.
- `utils/parsing.test.ts` — `DEFAULT_HIDDEN_TYPES`, `filtersActive` corner
  cases.
- `utils/colors.test.ts` — brand-palette collision guard.

### E2E (Playwright)

- `e2e/handoff/` — Tier-1 handoff matrix.
- `e2e/handoff-production/` — Tier-2 production-shaped fixtures. P2.5
  regression: "Digest → Library when target artifacts already loaded" —
  the stuck-timeout bug from Tier-3 (fixed by
  `loadSelected({preserveExpansion: true})` on `appendRelativeArtifacts`
  short-circuit + the FOCUS_RESOLVE_FRAME_BUDGET bump).
- `e2e/validation/` — Tier-3 real-corpus walks against
  `tests/fixtures/viewer-validation-corpus/v3` (synthetic, deterministic).

### Stack-test (docker compose + Playwright)

`tests/stack-test/` — brings up the api + viewer + mock-feeds containers,
runs the pipeline, then walks the UI against the real product corpus. Only
place where the full "boot to open" flow is tested. Runs in CI on push to
main and in `ci-ui-full` locally.

---

## Cross-references

**Design & rationale:**

- `docs/rfc/RFC-080-graph-visualization-extensions.md` — original design for
  Tier 5 lenses (aggregated edges, velocity halo, etc.).
- `docs/rfc/RFC-069-graph-exploration-toolkit.md` — the graph-exploration
  primitives (ego, expand-on-tap, matrix contracts).
- `docs/rfc/RFC-075-corpus-topic-clustering.md` — the theme-cluster enricher.

**Implementation specs:**

- `docs/architecture/VIEWER_GRAPH_SPEC.md` — handoff orchestrator, camera
  modes, matrix assertion layers, gesture overlay, cluster sibling load.
- `docs/architecture/VIEWER_FRONTEND_ARCHITECTURE.md` — the surrounding
  Vue app structure.

**Data / model:**

- `docs/guides/KNOWLEDGE_GRAPH_GUIDE.md` — node/edge kinds at the
  data layer; entity resolution.
- `docs/guides/GIL_KG_CIL_CROSS_LAYER.md` — how GI, KG, and CIL layers
  merge into the artifact the viewer receives.

**Change history:**

- `docs/wip/graph-v3/SUMMARY.md` — tier-by-tier chronology of the visual
  iteration (A/B/C/D → Tier 8 + gaps).
- `docs/wip/graph-v3/HARDEN-FOLLOWUPS-2026-07-17.md` — the HD1–HD20 harden
  pass audit + fixes.
- `docs/wip/graph-v3/REPRODUCIBILITY.md` — running the enrichers end-to-end
  against a corpus to reproduce the on-screen state.

**Related consumer surface:**

- `docs/guides/CONSUMER_LEARNING_PLAYER_GUIDE.md` — where sentiment lives
  (arc-tint), why it's not a graph lens.

---

## Reading order for a newcomer

Working through this file cover-to-cover is heavy. For a first pass:

1. **§Mental model** — three layers, composability.
2. **§Node encoding §1–§3** (shape, colour, size) — 80% of "why does it
   look like this at overview zoom".
3. **§Edge encoding §1** — tier system.
4. **§Lenses table** — one row per user-facing toggle.
5. **§Load modes** — Everything vs Top-down.
6. Deep-dive whatever you're editing.

The rest is reference — read when the specific concern comes up.
