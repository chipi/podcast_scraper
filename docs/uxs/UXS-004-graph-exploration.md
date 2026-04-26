# UXS-004: Graph Exploration

- **Status**: Active
- **Authors**: Podcast Scraper Team
- **Parent UXS**: [UXS-001: GI/KG Viewer](UXS-001-gi-kg-viewer.md) -- shared tokens,
  typography, layout, states
- **Related PRDs**:
  - [PRD-024: Graph Exploration Toolkit](../prd/PRD-024-graph-exploration-toolkit.md)
- **Related specifications** (design docs under `docs/rfc/`):
  - [Graph exploration toolkit](../rfc/RFC-069-graph-exploration-toolkit.md)
  - [GI/KG viewer v2](../rfc/RFC-062-gi-kg-viewer-v2.md)
  - [Corpus topic clustering](../rfc/RFC-075-corpus-topic-clustering.md) (optional **TopicCluster** overlay + rail context)
  - [Progressive graph expansion (cross-episode)](../rfc/RFC-076-progressive-graph-expansion.md) (`onetap` rail, `dbltap` expand/collapse, `POST /api/corpus/node-episodes`)
- **Implementation paths**:
  - `web/gi-kg-viewer/src/components/graph/GraphCanvas.vue`
  - `web/gi-kg-viewer/src/components/episode/EpisodeDetailPanel.vue` (**Open in graph** hand-off)
  - `web/gi-kg-viewer/src/utils/graphEpisodeMetadata.ts` (corpus episode → Cytoscape id; metadata + episode-id fallback)
  - `web/gi-kg-viewer/src/components/graph/GraphStatusLine.vue` (summary **counts** strip + bottom-bar **lens** row; `docs/architecture/VIEWER_GRAPH_SPEC.md`)
  - `web/gi-kg-viewer/src/components/graph/GraphBottomBar.vue` (minimap / re-layout / lens centre / zoom+PNG+optional Gestures / collapse)
  - `web/gi-kg-viewer/src/components/graph/GraphFilterBar.vue` + `chips/{GraphFeedChip,GraphTypesChip,GraphSourcesChip,GraphEdgesChip,GraphDegreeChip}.vue` (#658 chip-bar replaces the legacy `GraphFiltersPopover.vue` ⚙ popover with per-dimension chips)
  - `web/gi-kg-viewer/src/components/graph/GraphGestureOverlay.vue` (one-time gesture discovery overlay)
  - `web/gi-kg-viewer/src/components/graph/GraphNodeRailPanel.vue`
  - `web/gi-kg-viewer/src/components/graph/NodeDetail.vue`
  - `web/gi-kg-viewer/src/components/shared/TranscriptViewerDialog.vue`
  - `web/gi-kg-viewer/src/components/graph/GraphConnectionsSection.vue`
  - `web/gi-kg-viewer/src/components/graph/GraphNeighborhoodMiniMap.vue`
  - `web/gi-kg-viewer/src/components/explore/ExplorePanel.vue`
  - `web/gi-kg-viewer/src/utils/cyGraphStylesheet.ts`
  - `web/gi-kg-viewer/src/utils/cyCoseLayoutOptions.ts` (COSE tuning for topic clusters + semantic edge lengths)
  - `web/gi-kg-viewer/src/utils/parsing.ts` (Cytoscape node data: `shortLabel`, `recencyWeight`, `confidenceOpacity`, canonical `edgeType`)
  - `web/gi-kg-viewer/src/utils/topicClustersOverlay.ts` (corpus **TopicCluster** compound parents)
  - `web/gi-kg-viewer/src/stores/graphExplorer.ts` (graph-only time lens + layout prefs; seeded from Digest/Library lens once)
  - `web/gi-kg-viewer/src/utils/graphEpisodeSelection.ts` (graph lens filter + scored episode cap before corpus load)
  - `web/gi-kg-viewer/src/stores/graphFilters.ts`
  - `web/gi-kg-viewer/src/stores/graphNavigation.ts`
  - `web/gi-kg-viewer/src/stores/graphExpansion.ts` (cross-episode expand seed → appended paths)
- **Shell IA:** [VIEWER_IA.md](VIEWER_IA.md) — canonical shell layout, navigation axes, subject rail, status bar, first-run behavior

---

## Summary

For shell layout, the three navigation axes, subject rail persistence and clearing, status bar, and first-run empty corpus behavior, see **[VIEWER_IA.md](VIEWER_IA.md)**. This document specifies the **Graph** tab only (toolbar, canvas, overlays, node detail in the subject rail, and graph-specific chrome).

The Graph tab provides a Cytoscape-powered interactive graph canvas for exploring
merged GI/KG artifacts.

### Default graph load (corpus API + local files)

- **Graph lens (`graphLens`):** Independent from Digest/Library **Published on or after** (`corpusLens`). On first **Graph** tab visit per browser session, the graph lens is **seeded** from the shared corpus lens; if Library/Digest is **all time**, the graph defaults to **last 7 days** instead of loading the whole corpus. Changing Digest/Library filters **does not** change the graph lens after seed; changing the graph lens **does not** change Digest/Library.
- **Episode cap:** The merged graph loads at most **15** episodes (tunable in [UXS-001](UXS-001-gi-kg-viewer.md)) from the current graph lens window. Episodes are **ranked by score** (recency within the window + topic-cluster bonus when `topic_clusters.json` is available; tie-break newer publish), not “newest 15 only”. **`(capped)`** appears in the **counts** strip (`graph-status-line`) when more episodes matched the window than the cap (hidden while RFC-076 cross-episode expansion adds rows).
- **Stats strip + time lens (full merged graph):** **`data-testid="graph-status-line"`** is **counts only** — muted **`text-[10px]`** row at the **top** of the graph card (**Showing … · N episodes · M nodes · C components**; **`graph-status-component-count`**; **`graph-status-episode-count`** / **`graph-status-node-count`** with optional **`k`** suffix on large node counts). **`data-testid="graph-gesture-overlay-reopen"`** (**Gestures**) sits on the **same strip, right-aligned**, and reopens the gesture overlay **without** clearing `localStorage`. The **graph time lens** (**`data-testid="graph-status-lens-selector"`**, presets **7d / 30d / 90d / All**, **`data-testid="graph-status-since-input"`**, optional **`data-testid="graph-status-reset"`** after RFC-076 **expand**) renders in **`data-testid="graph-bottom-bar-centre"`** as **`data-testid="graph-status-line-controls"`** (same **Digest-style** active ring on presets / custom **Since**). Changing the lens clears RFC-076 expansion and reloads the graph from the API list (or re-filters local file picks). When there is **no** full merged-graph strip (e.g. some offline slices), **Gestures** stays in the **bottom bar** right zone instead. Normative test ids: [E2E surface map](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md) (**Graph shell** row).
- **Search highlight chip:** When search (or library) highlights are active, **`data-testid="graph-search-highlight-chip"`** appears in a **thin row between** the stats strip and the **Types** toolbar — not in the bottom bar.
- **Default node types:** **Quote**, **Speaker**, and **Episode** checkboxes start **off** for a cleaner first read; a **reset** affordance (**`data-testid="graph-types-reset"`**) inside the Types chip popover appears when the user deviates from defaults. After #658 each filter dimension is a separate chip (**Feed** / **Types** / **Sources** / **Edges** / **Degree**) — non-default values surface in the chip's own active label (e.g. `Types: 2 of 5 ▾`) instead of a separate ⚙ warning dot.

Full spec: [Viewer graph spec — Graph initial load](../architecture/VIEWER_GRAPH_SPEC.md#graph-initial-load).

When **`GET /api/corpus/topic-clusters`** returns clustering JSON,
the graph adds **TopicCluster** parent nodes (dashed compound outline) and sets
Cytoscape **`parent`** on member **Topic** nodes whose bare id matches **`topic:…`** members.
Payload **v2** uses **`graph_compound_parent_id`** (`tc:`…) for those parents; **`cil_alias_target_topic_id`**
is for CIL aliases only, not for graph ids.
This UXS defines the visual contract for graph chrome:
toolbar, bottom bar, filters popover, minimap, degree filter, node detail rail, and
neighborhood visualization. On the **main graph canvas**, node captions default to
the **side** of the disc (dynamic `text-margin-x`) with a canvas-tinted halo so edge
splines stay readable; **above** / **below** placement remain available via the shared
Cytoscape stylesheet builder. COSE and the rail mini-map use slightly looser spacing.
All tokens reference [UXS-001](UXS-001-gi-kg-viewer.md).

**Topic cluster context (phase 1):** When **`topic_clusters.json`** is loaded (API corpus
path), the **graph node detail rail** for a **Topic** that is a cluster member shows a **Topic cluster:**
line with the cluster **`canonical_label`** and a help tip. **Primary selection** stays on the Topic
node; **TopicCluster** compound parents remain graph grouping chrome.

When the user selects a **TopicCluster** compound node, the rail shows **Member topics** (rows from
`topic_clusters.json`: label, `topic:…` id, optional **Focus** when that Topic exists in the merged
graph, or a warning when the member is not in the current view). **Hide topics on graph** / **Show
topics on graph** toggles visibility of member Topic nodes on the Cytoscape canvas (compound outline
may remain). The header uses avatar **TC** and a **Topic cluster** badge line. **Cluster neighborhood**
(minimap) shows the compound, member topics, and **one hop outward from any member** (not ego of the
compound alone). **Connections to other nodes** lists edges from member topics to the rest of the
graph; each row can show **Via:** which member topic(s) that edge comes from when merged. Duplicate
neighbors by id are merged.

Later phases may extend **Show on graph** and search highlighting — see [UXS-005](UXS-005-semantic-search.md) and
[`wip-rfc-075-open-questions-followup`](../wip/wip-rfc-075-open-questions-followup.md).

**Progressive cross-episode expansion:** With a healthy corpus path, **double-activation** (Cytoscape **`dbltap`**; mouse: **double-click**, touch: **double-tap**) without Shift on an
eligible **Topic**, **Person**, or **Entity** node (canonical `topic:` / `person:` / `org:` id, degree
greater than one) loads additional episodes whose `bridge.json` lists that identity via
`POST /api/corpus/node-episodes`, appending GI/KG paths into the merged graph. A **second**
double-activation on the same node **collapses** that expansion by removing those appended paths.
**Single activation** uses Cytoscape **`onetap`** so the episode or graph-node rail opens after the
`dbltap` debounce window and does not fire on an expand gesture. A thin **strip**
above the graph (`data-testid="graph-expansion-truncation-line"`) shows truncation, empty-result,
or error text when applicable, with **Dismiss**.

**Cross-episode expand rings on the canvas:** Eligible nodes show a **teal** border only when the corpus library reports at least one **other** episode’s GI/KG for that identity that is **not** already merged into the graph (plain **dbl-click** merges those paths). After expand, the **seed** node shows a **blue** border until you collapse. **Ring semantics** and non-obvious gestures (Shift neighbourhood expand, Shift+drag box zoom, plain double-click expand/collapse) are explained in the **Gesture discovery overlay** below, not in a persistent toolbar sentence.

**RFC-076 state vs full graph reload:** A normal **`artifacts.loadSelected()`** (default) clears **`graphExpansion`** in the Pinia store **before** refetching merged GI/KG so `expandedBySeed` never points at artifact paths that are no longer in the selection. That applies to graph **lens** changes, corpus **auto-sync** when the capped episode slice changes, **Digest/Library → graph** handoffs (`loadRelativeArtifacts`), **Dashboard → Load into graph**, **Refresh graph** on the overview, and any other path that replaces the merged load without using expand/collapse internals. **`appendRelativeArtifacts`** and **`removeRelativeArtifacts`** (progressive expand / collapse) call **`loadSelected({ preserveExpansion: true })`** so the merged graph can reload while unrelated expand seeds stay consistent until the UI updates the record. Normative detail: [RFC-076 — Expansion reset vs full reload](../rfc/RFC-076-progressive-graph-expansion.md#expansion-reset-vs-full-reload).

---

## Camera framing and selection (merged graph)

**User-visible contract** (Cytoscape canvas + subject rail):

1. **Cross-panel focus** — When the app moves focus onto a node **from outside the canvas** (pending focus / `requestFocusNode`: Digest **topic band** hit rows that land on Graph, **Open in graph** from the Episode rail, Library → Graph with focus, Semantic Search **Show on graph**, etc.), after the merged graph finishes its layout pass the canvas **selects** that node, applies **selection dimming** (see *Selection dim* under **Graph visual styling** below), updates the **subject rail**, and runs one **short animated pan/zoom** so the focal element(s) are **centered** in the viewport. Some handoffs pass **extra graph ids** so the camera’s bounding box can include a related compound (e.g. TopicCluster) while **primary selection** stays on the main id — see [UXS-005 — Results](UXS-005-semantic-search.md#results) (**Show on graph**).

2. **Single-tap on the canvas** — A normal **single** activation on a node (Cytoscape **`tap`** + debounced **`onetap`**; opens the Episode or graph-node rail) runs the **same** camera animation as cross-panel focus. The newly inspected node should not stay parked at the edge of the viewport while the rail updates. **Double** activation remains **RFC-076** expand/collapse on eligible seeds (**`dbltap`**); **`onetap`** is debounced so a double-click expand does not open the rail first.

3. **Minimum zoom during focus animation** — The animation uses at least **`GRAPH_FOCUS_FRAME_MIN_ZOOM`** (**1.3×**, roughly **130%** in the bottom zoom readout) unless the user was already zoomed in further (`max(current zoom, constant)` in `GraphCanvas.vue`). This keeps hand-offs readable without zooming “into your face” as aggressively as earlier **1.6×** defaults.

4. **Stable episode neighbourhood after hand-off** — After **Open in graph** (or any path that focuses an **Episode** on the merged slice), **1-hop neighbourhood dimming** must remain on the intended episode and its neighbours once layout settles — not revert to **uniform full brightness** a second or two later. Corpus **metadata path** strings and graph **Episode** row text can diverge (punctuation, apostrophes, normalisation); the product resolves the Cytoscape episode node by **metadata match first**, then **stable corpus episode id**, and must **not** discard a **valid** stored graph episode id when a path-only match fails.

**Implementation anchors:** `GraphCanvas.vue` (`animateCameraToFocusedNode`, `tryApplyPendingFocus`, `onetap` handler), `EpisodeDetailPanel.vue`, `graphEpisodeMetadata.ts`, Pinia **`subject`** (`graphConnectionsCyId`) and **`graphNavigation`** (`pendingFocusNodeId`, optional `pendingFocusCameraIncludeRawIds`).

---

## Graph visual styling (Cytoscape)

Normative numbers and formulas live in the [Viewer graph spec — Graph visual styling](../architecture/VIEWER_GRAPH_SPEC.md#graph-visual-styling) (tracks [GitHub #608](https://github.com/chipi/podcast_scraper/issues/608)). This subsection states the **user-visible contract** only.

**Phase 1 (stylesheet + wiring):**

- **Node hierarchy:** Disc sizes follow type tiers (Insight / Topic largest; Episode / Speaker / Podcast smallest; **Entity_person** and **Entity_organization** map to the Person vs Entity tiers in the WIP table). **TopicCluster** compounds use a faint **`kg`** fill tint, dashed **`kg`** border, and padding so members read as one region.
- **Edges:** Each relationship is distinguished by stroke width, dash pattern, colour (**`gi`**, **`kg`**, **`primary`**, **`muted`** tokens), and arrowheads where applicable. Edge data uses **`edgeType`** (canonical uppercase in `parsing.ts`), not a generic `type` field. Unknown types fall back to a muted neutral stroke. **Synthetic** `_tc_cohesion` edges stay invisible. **No edge labels** on the canvas (type is read from stroke alone).
- **Selection dim:** When a node is selected, unrelated nodes and edges drop in opacity; the **focused** node stays at full strength (times **recency**); **1-hop neighbours** read slightly softer (**~85%** × recency) than the focus node (`graph-dimmed`, `graph-focused`, `graph-neighbour`, `graph-edge-dimmed`, `graph-edge-neighbour`). Dimming composes with **search-hit** and **RFC-076** expand rings (teal eligible / blue seed) without stripping those cues. **At most one** graph node is selected in normal use (**multi-select is not supported**).
- **Motion:** Opacity transitions use a short duration unless **`prefers-reduced-motion: reduce`**, in which case transitions are disabled when the stylesheet is built.
- **Zoom-driven labels:** Cytoscape has no min-zoom stylesheet selectors. **`GraphCanvas.vue`** (and the episode-rail **`GraphNeighborhoodMiniMap`**) call **`syncGraphLabelTierClasses`** from **`cyGraphLabelTier.ts`** so every node carries exactly one of **`graph-label-tier-none`**, **`graph-label-tier-short`**, or **`graph-label-tier-full`**, driven by **zoom** (main graph) or post-**fit** zoom (minimap). Constants **`GRAPH_LABEL_ZOOM_NONE_MAX`** (**0.5**) and **`GRAPH_LABEL_ZOOM_SHORT_MAX`** (**1.0**) match the WIP table. Tier 1–2 types use **`data(shortLabel)`** in the short band; full **`data(label)`** in the full band. Tier-1/2 set matches the WIP list (Insight, Topic, TopicCluster, Entity_person, Entity_organization).

**Phase 2 (data-driven accents, additive):**

- **Recency:** Nodes get **`recencyWeight`** from the episode **publish** date when resolvable; default **1.0** when missing. Combined with selection dim via a single opacity function on nodes.
- **Insight confidence:** **`confidenceOpacity`** on Insight nodes (default when the GI field is absent). **`background-opacity`** on the Insight fill reads this value.
- **Topic degree heat:** After each layout pass, **Topic** nodes get **`degreeHeat`** from graph degree (capped by **`maxDegree`**). **Border width is 0** when heat is **0** so hubs do not add a baseline ring next to expand/search styling; width ramps once heat is positive. High-heat nodes may gain **`graph-topic-heat-high`** (glow).

Tuning knobs for COSE semantics, label zoom breakpoints, and **`maxDegree`** are listed as **Open** under [UXS-001 — Tunable parameters](UXS-001-gi-kg-viewer.md#tunable-parameters).

---

## Gesture discovery overlay

A **one-time, dismissible** overlay on the graph canvas teaches gestures and ring colours the first time the user sees a **non-empty** merged graph in this browser profile, unless they already dismissed it (`localStorage` key **`ps_graph_hints_seen`** set to **`1`**). Persistence is **per browser profile** (until storage is cleared), not per browser session. Full visual and interaction spec: [Viewer graph spec — Graph gesture overlay](../architecture/VIEWER_GRAPH_SPEC.md#graph-gesture-overlay).

**Mounting context:** [`App.vue`](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/src/App.vue) mounts [`GraphTabPanel`](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/src/components/graph/GraphTabPanel.vue) only when the **Graph** main tab is active (`v-if="mainTab === 'graph'"`), and [`GraphCanvas`](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/src/components/graph/GraphCanvas.vue) mounts only when artifacts display — the overlay does not need a separate `mainTab` prop unless that shell layout changes.

**Dismiss:** primary control **Got it** (`data-testid="graph-gesture-overlay-dismiss"`), backdrop click outside the card (`@click.self` on the overlay root; card uses `@click.stop`), and **Escape** via a `window` `keydown` listener (capture) that ignores **Escape** when focus is outside the overlay (e.g. **Since** (`graph-status-since-input`) in the bottom bar lens row, **⚙** filters popover, or **Fit** / zoom in the bottom bar) so graph chrome stays predictable.

**Accessibility:** `role="dialog"`, `aria-modal="true"`, heading **Graph gestures** referenced with `aria-labelledby`, initial focus on **Got it** after open; on dismiss, focus returns to the graph canvas host for keyboard continuity.

**Surface map:** overlay root `data-testid="graph-gesture-overlay"`. **`data-testid="graph-gesture-overlay-reopen"`** (**Gestures**) reopens the card **without** clearing `localStorage`: **right** on the **stats** strip when a full merged graph is loaded, otherwise in the **bottom bar** right zone (same test id in both places).

---

## Graph chrome (toolbar, bottom bar, filters popover)

Normative layout and test ids: [GRAPH-CHROME-REDESIGN](../wip/GRAPH-CHROME-REDESIGN.md) and [E2E surface map](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md) (**Graph shell** row).

- **Filter chip bar (#658, `data-testid="graph-filter-bar"`):** Replaces the legacy `graph-toolbar-types` Types row + ⚙ popover. Per-dimension chips: **Feed** (`graph-chip-feed`), **Types** (`graph-chip-types`, popover hosts the per-type checkboxes + swatches + `graph-types-reset`), **Sources** (`graph-chip-sources`), **Edges** (`graph-chip-edges`), **Degree** (`graph-chip-degree`). Each chip's label switches from `Type ▾` (default) to `Type: detail ▾` when active. Pointerdown outside / Escape close popovers (shared `useFilterChipPopover` composable).
- **Bottom bar (`data-testid="graph-bottom-bar"`):** Under the canvas; optional collapse (**`graph-bottom-bar-toggle`** / **`graph-bottom-bar-expand`**, **`ps_graph_bottom_bar_collapsed`**, **Alt+B**). **Left:** **`graph-minimap-toggle`**, **`graph-relayout`**, **`graph-layout-cycle`** (cycles layout algorithm and re-layouts). **Centre (full merged graph):** **`graph-bottom-bar-centre`** — **`graph-status-line-controls`** / **`graph-status-lens-selector`** (lens presets, **Since**, optional **Reset**). **Right:** **`role="toolbar"`** — **Fit**, **−** / **+** / **100%**, optional **Gestures** (only when the stats strip is absent), **PNG** (2× full graph); toolbar **`aria-label`** omits “gestures” when that button is hidden. (Legacy floating zoom / layout strips on the canvas were removed in favour of this bar.)
- **Minimap:** Lower-left on the canvas host; **`graph-minimap-close`** (**×**) on the frame; visibility toggled from the bottom bar **⊞** control and the close button.

---

## Node detail (subject rail)

**Graph node subject rail (`GraphNodeRailPanel`, `NodeDetail` with `embed-in-rail`):** the
**`GraphNodeRailPanel`** mounts inside **`SubjectRail`** whenever **`subject.kind === 'graph-node'`**
(including when the main tab is **Digest**, **Library**, or **Dashboard**). Prefill / Explore handoffs
switch the **left** rail to the **Search** tab and fill **Search** or **Explore** filters; they do **not**
replace the right column. A pill **tablist** sits under the node header — **Details** (default) vs **Neighbourhood** — **`data-testid`** **`node-detail-rail-tab-details`** / **`node-detail-rail-tab-neighbourhood`**. The scroll body shows either node content (**Details**) or **`graph-connections-section`** (**Neighbourhood**); the active tab resets to **Details** when the focused graph node id changes. If the center id is not in the merged slice, **Neighbourhood** shows **`node-detail-rail-neighbourhood-unavailable`** instead of an empty panel. Graph **Type** plus **`entity_kind`**-style labels (**person**, **organization**, …) appear at the top of **Details** (**`node-detail-kind-row`**), not under the **h3** title. Insight **L** / **G** / **S** on graph neighbors (see long **Insight** paragraph below) live under **Neighbourhood**, not **Details**.

The header primary title (**h3**) expands vertically to fit the full label (`whitespace-pre-wrap`, `select-text`; no `line-clamp`). **Quote**, **Topic**, **Insight**, **Person**, **Speaker**, and **Entity** nodes set **`data-testid`** on that span (**`node-detail-full-quote`**, **`node-detail-full-topic`**, **`node-detail-full-insight`**, **`node-detail-full-person-entity`**) and expose a **`C`** chip stacked **under** **`?`** (**`node-detail-full-*-copy`**, same square footprint as **E** / **`?`**; native tooltip and **`aria-label`** **Copy title**, then **Copied to clipboard** / **Copy failed; try again** briefly). Other node types keep a native **`title`** on the span for hover. **Topic** optional **Aliases:** line (**`data-testid="node-detail-topic-aliases"`**) when **`properties.aliases`** is a non-empty string array (GI schema); aliases are omitted from the generic property list for topics. Below the header (above neighborhood / **Where this appears**), a compact gateway row mirrors Episode handoffs: **`data-testid="node-detail-topic-prefill-search"`** **Prefill semantic search** opens **Search** with the topic label as query (no feed filter); **`data-testid="node-detail-topic-explore-filter"`** **Set Explore topic filter** opens **Explore** with **Topic contains** filled and clears prior explore output — the user still runs **Explore**. Both are disabled when corpus/API health is not OK. Below that row, **Topic timeline** (**`data-testid="node-detail-topic-timeline"`**) opens a native **`dialog`** (**`data-testid="topic-timeline-dialog"`**) with the same modal shell as **Transcript** (`w-[min(100%,42rem)]`, `max-h-[min(92vh,48rem)]`, fixed header + scrollable body): it calls **`GET /api/topics/{topic_id}/timeline`** (CIL / RFC-072) using the graph node **id** and the resolved corpus path. Inner hooks include **`topic-timeline-loading`**, **`topic-timeline-error`**, **`topic-timeline-empty`**, **`topic-timeline-episodes`**. Disabled when health is not OK, **`cil_queries_api`** is false, or the corpus path is unset. **Person** / **Entity** scroll body: optional **`data-testid="node-detail-person-entity-role"`** (**SPOKEN_BY** / **SPOKE_IN** counts in the loaded slice), optional **`data-testid="node-detail-person-entity-aliases"`**, and (**`data-testid="node-detail-person-entity-prefill-search"`**, **`data-testid="node-detail-person-entity-explore-filter"`**) — compact labels **Speaker filter** / **Topic filter** (maps to **Speaker contains** vs **Topic contains**; see **`aria-label`** and **`?`** HelpTip). **`data-testid="node-detail-insight-details-tip"`** wraps a text-style **HelpTip** in the insight scroll body (below the header): the trigger is the underlined label **`Grounded`**, **`Not grounded`**, or **`Extraction details`** (when **`grounded`** is missing but lineage or other fields still populate the panel); **`aria-label`** matches that label. The panel holds a short **Grounding** explainer (**Grounded** = at least one supporting quote linked via **SUPPORTED_BY** in this GI; **Not grounded** = the artifact sets grounded to false with no such quotes), optional **Other fields** (**Type** / **Position in episode** / **Confidence**), and **Lineage** (**`model_version`**, **`prompt_version`**, optional **`extraction.extracted_at`**, artifact name). **`insight_type`**, **`position_hint`**, **`confidence`**, and **`grounded`** stay out of the generic property list when surfaced here. A gateway row (**`data-testid="node-detail-insight-prefill-search"`**, **`data-testid="node-detail-insight-explore-filters"`**) mirrors Topic handoffs: **Prefill semantic search** uses a truncated insight string for the vector index; **Set Explore filters** opens **Explore** with **Topic contains** and **Speaker contains** cleared, **Grounded only** and optional **Min confidence** aligned to this node — the user still runs **Explore**. **Episode on graph** and **Open source episode in Library** are not separate insight buttons; use **`Graph neighborhood and connections`** (**`data-testid="graph-connections-section"`**): for each neighbor, **L** (before **G**, **Episode** neighbors only) opens **Library** when metadata resolves (same rules as search result **L**); **G** focuses that neighbor on the graph; **S** (after **G**) prefills **Semantic search** with the neighbor’s primary text (truncated). **`data-testid="node-detail-insight-related-topics"`** (bordered region like Library **Similar episodes**) lists **Topic** neighbors via **`ABOUT`** / **`RELATED_TO`**; inner list host **`data-testid="node-detail-insight-related-topics-list"`** has no max-height or vertical scroll (the rail scrolls as a whole). Each row is one full-width control (**`node-detail-insight-related-topic-row`**): **click** focuses that **Topic** on the graph (same behavior as neighbor **G** on quotes). **`data-testid="node-detail-insight-supporting-quotes"`** lists **Quote** neighbors linked by **`SUPPORTED_BY`** (out from the insight), ordered by **`char_start`** then **`timestamp_start_ms`**, with truncated text and **G** to focus each quote; **`data-testid="node-detail-insight-view-transcript-all-quotes"`** (**Transcript (all quotes)**) opens the in-app transcript when every listed quote shares one **`transcript_ref`** and has finite **`char_start`** / **`char_end`**, highlighting all spans (the dialog may contain multiple **`transcript-viewer-highlight`** marks; **Passage** shows **N character spans (supporting quotes)** when **N** is greater than 1); more than five quotes collapse until **Show all N** (**`data-testid="node-detail-insight-supporting-quotes-toggle-expand"`**), reset when the selected node changes.
A body paragraph is omitted when it would only repeat that full primary label.
**`entity_kind: episode`** is not
shown as a subtitle under Insight (and similar) nodes -- the rail header already gives
the graph type. Header **E** (id), **`?`** diagnostics, and optional **Copy** are **stacked vertically**
(**E** above **`?`** above **`C`**, right-aligned) so the title column can use width for wrapping. The **subject** column header includes **Close subject panel** (**×**, **`data-testid="subject-rail-close"`**)
in addition to in-panel controls; type avatars reuse graph fill/border colors.
**Quote** transcript affordance: a compact **View transcript** (`button`, `aria-label` **View transcript**, **`data-testid="node-detail-view-transcript"`**) opens an in-app viewer (`dialog`, **`data-testid="transcript-viewer-dialog"`**, heading **Transcript**) via `GET /api/corpus/text-file`. The rail does not duplicate “open in new tab” or GI passage/audio lines — those sit in the dialog header: optional file subtitle, **Audio** timing, **Passage** (GI character span), **Open raw transcript in new tab** (**`data-testid="transcript-viewer-open-raw"`**), and a short note that highlight position is approximate when the served transcript variant differs from what GI indexed (e.g. cleaned vs raw). Body text uses monospace and `whitespace-pre-wrap`; `char_start` / `char_end` map to a highlighted range (`mark`, **`data-testid="transcript-viewer-highlight"`**) with smooth scroll. When a sibling `.segments.json` exists (Whisper-style list), a **Timeline** collapsible section (`details`, ordered list **`data-testid="transcript-viewer-timeline"`**) lists segment times and text; missing or invalid sidecar leaves that section absent. Files over the size cap (default **5 MiB**, chosen with headroom above typical dev transcripts in the tens of KiB) are not rendered in the dialog body; the user sees an explanation in the body and uses the header link for the raw file. When **`speaker_id`** is set, a muted line shows **Speaker:** *value* (below the button row when **View transcript** is available). When the quote has **`transcript_ref`** but no resolved speaker, **`GI_QUOTE_SPEAKER_UNAVAILABLE_HINT`** (**`data-testid="node-detail-quote-speaker-unavailable"`**) appears on the **same row** as **View transcript** when the API + corpus path yield an in-app **`href`**; if only a ref line is shown (no **`href`**), the same copy appears **below** that ref block (GitHub **#541** — same contract as Semantic Search supporting quotes, Explore, and the Search **Lifted GI insight** hint). When **`timestamp_start_ms`** and **`timestamp_end_ms`** are both **0**, the dialog **Audio** line still explains missing timed segments; see Development Guide (Transcript hash cache) and GitHub issue **543**.

---

## Explore panel (supporting quotes)

**Explore & query** (`ExplorePanel.vue`) can list supporting quotes under each insight. When a quote has text but **no** speaker fields, the same muted **#541** line and **`data-testid="supporting-quote-speaker-unavailable"`** apply as in [UXS-005 — Results](UXS-005-semantic-search.md#results) (semantic search result cards). Playwright: **`explore-supporting-quotes-mocks.spec.ts`**.

---

## Minimap

Fixed footprint (~7.5rem tall x ~10.5rem wide, capped vs short viewports) in the
lower-left of the graph canvas host (same `overflow-hidden` region as the main
Cytoscape surface), not a viewport-fixed tile and not over the app's right rail.

---

## Density

Use existing `text-[10px]` / `border-border` patterns so toolbar and bottom bar stay compact
and scannable; minimap is a fixed footprint in the lower-left of the graph canvas host.

---

## E2E contract

[E2E surface map](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md) --
graph shell row.

---

## Revision history

| Date       | Change                                                                                              |
| ---------- | --------------------------------------------------------------------------------------------------- |
| 2026-04-21 | Graph chrome: stats strip (counts + **Gestures** when full graph); lens **centre**; search chip.    |
| 2026-04-21 | Graph chrome: Types + **⚙** popover; bottom bar (minimap, re-layout, zoom, PNG, collapse).          |
| 2026-04-21 | 100%: `zoom(1)` then `center(:visible)`; clears debounced zoom-out recenter.                        |
| 2026-04-21 | Camera: hand-off + single-tap share focus anim; min zoom 1.3×; stable episode dim.                  |
| 2026-04-20 | Episode cap: scored pick (recency+cluster); store=graphExplorer; **Reset** + E2E ref.               |
| 2026-04-19 | Default graph load: graphLens, cap, stats strip; Q/S/E off (VIEWER_GRAPH_SPEC).                     |
| 2026-04-19 | RFC-076: loadSelected clears graphExpansion; preserveExpansion on append/remove.                    |
| 2026-04-10 | Initial content (in UXS-001)                                                                        |
| 2026-04-13 | Extracted from UXS-001 into standalone UXS-004                                                      |
| 2026-04-15 | Quote node: 0ms audio timing explained (#543)                                                       |
| 2026-04-15 | In-app transcript dialog + segments timeline (#546)                                                 |
| 2026-04-15 | Transcript rail: button-only; passage + raw link in dialog header                                   |
| 2026-04-15 | Quote rail + Explore: muted #541 hint (shared copy, UXS-005)                                        |
| 2026-04-15 | Topic node detail: full label + copy block; optional aliases line                                   |
| 2026-04-15 | Insight full-text + copy; no header title for Quote/Topic/Insight                                   |
| 2026-04-15 | Quote rail #541: **`GI_QUOTE_SPEAKER_UNAVAILABLE_HINT`** (same as Search / Explore)                 |
| 2026-04-16 | Quote node detail UXS: markdown cleanup + #541 hint layout; Explore E2E ref in panel section        |
| 2026-04-15 | Topic node detail: Search prefill + Explore topic-filter gateway row                                |
| 2026-04-15 | Topic node: CIL **Topic timeline** modal (#548), `cil_queries_api` shell flag                       |
| 2026-04-15 | Topic/cluster timeline: flat blocks, middot lines; date sort (newest default)                       |
| 2026-04-15 | Insight node: meta strip, Search/Explore handoffs, supporting quotes, source episode focus          |
| 2026-04-16 | Insight gateway: shorter labels; items-center with Topic rows; Episode on graph + aria-label        |
| 2026-04-16 | Topic/cluster timeline legend: **Where we looked:** / **How to read:** (two lines); library wording |
| 2026-04-16 | CIL timeline dialog: single-mode **h2** from graph node type; **Cluster timeline** unchanged        |
| 2026-04-16 | Topic timeline modal: legend in header HelpTip; teleport to dialog for showModal                    |
| 2026-04-16 | Topic cluster: per-row Timeline opens single-topic CIL (not merged cluster modal)                   |
| 2026-04-16 | Graph node rail: **Details** / **Neighbourhood** tabs; connections on Neighbourhood tab only        |
| 2026-04-16 | Graph node rail: **Graph** tab only; stash node id; restore **Details** on return                   |
| 2026-04-16 | NodeDetail: type + entity_kind row; top of Details (`node-detail-kind-row`)                         |
| 2026-04-15 | Insight detail: provenance, related topics, Library handoff, quote sort + collapse                  |
| 2026-04-15 | Insight: grounding + lineage in **i** HelpTip; related topics scroll + row opens topic on graph     |
| 2026-04-15 | Connections L/G/S on neighbors; insight rail drops episode graph + Library buttons                  |
| 2026-04-17 | Graph styling: WIP tiers, edges, dim, zoom label classes, Phase 2 accents                           |
| 2026-04-17 | UXS: default **side** labels; neighbour ~85%; minimap tier sync + reduced motion                    |
