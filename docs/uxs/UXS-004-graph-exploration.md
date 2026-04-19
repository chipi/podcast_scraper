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
  - `web/gi-kg-viewer/src/stores/graphExplorer.ts`
  - `web/gi-kg-viewer/src/stores/graphFilters.ts`
  - `web/gi-kg-viewer/src/stores/graphNavigation.ts`
  - `web/gi-kg-viewer/src/stores/graphExpansion.ts` (cross-episode expand seed → appended paths)

---

## Summary

The Graph tab provides a Cytoscape-powered interactive graph canvas for exploring
merged GI/KG artifacts. When **`GET /api/corpus/topic-clusters`** returns clustering JSON,
the graph adds **TopicCluster** parent nodes (dashed compound outline) and sets
Cytoscape **`parent`** on member **Topic** nodes whose bare id matches **`topic:…`** members.
Payload **v2** uses **`graph_compound_parent_id`** (`tc:`…) for those parents; **`cil_alias_target_topic_id`**
is for CIL aliases only, not for graph ids.
This UXS defines the visual contract for graph chrome:
toolbar, canvas overlays, minimap, degree filter, node detail rail, and
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
[`docs/wip/wip-rfc-075-open-questions-followup.md`](../wip/wip-rfc-075-open-questions-followup.md).

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

---

## Graph visual styling (Cytoscape)

Normative numbers and formulas live in [`docs/wip/GRAPH-VISUAL-STYLING.md`](../wip/GRAPH-VISUAL-STYLING.md) (implementation spec; tracks [GitHub #608](https://github.com/chipi/podcast_scraper/issues/608)). This subsection states the **user-visible contract** only.

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

A **one-time, dismissible** overlay on the graph canvas teaches gestures and ring colours the first time the user sees a **non-empty** merged graph in this browser profile, unless they already dismissed it (`localStorage` key **`ps_graph_hints_seen`** set to **`1`**). Persistence is **per browser profile** (until storage is cleared), not per browser session. Full visual and interaction spec: [`docs/wip/GRAPH-GESTURE-OVERLAY.md`](../wip/GRAPH-GESTURE-OVERLAY.md).

**Mounting context:** [`App.vue`](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/src/App.vue) mounts [`GraphTabPanel`](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/src/components/graph/GraphTabPanel.vue) only when the **Graph** main tab is active (`v-if="mainTab === 'graph'"`), and [`GraphCanvas`](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/src/components/graph/GraphCanvas.vue) mounts only when artifacts display — the overlay does not need a separate `mainTab` prop unless that shell layout changes.

**Dismiss:** primary control **Got it** (`data-testid="graph-gesture-overlay-dismiss"`), backdrop click outside the card (`@click.self` on the overlay root; card uses `@click.stop`), and **Escape** via a `window` `keydown` listener (capture) that ignores **Escape** when focus is outside the overlay (e.g. layout combobox or bottom zoom toolbar) so graph chrome stays predictable.

**Accessibility:** `role="dialog"`, `aria-modal="true"`, heading **Graph gestures** referenced with `aria-labelledby`, initial focus on **Got it** after open; on dismiss, focus returns to the graph canvas host for keyboard continuity.

**Surface map:** overlay root `data-testid="graph-gesture-overlay"`. Optional **Gestures** control in the bottom-right zoom cluster (`data-testid="graph-gesture-overlay-reopen"`) reopens the card **without** clearing `localStorage` (reminder after a quick dismiss).

---

## Toolbar (primary row)

When semantic search (or library-driven graph highlights) mark nodes on the graph, a compact **highlight count** chip may appear in this row (`text-[10px]`). There is **no** persistent line of Shift / double-click gesture copy here; gestures are covered by the **Gesture discovery overlay** and optional **Gestures** reopen control.

---

## Canvas overlay (bottom-right)

**Fit** (`primary`), zoom **-** / **+** / **100%** (100% = `zoom(1)`, pan unchanged),
**Gestures** (`data-testid="graph-gesture-overlay-reopen"`, opens the gesture discovery card without clearing `localStorage`), and **Export PNG** (rightmost, 2x full graph) in one `toolbar` with `aria-label`
"Graph fit, zoom, gestures, and export" (`role="toolbar"`); aria **Zoom out** / **Zoom in** on
**-** / **+**. Sits above the graph surface (`.graph-canvas`), not in the top chrome
row.

---

## Canvas overlay (upper-right)

`role="region"` `aria-label` "Graph layout, re-layout, and degree filter"
(`.graph-layout-controls`) -- tight vertical column (~6.75rem wide): full-width
**Re-layout**, **Layout** label stacked above select (cose, breadthfirst, circle,
grid; "Graph layout algorithm" combobox), **Degree** buckets in a 2-column grid,
compact **Clear** (degree) with `aria-label` "Clear degree filter" when active.
Node detail lives in the App **subject** column (`SubjectRail`), not over the canvas.

---

## Toolbar (chrome below primary)

Optional **Sources** row first (merged GI / KG, **Hide ungrounded**, **filters
active** when relevant); **Minimap** checkbox row; then **Edges** and **Types**
(per-type checkboxes + all / none, swatches match node fills, counts). No separate
panel above the graph.

---

## Node detail (subject rail)

**Graph node subject rail (`GraphNodeRailPanel`, `NodeDetail` with `embed-in-rail`):** the
**`GraphNodeRailPanel`** mounts inside **`SubjectRail`** whenever **`subject.kind === 'graph-node'`**
(including when the main tab is **Digest**, **Library**, or **Dashboard**). Prefill / Explore handoffs
switch the **left** rail to the **Search** tab and fill **Search** or **Explore** filters; they do **not**
replace the right column. A pill **tablist** sits under the node header — **Details** (default) vs **Neighbourhood** — **`data-testid`** **`node-detail-rail-tab-details`** / **`node-detail-rail-tab-neighbourhood`**. The scroll body shows either node content (**Details**) or **`graph-connections-section`** (**Neighbourhood**); the active tab resets to **Details** when the focused graph node id changes. If the center id is not in the merged slice, **Neighbourhood** shows **`node-detail-rail-neighbourhood-unavailable`** instead of an empty panel. Graph **Type** plus **`entity_kind`**-style labels (**person**, **organization**, …) appear at the top of **Details** (**`node-detail-kind-row`**), not under the **h3** title. Insight **L** / **G** / **S** on graph neighbors (see long **Insight** paragraph below) live under **Neighbourhood**, not **Details**.

The header primary title (**h3**) expands vertically to fit the full label (`whitespace-pre-wrap`, `select-text`; no `line-clamp`). **Quote**, **Topic**, **Insight**, **Person**, **Speaker**, and **Entity** nodes set **`data-testid`** on that span (**`node-detail-full-quote`**, **`node-detail-full-topic`**, **`node-detail-full-insight`**, **`node-detail-full-person-entity`**) and expose a **`C`** chip stacked **under** **`?`** (**`node-detail-full-*-copy`**, same square footprint as **E** / **`?`**; native tooltip and **`aria-label`** **Copy title**, then **Copied to clipboard** / **Copy failed; try again** briefly). Other node types keep a native **`title`** on the span for hover. **Topic** optional **Aliases:** line (**`data-testid="node-detail-topic-aliases"`**) when **`properties.aliases`** is a non-empty string array (GI schema); aliases are omitted from the generic property list for topics. Below the header (above neighborhood / **Where this appears**), a compact gateway row mirrors Episode handoffs: **`data-testid="node-detail-topic-prefill-search"`** **Prefill semantic search** opens **Search** with the topic label as query (no feed filter); **`data-testid="node-detail-topic-explore-filter"`** **Set Explore topic filter** opens **Explore** with **Topic contains** filled and clears prior explore output — the user still runs **Run explore**. Both are disabled when corpus/API health is not OK. Below that row, **Topic timeline** (**`data-testid="node-detail-topic-timeline"`**) opens a native **`dialog`** (**`data-testid="topic-timeline-dialog"`**) with the same modal shell as **Transcript** (`w-[min(100%,42rem)]`, `max-h-[min(92vh,48rem)]`, fixed header + scrollable body): it calls **`GET /api/topics/{topic_id}/timeline`** (CIL / RFC-072) using the graph node **id** and the resolved corpus path. Inner hooks include **`topic-timeline-loading`**, **`topic-timeline-error`**, **`topic-timeline-empty`**, **`topic-timeline-episodes`**. Disabled when health is not OK, **`cil_queries_api`** is false, or the corpus path is unset. **Person** / **Entity** scroll body: optional **`data-testid="node-detail-person-entity-role"`** (**SPOKEN_BY** / **SPOKE_IN** counts in the loaded slice), optional **`data-testid="node-detail-person-entity-aliases"`**, and (**`data-testid="node-detail-person-entity-prefill-search"`**, **`data-testid="node-detail-person-entity-explore-filter"`**) — compact labels **Speaker filter** / **Topic filter** (maps to **Speaker contains** vs **Topic contains**; see **`aria-label`** and **`?`** HelpTip). **`data-testid="node-detail-insight-details-tip"`** wraps a text-style **HelpTip** in the insight scroll body (below the header): the trigger is the underlined label **`Grounded`**, **`Not grounded`**, or **`Extraction details`** (when **`grounded`** is missing but lineage or other fields still populate the panel); **`aria-label`** matches that label. The panel holds a short **Grounding** explainer (**Grounded** = at least one supporting quote linked via **SUPPORTED_BY** in this GI; **Not grounded** = the artifact sets grounded to false with no such quotes), optional **Other fields** (**Type** / **Position in episode** / **Confidence**), and **Lineage** (**`model_version`**, **`prompt_version`**, optional **`extraction.extracted_at`**, artifact name). **`insight_type`**, **`position_hint`**, **`confidence`**, and **`grounded`** stay out of the generic property list when surfaced here. A gateway row (**`data-testid="node-detail-insight-prefill-search"`**, **`data-testid="node-detail-insight-explore-filters"`**) mirrors Topic handoffs: **Prefill semantic search** uses a truncated insight string for the vector index; **Set Explore filters** opens **Explore** with **Topic contains** and **Speaker contains** cleared, **Grounded only** and optional **Min confidence** aligned to this node — the user still runs **Run explore**. **Episode on graph** and **Open source episode in Library** are not separate insight buttons; use **`Graph neighborhood and connections`** (**`data-testid="graph-connections-section"`**): for each neighbor, **L** (before **G**, **Episode** neighbors only) opens **Library** when metadata resolves (same rules as search result **L**); **G** focuses that neighbor on the graph; **S** (after **G**) prefills **Semantic search** with the neighbor’s primary text (truncated). **`data-testid="node-detail-insight-related-topics"`** (bordered region like Library **Similar episodes**) lists **Topic** neighbors via **`ABOUT`** / **`RELATED_TO`**; inner list host **`data-testid="node-detail-insight-related-topics-list"`** has no max-height or vertical scroll (the rail scrolls as a whole). Each row is one full-width control (**`node-detail-insight-related-topic-row`**): **click** focuses that **Topic** on the graph (same behavior as neighbor **G** on quotes). **`data-testid="node-detail-insight-supporting-quotes"`** lists **Quote** neighbors linked by **`SUPPORTED_BY`** (out from the insight), ordered by **`char_start`** then **`timestamp_start_ms`**, with truncated text and **G** to focus each quote; **`data-testid="node-detail-insight-view-transcript-all-quotes"`** (**Transcript (all quotes)**) opens the in-app transcript when every listed quote shares one **`transcript_ref`** and has finite **`char_start`** / **`char_end`**, highlighting all spans (the dialog may contain multiple **`transcript-viewer-highlight`** marks; **Passage** shows **N character spans (supporting quotes)** when **N** is greater than 1); more than five quotes collapse until **Show all N** (**`data-testid="node-detail-insight-supporting-quotes-toggle-expand"`**), reset when the selected node changes.
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

Use existing `text-[10px]` / `border-border` patterns so the extra row stays compact
and scannable; minimap is a fixed footprint in the lower-left of the graph canvas host.

---

## E2E contract

[E2E surface map](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md) --
graph shell row.

---

## Revision history

| Date       | Change                                                                                              |
| ---------- | --------------------------------------------------------------------------------------------------- |
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
