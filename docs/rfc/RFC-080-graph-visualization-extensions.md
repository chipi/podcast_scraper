# RFC-080: Graph Visualization Extensions

**Status:** Draft (rewrite pass 1 — April 2026)
**Author:** Marko + Claude
**Repo:** `chipi/podcast_scraper`
**Related:** UXS-004, GRAPH-CHROME-REDESIGN, GRAPH-INITIAL-LOAD-V2,
RFC-062, RFC-075, RFC-076, #656 (confidence-weighted ABOUT edges)
**Stack:** Cytoscape 3.33.1, cytoscape-navigator 2.0.2

---

## Abstract

The GI/KG graph renders nodes and edges with basic visual differentiation
— colour by type, opacity by recency, border width by degree heat. Several
signals are already partially wired (`recencyWeight`, `confidenceOpacity`,
`degreeHeat`) but not fully exploited, and #656 Stage C just shipped
confidence-weighted per-edge styling on ABOUT edges
(`aboutConfidenceOpacity` / `aboutConfidenceWidth`) — the first data-driven
edge-style pass in the viewer.

This RFC proposes five visualization lenses that extend the canvas without
breaking its current behaviour. Each lens is a pure stylesheet or
layout-mode addition; none change the persistent graph schema. The user
switches lenses via the bottom bar; compatible lenses compose.

The RFC is structured as an implementation spec, not an exploration
exercise: each lens has an unambiguous contract, composes explicitly with
the other four, and carries a rollout toggle so a bad lens can be disabled
without a revert.

---

## What is already wired

Before specifying new lenses, the pre-existing data-driven styling:

| Signal | Data path | Currently drives | Scope |
|---|---|---|---|
| `recencyWeight` | `node.data('recencyWeight')` | Node opacity (0.4–1.0 decay) | All nodes |
| `confidenceOpacity` | `node.data('confidenceOpacity')` | Insight `background-opacity` | Insight nodes |
| `degreeHeat` | `node.data('degreeHeat')` | Topic `border-width` (1–4px) | Topic nodes |
| `properties.confidence` | `edge.data('properties').confidence` | ABOUT edge `line-opacity` + `width` (#656 Stage C via `aboutConfidenceOpacity` / `aboutConfidenceWidth`) | Insight→Topic ABOUT edges |
| `label` / `shortLabel` | `node.data(...)` | Node caption + zoom-tier short caption | All / tier nodes |

Edge types styled: `HAS_INSIGHT`, `ABOUT`, `SUPPORTED_BY`, `RELATED_TO`,
`MENTIONS`, `SPOKE_IN`, `HAS_MEMBER`.

Cytoscape classes already active: `graph-dimmed`, `graph-neighbour`,
`graph-focused`, `graph-expand-eligible`, `graph-expand-seed`,
`search-hit`, `graph-label-tier-none/short/full`, `graph-topic-heat-high`.

Every new lens below either (a) introduces a new data attribute, (b) adds
a new class namespace, or (c) adds a new layout — and must not clobber
any of the above.

---

## Graph schema recap (resolving V1 ambiguity)

**There is no Episode→Topic ABOUT edge in the persisted graph.** ABOUT
edges are always `Insight → Topic` (see `parsing.ts`, #664 cosine
confidence lives on these). An Episode "is about" a Topic only indirectly
— via one or more Insights the Episode emitted that each have their own
ABOUT edge to that Topic.

V1 below therefore **synthesises a new aggregated edge at render time**
(`Episode → Topic` with a derived `weight`). It is not persisted, not
emitted by the backend, and does not replace the existing per-Insight
ABOUT edges. Section V1 × Stage C specifies exactly when the aggregated
vs. per-edge ABOUT edges are visible.

---

## Visualization 1 — Aggregated Episode↔Topic edge by discussion depth

### What it is

Today, the graph shows a fan of Insight→Topic ABOUT edges. Each carries
Stage C confidence styling. But to answer *"which Episode covers this
Topic most deeply?"*, the user has to eyeball the per-Insight fan.

V1 introduces a new render-only edge class: `ABOUT_AGG` from
Episode → Topic, styled by `weight` = *count of Insights from that
Episode that have an ABOUT edge to that Topic*. Range 1..N, normalised
against the max weight in the current graph slice.

`ABOUT_AGG` is **not a schema edge.** It exists only in the Cytoscape
element set, built by `toCytoElements` alongside the real edges. It
carries a discriminator class `graph-edge-about-agg` so stylesheet
selectors never confuse it with real ABOUT edges.

Same shape for Episode↔Person (`SPOKE_IN_AGG`, weight = count of Quotes
from that Episode attributed to that Person).

### Composition with Stage C (per-Insight ABOUT)

Two edges can connect Episode A to Topic T conceptually:

- The aggregated `ABOUT_AGG` edge (Episode A → Topic T, weight = 12)
- Twelve per-Insight ABOUT edges (Insight i_k → Topic T, each with
  cosine confidence)

Default render mode: **only one of them is visible at a time**, driven
by the user's current zoom tier (`graph-label-tier-*`):

- `tier-none` / `tier-short` (zoomed out) → render `ABOUT_AGG` only;
  hide per-Insight ABOUT. The aggregate is the readable summary.
- `tier-full` (zoomed in) → render per-Insight ABOUT (Stage C styling);
  hide `ABOUT_AGG`. The user wants detail.

This is controlled by a single stylesheet display rule on each edge
class; no custom event handlers. Users can force-show both via a bottom-
bar toggle ("Show aggregate edges at all zoom levels") for power users.

### Width computation

```
weight = count of (Insight, Topic) pairs where
           Insight.episode_id == Episode.id AND
           an ABOUT edge exists Insight → Topic

scaled_width = mapData(weight, 1, maxWeightInSlice, 1.5, 5)
```

`maxWeightInSlice` is recomputed every time the element set rebuilds
(on graph load, RFC-076 expand, "Load into graph"). Normalisation is
**relative to the current slice**, not global — see *Open question:
stable vs. relative normalisation*.

### Data contract

```ts
edge.data = {
  id: `about_agg:${episodeId}->${topicId}`,
  source: episodeId,
  target: topicId,
  label: 'ABOUT_AGG',
  weight: number,  // 1..N
  contributingInsightIds: string[],  // for rail drill-in
}
```

A click on an `ABOUT_AGG` edge opens the Insight rail filtered to
`contributingInsightIds` — same UX as clicking a Topic node, just
pre-filtered.

### Implementation path

Native Cytoscape. No extension. Stylesheet adds selector
`edge.graph-edge-about-agg` with `mapData(weight, …)`. Element build
runs a new `aggregateEpisodeTopicEdges(merged)` step after the normal
merge in `toCytoElements`.

### Use case

Researcher loads 30 episodes. Zoomed out, they see fat Episode→Topic
edges where a podcast ran a 4-episode deep-dive on "AI regulation" —
each of those 4 Episodes has a thick `ABOUT_AGG` edge to "AI regulation".
Other episodes with passing mentions show thin edges. Zooming in, the
aggregated edges fade and the per-Insight fan appears, with Stage C
confidence styling showing *how certain* each Insight's claim about the
Topic was.

### Conflicts

- `degreeHeat` (Topic border width) — orthogonal; on nodes, not edges.
- Stage C `aboutConfidence*` — disjoint (they style per-Insight ABOUT;
  V1 styles aggregated ABOUT_AGG). Both exist in the stylesheet but
  target different selectors.

---

## Visualization 2 — Insight-node grounding + confidence tiers

### What is actually new here (redundancy note)

`confidenceOpacity` is already wired on Insight nodes and already drives
`background-opacity`. #656 Stage C added edge-side confidence styling
on ABOUT edges. V2 is **only two small additions** on Insight nodes:

1. **Ungrounded dashed border** — visually distinguish `grounded: false`
   beyond just "more transparent". Currently an ungrounded Insight just
   looks faint; V2 makes it explicit.
2. **Confidence tier classes** applied consistently at build time so
   stylesheet selectors can target tiers directly rather than everyone
   reading raw `confidenceOpacity`.

Everything else (the opacity scaling itself, the `confidenceOpacity`
data attribute) already exists. V2 is a completion, not a new signal.

### Implementation path

Native Cytoscape. At `toCytoElements` build time, assign classes:

- `insight-confidence-high` — confidence ≥ 0.7
- `insight-confidence-medium` — 0.4 ≤ confidence < 0.7
- `insight-confidence-low` — confidence < 0.4
- `insight-ungrounded` — `grounded === false` regardless of confidence

Stylesheet:

```css
node[type = "Insight"].insight-ungrounded {
  border-width: 1.5;
  border-style: dashed;
  border-color: var(--ps-warning);
  border-opacity: 0.7;
}
```

Tier classes are selector hooks; they don't override `confidenceOpacity`
(which keeps driving `background-opacity` as today). Future
visualizations can target tiers directly (e.g., "hide all low-confidence
insights" becomes `cy.nodes('.insight-confidence-low').hide()`).

### Use case

Journalist building a story filters the graph to "pharmaceutical
pricing" Topic. 12 Insight nodes. Some bright-solid (high+grounded),
some dim-dashed (low+ungrounded). At a glance she can triage which to
read first — the core contract of the GIL is visible on the canvas
surface.

### Conflicts

- `confidenceOpacity` on the node — same attribute, different style
  output (opacity vs. border). Compose cleanly.
- `search-hit` class — its yellow border should visually dominate the
  ungrounded dashed border when a node is both searched and ungrounded.
  Specificity: `.search-hit` selector sorted after
  `.insight-ungrounded` in the stylesheet.

---

## Visualization 3 — Timeline layout (date axis)

### What it is

A new entry in the layout cycle: `cose → breadthfirst → circle → grid →
timeline → cose`. Episode nodes are positioned on a horizontal time axis
(x = publish date), Topic / Person / Insight / Quote nodes are
positioned by *weighted connection to Episodes*. The result is a
temporal map: "how did this discourse develop over time?".

### Implementation path

Native Cytoscape using `preset` layout (explicit `{x, y}` per node).
**No secondary force step** — `preset` is positional only; there is no
"run forces after `preset`". (The earlier draft described a force step;
that was internally inconsistent and is dropped.)

### X-axis: quantile mapping, not linear

A naïve linear map `t = (date - min) / (max - min)` collapses tight
early clusters when one late episode shifts `max`. For podcast corpora
this is the common case (a back-catalogue dump followed by a single
recent episode).

**Use quantile mapping instead:** sort episode dates; the i-th episode
of N total gets `x = canvasWidth * i / (N - 1)`. Dates are preserved
ordinally but canvas space is distributed evenly — no single outlier
can compress the rest.

If the user wants linear-date, a bottom-bar sub-toggle switches between
`quantile` (default) and `linear` axes. Quantile is the safe default;
linear is available for specific analysis.

### Episode positions

```ts
const episodes = episodeNodes.sort(byDate)
episodes.forEach((n, i) => {
  positions[n.id()] = {
    x: canvasWidth * i / Math.max(1, episodes.length - 1),
    y: canvasMidY + jitterForCollision(n, i),  // -40..+40
  }
})
```

Y-jitter is deterministic (hash of node id) so the layout is stable
across re-runs.

### Topic / Person positions

After Episode x-positions are fixed, compute:

```ts
topic.x = mean(connectedEpisodes.map(e => positions[e.id()].x))
topic.y = canvasMidY - 100 + deterministicJitter(topic.id)
```

Persons: same, but y-position above the Episode band (`canvasMidY -
180`). Topics: below (`canvasMidY + 100`). Keeps bands visually
distinct.

### Insights / Quotes

Positioned near their parent Episode (small signed offset from Episode
position so children don't all land on the parent's exact (x, y)).

### Missing dates

Episodes with no `publishDate` cluster at a single parking spot
(leftmost-minus-60px, y = canvasMidY) with a muted marker. **Not**
"positioned at rightmost as if recent" — that's a silent lie. A bottom-
bar note reads: `N episodes have no date (parked at left)`.

### Data contract

```ts
episode.data('publishDate'): number  // Unix seconds (not ms), set at build
                                      // time from episode metadata.
                                      // null when missing.
```

`publishDate` is already in the corpus catalog API response; wire it
through `toCytoElements`.

### Performance

For N = 500 nodes the quantile sort + weighted-average pass is O(N log N)
+ O(E) where E is edge count. Well under 50ms for realistic corpora.
`preset` apply is O(N).

### Use case

Media analyst tracking coverage of a political event over 6 months,
30 episodes. Timeline layout shows: *data privacy* Topic clusters
left (peaked 3 months ago), *AI regulation* clusters right (recent),
one Person sits dead-centre (appeared consistently). Answering this
question was impossible in any other layout mode.

### Conflicts

- **V4 Radial mode:** mutually exclusive. Entering Radial mode while
  Timeline is active: Timeline's `preset` positions are saved for
  restore (see V4 *Storing state*); on Radial exit, Timeline is
  re-applied.
- **V5 node-size-by-degree:** composes cleanly — bigger Topics at
  their x-position is a good signal.
- **RFC-076 expand:** new nodes arriving mid-Timeline need re-layout.
  Trigger `timelineLayout()` on expand completion.

---

## Visualization 4 — Radial focus mode

### What it is

A mode (not a layout). Selected node sits at centre, 1-hop neighbours on
inner ring, 2-hop on outer ring, everything beyond hidden (not dimmed
— fully `display: none`).

Enter: double-click canvas background with a node selected, or the
"Radial" toggle in the bottom bar. Exit: `Escape` or toggle again — the
full graph is restored to its prior positions exactly.

### Implementation path

Native Cytoscape `preset` layout with mathematically computed radial
positions.

### Compound node handling (TopicCluster at centre)

If the selected centre is a compound node (TopicCluster with member
Topic children):

- Centre = the compound parent at (0, 0)
- Ring 1 = union of: compound children + external 1-hop neighbours
  of the compound itself + external 1-hop neighbours of any child
- Ring 2 = 2-hop from ring 1 (standard)

Compound members keep their parent-child relation in Cytoscape (they
render inside the compound bounding box if the compound is shown as a
hull, otherwise as standalone nodes). Ring 1 is sized to include both
the hull and the external neighbours — radius scales accordingly.

### Ring radius adapts to V5 node sizes

If V5 is active, nodes on the inner ring may be large (high-degree
Topics scale to 60px). Constant `r1 = 120` would overlap them.

```ts
const maxNodeRadius = Math.max(...ring1.map(n => n.width() / 2))
const r1 = Math.max(120, maxNodeRadius * 2.5)  // adaptive
const r2 = r1 * 2  // ring 2 always 2x inner
```

### Storing state for clean restore

```ts
// On enter:
const saved = new Map<string, {x: number, y: number}>()
cy.nodes().forEach(n => saved.set(n.id(), { ...n.position() }))
// plus save: current layout name, elements with display: none from
// other filters (search, dim), active classes per element.

// On exit:
restore positions + restore display + re-apply saved classes.
```

State is held in a store (pinia) keyed by graph-instance-id so Radial
exit after a graph reload falls back to "show everything" instead of
restoring stale positions.

### Interaction matrix (Radial × other lenses)

| Other lens | Interaction |
|---|---|
| RFC-076 expand | Disabled while Radial active (re-enabled on exit) — expand needs the full graph |
| V3 Timeline | Radial overrides; Timeline positions saved + restored on exit |
| V5 Node size | Composes — ring radius adapts (see above) |
| V1 ABOUT_AGG | Composes — aggregated edges visible on inner ring if zoom tier allows |
| V2 Confidence tiers | Composes — visible as normal |
| search-hit | Keeps glow; if the searched node is outside ring 2, it's hidden with the rest |

### Accessibility

- `aria-live="polite"` region announces "Radial view centred on {label}"
- `data-testid="graph-radial-mode-active"` on bottom bar when active
- Escape key exits regardless of focus location
- Focus is moved to the centre node on enter so screen reader announces
  it

### Use case

Busy graph: 15 episodes, 60 Topics, 200 Insights. User clicks *machine
learning* Topic → dims its neighbours. Still cluttered. Enters Radial
view: *machine learning* at centre, connected Episodes + Insights on
inner ring, their neighbours on outer ring. Hidden everything else.
Exit → full graph restored. The user explored without destroying their
context.

---

## Visualization 5 — Node size by degree (Topic + Episode)

### What it is

Replace the subtle `degreeHeat` border-width signal with **node size**.
Topic and Episode base sizes become `mapData` functions of `degreeHeat`
instead of fixed values.

```
Topic: mapData(degreeHeat, 0, 1, 28, 60)  // was fixed 40
Episode: mapData(degreeHeat, 0, 1, 14, 28)  // was fixed 18
```

Border-width heat stays as a secondary signal.

### Implementation path

Native Cytoscape. One stylesheet change. `degreeHeat` is already
computed and set at build time; this only changes how it's consumed.

### Performance implication (important)

Cytoscape's render cache keys off `width`/`height`. Making them
data-driven via `mapData` invalidates the cache on every `degreeHeat`
update. Since `degreeHeat` only changes on graph rebuild (not on pan/
zoom), this is **not** a per-frame cost — but it does mean the first
paint after a RFC-076 expand takes ~5–10% longer than before for large
graphs. Acceptable trade-off; documented here so future "graph feels
slower after V5" debugging starts in the right place.

### Label placement auto-adjusts

Current `cytoscapeSideLabelMarginXCallback` already reads `ele.width()`
dynamically. No change needed, but verify with real data at both
extremes (14px Episode and 60px Topic in the same graph).

### Use case

First-time user opens the graph. In 5 seconds — without reading a
single label — they see the 3 dominant Topics of the corpus as the
biggest nodes. "Executive summary at a glance."

### Conflicts

- V4 Radial: ring radii adapt (see V4 section).
- Search-hit glow: proportional to node size, so large searched nodes
  get a bigger glow radius. Already handled by the selector — verified.
- `graph-topic-heat-high` class (existing): currently adds a secondary
  visual accent on top-heat Topics. Keep; it stays meaningful even when
  size already signals degree.

---

## Interaction matrix (all lenses)

| Pair | Composes? | Notes |
|---|---|---|
| V1 + V2 | Yes | V1 styles edges, V2 styles Insight nodes — disjoint |
| V1 + V3 | Yes | ABOUT_AGG visible at zoomed-out tiers during Timeline |
| V1 + V4 | Yes | ABOUT_AGG shown if contributing edges fall in Radial window |
| V1 + V5 | Yes | Disjoint (edges vs. nodes) |
| V2 + V3 | Yes | Disjoint |
| V2 + V4 | Yes | Disjoint |
| V2 + V5 | Yes | Size and border compose |
| V3 + V4 | **Mutex** | V4 overrides V3; V3 restored on V4 exit |
| V3 + V5 | Yes | Biggest Topics at their x-position = strong signal |
| V4 + V5 | Yes | Ring radius adapts to node size |
| V1 + Stage C `aboutConfidence*` | Yes (tier-gated) | Only one edge class visible per zoom tier — see V1 × Stage C |
| Any + RFC-076 expand | Yes except V4 | V4 disables expand |
| Any + `search-hit` | Yes | Search highlight dominates visually |

---

## Rollout — per-lens feature flags

Each lens ships behind a flag in the viewer's feature-flags store
(pinia, persisted to `localStorage`). The bottom bar gains a lens menu
that toggles each independently. Default states at ship:

- V1 (ABOUT_AGG) — **off** initially. Schema-adjacent, needs corpus-
  validation on 3+ real corpora before default-on.
- V2 (Insight grounding + tiers) — **on**. Pure additive styling, low
  risk.
- V3 (Timeline layout) — **on** (available in layout cycle). Not the
  default layout; user opt-in per view.
- V4 (Radial mode) — **on** but only reachable via explicit toggle /
  double-click. Low risk because it's a mode, not a default layout.
- V5 (Node size by degree) — **off** initially. Visually invasive;
  A/B with `off` for one week on a staging corpus before default-on.

Flags are read at `toCytoElements` + stylesheet-build time. A bad lens
can be toggled off in session without reload.

---

## Performance budget

Target: **1000 nodes + 3000 edges**, no sustained regression beyond
current baseline.

| Lens | Build-time cost | Render cost |
|---|---|---|
| V1 | O(E) aggregation pass — ~30ms for E=3000 | Doubles edge count for aggregated pairs; offset by tier-gated display |
| V2 | O(N) class assignment — negligible | Same as today |
| V3 | O(N log N) sort + O(E) weighted mean — ~40ms for N=1000 | `preset` apply: O(N), same as any other layout |
| V4 | O(N) hide + O(ring1 + ring2) position — ~10ms | Fewer visible nodes = faster repaint |
| V5 | O(1) stylesheet change | Cache invalidation on rebuild only (see V5 perf note) |

First paint regression budget: +100ms vs. current baseline with all
lenses default-on (V1 is the biggest contributor at +~30ms when it
aggregates a dense corpus). Measured via existing
`PipelineStageChart`-style perf probe — add a `graph-build-ms` metric.

---

## Accessibility

- Each lens toggle in the bottom bar has an accessible name
  (`aria-label="Toggle Radial mode"`).
- Mode changes fire `aria-live="polite"` announcements:
  `"Radial view centred on {label}"`, `"Timeline layout applied"`,
  `"Aggregated edges shown"`.
- Lenses that hide content (V4) restore focus to the centre node so
  keyboard users don't lose their place.
- Contrast: ungrounded dashed border (V2) uses `var(--ps-warning)` —
  already passes WCAG AA in both light and dark tokens.

---

## Files to touch

```
web/gi-kg-viewer/src/utils/cyGraphStylesheet.ts
  — V1: edge.graph-edge-about-agg selector + mapData
  — V2: insight-ungrounded + confidence-tier selectors
  — V5: node width/height → mapData(degreeHeat)

web/gi-kg-viewer/src/utils/parsing.ts (or a new util)
  — V1: aggregateEpisodeTopicEdges() / aggregateEpisodePersonEdges()
  — V2: assignConfidenceTiers() + grounding class
  — V3: wire publishDate into Episode node.data

web/gi-kg-viewer/src/components/graph/GraphCanvas.vue
  — V3: timelineLayout()
  — V4: radialLayout() + enter/exit + state snapshot

web/gi-kg-viewer/src/components/graph/GraphBottomBar.vue
  — V3: Timeline entry in layout cycle
  — V4: Radial toggle + aria-live region
  — All: lens feature-flag menu

web/gi-kg-viewer/src/stores/graphLenses.ts (new)
  — Per-lens flags, persisted

web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md
  — graph-radial-mode-active, graph-timeline-layout-active,
    graph-lens-menu-*
```

---

## Implementation phases

The five lenses split cleanly into three shippable PRs:

### Phase 1 — Stylesheet extensions (low risk, high value)

V1 (Aggregated ABOUT_AGG / SPOKE_IN_AGG edges) + V2 (Insight grounding
+ tiers) + V5 (Node size by degree).

All stylesheet-or-aggregation-only. One PR. One corpus-validation
round before default-on decisions per V1/V5.

### Phase 2a — Radial focus mode

V4. Self-contained: mode toggle + preset positions + state snapshot +
aria-live. Does not require any Phase 1 change to land first, so can
ship in parallel.

### Phase 2b — Timeline layout

V3. New layout-cycle entry + `publishDate` wiring + quantile-mapped
positions + missing-date parking. Requires `publishDate` in
`toCytoElements` — touches parsing, so ships after Phase 1 to avoid a
merge headache.

Each phase is its own GitHub issue (see companion issue split).

---

## Open questions

**V1 weight normalisation — stable vs. slice-relative.** Slice-relative
(current spec) is most visually informative but edges visually "shrink"
when a new episode with a high-count Topic joins. Stable (pre-computed
max across full catalog) is less informative but doesn't flicker on
expand. Decide after first corpus-level validation.

**V3 y-band collisions.** Topics at similar weighted x will overlap
unless we y-jitter aggressively. Current deterministic jitter in ±40px
may not be enough at N=500. Alternative: cluster by x-bucket (30-day
bins), stack within bucket. Evaluate.

**V5 at small canvas sizes.** 60px Topic nodes fill a 400×400 mobile
viewport fast. Consider clamping the max mapData output to
`min(60, canvasWidth * 0.12)`.

**Label truncation at extreme sizes.** Existing side-label callback
reads `ele.width()`, but verify labels don't clip their own nodes at
the 14px end (Episode, min).

---

## Appendix — extension packages (not in scope)

The five lenses above are all native. Two extensions are worth a
separate spike after ship:

- **`cytoscape-fcose`** — drop-in replacement for `cose` with better
  cluster separation. Candidate for default layout.
- **`cytoscape-layers`** — enables a cluster-density heatmap
  background. Currently unimplementable.

Explicitly out of scope for this RFC; track in separate exploration
tickets.
