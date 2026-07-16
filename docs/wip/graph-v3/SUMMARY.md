# graph-v3 — visual iteration on gi-kg-viewer

*Working branch: `feat/graph-v3` off `main`. Not merged.*

## Trigger

Operator delivered four analyses (graph-visual-improvement, gikgviewergraphredesign,
graphfilterrail, graphvisualanalysis) comparing the viewer against `codeindex`,
Maltego, InfraNodus, Obsidian Graph, Palantir Gotham, and the 2025 Graph Drawing
contest *Dark* visualisation. Ask: honest read against current state, then start
with easy wins, screenshots before/after, perf assessment per change.

## Read against current state

Half of what the docs flagged as "critical missing" was already in
`buildGiKgCyStylesheet` — either shipped or behind an opt-in flag:

- Edge colour by relationship type — fully wired for
  HAS_INSIGHT / ABOUT / SUPPORTED_BY / RELATED_TO / MENTIONS / SPOKE_IN /
  SPOKEN_BY / HAS_MEMBER / aggregates.
- Confidence encoding — `insight-ungrounded` dashed border, ABOUT edge opacity
  mapped to confidence 0.25–1.0.
- Temporal fade — `recencyWeight → opacity` (floor 0.4).
- Label zoom-gating — `graph-label-tier-{none,short,full}` classes.
- Node size by degree — `enableNodeSizeByDegree` option existed, off by default.
- fcose — default layout, chosen for perf (#967) over cose.
- Timeline layout — implemented as `timelineLayoutSpec` (RFC-080 V3).

Genuine gaps: darker canvas, default edge opacity, degree sizing default-off,
hover-dim, community detection, bridge betweenness ring, brand palette repaint.

## Shipped in A/B/C/D

| # | Change | Files | Notes |
|---|---|---|---|
| A | `--ps-graph-canvas` token (dark `#0a0d10`, light falls through to `--ps-canvas`) | `theme/tokens.css`, `components/graph/GraphCanvas.vue` scoped style | Shell chrome, `theme.spec.ts` assertions, and `canvasExportBg` untouched. Halo mismatch deferred. |
| B | Default edge opacity theme-aware — `0.3` dark, `0.5` light | `utils/cyGraphStylesheet.ts` | `isLightThemeActive()` reads `data-theme`, falls back to `matchMedia`. Neighbour `0.9` / dimmed `0.2` preserved. |
| C | `DEFAULT_FLAGS.nodeSizeByDegree = true` | `stores/graphLenses.ts` + tests + module docstring | Users retain explicit toggle via localStorage; only "never touched" default flips. |
| C-fix | `mapData(degreeHeat)` scoped to `[type][degreeHeat]` selector | `utils/cyGraphStylesheet.ts` + tests | Silences per-element Cytoscape warnings on Episode nodes without `degreeHeat`. Nodes without the field fall through to fixed base size. |
| D | Hover-dim wiring | `components/graph/GraphCanvas.vue` | `cy.on('mouseover'/'mouseout', 'node')` reuses `applyGraphSelectionDimFromNode` / `clearGraphSelectionDim`. Guarded by `:selected` count so selection dim wins over hover. |

## Perf

- LCP on prod-v2 (833 nodes, DPR-2, 1440×900) after A+B+C: **1561ms**. Trace at
  `traces/03-C-first-paint.json.json.gz`.
- No pre-C LCP baseline captured — regression judged by feel; layout settled
  visibly at the same pace as pre-change.
- Hover-dim adds a single-frame class toggle; existing 120ms opacity
  transition on nodes + edges smooths cursor sliding between neighbours.

## Theme coverage

Both themes visually verified against prod-v2:

- `screenshots/00-baseline-idle.png` → `03-ABC-size-by-degree-idle.png`: dark theme evolution.
- `screenshots/04-ABC-light-theme-idle.png`: light theme after A+B+C.
- `screenshots/05-ABCD-dark-idle.png` / `06-ABCD-dark-hover.png`: dark hover.
- `screenshots/07-ABCD-light-hover.png`: light hover — 0.5→0.9 lift (1.8×) reads.

## Tests

`web/gi-kg-viewer` vitest — 37 tests pass (`graphLenses` + `cyGraphStylesheet`
suites both updated to reflect the new default and the `[type][degreeHeat]`
specialization selector). `theme.spec.ts` (e2e) untouched — `--ps-canvas`
value preserved.

## Not done

- Halo colour still resolves `--ps-canvas` (not `--ps-graph-canvas`); ~3 shade
  mismatch behind labels on the darker bg.
- `canvasExportBg()` still reads `--ps-canvas`; PNG export bg diverges from
  on-screen bg on dark theme.
- No perf trace for baseline — the C trace is the only reference.
- fcose repulsion / edge-length tuning unchanged; docs' `nodeRepulsion: 8000 /
  gravity: 0.08` proposals not applied (units differ from current 880_000 base;
  needs an isolated tuning session on real corpora).

## Open questions carried forward

1. Direction: Maltego / InfraNodus vs codeindex ceiling. Operator gut says
   Maltego / InfraNodus stepping-stone.
2. Left rail vs chip bar for Graph. Deferred — chip bar (#658) stays for now.
3. Next tier scope. Operator direction (2026-07-16): halos, sizes, circles and
   lines — move away from same-size circles and same-thickness lines. Colour +
   shape variance next, not community detection yet.

## Tier 2 — E/F/G/H

| # | Change | Files | Notes |
|---|---|---|---|
| E | Halo + PNG-export bg follow `--ps-graph-canvas` | `utils/cyGraphStylesheet.ts`, `components/graph/GraphCanvas.vue`, `PS_TOKEN_FALLBACKS` | Resolves A deferral. Verified: `text-outline` / `text-background` now `rgb(10,13,16)`. |
| F | Semantic shape per node type | `utils/cyGraphStylesheet.ts` `shapeByType` | Podcast=hexagon, Episode=round-rectangle, Insight=diamond, Quote=round-tag, Speaker=round-diamond, Entity_organization=round-rectangle. Person / Entity_person / Topic stay ellipse. TopicCluster keeps `roundrectangle` compound. Speaker + Quote emit 0 in prod-v2 slice — code-verified only, live-verify pending on a corpus that emits them. |
| G | Edge width tiers widened | `utils/cyGraphStylesheet.ts` | Structural 2 → 2.75 (HAS_INSIGHT, SPOKE_IN, HAS_EPISODE). HAS_MEMBER 1.5 → 2. Evidence + discovery 1 → 0.75-0.85 (SUPPORTED_BY, RELATED_TO, MENTIONS). Descriptive confidence-mapped unchanged (already scaled). Compact profile scales proportionally. Aggregate mapData(weight) untouched. |
| H | Episode base 18 → 22, aspect W 1.35 → renders 30×22 card | `utils/cyGraphStylesheet.ts` `NODE_ASPECT_W`, `scaledNodeWidth` | mapData(degreeHeat) specialization scales width vs height independently against per-type bases so aspect ratio is preserved across the 0.7×–1.5× range. |

## Not done (still)

- Speaker + Quote shape live-verify — pending a corpus that emits either.
- fcose repulsion / edge-length tuning still unchanged. Docs' `nodeRepulsion: 8000 /
  gravity: 0.08` proposals not applied (units differ from current 880_000 base;
  needs isolated tuning session).
