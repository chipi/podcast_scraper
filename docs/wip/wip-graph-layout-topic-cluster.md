# Graph layout: TopicCluster footprint and neighborhood (plan notes)

**Status:** Cluster compaction tuned (section 1); minimap layout still pending (section 2).

## 1. Cluster compound: tighter footprint on main canvas

**Issue:** For 2-4 nodes inside a TopicCluster compound, the cluster **region** can occupy
~1/4 of the whole graph -- poor use of space and visually "pulls" the layout.

**Applied (cyCoseLayoutOptions.ts):**

- **Member repulsion** cut from 420k to 180k (main) / 57k to 24k (compact) -- members
  pack much tighter inside the compound.
- **Intra-cluster ideal edge length** cut from 58 to 36 (main) / 32 to 20 (compact) --
  connected members pull closer together.
- **Gravity** raised from 0.15 to 0.18 (main) / 0.28 to 0.32 (compact) -- slightly
  stronger pull toward center reduces overall sprawl.
- **Nesting factor** raised from 1.38 to 1.52 -- cross-boundary edges stretch more
  relative to intra-cluster edges, keeping external nodes further from the compound
  while internals stay tight.
- **numIter** set to 2500 (explicit) -- ensures the simulation converges well at the
  lower repulsion values.
- Compound **padding** (`cyGraphStylesheet.ts`) left at 6px/3px -- already minimal; the
  issue was internal node spacing, not border padding.

## 2. Neighborhood minimap: 2D layout, not a single line

**Issue:** In the **local neighborhood** preview, nodes feel **all on one horizontal
line** -- too tight 1D; want a **normal COSE** (or equivalent) feel and **more 2D** spread.

**Current code:** `web/gi-kg-viewer/src/components/graph/GraphNeighborhoodMiniMap.vue`
uses **`breadthfirst`** with `directed: true` and `roots` -- that tends to produce
**tree / strip** layouts (often one row or column).

**Direction:**

- Switch minimap layout to **`cose`** (align with main graph's `layoutOptionsFor('cose')`
  in `web/gi-kg-viewer/src/components/graph/GraphCanvas.vue`) or **`fcose`** if available
  in the bundled cytoscape build -- tune `nodeRepulsion`, `idealEdgeLength`, `gravity`
  for **small** element counts so the preview is 2D without huge spread.
- Keep `fit` after layout; preserve selection/highlight behavior.
- Re-check TopicCluster neighborhood path (`topicClusterNeighborhood` prop) vs generic
  ego slice so both look reasonable.

## 3. Relationship to sibling-episode auto-load

Independent: episode merge behavior is data volume; this doc is **pure
layout/presentation**. They can ship in either order.

## 4. UX / E2E

When layout strings or minimap behavior become E2E-visible, update
`web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md` and the relevant UXS if the visual contract
changes.
