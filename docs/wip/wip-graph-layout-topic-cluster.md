# Graph layout: TopicCluster footprint and neighborhood (plan notes)

**Status:** Requirements from graph review; implement when scheduling viewer graph work.

## 1. Cluster compound: tighter footprint on main canvas

**Issue:** For 2–4 nodes inside a TopicCluster compound, the cluster **region** can occupy ~1/4 of the whole graph — poor use of space and visually “pulls” the layout.

**Direction:**

- Reduce **wasted** compound bbox: review Cytoscape compound **`padding`** for `node[type = "TopicCluster"]` in [`web/gi-kg-viewer/src/utils/cyGraphStylesheet.ts`](web/gi-kg-viewer/src/utils/cyGraphStylesheet.ts) (currently `8px` / `12px` in compact vs normal).
- Prefer **member topics laid out close** inside the compound: consider COSE **nested** / compound-aware tuning, or post-layout `fit` only the compound’s children, or lower `idealEdgeLength` **within** compounds if the layout engine supports subgraph options.
- Validate with small clusters (2–4 members) vs large graphs so the main graph does not over-expand compounds when many edges exist.

## 2. Neighborhood minimap: 2D layout, not a single line

**Issue:** In the **local neighborhood** preview, nodes feel **all on one horizontal line** — too tight 1D; want a **normal COSE** (or equivalent) feel and **more 2D** spread.

**Current code:** [`GraphNeighborhoodMiniMap.vue`](web/gi-kg-viewer/src/components/graph/GraphNeighborhoodMiniMap.vue) uses **`breadthfirst`** with `directed: true` and `roots` — that tends to produce **tree / strip** layouts (often one row or column).

**Direction:**

- Switch minimap layout to **`cose`** (align with main graph’s [`layoutOptionsFor('cose')`](web/gi-kg-viewer/src/components/graph/GraphCanvas.vue)) or **`fcose`** if available in the bundled cytoscape build — tune `nodeRepulsion`, `idealEdgeLength`, `gravity` for **small** element counts so the preview is 2D without huge spread.
- Keep `fit` after layout; preserve selection/highlight behavior.
- Re-check TopicCluster neighborhood path (`topicClusterNeighborhood` prop) vs generic ego slice so both look reasonable.

## 3. Relationship to sibling-episode auto-load

Independent: episode merge behavior is data volume; this doc is **pure layout/presentation**. They can ship in either order.

## 4. UX / E2E

When layout strings or minimap behavior become E2E-visible, update [`web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md`](web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md) and the relevant UXS if the visual contract changes.
