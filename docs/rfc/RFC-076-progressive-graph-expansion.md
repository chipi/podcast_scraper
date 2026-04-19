# RFC-076: Progressive Graph Expansion (Cross-Episode Explore)

- **Status**: Completed (v2.6.0)
- **Tracking**: [GitHub #581](https://github.com/chipi/podcast_scraper/issues/581)
- **Related RFCs**:
  - [RFC-062](RFC-062-gi-kg-viewer-v2.md) (viewer stack)
  - [RFC-069](RFC-069-graph-exploration-toolkit.md) (graph chrome, Shift+`dbltap` / 1-hop ego)
  - [RFC-072](RFC-072-canonical-identity-layer-cross-layer-bridge.md) (bridge identities)
  - [RFC-075](RFC-075-corpus-topic-clustering.md) (`topic_clusters.json` is separate from bridge truth)
- **Related UXS**:
  - [UXS-004](../uxs/UXS-004-graph-exploration.md)
  - [UXS-001](../uxs/UXS-001-gi-kg-viewer.md) (shell)

## Abstract

This RFC specifies **progressive cross-episode graph expansion** in the GI/KG viewer: the graph uses **single activation** (**`onetap`**) to open the rail detail and **double-activation** (**`dbltap`**; mouse: double-click, touch: double-tap) to load other episodes that share a canonical bridge identity (Topic / Person / Entity). A **second** double-activation on the same seed **collapses** that expansion by unloading the artifacts that were appended. **Shift** during double-activation (Shift+**`dbltap`**) keeps the existing 1-hop ego neighborhood toggle from RFC-069 (same event names as in **Goals** below).

The server exposes **`POST /api/corpus/node-episodes`**, implemented by scanning `*.bridge.json` files under the corpus (bridge-only reads for performance) and returning corpus-relative GI/KG paths. The viewer reuses **`appendRelativeArtifacts`** / **`loadSelected`** so merge, layout, and deduplication stay unchanged.

## Problem

The Graph tab merges loaded GI/KG files but has no affordance to discover **which other episodes** reference the same `topic:…`, `person:…`, or `org:…` identity from the canvas. Users must use the Library or other tools instead of graph-native exploration.

## Goals

1. **Click model**: `onetap` on a node opens the episode or graph-node rail (debounced so `dbltap` expand does not open the rail first). `dbltap` (without Shift) runs expand or collapse. Shift+`dbltap` keeps ego behavior.
2. **Server**: `POST /api/corpus/node-episodes` returns all matching episodes by default, optional `max_episodes` cap with `truncated` + `total_matched` in the response.
3. **Viewer**: Expand appends artifact paths; collapse removes exactly the paths recorded for that seed and reloads the graph.
4. **Truncation UX**: A second line under the graph tab (`data-testid="graph-expansion-truncation-line"`) communicates caps, errors, or empty results.

## Non-goals

- PostgreSQL or any DB-backed index (future work; filesystem-only v1).
- Visual hide of merged nodes without reload (merge pipeline does not preserve per-file node IDs after unify).
- Semantic (FAISS) search for identity membership (wrong semantics).
- Expanding **Episode** nodes when they are already fully loaded (v1 no-op).

## Security

Same pattern as existing CIL corpus routes: `root_path` is user-influenced; **`anchor_path`** is the server `output_dir`. `episodes_for_bridge_node_id` / `iter_cil_bridge_bundles` only walk under `anchor_path` and filter by `root_path` prefix (see `cil_queries.py`).

## API

**`POST /api/corpus/node-episodes`**

Request JSON:

- `path` (optional): corpus root; omit when server default output_dir applies.
- `node_id` (required): viewer id or canonical id; normalized with `canonical_cil_entity_id`.
- `max_episodes` (optional): positive integer; omit or null for all matches. Results sorted by `gi_relative_path` then truncated.

Response JSON:

- `path`: resolved corpus root string.
- `node_id`: canonical id used for matching.
- `episodes`: list of `{ gi_relative_path, kg_relative_path, bridge_relative_path, episode_id }`.
- `truncated`: boolean.
- `total_matched`: integer when `truncated` is true (pre-cap count); otherwise null.

## Viewer behavior

### Eligibility (expand)

Expand runs only when:

- The node is not an **Episode** rail shape (episode nodes: v1 no-op).
- `RawGraphNode.type` is **Topic**, **Person**, or **Entity**.
- After stripping layer prefixes, the id matches `^(person|org|topic):`.
- Cytoscape **degree** is greater than 1 (isolated nodes do not expand).

### Collapse

`graphExpansion` records `seedCyId -> addedRelPaths[]`. Collapse calls **`removeRelativeArtifacts`** on those paths and clears the record. Full graph reload matches expand cost (accepted for v1).

### Visual affordance (viewer)

Topic / Person / Entity nodes that pass the same structural rules as **Eligibility (expand)** get a **teal** Cytoscape node border **only after** a debounced `POST /api/corpus/node-episodes` probe finds at least one matching episode whose GI/KG paths are **not** already in the viewer’s merged artifact selection (so the ring means “double-click can merge corpus material that is not on the graph yet,” not merely “degree &gt; 1”). The expansion **seed** after a successful expand shows a **blue** border until collapsed. Styling uses classes `graph-expand-eligible` / `graph-expand-seed`; the graph toolbar hint summarizes the rings.

### Performance

Server cost is **O(number of bridge files)** with one JSON read per bridge. Viewer cost is dominated by **`loadSelected`** over all selected artifacts after each append chunk.

## Testing

- Python unit: `tests/unit/podcast_scraper/server/test_cil_queries_node_episodes.py`
- Python integration: `tests/integration/server/test_corpus_node_episodes_integration.py`
- Viewer Vitest: `corpusLibraryApi.test.ts` for `fetchNodeEpisodes`
- Playwright (GI/KG viewer): `web/gi-kg-viewer/e2e/graph-expansion-mocks.spec.ts` — drives **`dbltap`** expand/collapse by sending **two quick click cycles** on `.graph-canvas` at the node’s rendered position (same Cytoscape input path as a user **double-click** / **double-tap**); it does not call `cy.emit` from test code. Expand **`POST`** bodies are expected to include **`max_episodes`** set to the viewer constant **`GRAPH_NODE_EPISODES_EXPAND_MAX`** (same value as Vitest `corpusLibraryApi` tests) on every mocked expand path that receives a request.

## Resolved decisions

| Topic | Decision |
| --- | --- |
| Rail vs expand gesture | `onetap` opens rail; `dbltap` expands/collapses (mouse double-click / touch double-tap) |
| Collapse semantics | Unload appended artifact paths; not CSS hide |
| Server scan | `iter_cil_bridge_bundles` (bridge-only reads) |
| Episode `dbltap` | No-op for v1 |

## Future work

- Inverted index or DB (issue #40) for sub-linear lookups.
- Incremental graph merge without clearing `parsedList` to remove reload flash.
- Episode expand when only partial artifacts are loaded.
