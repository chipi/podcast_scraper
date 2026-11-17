# ADR-183: The graph is read through one engine: a structured query and presets, returning one slice shape

- **Status**: Proposed (the database session, 2026-11-17)
- **Date**: 2026-11-17
- **Authors**: Marko Dragoljevic
- **Related**: [ADR-169](ADR-169-the-database-is-the-source-of-truth.md) (Postgres in every
  edition), [ADR-170](ADR-170-private-repo-split-platform-common-apps.md) (the split),
  [ADR-173](ADR-173-how-the-app-uses-database-connections.md) (connections),
  [ADR-178](ADR-178-ontology-v3-in-two-halves.md) (ontology v3),
  [ontology-v3.md](../architecture/corpus/ontology-v3.md), [ontology.md](../architecture/corpus/ontology.md)
  (the two-tier edge contract)

## Context & Problem Statement

The viewer builds its graph in the browser. It fetches each item's KG file (and any other graph
file a build adds) through `/api/artifacts`, merges them, and caps the merge because the browser
cannot afford more:

- 22 items by default, because about 1.5k nodes draw at about 37 fps and 2.9k at about 20 fps;
- up to 2,048 items appended in chunks of 12 when a node is expanded.

The server keeps its own in-memory graph per process (`search/corpus_graph.py`), which the
relational routes read.

After the corpus moves into the database (ADR-169), there are no graph files to fetch. A build of the
viewer, an agent and the app's own routes need one way to read the graph from the corpus tables. It
must be general enough for questions no screen has been written for, and bounded enough to run on
one shared Postgres.

## Decision

1. **One engine, two consumers.** One module of functions over the corpus tables reads the graph.
   - **The graph API** serves graph surfaces and agents: a structured query and named presets, all
     returning `graph-slice/1`.
   - **Preset routes** (the app's routes, and the relational routes) keep their own shapes. They call
     the engine's functions where they walk the graph and a plain indexed query where they do not.
   - No app route exposes traversal parameters.
   - Agents call the engine in process, not over HTTP.
2. **The engine reads every corpus table that holds graph data.** Items, entities, KG edges,
   insights, evidence and their edges, and groupings (clusters, storylines and the levels above
   them).
   - A build that writes no insights or groupings has empty tables, so its slice holds none: by
     data, not by code.
   - An extension can add a score to how `slice.recent` chooses items (`Extension.groupings`).
   - An extension can later add a layer from a table outside the corpus schema. None is built now.
3. **The structured query** (`POST /api/graph/query`) takes parameters, never a query language.
   - **`seeds`:** node ids (at most 50), item ids (at most 200), or one group id.
   - **`path`:** at most 3 steps, or a named path. A step is
     `{edges: [type, class or "*"], direction: out | in | any, types: [...]}`; an entry in `edges` may
     carry its own direction. `path: []` returns the seeds' own rows.
   - **`keep`:** the node types returned. `Quote` (evidence) only when named, because its text is
     heavy.
   - **`filters`:** a time window, sources, content types.
   - **`order`:** `nearness`, `recency` or `weight`.
   - **`budget`:** nodes, default 1,500, at most 3,000. **`edge_budget`:** default 10,000, at most
     20,000.
   - **`exclude_items`:** items the client already holds, at most 2,000.
   - **`with: counts`:** adds `item_count` per node.

   Every parameter is bounded. Beyond a bound, or for an unknown type, path or grouping kind, the
   answer is 422 in FastAPI's `{detail: [{loc, msg, type}]}`, with `loc` naming the field. An unknown
   seed is not an error: the answer is 200 with `unknown_seeds`.
4. **Named paths** are the engine's constants. They may be unions of step lists chosen by the seed's
   type, and may be longer than a client's 3 steps:

   | Name | Seed | Steps |
   | --- | --- | --- |
   | `node.items` | topic | `MENTIONS` out; `ABOUT` in, then `HAS_INSIGHT` in |
   | | person | `MENTIONS` out; `PARTICIPATES_IN` out; `MENTIONS_PERSON` in, then `HAS_INSIGHT` in; `ATTRIBUTED_TO` in, `SUPPORTED_BY` in, `HAS_INSIGHT` in |
   | | organisation | `MENTIONS` out; `MENTIONS_ORG` in, then `HAS_INSIGHT` in |
   | | object | `MENTIONS` out; `MENTIONS_OBJECT` in, then `HAS_INSIGHT` in |
   | `person.topics` | person | `STATED_BY` in, then `ABOUT` out |
   | `person.co_speakers` | person | `STATED_BY` in, `ABOUT` out, `ABOUT` in, `STATED_BY` out |
   | `topic.people` | topic | `ABOUT` in, then `STATED_BY` out |
   | `topic.entities` | topic | `ABOUT` in, then `MENTIONS_PERSON` and `MENTIONS_ORG` out |
   | `org.topics` | organisation | `MENTIONS_ORG` in, then `ABOUT` out |

5. **Presets** are named queries, cached per corpus revision:
   - `slice.recent`: window, feeds, content types, `min_items`, `expand_groups`, `exclude_items`. The
     window widens (7d, 30d, 90d, all) while fewer than `min_items` items match; the response returns
     `applied_window`.
   - `slice.items`: item ids.
   - `slice.neighbourhood`: a node and a named path.
   - `slice.group`: one group of any kind.
   - `groups.summary`: every group of a kind, with the weights between them.

   A query asked often becomes a preset.
6. **One response shape, `graph-slice/1`:**
   - `nodes`: a bare canonical id, the type (ontology v3), a joined label, `properties`, `item_id` for
     an item-owned node, `groups`, `held: true` for a stub, and `item_count` when asked;
   - `edges`: type, class, `from`, `to`, `weight`, `item_id` for an item-scoped edge, `properties`;
   - `items`: the items the slice draws from, with the source's id and title;
   - `groups`: a catalogue of every group referenced (id, kind, label, parent);
   - `truncated`: reasons (`budget`, `edge_budget`, `fanout`) and what was left out;
   - `corpus_revision`, `unknown_seeds`, `applied_window`.

   **`exclude_items`:** the held items' own nodes and edges are not sent again. A shared node reached
   through a new item comes back as a stub. Item-scoped edges carry `item_id`, so the response's
   `items` is the record of an expand, and a client collapses one by dropping those items' edges.
   **When `keep` drops what a path passes through,** one aggregated edge joins the kept ends (type =
   the path's name, class discovery, weight = the number of paths, no `item_id`). A slice is never
   paged: budgets, `truncated` and `exclude_items` take its place.
7. **Edges, their storage and their class:**

   | Edge | Direction | Class | Read from |
   | --- | --- | --- | --- |
   | `HAS_ITEM` | Source → ContentItem | structural | the item's source |
   | `HAS_INSIGHT` | ContentItem → Insight | structural | the insight's item |
   | `PUBLISHED_BY` | Source → Organization | structural | the source's publisher |
   | `CONTRIBUTES_TO` | Person → Source | structural | roles |
   | `PARTICIPATES_IN` | Person → ContentItem | attribution | roles |
   | `MENTIONS` | Topic, Person, Organization, Object → ContentItem | discovery | item entities |
   | `RELATED_TO` | Topic ↔ Topic | discovery | topic relations (reserved) |
   | `ABOUT` | Insight → Topic | descriptive | insight mentions |
   | `MENTIONS_PERSON`, `MENTIONS_ORG`, `MENTIONS_OBJECT` | Insight → Person, Organization, Object | descriptive | insight mentions |
   | `SUPPORTED_BY` | Insight → evidence (Quote) | evidentiary | insight support |
   | `ATTRIBUTED_TO` | evidence → Person | attribution | the evidence's speaker; `SPOKEN_BY` is its audio form |
   | `STATED_BY` | Insight → Person | attribution | derived: the strict majority of the insight's evidence, kept per item in `insight_attribution` |
   | `IN_GROUP` | Topic or group → group | structural | group membership |

8. **The walk is bounded level by level.**
   - One statement per step, over edge tables indexed `(from_id, type)` and `(to_id, type)`. Each
     frontier node is joined with at most 200 edges per type, in the query's order, and the frontier
     is cut to the remaining budget after each step. One recursive statement is not used: it cannot
     cap a level's frontier, so a hub would build its whole reach first.
   - The statement count is a fixed function of the path and the options, stated per preset with the
     implementation and asserted by a test.
   - Each request runs in one read-only, repeatable-read transaction (one snapshot across the step
     statements; ADR-173's `SET LOCAL` keeps no session state) with a 5 s statement timeout. A
     timeout is 503 with `retry: narrower`, never a partial slice.
   - The graph API reads from a read-only replica where one exists. A replica can cancel a snapshot
     on a conflict (SQLSTATE 40001); the engine retries once on the primary, so the worst case is
     two deadlines.
9. **Budgets fill in two modes.**
   - **Item slices:** items in the query's order, then round-robin: each item's KG nodes first (capped
     like any fan-out), then insights by rank across the items. Admitting an insight admits and
     counts the nodes it reaches.
   - **Path walks:** the frontier is cut per step.
   - **Edges:** the induced edges among returned nodes, by weight, then item recency, then id. A node
     may be left with no edge when `edge_budget` cuts, and `truncated` says so.
10. **Discovery and freshness.**
    - `GET /api/graph/catalogue`, cached per revision, lists the format, presets, node types, edge
      types with class and direction, named paths with their steps, and the bounds.
    - `/api/health` gains `graph_api` and `corpus_revision`.
    - Presets carry an ETag `"<revision>:<hash>"`, and a matching `If-None-Match` gets 304. A client
      never unions slices from different revisions.
11. **Exposure.**
    - On the public operator surface, `POST /api/graph/query` is where an outsider can spend database
      time. The static bounds, the deadline, and at most 4 generic queries at once per process
      (429 `retry: later` beyond) protect it. Presets are cached and not counted.

## Consequences

- **Positive:**
  - one read for every graph surface;
  - canonical ids end the browser's merge, prefixes and caps;
  - agents and screens ask the same engine;
  - the in-process graph copies leave the API's memory.
- **Negative:**
  - the clients' graph code is rewritten before the files go;
  - the bounds' numbers (the fan-out cap, the deadline, the budgets) are chosen before production
    measures them;
  - `STATED_BY`'s majority rule returns fewer person-to-insight links than today's any-quote link,
    so the relational routes' rows change when they move onto the engine.
- **Neutral:**
  - preset routes keep their contracts;
  - the files stay the source until the cut-over, and the graph API exists on the database only.

## Alternatives considered

- **Rebuild each item's graph files from the database.** Rejected: work to delete again, and it
  keeps the browser merge and its caps.
- **GraphQL.** Rejected: its flexibility is per field, not per traversal, so a walk still needs
  custom resolvers, and statement budgets per request are harder to hold.
- **A graph query language** (Cypher through Apache AGE, or text composed by the client). Rejected:
  no AGE without a measured need, and a text query cannot be bounded before it runs.
- **One recursive statement per request.** Rejected: a level's frontier cannot be capped.
- **Paging a slice.** Rejected: a page cuts edges from their nodes.
- **A partial slice on timeout.** Rejected for now: a cancelled statement aborts the transaction. A
  savepoint per step can return completed levels later, if timeouts are measured.

## Implementation Notes

- Measure first: out-degree per node type and edge type (p50, p95, p99, top hubs), the derived
  tables' build costs, and each surface's slice size and time against p95 at or under 1 s for
  presets. The fan-out cap is then set at a percentile that caps only hubs.
- Parity before the files go: for a fixed set of slices, the engine's nodes and edges equal today's
  browser merge of the same files; the relational routes' parity lists the rows that change
  (`STATED_BY`, and what the in-process graph's host reconciliation and identity map did at serve
  time).
- Not built now: extension layers from tables outside the corpus schema; enrichment edges drawn on
  the canvas.

## References

- [ontology-v3.md](../architecture/corpus/ontology-v3.md) and [ontology.md](../architecture/corpus/ontology.md):
  node and edge types, classes, directions.
- `search/corpus_graph.py`, `search/relational_queries.py`: the in-process graph and the relational
  functions this replaces.
