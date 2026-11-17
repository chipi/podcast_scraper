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

The server keeps its own in-memory graph per process (`search/corpus_graph.py`,
`get_corpus_graph`), which the relational routes read. It derives a `STATES` edge (person to
insight, whenever any of the insight's quotes is the person's) and applies three transforms at
build time: host reconciliation, the identity map and `_spell_people`.

After the corpus moves into the database (ADR-169), there are no graph files to fetch. A build of the
viewer, an agent and the app's own routes need one way to read the graph from the corpus tables. It
must be general enough for questions no screen has been written for, and bounded enough to run on
one shared Postgres.

## Decision

1. **One engine, two consumers.** One module of functions over the corpus tables reads the graph.
   - **The graph API** serves graph surfaces and agents outside the process: a structured query and
     named presets, all returning `graph-slice/1`. Code inside the process (the relational routes,
     an in-process agent server) calls the same functions directly.
   - **Preset routes** (the app's routes, and the relational routes) keep their own shapes. They call
     the engine's functions where they walk the graph and a plain indexed query where they do not.
   - No app route exposes traversal parameters.
2. **The engine reads every corpus table that holds graph data.** Items and sources, the KG tables
   (`item_entities`, `person_roles`, `topic_relations`, `kg_edges`), insights, evidence and their edges (`insight_support`, `insight_mentions`),
   `insight_attribution`, and groupings (`clusters`, `cluster_members`, `cluster_links`).
   - A build that writes no insights or groupings has empty tables, so its slice holds none: by
     data, not by code.
   - An extension can add a score to how `slice.recent` chooses items (`Extension.groupings`).
   - An extension can later add a layer from a table outside the corpus schema. None is built now.
3. **Two new corpus tables,** whose DDL is in public's corpus Alembic history whoever writes the
   rows:
   - **`insight_attribution(insight_id, person_id)`**, indexed both ways, served as `STATED_BY`. An
     insight's person is the strict majority of its evidence's people. It is kept per item, not
     rebuilt per revision: public's derivation function runs for an item in the same transaction
     that commits the item's evidence, and every evidence writer (a stage, a data migration) calls
     it for the items it touched. An identity publish rebuilds the whole table.
   - **`cluster_links(build_id, kind, a, b, weight)`**: per grouping kind, the weight between two
     groups is the number of descriptive and discovery edges between member topics of different
     groups. The clustering build writes it with its build id and publishes it in the same pointer
     swap as the groups, so `groups.summary` never mixes two builds. Clusters keep pairs above a
     minimum weight recorded on the build row; the levels above keep all pairs. It is corpus-wide,
     so its numbers are not today's client numbers, which count only what is loaded.
4. **`corpus_revision`** is a counter every committed corpus write moves. Each request reads it
   inside its transaction (decision 9), so a slice cannot straddle a bump with nodes and edges that
   disagree. `/api/health` publishes it.
5. **The structured query** (`POST /api/graph/query`) takes parameters, never a query language.
   - **`seeds`:** node ids (at most 50), item ids (at most 200), or one group id.
   - **`path`:** at most 3 steps, or a named path. A step is
     `{edges: [type, class or "*"], direction: out | in | any, types: [...]}`; an entry in `edges` may
     carry its own direction. `path: []` returns the seeds' own rows.
   - **`keep`:** the node types returned. `Quote` (evidence) only when named, because its text is
     heavy. Groups are not nodes (decision 8), so `keep` and `types` cannot name one.
   - **`filters`:** a time window, sources, content types.
   - **`order`:** `nearness`, `recency` or `weight`.
   - **`budget`:** nodes, default 1,500, at most 3,000. **`edge_budget`:** default 10,000, at most
     20,000.
   - **`exclude_items`:** items the client already holds, at most 2,000.
   - **`with: counts`:** adds `item_count` per node: the number of items in the whole corpus that
     carry the node, not its degree on the canvas.

   Every parameter is bounded. Beyond a bound, or for an unknown edge type, node type, path name or
   grouping kind, the answer is 422 in FastAPI's `{detail: [{loc, msg, type}]}`, with `loc` naming
   the field. An unknown seed is not an error: the answer is 200 with `unknown_seeds`.
6. **Named paths** are the engine's constants. They may be unions of step lists chosen by the seed's
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

   `node.items` returns what `POST /api/corpus/node-episodes` returns today.
7. **Presets** are named queries. Every preset takes `budget` and `exclude_items` besides its own
   parameters:

   | Preset | Parameters | Defined as |
   | --- | --- | --- |
   | `slice.recent` | window, sources, content types, `min_items` (default 10), `expand_groups` (default 0) | the newest items, by recency plus the extension score. The window widens (7d, 30d, 90d, all) while fewer than `min_items` match, never past all, and the response returns `applied_window`. With `expand_groups: n`, every cluster with a member topic in the slice adds its n newest other items, inside the same budget, after the primary items |
   | `slice.items` | item ids | those items' graph |
   | `slice.neighbourhood` | node id, a named path | the node and what the path reaches, with the items that connect them |
   | `slice.group` | one group id of any kind | the group's member topics or child groups, the items that carry them (newest first), and their graph |
   | `groups.summary` | a grouping kind | every group of that kind (label, parent, child count) and `cluster_links`' weights between them |

   A query asked often becomes a preset.
8. **One response shape, `graph-slice/1`:**
   - `format`, `corpus_revision`, the preset or query with its parameters, `applied_window` where it
     applies, `unknown_seeds`;
   - `nodes`: a bare canonical id, the type (ontology v3), a joined label (never derived from the
     id), `properties`, `item_id` for an item-owned node, `groups` (the ids of the groups it is in),
     `held: true` for a stub, and `item_count` when asked;
   - `edges`: the induced edges among the returned nodes (every stored edge with both ends returned,
     whether or not the walk crossed it), by weight, then item recency, then id, cut at
     `edge_budget`; each with type, class, `from`, `to`, `weight` (the one field every aggregation reads), `item_id`
     for an item-scoped edge (decision 10's table), `properties`;
   - `items`: every item the slice draws from, with `item_id`, `content_type`, `published_at`,
     `source_id` and the source's title;
   - `groups`: a catalogue of every group referenced (`id`, `kind`, `label`, `parent_id`). Groups
     live here, not in `nodes`: `IN_GROUP`'s `to` and a group's child groups resolve in this
     catalogue;
   - `truncated`: `reasons` (any of `budget`, `edge_budget`, `fanout`), `nodes_left` and
     `items_left` where known, and `fanout_capped` (the nodes whose edges were capped).

   **`exclude_items`:** the held items' `ContentItem` nodes, their item-owned nodes and their
   item-scoped edges are not sent again. A shared node (a topic, a person) reached through a new
   item comes back as a stub (`id`, `type`, `held: true`), so the new edge does not dangle. A client
   collapses an expand by dropping the edges of the items the expand returned, then the nodes left
   with no edge; the response's `items` is the collapse record.

   **When `keep` drops what a path passes through,** one aggregated edge joins each pair of kept
   ends: type = the path's name, class discovery, weight = the number of paths. It carries no
   `item_id`, is never part of a collapse record, and goes when either end goes.

   A client that shows several slices merges them by id and never unions slices from different
   revisions. A slice is never paged: budgets, `truncated` and `exclude_items` take its place.
9. **The walk is bounded level by level.**
   - One statement per step, over the step's edge tables (any of `item_entities`, `person_roles`,
     `topic_relations`, `kg_edges`, `insight_support`, `insight_mentions`, `insight_attribution`,
     `cluster_members`) as one `UNION ALL`, through indexes `(from_id, type)` and `(to_id, type)`.
     A union path runs one statement per level: each frontier node carries its branch, so the next
     level continues it along its own branch only. Each frontier node is joined `LATERAL` with at most 200
     edges per edge type, in the query's order, and the frontier is cut to the remaining budget after
     each step. The worst case per statement is the budget times that cap. One recursive statement is
     not used: it cannot cap a level's frontier, so a hub would build its whole reach first.
   - Each request runs in one read-only, repeatable-read transaction on one connection (one
     snapshot across the step statements; ADR-173's `SET LOCAL` keeps no session state), with a 5 s
     statement timeout. Nothing but the request's own statements runs inside it.
   - A timeout is 503 with `retry: narrower` and no `Retry-After`, never a partial slice.
   - Beyond a static bound the request is refused (422); beyond a data bound the slice is cut, and
     `truncated` says where.
   - **Budgets fill in two modes.** Item slices: items in the request's order, then round-robin
     across them, each item's KG nodes first (capped like any fan-out), then insights by rank;
     admitting an insight admits and counts the nodes it reaches. Path walks: the frontier is cut
     after each step. Edges: decision 8's order; a node may be left with no edge when
     `edge_budget` cuts, and `truncated` says so.
   - The graph API reads from a read-only replica where one exists. A replica can cancel a snapshot
     on a conflict (SQLSTATE 40001); the engine retries once on the primary, so the worst case is
     two deadlines.
   - **The statement count is a fixed function of the request**, and a test asserts it per preset.
     Counted: the statements that read rows. Not counted: the transaction's `SET` lines. The
     revision read is folded into the first statement. With `s` the steps (a union path's longest
     branch) and the fixed tail = 4 (nodes' rows, items' rows, the groups' catalogue, the induced
     edges):

     | Request | Statements |
     | --- | --- |
     | the query | 1 (revision and seeds) + `s` + 4, +1 when `keep` drops intermediates (aggregated edges), +1 with `counts` |
     | `slice.items` | 1 + 2 + 4, +1 with `counts` |
     | `slice.recent` | 1 (revision and the window search, every widening in one statement) + 2 + 4; with `expand_groups`, +1 to choose the added items and +2 for their edges |
     | `slice.neighbourhood` | 1 + the named path's longest branch + 4, +1 for aggregated edges |
     | `slice.group` | 1 (revision, the group and its members) + 1 (the items carrying them) + 2 + 4 |
     | `groups.summary` | 2 (revision and the groups; their links) |

     An item slice's 2 steps are the items' own edges, then their insights' edges. The second
     fetches every candidate insight's edges (within the fan-out cap), and the engine admits
     insights by rank afterwards (the budget modes above); admitting one at a time would not be two
     statements. `slice.recent`'s window search returns a bounded candidate list, newest first and
     at most the budget's worth of items, which the extension score re-ranks in the engine.

10. **Edges, their storage, their class, and which carry `item_id`:**

    | Edge | Direction | Class | Read from | Item-scoped |
    | --- | --- | --- | --- | --- |
    | `HAS_ITEM` | Source → ContentItem | structural | `content_items` | yes |
    | `HAS_INSIGHT` | ContentItem → Insight | structural | `insights` | yes |
    | `PUBLISHED_BY` | Source → Organization | structural | `sources` | no |
    | `CONTRIBUTES_TO` | Person → Source | structural | `person_roles` | no |
    | `PARTICIPATES_IN` | Person → ContentItem | attribution | `person_roles` | yes |
    | `MENTIONS` | Topic, Person, Organization, Object → ContentItem | discovery | `item_entities` | yes |
    | `RELATED_TO` | Topic ↔ Topic | discovery | `topic_relations` (reserved) | no |
    | `ABOUT` | Insight → Topic | descriptive | `insight_mentions` | yes |
    | `MENTIONS_PERSON`, `MENTIONS_ORG`, `MENTIONS_OBJECT` | Insight → Person, Organization, Object | descriptive | `insight_mentions` | yes |
    | `SUPPORTED_BY` | Insight → evidence (Quote) | evidentiary | `insight_support` | yes |
    | `ATTRIBUTED_TO` | evidence → Person | attribution | the evidence's speaker; `SPOKEN_BY` is its audio form | yes |
    | `STATED_BY` | Insight → Person | attribution | `insight_attribution` (derived, decision 3) | yes |
    | `IN_GROUP` | Topic or group → group | structural | `cluster_members` (topic → group); the group's parent link (group → group) | no |

    - **The vocabulary** is ontology v3 plus three names this ADR adds to the ontology docs:
      `MENTIONS_OBJECT` (written and read today, missing from the docs' edge tables), `STATED_BY`
      (derived) and `IN_GROUP` (membership, whose `to` is a group in the response's catalogue).
    - **`STATED_BY` replaces the in-memory graph's `STATES`,** with a stricter rule: majority, not
      any quote. No weaker edge is named; a reader who wants anyone whose words support an insight
      follows `SUPPORTED_BY` out, then `ATTRIBUTED_TO` out.
11. **Discovery, caching and freshness.**
    - `GET /api/graph/catalogue`, cached per revision, lists the format, presets, node types, edge
      types with class and direction, named paths with their steps, and the bounds.
    - `/api/health` gains `graph_api` and `corpus_revision`.
    - Presets are computed once per revision in each process, keyed on the preset and its
      parameters, with one build per key at a time. Their ETag is `"<revision>:<hash>"`, and a
      matching `If-None-Match` gets 304. Generic queries are computed per request within their
      bounds, with no HTTP cache.
12. **Exposure.** On the public operator surface, `POST /api/graph/query` is where an outsider can
    spend database time. The static bounds, the deadline, and at most 4 generic queries at once per
    process (429 `retry: later` beyond) protect it. Presets are cached and not counted.
13. **What public removes, after the cut-over:**
    - `GET /api/artifacts` and `GET /api/artifacts/{path}` for graph files, and the viewer's merge
      behind them (`stores/artifacts.ts`);
    - `POST /api/corpus/node-episodes`, a wrapper over `node.items` until then (the digest's
      `_linkable_topic_id`, `routes/corpus_digest.py`, calls its function and moves to `node.items`
      first);
    - `POST /api/corpus/resolve-episode-artifacts`, which maps item ids to file paths;
    - `get_corpus_graph` with its derived `STATES`, and the viewer's prefix stripping, sibling caps
      and per-file merge;
    - the viewer's type names change to v3 (`colors.ts`, `visualGroup.ts`); a one-table adapter may
      bridge the client until the cut-over and is deleted with the file path.

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
  - `STATED_BY`'s majority rule returns fewer person-to-insight links than `STATES`, so the
    relational routes' rows change when they move onto the engine;
  - every evidence writer gains a call to the attribution derivation.
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
- **Insights and groupings as extension layers.** Rejected: their tables are the platform's corpus
  schema; a hook would carry the file-era seam into the row era.
- **Insight attribution as a view.** Rejected: a GROUP BY over every quote of a person at each step,
  with no index to probe.
- **One recursive statement per request.** Rejected: a level's frontier cannot be capped.
- **Paging a slice.** Rejected: a page cuts edges from their nodes.
- **A partial slice on timeout.** Rejected for now: a cancelled statement aborts the transaction. A
  savepoint per step can return completed levels later, if timeouts are measured.

## Implementation Notes

- Measure first: out-degree per node type and edge type (p50, p95, p99, top hubs), the two new
  tables' build costs, and each surface's slice size and time against p95 at or under 1 s for
  presets. The fan-out cap is then set at a percentile that caps only hubs.
- Parity before the files go: for a fixed set of slices, the engine's nodes and edges equal today's
  browser merge of the same files, after mapping ids and the v3 vocabulary (`groups.summary`
  excepted: its weights are corpus-wide). The relational routes' parity lists the rows that change:
  `STATES` to `STATED_BY`, and what the in-memory graph's three build-time transforms (host
  reconciliation, the identity map, `_spell_people`) did at serve time, each either stored in the
  rows already or listed as a change.
- The ontology docs gain `MENTIONS_OBJECT`, `STATED_BY` and `IN_GROUP` with this ADR's acceptance.
- Not built now: extension layers from tables outside the corpus schema; enrichment edges drawn on
  the canvas.

## References

- [ontology-v3.md](../architecture/corpus/ontology-v3.md) and [ontology.md](../architecture/corpus/ontology.md):
  node and edge types, classes, directions.
- `search/corpus_graph.py`, `search/relational_queries.py`: the in-memory graph and the relational
  functions this replaces.
- `server/routes/corpus_library.py`, `server/routes/artifacts.py`: the routes decision 13 removes.
