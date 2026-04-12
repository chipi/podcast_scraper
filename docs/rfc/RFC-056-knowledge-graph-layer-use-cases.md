# RFC-056: Knowledge Graph Layer – Use Cases & End-to-End Consumption

- **Status**: Completed — single-layer consumption patterns retained; cross-layer use
  cases (opinion tracking, guest intelligence, controversy detection) moved to
  [RFC-072](RFC-072-canonical-identity-layer-cross-layer-bridge.md)
- **Authors**: Podcast Scraper Team
- **Stakeholders**: Core team, downstream consumers, integrators
- **Execution Timing**: **Parallel with RFC-055 implementation** — Consumption patterns,
  CLI contracts, and query shapes evolve as the KG artifact and `kg` namespace land.

  Depends on RFC-055 for per-episode KG JSON and ontology.

- **Related PRDs**:
  - `docs/prd/PRD-019-knowledge-graph-layer.md` (Knowledge Graph Layer — **KG**)
  - `docs/prd/PRD-017-grounded-insight-layer.md` (**separate** — GIL / `gi`; not KG)
  - `docs/prd/PRD-018-database-projection-gil-kg.md` (optional relational serving via RFC-051)
- **Related RFCs**:
  - `docs/rfc/RFC-055-knowledge-graph-layer-core.md` (**primary dependency** — artifact,
    schema, config, `kg` vs `gi` separation)

  - `docs/rfc/RFC-051-database-projection-gil-kg.md` (Postgres projection for KG tables
    when enabled)

  - `docs/rfc/RFC-053-adaptive-summarization-routing.md` (optional `route_kg_extraction`
    alignment with episode profile)

  - `docs/rfc/RFC-004-filesystem-layout.md` (output layout and run scoping)

  - `docs/rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md`
    (cross-layer use cases — Position Tracker, Guest Brief)
- **Related Documents**:
  - `docs/architecture/kg/ontology.md` — Human-readable ontology (**v1 frozen**, GitHub #464)
  - `docs/architecture/kg/kg.schema.json` — Normative JSON Schema (v1 frozen, #464)
  - `docs/guides/KNOWLEDGE_GRAPH_GUIDE.md` — User-facing guide (living document)

## Abstract

This RFC defines how the **Knowledge Graph Layer (KG)** delivers user value **after**
per-episode extraction. RFC-055 specifies **what** is stored (nodes, edges, provenance);
this RFC specifies **how** operators and developers **consume** that data: exploration
patterns, export and merge strategies, optional database-backed queries, and the **`kg`**
CLI surface (aligned with PRD-019).

**Relationship to GIL (RFC-050):** GIL optimizes for **insights + verbatim evidence**
(trust and navigation). KG optimizes for **entities, topics, and typed relationships**
(linking and discovery). Consumption shapes differ: GIL responses center on
`insights[]` and `supporting_quotes[]`; KG responses center on **graph traversals**
and aggregations over **nodes** and **edges** (see §Output contracts).

## Problem Statement

Without explicit consumption design:

- **Integrators** do not know how to merge episode files into a corpus view
- **CLI users** lack a documented **`kg`** workflow comparable to **`gi`** discoverability
- **Success criteria** for “KG is useful” stay vague compared to GIL’s Insight Explorer
  narrative

This RFC closes that gap at the **specification** level; implementation fills in exact
commands and flags.

## Goals

1. **Define KG-centric use cases** that are distinct from GIL (no grounding requirement)
2. **Specify query and aggregation patterns** that work over per-episode files (and
   optionally RFC-051 tables)

3. **Establish illustrative output contracts** for common operations (not every future
   query)

4. **Align CLI naming** with RFC-055: **`kg`** for graph operations, **`gi`** unchanged
   for grounded insights

## Non-Goals (this RFC)

- Replacing or embedding KG inside `gi.json`
- Perfect **entity resolution** across the open web (see PRD-019 non-goals)
- Natural-language query translation (post-v1; structured access first)

## Design Principles

1. **Episode-local production, global consumption**: Same pattern as GIL — artifacts are
   written per episode; consumers build logical union or DB views.

2. **Structure over evidence**: User value is **who/what links to what**, not mandatory
   quote spans (those remain GIL’s contract).

3. **Stable IDs within scope**: Episode-local or feed-scoped identifiers are acceptable for
   v1; cross-corpus merging rules live in RFC-055 / ontology.

4. **Optional join to GIL**: Linking KG nodes to `insight_id` is **out of scope for v1**
   unless explicitly added later.

## Minimal v1 Use Cases

### UC1: Cross-episode theme and entity exploration

**User intent:** See which **entities** or **topics** recur across episodes from the same
show (or run), to prioritize listening or analysis.

**Consumption pattern:**

- Scan per-episode KG JSON under the run output tree **or** query KG tables if PRD-018 /
  RFC-051 export is enabled.

- Aggregate by **node label** or **normalized key** (exact strategy per ontology).
- Present counts and episode lists; optional co-occurrence via edges.

**Illustrative response shape (logical — not a mandatory wire format):**

```json
{
  "scope": "podcast:planet-money",
  "entity": {
    "id": "kg:entity:federal-reserve",
    "label": "Federal Reserve",
    "type": "organization"
  },
  "episode_count": 12,
  "episodes": [
    {
      "episode_id": "episode:abc123",
      "title": "Why the Fed raised rates",
      "mention_count": 3
    }
  ]
}
```python

**Success criteria:**

- User can answer “where did this entity show up?” across processed episodes
- Results are reproducible from files alone (no DB required)

### UC2: Structured export for downstream tools

**User intent:** Load KG JSON into **RAG**, search, or visualization tools with a
**versioned schema**.

**Consumption pattern:**

- Read `docs/architecture/kg/kg.schema.json`-valid artifacts (when published)
- Optionally run **`kg export`** (or equivalent) to emit **NDJSON**, **single merged
  graph snapshot**, or **SQL** insert stubs — exact flags specified at implementation time

**Success criteria:**

- Documented path from disk artifact → consumer pipeline (see
  `docs/guides/KNOWLEDGE_GRAPH_GUIDE.md`)

- Schema validation in CI when KG is exercised in tests (per RFC-055)

### UC3: Parallel operation with GIL

**User intent:** Same pipeline run produces **`gi.json`** (GIL) and **KG artifact** when
both flags are on, without cross-contamination.

**Consumption pattern:**

- Operators enable **`generate_gi`** and **`generate_kg`** independently or together
- Downstream jobs read **two files** per episode; routing and naming stay **`gi`** vs
  **`kg`** (RFC-055)

**Success criteria:**

- Disabling one feature does not alter the other’s semantics (PRD-019 FR1)

### UC4: Operator inspection (CLI)

**User intent:** Inspect KG contents from the terminal the same way operators inspect GIL.

**Consumption pattern (implemented in CLI):**

- **`kg validate`**: Validate KG JSON against published schema (`--strict` for full JSON Schema).
- **`kg inspect`**: Summarize nodes/edges for one episode or path.
- **`kg export`**: NDJSON or merged JSON bundle over a run output tree.
- **`kg entities`** / **`kg topics`**: File-based roll-up and topic co-occurrence (RFC-056 query patterns).

Exact flags are documented in `docs/guides/KNOWLEDGE_GRAPH_GUIDE.md` and `docs/api/CLI.md`.
Subcommands live under the **`kg`** namespace per RFC-055.

## Query Patterns

| Pattern | Description | Typical inputs |
| --- | --- | --- |
| **Entity roll-up** | Count episodes and mentions per entity | Entity id or label |
| **Topic co-occurrence** | Pairs of topics that appear in same episode | Optional minimum support |
| **Subgraph slice** | Nodes/edges for one episode | `episode_id` |
| **Feed corpus view** | Union of episodes under one podcast/run | Output root or DB |

Natural-language or embedding-based search over KG is **out of scope for v1** unless
added in a later RFC.

## Relational consumption (optional)

When [RFC-051](RFC-051-database-projection-gil-kg.md) is enabled, consumers may use SQL
for roll-ups and joins instead of scanning JSON. **Table shapes and migrations** belong in
RFC-051 / PRD-018; this RFC only requires that **semantic separation** between GIL and KG
projections is preserved (separate tables or namespaces).

## Testing & acceptance hooks

- **Unit**: Node/edge builders, ID stability, schema validation (RFC-055)
- **Integration**: Transcript → KG artifact → validate
- **E2E (optional)**: Config path with `generate_kg: true` and artifact assertions

Consumer-focused acceptance tests may mirror GIL acceptance layout under
`config/acceptance/` when KG configs exist (directory name **`kg/`** recommended for
symmetry with `gi/`).

## Rollout

- Keep `docs/guides/KNOWLEDGE_GRAPH_GUIDE.md` updated as commands and paths stabilize
- Link RFC-056 from PRD-019 and RFC-055
- When CLI is live, update `docs/api/CLI.md` and `docs/api/CONFIGURATION.md`

## References

- [PRD-019: Knowledge Graph Layer](../prd/PRD-019-knowledge-graph-layer.md)
- [RFC-055: KG Core Concepts & Data Model](RFC-055-knowledge-graph-layer-core.md)
- [RFC-050: GIL Use Cases](RFC-050-grounded-insight-layer-use-cases.md) (analogous split)
- [PRD-018: Database Projection](../prd/PRD-018-database-projection-gil-kg.md)
