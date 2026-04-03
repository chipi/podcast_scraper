# PRD-018: Database Projection for GIL and KG

- **Status**: 📋 Draft
- **Authors**: Podcast Scraper Team
- **Related RFCs**:
  - RFC-044 (Model Registry — prerequisite)
  - RFC-042 (Hybrid ML Platform — prerequisite)
  - RFC-049 (Core GIL Concepts — GIL dependency)
  - RFC-050 (Use Cases & Insight Explorer — GIL)
  - RFC-051 (Database Projection — technical design for GIL and KG)
  - RFC-055 (Knowledge Graph — artifact model for KG projection)
  - RFC-056 (Knowledge Graph — use cases and consumption patterns for KG queries)
- **Related Issues**: #31, #40, #50
- **Related PRDs** (in scope):
  - [PRD-017: Grounded Insight Layer](PRD-017-grounded-insight-layer.md) — source `gi.json` and GIL semantics
  - [PRD-019: Knowledge Graph Layer](PRD-019-knowledge-graph-layer.md) — source KG artifacts per RFC-055 (`kg` export and tables are **in scope** here; implementation may ship after GIL export when KG artifacts exist)
- **Related Documents**:
  - `docs/gi/ontology.md`, `docs/gi/gi.schema.json` — GIL ontology and schema
  - `docs/kg/ontology.md`, `docs/kg/kg.schema.json` — KG ontology and schema (RFC-055)

## Summary

This PRD defines **database projection** for **both** the **Grounded Insight Layer (GIL)** and the **Knowledge Graph Layer (KG)**. It enables fast, queryable access by projecting **file-based artifacts** into a relational database (Postgres). Files remain the **canonical, auditable** source of truth; the database is a **derived serving layer** for SQL, indexing, and incremental upserts.

**GIL projection** covers `gi.json` (insights, quotes, grounding, topics, speakers) and supports **UC1–UC5** / Insight Explorer–style queries (RFC-050) without scanning every episode directory.

**KG projection** covers per-episode **KG artifacts** defined in RFC-055 (entities, topics, typed edges, episode anchoring). It supports cross-episode **linking and discovery** queries that are **not** the same as GIL’s evidence-first contract. GIL and KG use **separate tables and provenance columns** — no merging of the two layers into a single JSON blob or a single undifferentiated “graph” table without typed source (see RFC-051).

Together, this transforms the project into a better **tooling surface** for analysts, notebook users, and integrators, while preserving on-disk layouts and independent feature flags (`generate_gi` vs KG generation per RFC-055).

## Background & Context

The Grounded Insight Layer (PRD-017) produces
structured `gi.json` files per episode, enabling
evidence-backed insight retrieval. However, file-based
access has limitations:

- **Scalability**: Scanning N folders and parsing N
  JSON files becomes slow at scale (50 episodes is
  fine; 5,000 is painful)
- **Query Performance**: No indexing, filtering, or
  pagination for global queries
- **Integration Friction**: Downstream tools (CLI,
  notebooks, web UIs, agents) must implement file
  scanning logic
- **Iteration Speed**: Re-running extraction or
  comparing runs requires manual file operations

**The Core Problem**: Files are canonical and auditable,
but they're not optimized for fast queries and tooling.

**The Solution**: Export to Postgres as a "projection
layer" that preserves provenance while enabling
SQL-based queries, indexing, and incremental updates.
Files remain the source of truth; the database becomes
the serving layer.

**How it relates to existing features:**

- **Grounded Insight Layer (PRD-017)**: Database export
  projects GIL data from `gi.json` files into tables
- **Metadata Generation (PRD-004)**: Episode metadata
  is also exported for complete episode context
- **Transcript Pipeline (PRD-001)**: Transcript paths
  are stored for evidence resolution
- **GIL Use Cases (RFC-050)**: Database export makes
  UC1–UC5 queries fast and ergonomic
- **Knowledge Graph (PRD-019, RFC-055)**: When KG
  artifacts exist, database export can project **KG
  nodes and edges** for fast entity/topic/relationship
  queries across episodes — **same projection product**
  as GIL, **separate tables** (see RFC-051)

## Goals

1. **Enable Fast Queries**: Transform file scanning
   into indexed SQL queries for UC1–UC5 **(GIL)** and
   for cross-episode **KG** traversals defined in RFC-055
2. **Support Notebook Research Workflows**: Enable
   power users to build topic dossiers, speaker
   profiles, and insight timelines **(GIL)**; enable
   entity-centric and relationship queries **(KG)** where
   implemented
3. **Provide Stable Integration Interface**: Give
   downstream tools (CLI, notebooks, web UIs, agents)
   a consistent SQL interface for **both** layers
4. **Preserve Provenance**: Every database row is
   traceable back to source files (`gi.json`, KG
   artifact path, transcript evidence) and extraction
   metadata
5. **Enable Incremental Growth**: Support upserting
   new episodes without reprocessing historical data
6. **Keep GIL and KG Semantically Separate**: Same
   Postgres instance may host both projections; schemas
   and CLI entrypoints (`gi export` vs `kg export`, or
   a documented combined mode) **must not** conflate
   grounding contracts with KG linking rules

## Non-Goals (v1)

- Replacing file-based outputs as the source of truth
  (files remain canonical)
- Collapsing GIL and KG into a single on-disk artifact
  or a single undifferentiated graph table (both may
  coexist in DB with clear `source` / table naming)
- Graph database as a hard dependency (relational
  projection is sufficient for v1; optional graph store
  remains a future evaluation per RFC-055)
- End-user UI or dashboards (deferred to post-v1)
- Real-time or streaming ingestion (batch export only)
- Storing full transcript text in database (pointers
  only for v1)
- Advanced analytics (trends, sentiment) — deferred
  to post-v1
- Cross-run GIL merging logic (episode-scoped in v1)
- **Guaranteed same release** for GIL export and KG
  export (KG projection may follow once `kg.schema.json`
  and pipeline outputs are stable)

## Personas

- **Power Users (Researchers, Analysts, Founders)**:
  Need instant topic/speaker exploration, insight
  retrieval with supporting quotes, and the ability to
  build research workflows in notebooks
- **Builders (Developers, Agent Builders)**: Need a
  stable SQL interface for integrating podcast data
  into CLI tools, web UIs, and RAG pipelines
- **Iterators (ML Engineers, Pipeline Developers)**:
  Need to re-run extraction, re-export, and compare
  runs without manual file operations

## User Stories

- _As a researcher, I can query all episodes
  discussing a topic in milliseconds so that I can
  build topic dossiers without waiting for file
  scans._
- _As an analyst, I can retrieve insights with
  supporting quotes using SQL so that I can cite
  findings with confidence._
- _As a developer, I can build a CLI tool on top of
  Postgres so that I can provide instant topic/speaker
  queries with evidence._
- _As a power user, I can build notebook workflows
  that query Postgres so that I can create speaker
  profiles, insight timelines, and quality audits._
- _As an ML engineer, I can re-export after model
  changes so that I can compare extraction quality
  across runs._

## Functional Requirements

### FR1: Export Command & Interface

- **FR1.1**: Provide **`gi export`** and **`kg export`**
  CLI commands (or one orchestrated command with
  explicit layer flags) that scan the output directory
  and export to Postgres
- **FR1.2**: Support `--target postgres` with `--dsn`
  connection string
- **FR1.3**: Support `--output-dir` to specify root
  output directory
- **FR1.4**: Support `--rebuild` flag to drop and
  recreate all tables
- **FR1.5**: Support incremental mode (default) that
  only ingests new episodes

### FR2: Episode Metadata Export

- **FR2.1**: Export episode-level records (id, title,
  publish_date, podcast_id, file paths)
- **FR2.2**: Store paths to `metadata.json`,
  `transcript.json`, `summary.json`, `gi.json`
- **FR2.3**: Include `schema_version` and
  `ingestion_run_id` for lineage tracking

### FR3: GIL Node Export

- **FR3.1**: Export global nodes (speakers, topics)
  with deduplicated, slug-based IDs
- **FR3.2**: Export episode-scoped nodes (insights,
  quotes) with episode_id foreign key
- **FR3.3**: Preserve all GIL node properties
  (text, label, confidence, grounded status, etc.)
- **FR3.4**: Include confidence scores and provenance
  metadata (model_version, prompt_version)

### FR4: GIL Edge Export

- **FR4.1**: Export relationship tables
  (`insight_topics`, `insight_support`)
- **FR4.2**: Preserve edge properties (confidence)
  where applicable
- **FR4.3**: Support filtering by confidence
  thresholds in queries

### FR5: Evidence & Provenance Tracking

- **FR5.1**: Store transcript references (path,
  char_start, char_end) for quotes
- **FR5.2**: Store timestamp ranges
  (timestamp_start_ms, timestamp_end_ms) for quotes
- **FR5.3**: Include `model_version` and
  `prompt_version` for ML-derived data
- **FR5.4**: Ensure every row is traceable back to
  `gi.json` path and episode_id

### FR6: Query Support for UC1–UC5

- **FR6.1**: Support UC1 (Cross-Podcast Topic
  Research) via `topics` + `insight_topics` +
  `insights` + `episodes` joins
- **FR6.2**: Support UC2 (Speaker-Centric Insight
  Mapping) via `speakers` → `quotes` →
  `insight_support` → `insights` joins
- **FR6.3**: Support UC3 (Evidence-backed Quote/Insight
  Retrieval) via `insights` + `insight_support` +
  `quotes` joins
- **FR6.4**: Support UC4 (Semantic Question Answering)
  via topic → insight filtering
- **FR6.5**: Support UC5 (Insight Explorer) via
  `topics` → `insight_topics` → `insights` →
  `insight_support` → `quotes` joins

### FR7: Incremental Updates & Rebuilds

- **FR7.1**: Support incremental ingestion (only new
  episodes by default)
- **FR7.2**: Support full rebuild (drop and recreate
  all tables)
- **FR7.3**: Handle schema version changes gracefully
  (warn or fail on mismatch)
- **FR7.4**: Support run comparison via
  `ingestion_run_id` tracking

### FR8: KG Artifact Export (RFC-055)

- **FR8.1**: Provide **`kg export`** (or equivalent
  documented command) that scans the output directory
  for per-episode **KG artifacts** and projects them
  into Postgres tables defined in RFC-051
- **FR8.2**: Store **`kg_path`** (and optional
  **`kg_schema_version`**) on episode records when KG
  data is present; rows must be traceable to the source
  KG JSON file
- **FR8.3**: Export **KG nodes** and **KG edges** (or
  normalized equivalents) with episode scope and stable
  IDs per RFC-055 — separate from `insights` /
  `quotes` / GIL join tables
- **FR8.4**: Support incremental and rebuild modes
  analogous to GIL export; idempotent upserts per
  episode
- **FR8.5**: Validate against **`docs/kg/kg.schema.json`**
  when present (mirror `make validate-gi-schema` story)

### FR9: Coexistence and Independence

- **FR9.1**: Exporting **GIL only**, **KG only**, or
  **both** must be supported when the corresponding
  artifacts exist (no requirement to enable both
  pipeline features to run export)
- **FR9.2**: Optional **cross-layer links** (e.g. KG node
  referencing a GIL `insight_id`) remain **out of scope
  for v1** unless added by a follow-up RFC; database
  schema should not assume such links

## Success Metrics

- **Query Performance**: UC1–UC5 queries complete in
  < 100ms for datasets up to 5,000 episodes; KG
  relational queries meet targets agreed in RFC-051
  for comparable corpus sizes
- **Export Speed**: Export completes in < 5 minutes for
  1,000 episodes (GIL and/or KG, within an order of
  magnitude of each other for similar row counts)
- **Provenance Accuracy**: 100% of database rows are
  traceable to source `gi.json` and/or KG artifact files
- **Notebook Workflow Adoption**: Positive feedback
  from power users on research workflow speed
- **Integration Success**: CLI tools and notebooks can
  query Postgres without file scanning

## Dependencies

- **PRD-017**: Grounded Insight Layer (GIL data to
  export)
- **PRD-019** / **RFC-055**: Knowledge Graph artifact
  model and schema (KG data to export when implemented)
- **RFC-044**: Model Registry (model metadata tracked
  in provenance fields)
- **RFC-042**: Hybrid ML Platform (produces pipeline
  outputs including `gi.json` when GIL is enabled)
- **RFC-049**: Core GIL Concepts & Data Model (defines
  ontology and `gi.json` schema)
- **RFC-050**: GIL Use Cases & Insight Explorer
  (defines UC1–UC5 query patterns)
- **RFC-051**: Database Projection (technical
  implementation details for GIL and KG tables)

## Constraints & Assumptions

**Constraints:**

- Must not replace file-based outputs as source of
  truth
- Must preserve full provenance (traceable to
  `gi.json` and transcript evidence)
- Must support incremental updates without
  reprocessing historical data
- Must be idempotent and rebuildable from disk
- Must not require graph database as hard dependency

**Assumptions:**

- Users have access to Postgres instance (local or
  remote)
- Database schema can be created/managed by export
  command
- Transcript text can remain on disk (pointers only
  in v1)
- Users understand SQL or can use provided query
  examples
- GIL extraction (RFC-049) produces valid `gi.json`
  files conforming to schema when GIL export is used
- KG pipeline (RFC-055) produces valid KG artifacts when
  KG export is used

## Design Considerations

### Transcript Text Storage

- **Option A**: Store only pointers (transcript path +
  spans) — **DECIDED for v1**
  - **Pros**: Simpler, smaller DB, canonical stays on
    disk, easier migration
  - **Cons**: Quote viewer must read disk files
    (acceptable for v1)
  - **Decision**: Option A for v1. If query latency
    for transcript resolution exceeds 100ms per
    quote, add a `quote_text` denormalized column
    in v1.1.

### Database Target Priority

- **Option A**: Postgres primary, others future —
  **DECIDED**
  - **Pros**: Structured joins, strong constraints,
    simple ops, covers UC1–UC5
  - **Cons**: May need columnar store for aggregations
    later
  - **Decision**: Option A (Postgres v1, graph
    database evaluated at ~1000+ episodes)

### Incremental vs Full Rebuild

- **Option A**: Support both modes — **DECIDED**
  - **Pros**: Fast updates for new episodes, full
    control for schema changes
  - **Cons**: More complex export logic
  - **Decision**: Option A (incremental default,
    rebuild flag for schema changes)

## Integration with Existing Features

Database export enhances **GIL** and **KG** by:

- **GIL File Output**: Projects `gi.json` files into
  queryable tables without changing file format
- **KG File Output**: Projects KG artifacts (RFC-055)
  into queryable tables without merging into `gi.json`
- **Episode Metadata**: Combines episode records with
  GIL and/or KG paths for complete context
- **Transcript Evidence**: Stores pointers to
  transcript files for quote resolution (GIL); KG may
  reference spans per RFC-055
- **Use Case Queries**: Makes UC1–UC5 queries fast
  and ergonomic via SQL (GIL); supports KG discovery
  queries per RFC-051

## Example Use Cases

### UC1: Cross-Podcast Topic Research (Postgres)

**Question**: "Show me all episodes discussing AI
Regulation since Jan 2026"

```sql
SELECT e.id, e.title, e.publish_date, e.podcast_id
FROM episodes e
JOIN insight_topics it ON e.id = it.episode_id
JOIN topics t ON it.topic_id = t.id
WHERE t.label = 'AI Regulation'
  AND e.publish_date >= '2026-01-01'
ORDER BY e.publish_date DESC;
```

**Benefit**: Instant topic-wide views, fast filters by
date/podcast/confidence, easy pagination

### UC2: Speaker-Centric Insight Mapping (Postgres)

**Question**: "What topics does Sam Altman cover, and
what are his top insights?"

```sql
SELECT t.label,
       COUNT(DISTINCT i.id) AS insight_count
FROM speakers s
JOIN quotes q ON s.id = q.speaker_id
JOIN insight_support isup ON q.id = isup.quote_id
JOIN insights i ON isup.insight_id = i.id
JOIN insight_topics it ON i.id = it.insight_id
JOIN topics t ON it.topic_id = t.id
WHERE s.name = 'Sam Altman'
GROUP BY t.label
ORDER BY insight_count DESC
LIMIT 10;
```

**Benefit**: Speaker profiles become one query, easy
ranking by confidence or recency

### UC3: Evidence-backed Quote/Insight Retrieval

**Question**: "Give me the supporting quotes +
timestamps for insight X"

```sql
SELECT i.text AS insight_text,
       q.text AS quote_text,
       q.timestamp_start_ms,
       q.timestamp_end_ms,
       q.transcript_ref,
       q.char_start,
       q.char_end,
       s.name AS speaker_name
FROM insights i
JOIN insight_support isup ON i.id = isup.insight_id
JOIN quotes q ON isup.quote_id = q.id
LEFT JOIN speakers s ON q.speaker_id = s.id
WHERE i.id = 'insight:episode:abc123:a1b2c3d4';
```

**Benefit**: Fetch insights with evidence without
opening JSON, build evidence viewer UI/CLI

### UC5: Insight Explorer (Postgres)

**Question**: "What are the top insights about AI
Regulation with supporting evidence?"

```sql
SELECT i.text AS insight,
       i.confidence,
       i.grounded,
       q.text AS supporting_quote,
       q.timestamp_start_ms,
       q.timestamp_end_ms,
       s.name AS speaker,
       e.title AS episode_title
FROM topics t
JOIN insight_topics it ON t.id = it.topic_id
JOIN insights i ON it.insight_id = i.id
JOIN insight_support isup ON i.id = isup.insight_id
JOIN quotes q ON isup.quote_id = q.id
LEFT JOIN speakers s ON q.speaker_id = s.id
JOIN episodes e ON i.episode_id = e.id
WHERE t.label = 'AI Regulation'
  AND i.grounded = true
ORDER BY i.confidence DESC
LIMIT 20;
```

**Benefit**: Full Insight Explorer query — insights,
quotes, speakers, episodes in one result set

### Notebook Research Workflows

**Topic Dossier**:

```python
import pandas as pd
import sqlalchemy

# Connect to Postgres
engine = sqlalchemy.create_engine("postgresql://...")

# Get insights + quotes + speakers about a topic
query = """
SELECT e.title, e.publish_date,
       i.text AS insight, i.confidence,
       q.text AS quote,
       q.timestamp_start_ms,
       s.name AS speaker
FROM topics t
JOIN insight_topics it ON t.id = it.topic_id
JOIN insights i ON it.insight_id = i.id
JOIN insight_support isup ON i.id = isup.insight_id
JOIN quotes q ON isup.quote_id = q.id
LEFT JOIN speakers s ON q.speaker_id = s.id
JOIN episodes e ON i.episode_id = e.id
WHERE t.label = 'AI Regulation'
ORDER BY i.confidence DESC
LIMIT 50;
"""

df = pd.read_sql(query, engine)
# Build topic dossier with insights, quotes, evidence
```

**Speaker Profile**:

```python
# What does a speaker discuss most, with top insights
query = """
SELECT t.label,
       COUNT(DISTINCT i.id) AS insight_count,
       AVG(i.confidence) AS avg_confidence
FROM speakers s
JOIN quotes q ON s.id = q.speaker_id
JOIN insight_support isup ON q.id = isup.quote_id
JOIN insights i ON isup.insight_id = i.id
JOIN insight_topics it ON i.id = it.insight_id
JOIN topics t ON it.topic_id = t.id
WHERE s.name = 'Sam Altman'
GROUP BY t.label
ORDER BY insight_count DESC;
"""
```

**Quality Audit**:

```python
# Show ungrounded insights or low-confidence quotes
query = """
SELECT i.id, i.text, i.confidence, i.grounded,
       e.title
FROM insights i
JOIN episodes e ON i.episode_id = e.id
WHERE i.grounded = false
   OR i.confidence < 0.5
ORDER BY i.confidence ASC;
"""
```

## Resolved Questions

All product questions have been resolved. Decisions
are recorded here for traceability.

1. **Transcript Text Storage**: Should transcript text
   be stored verbatim or referenced only?
   **Pointers only for v1.** The `quotes` table stores
   `char_start`, `char_end`, `timestamp_start_ms`,
   `timestamp_end_ms`, and `transcript_ref`. Full text
   is resolved at query time by reading the transcript
   file. If latency exceeds 100ms per quote, add a
   `quote_text` denormalized column in v1.1.

2. **Entity/Topic Normalization**: How aggressively
   should normalization occur before export?
   **Slug-based normalization for v1.** Topics are
   normalized to lowercase slugs (e.g., "Machine
   Learning" → `machine-learning`). Semantic dedup
   deferred to v1.1 when sentence embeddings
   (RFC-042) are available.

3. **Graph-Native Store**: When does a graph-native
   store become justified?
   **Not for v1; trigger at ~1000 episodes.** Postgres
   with proper indexes handles the v1 query patterns.
   Graph databases evaluated when multi-hop traversals
   or graph algorithms become common queries.

## Related Work

- **PRD-017**: Grounded Insight Layer — defines the
  GIL data that this PRD exports
- **PRD-019**: Knowledge Graph Layer — defines the KG
  data that this PRD exports (RFC-055 artifact contract)
- **RFC-044**: Model Registry — model metadata tracked
  in provenance fields
- **RFC-042**: Hybrid ML Platform — produces the
  `gi.json` files
- **RFC-049**: Core GIL Concepts & Data Model
- **RFC-050**: GIL Use Cases & Insight Explorer
- **RFC-051**: Database Projection — technical design
- **Issue #31**: Metadata persistence & structure
- **Issue #40**: Data storage / DB integration
- **Issue #50**: Querying & downstream usage

## Release Checklist

- [ ] PRD reviewed and approved
- [ ] RFC-051 finalized with technical design (GIL + KG)
- [ ] Database schema designed (GIL: insights, quotes,
      insight_support, insight_topics; KG: per RFC-051)
- [ ] Export command implemented (`gi export`; `kg
      export` or documented combined mode)
- [ ] Tests cover export, incremental updates, and
      rebuilds (GIL; KG when artifacts available)
- [ ] UC1–UC5 queries validated against Postgres (GIL)
- [ ] KG query examples validated when KG projection
      ships
- [ ] Notebook examples created and documented
- [ ] CLI integration verified (`gi explore`, etc.; KG
      CLI per RFC-055 when applicable)
- [ ] Documentation updated (README, query examples)
- [ ] Integration with GIL pipeline verified; KG
      pipeline verified when KG export ships
