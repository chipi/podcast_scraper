# RFC-051: Database Projection (GIL & Knowledge Graph)

- **Status**: Draft
- **Authors**: Podcast Scraper Team
- **Stakeholders**: Core team, ML engineers, downstream
  consumers, power users
- **Execution Timing**: **Phase 3b (parallel with
  RFC-049)** — GIL database projection alongside GIL
  core extraction (depends on RFC-049 for `gi.json`).
  **KG projection** depends on RFC-055 for KG artifact
  shape and may ship in the same codebase as GIL export
  once `kg.schema.json` and pipeline outputs exist.
- **Related PRDs**:
  - `docs/prd/PRD-017-grounded-insight-layer.md`
    (Grounded Insight Layer — `gi.json` source)
  - `docs/prd/PRD-018-database-projection-gil-kg.md`
    (Database Projection for GIL and KG — product scope)
  - `docs/prd/PRD-019-knowledge-graph-layer.md`
    (Knowledge Graph Layer — KG artifact source; **in
    scope** here as relational tables + `kg export`,
    separate from GIL tables)
- **Related RFCs**:
  - `docs/rfc/RFC-044-model-registry.md`
    (prerequisite — model metadata for provenance)
  - `docs/rfc/RFC-042-hybrid-summarization-pipeline.md`
    (prerequisite — ML platform that produces GIL data)
  - `docs/rfc/RFC-049-grounded-insight-layer-core.md`
    (Core Concepts & Data Model — primary dependency)
  - `docs/rfc/RFC-050-grounded-insight-layer-use-cases.md`
    (Use Cases & Consumption — parallel)
  - `docs/rfc/RFC-055-knowledge-graph-layer-core.md`
    (KG artifact model — prerequisite for KG projection)
  - `docs/rfc/RFC-052-locally-hosted-llm-models-with-prompts.md`
    (prompt quality for extraction provenance tracking)
- **Related Issues**:
  - Issue #31: Metadata persistence & structure
  - Issue #40: Data storage / DB integration
  - Issue #50: Querying & downstream usage
- **Related Documents**:
  - `docs/architecture/gi/ontology.md`, `docs/architecture/gi/gi.schema.json` — GIL
  - `docs/architecture/kg/ontology.md`, `docs/architecture/kg/kg.schema.json` — KG
  - `docs/architecture/ARCHITECTURE.md` - System architecture

## Abstract

This RFC defines how episode-level outputs, **Grounded Insight Layer (GIL)** data, and **Knowledge Graph (KG)** data are projected from **file-based artifacts** into a queryable data store (e.g., Postgres). **GIL** and **KG** are separate product contracts (PRD-017 vs PRD-019); this RFC gives them **separate relational projections** in the same database when both are enabled.

**GIL:** The key tables are **insights**, **quotes**, and **insight_support** (grounding join), plus topics/speakers as already outlined. Goal: fast **Insight Explorer** / UC1–UC5 queries (RFC-050) without scanning every folder.

**KG:** Additional tables (or clearly prefixed names) hold **KG nodes and edges** per RFC-055 (episode-anchored graph, entity/topic/relationship semantics distinct from GIL’s Insight/Quote grounding). Goal: fast cross-episode **linking and discovery** queries without conflating KG edges with `SUPPORTED_BY`.

Conceptually, exporting to Postgres provides a fast "projection" of file-based episode bundles. The files stay the semantic truth; Postgres is the serving layer. On disk, `output/episode_x/...` remains canonical. In Postgres, normalized tables enable millisecond queries **per layer**.

**Architecture Alignment:** This RFC aligns with existing architecture by:

- Preserving file-based outputs as canonical source of truth
- Enabling incremental updates without reprocessing historical data
- Providing stable SQL interface for downstream tools (CLI, notebooks, web UIs, agents)
- Maintaining full provenance traceability to `gi.json`, KG artifacts, and transcript evidence
- **Supporting the Insight Explorer pattern** (GIL) and **KG graph queries** (RFC-055) as **independent** projections

## Problem Statement

The Grounded Insight Layer (RFC-049, RFC-050) produces structured `gi.json` files per episode, containing **insights** and **supporting quotes** with grounding relationships. However, file-based access has limitations:

- **Scalability**: Scanning N folders and parsing N JSON files becomes slow at scale (50 episodes is fine; 5,000 is painful)
- **Query Performance**: No indexing, filtering, or pagination for cross-episode insight queries
- **Integration Friction**: Downstream tools (CLI, notebooks, web UIs, agents) must implement file scanning logic
- **Insight Explorer**: The canonical query (UC5) requires joining insights → quotes → episodes across many files

**The Core Problem**: Files are canonical and auditable, but they're not optimized for the **Insight Explorer** pattern.

**The Solution**: Export to Postgres as a "projection layer" with dedicated tables for **insights**, **quotes**, and **insight_support** (grounding join table). This makes the Insight Explorer query fast and ergonomic.

**KG (separate concern):** When the pipeline emits **KG artifacts** (RFC-055), file scanning across episodes is equally painful for **entity/topic/relationship** questions. The same relational projection approach applies: **KG nodes**, **KG edges**, and episode linkage — **not** stored as a second copy inside GIL tables.

**Use Cases (GIL):**

1. **UC1 – Cross-Podcast Topic Research**: Query insights about a topic with supporting quotes in milliseconds
2. **UC2 – Speaker-Centric Insight Mapping**: Build speaker profiles with insights they support via SQL joins
3. **UC3 – Evidence-Backed Retrieval**: Fetch insights with quotes and timestamps without opening JSON files
4. **UC4 – Semantic Question Answering**: Answer focused questions via deterministic SQL queries
5. **UC5 – Insight Explorer**: The canonical query that proves the layer works—fast via indexed tables

**Use Cases (KG):** Illustrative (exact queries depend on RFC-055 v1 node/edge types): “Which episodes mention entity X?”, “What relationships co-occur with topic Y?”, cross-episode rollups for **linking** (not grounding).

## Goals

1. **Enable Fast Insight Queries**: Transform file scanning into indexed SQL queries for UC1–UC5 (GIL)
2. **Enable Fast KG Queries**: Transform file scanning into indexed SQL for KG traversals (RFC-055)
3. **Support Insight Explorer Pattern**: Make the canonical GIL query (insights + quotes + timestamps) fast
4. **Support Notebook Research Workflows**: Enable power users to build topic dossiers, speaker profiles (GIL); entity/topic analytics (KG)
5. **Provide Stable Integration Interface**: Give downstream tools (CLI, notebooks, web UIs, agents) consistent SQL for **both** layers where implemented
6. **Preserve Provenance**: Every database row traceable to `gi.json` and/or KG artifact path, transcript evidence where applicable, extraction metadata
7. **Enable Incremental Growth**: Support upserting new episodes without reprocessing historical data
8. **Keep Layers Separated**: GIL and KG tables remain distinct; optional cross-links between layers stay out of v1 unless a follow-up RFC adds them

## Constraints & Assumptions

**Constraints:**

- Must not replace file-based outputs as source of truth (files remain canonical)
- Must preserve full provenance (traceable to `gi.json` and transcript evidence)
- Must support incremental updates without reprocessing historical data
- Must be idempotent and rebuildable from disk
- Must not require graph database as hard dependency (relational projection is sufficient)

**Assumptions:**

- Users have access to Postgres instance (local or remote)
- Database schema can be created/managed by export command
- Transcript text can remain on disk (pointers only in v1)
- Users understand SQL or can use provided query examples
- Export command runs after GIL extraction completes (and after KG artifact generation when KG export is used)

## Design & Implementation

### 1. Export Command & Interface

**GIL export (conceptual):**

```bash
podcast_scraper gi export \
  --output-dir ./output \
  --target postgres \
  --dsn postgresql://user:pass@host:5432/dbname \
  [--rebuild]  # Optional: drop and recreate GIL-related tables
```

**KG export (conceptual):**

```bash
podcast_scraper kg export \
  --output-dir ./output \
  --target postgres \
  --dsn postgresql://user:pass@host:5432/dbname \
  [--rebuild]  # Optional: drop and recreate KG-related tables
```

**Combined operation:** A single orchestrated command that runs **both** projections in one process (shared `episodes` upsert, separate GIL vs KG table writes) is allowed if documented; flags must keep **GIL-only** and **KG-only** modes for operators who enable only one feature.

**GIL export responsibilities:**

- Validate `gi.json` against `docs/architecture/gi/gi.schema.json`
- Load or update episode-level records (including `gi_path`, `gi_schema_version`)
- Upsert global nodes (topics, speakers) used by GIL
- Insert episode-scoped nodes (insights, quotes)
- Insert edge projections (insight_topics, insight_support)
- Track `ingestion_run_id` for run comparison

**KG export responsibilities:**

- Validate KG artifact against `docs/architecture/kg/kg.schema.json` when present
- Upsert episode-level `kg_path`, `kg_schema_version` when KG data exists
- Insert **kg_nodes**, **kg_edges** (names illustrative; exact columns follow RFC-055 types)
- Track `ingestion_run_id` (may share run id with GIL export in combined mode)

**Incremental vs Full Rebuild:**

- **Default (Incremental)**: Scan output directory, ingest episodes not yet present in data store
- **Rebuild**: Drop and recreate all tables, re-ingest from disk

### 2. Data Model (Relational Projection – v1)

**Core Tables:**

**`episodes`**

- `id` (PK) - Episode identifier
- `podcast_id` - Podcast identifier
- `title` - Episode title
- `publish_date` - Publication date
- `metadata_path` - Path to metadata.json
- `transcript_path` - Path to transcript.json
- `summary_path` - Path to summary.json (optional)
- `gi_path` - Path to `gi.json` (nullable if GIL not generated for this episode)
- `gi_schema_version` - GIL `gi.schema.json` version when `gi_path` is set
- `kg_path` - Path to per-episode KG artifact (nullable until RFC-055 output exists)
- `kg_schema_version` - KG schema version when `kg_path` is set
- `ingestion_run_id` - Export run identifier (last successful projection)

**`speakers`**

- `id` (PK) - Speaker identifier (global, deduplicated)
- `name` - Speaker name

**`topics`**

- `id` (PK) - Topic identifier (global, deduplicated)
- `label` - Topic label

**`insights`** (NEW - Core GIL Table)

- `id` (PK) - Insight identifier (episode-scoped)
- `episode_id` (FK) - Reference to episodes
- `text` - Insight text (takeaway)
- `grounded` - Boolean: has ≥1 supporting quote
- `confidence` - Extraction confidence (0.0-1.0)
- `model_version` - Model version used for extraction
- `prompt_version` - Prompt version used for extraction
- `ingestion_run_id` - Export run identifier

**`quotes`** (NEW - Evidence Table)

- `id` (PK) - Quote identifier (episode-scoped)
- `episode_id` (FK) - Reference to episodes
- `speaker_id` (FK, nullable) - Reference to speakers (nullable if no diarization)
- `text` - Verbatim quote text
- `char_start` - Character start in transcript
- `char_end` - Character end in transcript
- `timestamp_start_ms` - Timestamp start (milliseconds)
- `timestamp_end_ms` - Timestamp end (milliseconds)
- `transcript_ref` - Transcript file reference
- `ingestion_run_id` - Export run identifier

**Relationship Tables:**

**`insight_support`** (NEW - Grounding Join Table: SUPPORTED_BY edge)

- `insight_id` (FK) - Reference to insights
- `quote_id` (FK) - Reference to quotes
- PRIMARY KEY (insight_id, quote_id)

**`insight_topics`** (ABOUT edge: Insight → Topic)

- `insight_id` (FK) - Reference to insights
- `topic_id` (FK) - Reference to topics
- `confidence` - Edge confidence (optional)

### 2B. KG Tables (RFC-055)

GIL tables above model **insights, quotes, and grounding**. KG uses **separate** tables so `MENTIONS` / `RELATED_TO` (or RFC-055 equivalents) never collide with `SUPPORTED_BY` semantics.

**Illustrative names (finalize with `kg.schema.json`):**

**`kg_nodes`**

- `id` (PK) - Stable node id (episode-scoped or global per RFC-055)
- `episode_id` (FK) - Episode anchor
- `node_type` - e.g. Episode, Entity, Topic (RFC-055)
- `payload` - JSONB or typed columns for label, slug, properties
- `ingestion_run_id` - Export run identifier

**`kg_edges`**

- `id` (PK) - Edge id
- `episode_id` (FK) - Episode anchor (v1 per-episode graph)
- `src_node_id` (FK) - Source node
- `dst_node_id` (FK) - Destination node
- `edge_type` - e.g. MENTIONS, RELATED_TO (RFC-055)
- `properties` - JSONB optional (offsets, confidence)
- `ingestion_run_id` - Export run identifier

**Deferred / v1.1:**

**`entities` in GIL sense** (RFC-049 entity nodes) may remain deferred separately from **KG Entity** nodes; do not merge without an explicit RFC.

**`entities`** (Deferred - GIL entity extraction deferred to v1.1 per RFC-049)

- `id` (PK) - Entity identifier (global, deduplicated)
- `name` - Entity name
- `entity_type` - Entity type (person, company, product, place)

### 3. Identity & Deduplication Rules

- **Episode ID**: Authoritative and immutable (derived from RSS GUID)
- **Insight IDs**: Episode-scoped (format: `insight:<episode_id>:<hash>`)
- **Quote IDs**: Episode-scoped (format: `quote:<episode_id>:<hash>` or `quote:<episode_id>:<char_start>-<char_end>`)
- **Speakers, Topics**: Use stable global IDs (slug-based normalization)
- **Deduplication Logic**: Lives outside the exporter (resolved before export)

### 4. Versioning & Lineage

Every exported row SHOULD include:

- `gi_schema_version` / `kg_schema_version` (as applicable) — line up with `gi.json` vs KG artifact
- `model_version` - Model version (if ML-derived)
- `ingestion_run_id` - Export run identifier

This enables:

- **Rebuilds**: Full regeneration from disk
- **A/B Comparisons**: Compare extraction quality across runs
- **Rollbacks**: Revert to previous export if needed

### 5. Evidence & Provenance Tracking

**Key Rule**: Every row in Postgres should be traceable back to:

- `episode_id` - Source episode
- `gi_path` - Path to source `gi.json`
- `transcript_ref` + `char_start`/`char_end` - Transcript evidence span
- `timestamp_start_ms`/`timestamp_end_ms` - Temporal evidence span
- `schema_version` / `model_version` / `ingestion_run_id` - Extraction metadata

**Transcript Text Storage Decision (v1):**

- **Option A**: Store only pointers (transcript path + spans) - **RECOMMENDED v1**
  - **Pros**: Simpler, smaller DB, canonical stays on disk, easier migration
  - **Cons**: Claim viewer must read disk files (acceptable for v1)
  - **Decision**: Option A for v1

- **Option B**: Store transcript text in database (deferred to post-v1)
  - **Pros**: Evidence resolution is a single query, fully DB-only serving
  - **Cons**: Big database, duplication, more migration pain

### 6. How Postgres Supports UC1–UC5

#### UC1 – Cross-Podcast Topic Research

**Question Examples:**

- "Show me insights about AI Regulation"
- "Which speakers have quotes supporting these insights?"

**GIL Terms:**

- Topic → Insight via insight_topics
- Insight → Quote via insight_support
- Quote → Speaker via speaker_id

**Postgres Implementation:**

- `topics` + `insight_topics` + `insights` + `insight_support` + `quotes` + `speakers`

**Benefit:**

- Instant topic-wide insight views with supporting quotes
- Easy pagination ("show next 50 insights")

**Example SQL:**

```sql
SELECT i.id, i.text, i.grounded, i.confidence, e.title, e.publish_date
FROM insights i
JOIN insight_topics it ON i.id = it.insight_id
JOIN topics t ON it.topic_id = t.id
JOIN episodes e ON i.episode_id = e.id
WHERE t.label = 'AI Regulation'
  AND i.grounded = true
ORDER BY i.confidence DESC;
```

#### UC2 – Speaker-Centric Insight Mapping

**Question Examples:**

- "What topics does Speaker X have quotes about?"
- "What insights do their quotes support?"

**GIL Terms:**

- Speaker → Quote via speaker_id
- Quote → Insight via insight_support
- Insight → Topic via insight_topics

**Postgres Implementation:**

- `speakers` → `quotes` → `insight_support` → `insights` → `insight_topics` → `topics`

**Benefit:**

- Speaker profiles with insights they support (via their quotes)
- Easy ranking: "top insights by confidence", "topics with most quotes"

**Example SQL:**

```sql
SELECT t.label, COUNT(DISTINCT i.id) as insight_count, COUNT(DISTINCT q.id) as quote_count
FROM speakers s
JOIN quotes q ON s.id = q.speaker_id
JOIN insight_support isup ON q.id = isup.quote_id
JOIN insights i ON isup.insight_id = i.id
JOIN insight_topics it ON i.id = it.insight_id
JOIN topics t ON it.topic_id = t.id
WHERE s.name = 'Sam Altman'
GROUP BY t.label
ORDER BY quote_count DESC
LIMIT 10;
```

#### UC3 – Evidence-Backed Quote/Insight Retrieval

**Question Examples:**

- "Give me the exact quotes supporting this insight"
- "Show me timestamps for each quote"

**GIL Terms:**

- Insight → Supporting Quotes via insight_support
- Quote has transcript evidence

**Postgres Implementation:**

- `insights` → `insight_support` → `quotes` (with char_start/char_end, timestamps)

**Benefit:**

- Fetch insights with all supporting quotes in one query
- Build insight viewer UI/CLI that resolves evidence reliably

**Example SQL:**

```sql
SELECT i.text as insight_text, i.grounded, i.confidence,
       q.text as quote_text, q.timestamp_start_ms, q.timestamp_end_ms,
       q.char_start, q.char_end, s.name as speaker_name,
       e.transcript_path
FROM insights i
JOIN insight_support isup ON i.id = isup.insight_id
JOIN quotes q ON isup.quote_id = q.id
LEFT JOIN speakers s ON q.speaker_id = s.id
JOIN episodes e ON i.episode_id = e.id
WHERE i.id = 'insight:episode:abc123:a1b2c3d4';
```

#### UC4 – Semantic Question Answering (v1-Scoped)

**Question Examples:**

- "What insights are there about AI Regulation?"
- "Which insights since Jan 2026 have the most evidence?"

**GIL Terms:**

- Topic selection → insight_topics → insights
- Filter by time/confidence/grounding

**Postgres Implementation:**

- `topics` + `insight_topics` + `insights` + `episodes` (for date)

**Benefit:**

- Deterministic "semantic" queries without LLM reasoning
- Fast enough to support interactive tools and future UI

**Example SQL:**

```sql
SELECT i.text, i.grounded, i.confidence, e.title,
       COUNT(isup.quote_id) as supporting_quote_count
FROM topics t
JOIN insight_topics it ON t.id = it.topic_id
JOIN insights i ON it.insight_id = i.id
JOIN episodes e ON i.episode_id = e.id
LEFT JOIN insight_support isup ON i.id = isup.insight_id
WHERE t.label = 'AI Regulation'
  AND e.publish_date >= '2026-01-01'
GROUP BY i.id, i.text, i.grounded, i.confidence, e.title
ORDER BY supporting_quote_count DESC, i.confidence DESC;
```

#### UC5 – Insight Explorer (The Canonical Query)

**Question Examples:**

- "Show me all insights about AI Regulation with supporting quotes"
- "Give me the full insight report for a topic"

**GIL Terms:**

- Topic → Insights → Supporting Quotes → Speakers + Episodes + Timestamps

**Postgres Implementation:**

- Full join across all GIL tables

**Benefit:**

- The canonical query that proves the layer works
- Delivers insights + quotes + timestamps in one fast query

**Example SQL (Insight Explorer):**

```sql
WITH topic_insights AS (
  SELECT i.id, i.text, i.grounded, i.confidence, i.episode_id
  FROM insights i
  JOIN insight_topics it ON i.id = it.insight_id
  JOIN topics t ON it.topic_id = t.id
  WHERE t.label = 'AI Regulation'
    AND i.grounded = true
  ORDER BY i.confidence DESC
  LIMIT 20
)
SELECT
  ti.text as insight_text,
  ti.grounded,
  ti.confidence,
  e.title as episode_title,
  e.publish_date,
  q.text as quote_text,
  q.timestamp_start_ms,
  q.timestamp_end_ms,
  s.name as speaker_name
FROM topic_insights ti
JOIN episodes e ON ti.episode_id = e.id
LEFT JOIN insight_support isup ON ti.id = isup.insight_id
LEFT JOIN quotes q ON isup.quote_id = q.id
LEFT JOIN speakers s ON q.speaker_id = s.id
ORDER BY ti.confidence DESC, q.timestamp_start_ms ASC;
```

### 7. Notebook Research Workflows

Once data is in Postgres, power users can build research workflows:

**Topic Dossier (Insights + Quotes):**

```python
import pandas as pd
import sqlalchemy

engine = sqlalchemy.create_engine("postgresql://...")

query = """
SELECT i.text as insight, i.grounded, i.confidence,
       q.text as quote, q.timestamp_start_ms, s.name as speaker,
       e.title as episode
FROM topics t
JOIN insight_topics it ON t.id = it.topic_id
JOIN insights i ON it.insight_id = i.id
JOIN episodes e ON i.episode_id = e.id
LEFT JOIN insight_support isup ON i.id = isup.insight_id
LEFT JOIN quotes q ON isup.quote_id = q.id
LEFT JOIN speakers s ON q.speaker_id = s.id
WHERE t.label = 'AI Regulation'
ORDER BY i.confidence DESC
LIMIT 100;
"""

df = pd.read_sql(query, engine)
# Build topic dossier with insights, supporting quotes, speakers
```

**Speaker Profile (Insights Supported):**

```python
query = """
SELECT t.label, COUNT(DISTINCT i.id) as insight_count,
       COUNT(DISTINCT q.id) as quote_count
FROM speakers s
JOIN quotes q ON s.id = q.speaker_id
JOIN insight_support isup ON q.id = isup.quote_id
JOIN insights i ON isup.insight_id = i.id
JOIN insight_topics it ON i.id = it.insight_id
JOIN topics t ON it.topic_id = t.id
WHERE s.name = 'Sam Altman'
GROUP BY t.label
ORDER BY quote_count DESC;
"""
```

**Grounding Quality Audit:**

```python
query = """
SELECT i.id, i.text, i.grounded, i.confidence, e.title,
       COUNT(isup.quote_id) as quote_count
FROM insights i
JOIN episodes e ON i.episode_id = e.id
LEFT JOIN insight_support isup ON i.id = isup.insight_id
WHERE i.grounded = false
   OR i.confidence < 0.5
GROUP BY i.id, i.text, i.grounded, i.confidence, e.title
ORDER BY i.confidence ASC;
"""

# Find insights with grounding issues
```

**Quote Validity Audit:**

```python
query = """
SELECT q.id, q.text, q.char_start, q.char_end,
       q.timestamp_start_ms, e.transcript_path
FROM quotes q
JOIN episodes e ON q.episode_id = e.id
WHERE q.char_start IS NULL
   OR q.char_end IS NULL
   OR q.timestamp_start_ms IS NULL;
"""

# Find quotes with missing evidence pointers
```

### 8. Data Store Options

**Postgres (Primary v1 Target)**
- Structured joins
- Strong constraints
- Simple ops
- Covers UC1–UC4

**ClickHouse (Future)**
- Aggregations and trends
- Columnar storage for analytics

**Elasticsearch / OpenSearch (Future)**
- Full-text search over transcripts and claims
- Advanced search features

All targets share the same conceptual export model.

### 9. Integration Points

**Workflow Integration:**

Database export integrates with the existing pipeline as **optional post-steps**:

1. **After GIL Extraction**: `gi export` runs after `gi.json` files are generated (when enabled)
2. **After KG Generation**: `kg export` runs after per-episode KG artifacts exist (when enabled)
3. **Co-Located with Existing Outputs**: Export reads from the same episode directories
4. **Optional Step**: Export can be enabled/disabled via config; **GIL and KG toggles remain independent** (PRD-017 / PRD-019)

**Module Boundaries:**

- **Export Module**: Database export module (follows existing project patterns)
- **Storage Module**: Uses existing filesystem utilities (no new I/O abstractions)
- **Schema Validation**: Reuses **`gi.schema.json`** validation for GIL; **`kg.schema.json`** validation for KG when available

**Configuration Integration:**

Export should be controlled via `Config` model (exact field names TBD in implementation PR):

- `export_gi_to_db` / `export_kg_to_db` (or combined `export_to_db` with layer flags) — Enable/disable per layer
- `db_target: str` - Target database ('postgres', …)
- `db_dsn: Optional[str]` - Database connection string
- `db_rebuild: bool` - Rebuild mode (drop and recreate; may be scoped per layer)

### 10. Failure Modes

- **Invalid gi.json**: Skip episode, log error, continue with other episodes
- **Invalid KG artifact**: Skip episode for KG tables, log error; GIL export may still proceed
- **Partial episode output**: Ingest what is available, mark incomplete
- **Schema mismatch**: Fail fast with clear error message
- **Database connection failure**: Fail with clear error, do not proceed

## Key Decisions

1. **Files Are Canonical, Database Is Projection**
   - **Decision**: Files remain source of truth; database is derived, rebuildable view
   - **Rationale**: Preserves auditability, enables reprocessing, maintains co-location pattern

2. **Insights + Quotes + Grounding Tables**
   - **Decision**: Add dedicated tables for insights, quotes, and insight_support (grounding join)
   - **Rationale**: Enables Insight Explorer query; makes grounding relationships queryable

3. **Relational Projection Over Graph Database**
   - **Decision**: Use Postgres relational tables, not graph database (Neo4j, ArangoDB)
   - **Rationale**: Simpler ops, covers UC1–UC5, no new infrastructure, SQL is familiar

4. **Pointers Only for Transcript Text (v1)**
   - **Decision**: Store transcript path + spans, not full text
   - **Rationale**: Simpler, smaller DB, canonical stays on disk, easier migration

5. **Incremental Updates by Default**
   - **Decision**: Default to incremental mode, support rebuild flag
   - **Rationale**: Fast updates for new episodes, full control for schema changes

6. **Provenance Tracking Required**
   - **Decision**: Every row must be traceable to `gi.json`, transcript evidence, extraction metadata
   - **Rationale**: Enables trust, debugging, explainability, run comparison

7. **GIL and KG Projections Stay Separate**
   - **Decision**: Distinct tables (and CLI entrypoints) for GIL vs KG; episode row may hold both `gi_path` and `kg_path`
   - **Rationale**: Matches PRD-017 vs PRD-019 separation; avoids mixing grounding edges with KG linking edges

## Alternatives Considered

1. **Graph Database (Neo4j, ArangoDB)**
   - **Description**: Store **KG** (or unified graph) in native graph database
   - **Pros**: Faster graph queries, built-in graph operations, natural KG representation
   - **Cons**: Requires separate infrastructure, harder to debug, breaks co-location pattern, adds complexity
   - **Why Rejected**: Relational projection covers UC1–UC4, simpler ops, no new infrastructure needed

2. **Store Full Transcript Text in Database**
   - **Description**: Store transcript text in database for single-query evidence resolution
   - **Pros**: Evidence resolution is a single query, fully DB-only serving
   - **Cons**: Big database, duplication, more migration pain, breaks canonical file pattern
   - **Why Rejected**: Pointers are sufficient for v1, keeps DB smaller, preserves file-based canonical model

3. **Global Graph Index File**
   - **Description**: Build global index JSON file instead of database
   - **Pros**: No database dependency, simpler deployment
   - **Cons**: Still requires file scanning, no indexing, no SQL interface
   - **Why Rejected**: Doesn't solve query performance problem, no SQL interface for downstream tools

4. **Separate Export Service**
   - **Description**: Build separate service/daemon for continuous export
   - **Pros**: Real-time updates, decoupled from pipeline
   - **Cons**: Adds infrastructure, complexity, not needed for v1
   - **Why Rejected**: Batch export is sufficient for v1, simpler to maintain

## Testing Strategy

**Test Coverage:**

- **Unit Tests**: Test export command, schema creation, data transformation, ID generation
- **Integration Tests**: Test export with real `gi.json` files, validate SQL queries, test incremental updates
- **E2E Tests**: Test full workflow from GIL extraction → export → query → results

**Test Organization:**

- Unit tests: `tests/unit/test_gi_export_*.py`,
  `tests/unit/test_kg_export_*.py` (when implemented)
- Integration tests: `tests/integration/test_gi_export_*.py`,
  `tests/integration/test_kg_export_*.py`
- E2E tests: `tests/e2e/test_gi_export_*.py`,
  `tests/e2e/test_kg_export_*.py`

**Test Execution:**

- Run in CI as part of standard test suite
- Use test Postgres instance (Docker or testcontainers)
- Validate UC1–UC5 queries return expected results
- Test incremental updates and rebuilds

## Rollout & Monitoring

**Rollout Plan:**

- **Phase 1**: Implement export command and Postgres schema
- **Phase 2**: Add incremental update logic
- **Phase 3**: Add UC1–UC4 query validation
- **Phase 4**: Documentation and notebook examples

**Monitoring:**

- Track export success rate (episodes exported successfully)
- Monitor export performance (time to export N episodes)
- Track query performance (UC1–UC4 query latency)
- Monitor database size growth

**Success Criteria:**

1. Episode + GIL data (insights, quotes) can be exported to Postgres
2. Episode + KG data (nodes, edges) can be exported when KG artifacts exist
3. UC1–UC5 queries (including Insight Explorer) run faster via DB than file scan
4. Grounding relationships are queryable via `insight_support` table
5. Data store can be rebuilt from disk with no data loss
6. All database rows are traceable to source `gi.json` and/or KG artifact files
7. Notebook workflows can query insights + quotes successfully; KG examples documented when shipped

## Relationship to Other RFCs

This RFC (RFC-051) serves **GIL** (with RFC-049/050) and **KG**
(with RFC-055) under the broader ML platform.

**Dependency Chain (conceptual):**

```text
RFC-044 (Model Registry)        → infra
    ▼
RFC-042 (Hybrid ML Platform)    → models
    ├── RFC-052 (LLM Prompts)   → prompt quality
    ▼
RFC-049 (GIL Core)              → gi.json
    ├── RFC-050 (Use Cases)     → consumption
    ├── RFC-051 (this RFC)      → GIL serving (Postgres)
    ▼
RFC-055 (KG Core)               → KG artifacts
    └── RFC-051 (this RFC)      → KG serving (Postgres)
    ▼
RFC-053 (Adaptive Routing)      → optimization
```

**GIL Initiative RFCs:**

1. **RFC-049: Core GIL Concepts & Data Model** —
   Defines ontology, grounding contract, and storage
2. **RFC-050: GIL Use Cases & Consumption** — Defines
   query patterns and Insight Explorer
3. **RFC-051 (This RFC): Database Projection** —
   Projects **GIL and KG** data into Postgres for fast queries
4. **PRD-017: Grounded Insight Layer** — Product
   requirements
5. **PRD-018: Database Export** — Product requirements
   for database export (**GIL + KG**)
6. **RFC-055: Knowledge Graph — Core** — Artifact
   model for KG projection tables
7. **PRD-019: Knowledge Graph Layer** — Product
   requirements for KG

**Prerequisite RFCs:**

- **RFC-044: Model Registry** — Model metadata
   tracked in `model_version` provenance fields
- **RFC-042: Hybrid ML Platform** — Produces pipeline
   outputs including `gi.json` when GIL is enabled

**Complementary RFCs:**

- **RFC-052: Locally Hosted LLM Models** — Prompts
   tracked in `prompt_version` provenance fields

**Key Distinction:**

- **RFC-049**: *How* GIL knowledge is extracted and stored (`gi.json`)
- **RFC-050**: *How* GIL knowledge is consumed
- **RFC-055**: *How* KG knowledge is structured on disk
- **RFC-051**: *How* GIL and KG are projected for fast
  SQL queries (this RFC)

## Benefits

1. **Fast Insight Queries**: UC1–UC5 queries complete in milliseconds instead of seconds/minutes
2. **Insight Explorer Support**: The canonical query (insights + quotes + timestamps) is fast
3. **Grounding is Queryable**: insight_support table makes grounding relationships easy to explore
4. **Notebook Workflows**: Power users can build research workflows without file scanning
5. **Stable Interface**: Downstream tools (CLI, notebooks, web UIs, agents) get consistent SQL interface
6. **Incremental Growth**: New episodes can be ingested without reprocessing historical data
7. **Provenance Preserved**: Every database row is traceable to source files and evidence

## Migration Path

N/A - This is a new feature, not a migration from an existing system.

## Resolved Questions

All design questions have been resolved. Decisions are
recorded here for traceability.

1. **Transcript Text Storage**: Should transcript text
   be stored verbatim or referenced only?
   **Pointers only for v1.** The `quotes` table stores
   `char_start`, `char_end`, `timestamp_start_ms`,
   `timestamp_end_ms`, and `transcript_ref` (file path
   or URI). Full text is resolved at query time by
   reading the transcript file. This keeps DB size
   small (~1 KB per quote vs ~50 KB for embedded
   text). If query latency for transcript resolution
   exceeds 100ms per quote, a `quote_text` column
   can be added as a denormalized cache in v1.1.

2. **Entity/Topic Normalization**: How aggressively
   should entity/topic normalization occur?
   **Slug-based normalization for v1.** Topics are
   normalized to lowercase slugs (e.g.,
   "Machine Learning" → `machine-learning`). This
   catches exact duplicates and simple variations.
   Semantic deduplication (e.g., "ML" = "Machine
   Learning") is deferred to v1.1 when sentence
   embeddings (RFC-042 §11.4.2) are available for
   similarity scoring. Entity normalization (Wikidata
   linking, etc.) is deferred to v1.1+ alongside
   entity extraction itself.

3. **Graph-Native Store**: When does a graph-native
   store become justified?
   **Not for v1; trigger at ~1000 episodes or
   complex path queries.** Postgres with proper
   indexes handles the GIL v1 query patterns
   (insight → quote, topic → insight) efficiently.
   Graph databases become valuable when: (a) episode
   count exceeds ~1000 and JOIN depth grows, (b) new
   query patterns require multi-hop traversals
   (e.g., "speakers who discussed topics in common
   across episodes"), or (c) real-time graph
   algorithms (PageRank, community detection) are
   needed. At that point, evaluate Neo4j or Apache
   AGE (Postgres extension) for minimal migration.

---

## Conclusion

RFC-051 provides the **relational projection layer**
that makes **GIL** queryable at scale and **KG**
queryable when RFC-055 artifacts exist. By mapping the
GIL ontology (insights, quotes, topics, speakers, and
their relationships) and the KG graph (nodes and edges
per RFC-055) into **separate**, well-indexed Postgres
tables, it enables fast queries for RFC-050 use cases
**and** KG discovery workloads without conflating the
two layers.

**Key design choices:**

- **Pointers over text** — keeps the DB lean; full
  transcript text is resolved at query time (GIL quotes)
- **Provenance tracking** — `model_version` and
  `ingestion_run_id` enable quality audits and
  regression detection
- **Slug-based normalization** — simple but effective
  deduplication for v1 (GIL topics; KG per RFC-055)
- **Relational first** — Postgres handles the v1 query
  complexity; native graph stores remain optional for
  later evaluation

**RFC-051 aligns with RFC-049 / RFC-050 for GIL and with
RFC-055 for KG. GIL delivery path: extraction (049) →
consumption (050) → serving (051). KG delivery path:
artifact contract (055) → serving (051).**

## References

- **Related PRD**: `docs/prd/PRD-017-grounded-insight-layer.md`
- **Related PRD**: `docs/prd/PRD-018-database-projection-gil-kg.md`
- **Related PRD**: `docs/prd/PRD-019-knowledge-graph-layer.md`
- **Related RFC**: `docs/rfc/RFC-049-grounded-insight-layer-core.md`
- **Related RFC**: `docs/rfc/RFC-050-grounded-insight-layer-use-cases.md`
- **Related RFC**: `docs/rfc/RFC-055-knowledge-graph-layer-core.md`
- **Ontology Specification (GIL)**: `docs/architecture/gi/ontology.md`
- **Schema Specification (GIL)**: `docs/architecture/gi/gi.schema.json`
- **Ontology (KG)**: `docs/architecture/kg/ontology.md`
- **Schema (KG)**: `docs/architecture/kg/kg.schema.json`
- **Architecture**: `docs/architecture/ARCHITECTURE.md`
- **Related Issues**: #31, #40, #50
