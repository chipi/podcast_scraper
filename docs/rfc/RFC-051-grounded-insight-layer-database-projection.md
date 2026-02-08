# RFC-051: Grounded Insight Layer – Database Projection

- **Status**: Draft
- **Authors**: Podcast Scraper Team
- **Stakeholders**: Core team, ML engineers, downstream
  consumers, power users
- **Execution Timing**: **Phase 3b (parallel with
  RFC-049)** — Database projection developed alongside
  GIL core extraction. Depends on RFC-049 for `kg.json`
  schema and data model.
- **Related PRDs**:
  - `docs/prd/PRD-017-grounded-insight-layer.md`
    (Grounded Insight Layer)
  - `docs/prd/PRD-018-database-projection-grounded-insight-layer.md`
    (Database Projection for GIL)
- **Related RFCs**:
  - `docs/rfc/RFC-044-model-registry.md`
    (prerequisite — model metadata for provenance)
  - `docs/rfc/RFC-042-hybrid-summarization-pipeline.md`
    (prerequisite — ML platform that produces GIL data)
  - `docs/rfc/RFC-049-grounded-insight-layer-core.md`
    (Core Concepts & Data Model — primary dependency)
  - `docs/rfc/RFC-050-grounded-insight-layer-use-cases.md`
    (Use Cases & Consumption — parallel)
  - `docs/rfc/RFC-052-locally-hosted-llm-models-with-prompts.md`
    (prompt quality for extraction provenance tracking)
- **Related Issues**:
  - Issue #31: Metadata persistence & structure
  - Issue #40: Data storage / DB integration
  - Issue #50: Querying & downstream usage
- **Related Documents**:
  - `docs/kg/ontology.md` - Human-readable ontology
  - `docs/kg/kg.schema.json` - Machine-readable schema
  - `docs/ARCHITECTURE.md` - System architecture

## Abstract

This RFC defines how episode-level outputs and **Grounded Insight Layer (GIL)** data are projected from file-based artifacts into a queryable data store (e.g., Postgres). The goal is to enable fast **Insight Explorer** queries without changing the canonical file-based model.

The key addition is tables for **insights**, **quotes**, and **insight_support** (the grounding join table). This enables the core user value: query a topic → get insights with supporting quotes → navigate to timestamps—all in milliseconds.

Conceptually, exporting to Postgres provides a fast, queryable "projection" of file-based episode bundles + GIL data. The files stay the semantic truth; Postgres becomes the serving layer that makes UC1–UC5 cheap and ergonomic. On disk, `output/episode_x/...` is the canonical, auditable record. In Postgres, a normalized, indexed view enables millisecond queries.

**Architecture Alignment:** This RFC aligns with existing architecture by:

- Preserving file-based outputs as canonical source of truth
- Enabling incremental updates without reprocessing historical data
- Providing stable SQL interface for downstream tools (CLI, notebooks, web UIs, agents)
- Maintaining full provenance traceability to `kg.json` and transcript evidence
- **Supporting the Insight Explorer pattern** (insights + quotes + timestamps)

## Problem Statement

The Grounded Insight Layer (RFC-049, RFC-050) produces structured `kg.json` files per episode, containing **insights** and **supporting quotes** with grounding relationships. However, file-based access has limitations:

- **Scalability**: Scanning N folders and parsing N JSON files becomes slow at scale (50 episodes is fine; 5,000 is painful)
- **Query Performance**: No indexing, filtering, or pagination for cross-episode insight queries
- **Integration Friction**: Downstream tools (CLI, notebooks, web UIs, agents) must implement file scanning logic
- **Insight Explorer**: The canonical query (UC5) requires joining insights → quotes → episodes across many files

**The Core Problem**: Files are canonical and auditable, but they're not optimized for the **Insight Explorer** pattern.

**The Solution**: Export to Postgres as a "projection layer" with dedicated tables for **insights**, **quotes**, and **insight_support** (grounding join table). This makes the Insight Explorer query fast and ergonomic.

**Use Cases:**

1. **UC1 – Cross-Podcast Topic Research**: Query insights about a topic with supporting quotes in milliseconds
2. **UC2 – Speaker-Centric Insight Mapping**: Build speaker profiles with insights they support via SQL joins
3. **UC3 – Evidence-Backed Retrieval**: Fetch insights with quotes and timestamps without opening JSON files
4. **UC4 – Semantic Question Answering**: Answer focused questions via deterministic SQL queries
5. **UC5 – Insight Explorer**: The canonical query that proves the layer works—fast via indexed tables

## Goals

1. **Enable Fast Insight Queries**: Transform file scanning into indexed SQL queries for UC1–UC5
2. **Support Insight Explorer Pattern**: Make the canonical query (insights + quotes + timestamps) fast
3. **Support Notebook Research Workflows**: Enable power users to build topic dossiers, speaker profiles
4. **Provide Stable Integration Interface**: Give downstream tools (CLI, notebooks, web UIs, agents) consistent SQL
5. **Preserve Provenance**: Every database row traceable to `kg.json`, transcript evidence, extraction metadata
6. **Enable Incremental Growth**: Support upserting new episodes without reprocessing historical data

## Constraints & Assumptions

**Constraints:**

- Must not replace file-based outputs as source of truth (files remain canonical)
- Must preserve full provenance (traceable to `kg.json` and transcript evidence)
- Must support incremental updates without reprocessing historical data
- Must be idempotent and rebuildable from disk
- Must not require graph database as hard dependency (relational projection is sufficient)

**Assumptions:**

- Users have access to Postgres instance (local or remote)
- Database schema can be created/managed by export command
- Transcript text can remain on disk (pointers only in v1)
- Users understand SQL or can use provided query examples
- Export command runs after KG extraction completes

## Design & Implementation

### 1. Export Command & Interface

**Export Command (Conceptual):**

```bash
kg export \
  --output-dir ./output \
  --target postgres \
  --dsn postgresql://user:pass@host:5432/dbname \
  [--rebuild]  # Optional: drop and recreate all tables
```

**Export Responsibilities:**

- Validate `kg.json` against schema
- Load or update episode-level records
- Upsert global nodes (topics, speakers)
- Insert episode-scoped nodes (insights, quotes)
- Insert edge projections (insight_topics,
  insight_support)
- Track `ingestion_run_id` for run comparison

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
- `kg_path` - Path to kg.json
- `schema_version` - KG schema version
- `ingestion_run_id` - Export run identifier

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

**Deferred to v1.1:**

**`entities`** (Deferred - Entity extraction deferred to v1.1)

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

- `schema_version` - KG schema version
- `model_version` - Model version (if ML-derived)
- `ingestion_run_id` - Export run identifier

This enables:

- **Rebuilds**: Full regeneration from disk
- **A/B Comparisons**: Compare extraction quality across runs
- **Rollbacks**: Revert to previous export if needed

### 5. Evidence & Provenance Tracking

**Key Rule**: Every row in Postgres should be traceable back to:

- `episode_id` - Source episode
- `kg_path` - Path to source `kg.json`
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

KG export should be integrated into the existing workflow pipeline:

1. **After KG Extraction**: Export runs after `kg.json` files are generated
2. **Co-Located with Existing Outputs**: Export reads from same episode directories
3. **Optional Step**: Export can be enabled/disabled via config

**Module Boundaries:**

- **Export Module**: New module for database export (follows provider pattern)
- **Storage Module**: Uses existing filesystem utilities (no new I/O abstractions)
- **Schema Validation**: Reuses existing KG schema validation

**Configuration Integration:**

KG export should be controlled via `Config` model:

- `export_to_db: bool` - Enable/disable database export
- `db_target: str` - Target database ('postgres', 'clickhouse', 'elasticsearch')
- `db_dsn: Optional[str]` - Database connection string
- `db_rebuild: bool` - Rebuild mode (drop and recreate)

### 10. Failure Modes

- **Invalid kg.json**: Skip episode, log error, continue with other episodes
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
   - **Decision**: Every row must be traceable to `kg.json`, transcript evidence, extraction metadata
   - **Rationale**: Enables trust, debugging, explainability, run comparison

## Alternatives Considered

1. **Graph Database (Neo4j, ArangoDB)**
   - **Description**: Store KG in native graph database
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
- **Integration Tests**: Test export with real `kg.json` files, validate SQL queries, test incremental updates
- **E2E Tests**: Test full workflow from KG extraction → export → query → results

**Test Organization:**

- Unit tests: `tests/unit/test_kg_export_*.py`
- Integration tests: `tests/integration/test_kg_export_*.py`
- E2E tests: `tests/e2e/test_kg_export_*.py`

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

1. ✅ Episode + GIL data (insights, quotes) can be exported to Postgres
2. ✅ UC1–UC5 queries (including Insight Explorer) run faster via DB than file scan
3. ✅ Grounding relationships are queryable via insight_support table
4. ✅ Data store can be rebuilt from disk with no data loss
5. ✅ All database rows are traceable to source `kg.json` files
6. ✅ Notebook workflows can query insights + quotes successfully

## Relationship to Other RFCs

This RFC (RFC-051) is part of the Grounded Insight
Layer initiative and the broader ML platform:

**Dependency Chain:**

```text
RFC-044 (Model Registry)        → infra
    ▼
RFC-042 (Hybrid ML Platform)    → models
    ├── RFC-052 (LLM Prompts)   → prompt quality
    ▼
RFC-049 (GIL Core)              → extraction
    ├── RFC-050 (Use Cases)     → consumption
    ├── RFC-051 (this RFC)      → serving
    ▼
RFC-053 (Adaptive Routing)      → optimization
```

**GIL Initiative RFCs:**

1. **RFC-049: Core GIL Concepts & Data Model** —
   Defines ontology, grounding contract, and storage
2. **RFC-050: GIL Use Cases & Consumption** — Defines
   query patterns and Insight Explorer
3. **RFC-051 (This RFC): Database Projection** —
   Projects GIL data into Postgres for fast queries
4. **PRD-017: Grounded Insight Layer** — Product
   requirements
5. **PRD-018: Database Export** — Product requirements
   for database export

**Prerequisite RFCs:**

6. **RFC-044: Model Registry** — Model metadata
   tracked in `model_version` provenance fields
7. **RFC-042: Hybrid ML Platform** — Produces the
   `kg.json` files that this RFC exports

**Complementary RFCs:**

8. **RFC-052: Locally Hosted LLM Models** — Prompts
   tracked in `prompt_version` provenance fields

**Key Distinction:**

- **RFC-049**: *How* knowledge is extracted and stored
- **RFC-050**: *How* knowledge is consumed
- **RFC-051**: *How* knowledge is projected for fast
  queries (this RFC)

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
that makes the Grounded Insight Layer queryable at
scale. By mapping the GIL ontology (insights, quotes,
topics, speakers, and their relationships) into
well-indexed Postgres tables, it enables fast queries
for all five use cases defined in RFC-050.

**Key design choices:**

- **Pointers over text** — keeps the DB lean; full
  transcript text is resolved at query time
- **Provenance tracking** — `model_version` and
  `ingestion_run_id` enable quality audits and
  regression detection
- **Slug-based normalization** — simple but effective
  deduplication for v1
- **Relational first** — Postgres handles the v1 query
  complexity; graph databases are a future option

**RFC-051 runs in parallel with RFC-049 (core
extraction) and RFC-050 (consumption). Together they
form the GIL delivery path: extraction (049) →
consumption patterns (050) → fast serving (051).**

## References

- **Related PRD**: `docs/prd/PRD-017-grounded-insight-layer.md`
- **Related PRD**: `docs/prd/PRD-018-database-projection-grounded-insight-layer.md`
- **Related RFC**: `docs/rfc/RFC-049-grounded-insight-layer-core.md`
- **Related RFC**: `docs/rfc/RFC-050-grounded-insight-layer-use-cases.md`
- **Ontology Specification**: `docs/kg/ontology.md`
- **Schema Specification**: `docs/kg/kg.schema.json`
- **Architecture**: `docs/ARCHITECTURE.md`
- **Related Issues**: #31, #40, #50
