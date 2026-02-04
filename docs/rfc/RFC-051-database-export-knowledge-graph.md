# RFC-051: Podcast Knowledge Graph – Episode & KG Data Store Projection

- **Status**: Draft
- **Authors**: Podcast Scraper Team
- **Stakeholders**: Core team, ML engineers, downstream consumers, power users
- **Related PRDs**:
  - `docs/prd/PRD-017-podcast-knowledge-graph.md` (Knowledge Graph)
  - `docs/prd/PRD-018-database-export-knowledge-graph.md` (Database Export)
- **Related RFCs**:
  - `docs/rfc/RFC-049-podcast-knowledge-graph-core.md` (Core Concepts & Data Model)
  - `docs/rfc/RFC-050-podcast-knowledge-graph-use-cases.md` (Use Cases & Consumption)
- **Related Issues**:
  - Issue #31: Metadata persistence & structure
  - Issue #40: Data storage / DB integration
  - Issue #50: Querying & downstream usage
- **Related Documents**:
  - `docs/kg/ontology.md` - Human-readable ontology specification
  - `docs/kg/kg.schema.json` - Machine-readable schema
  - `docs/ARCHITECTURE.md` - System architecture

## Abstract

This RFC defines how episode-level outputs and Knowledge Graph (KG) metadata are projected from file-based artifacts into a queryable data store (e.g., Postgres, ClickHouse, Elasticsearch). The goal is to enable fast querying, experimentation, and analytics without changing the canonical file-based KG model.

Conceptually, exporting to Postgres provides a fast, queryable "projection" of file-based episode bundles + KG. The KG stays the semantic truth; Postgres becomes the serving layer that makes UC1–UC4 cheap and ergonomic. On disk, `output/episode_x/...` is the canonical, auditable record. In Postgres, a normalized, indexed view enables millisecond queries and tool/UI development.

**Architecture Alignment:** This RFC aligns with existing architecture by:
- Preserving file-based outputs as canonical source of truth
- Enabling incremental updates without reprocessing historical data
- Providing stable SQL interface for downstream tools (CLI, notebooks, web UIs, agents)
- Maintaining full provenance traceability to `kg.json` and transcript evidence

## Problem Statement

The Knowledge Graph (RFC-049, RFC-050) produces structured `kg.json` files per episode, enabling semantic querying and evidence-backed exploration. However, file-based access has limitations:

- **Scalability**: Scanning N folders and parsing N JSON files becomes slow at scale (50 episodes is fine; 5,000 is painful)
- **Query Performance**: No indexing, filtering, or pagination for global queries
- **Integration Friction**: Downstream tools (CLI, notebooks, web UIs, agents) must implement file scanning logic
- **Iteration Speed**: Re-running extraction or comparing runs requires manual file operations

**The Core Problem**: Files are canonical and auditable, but they're not optimized for fast queries and tooling.

**The Solution**: Export to Postgres as a "projection layer" that preserves provenance while enabling SQL-based queries, indexing, and incremental updates. Files remain the source of truth; the database becomes the serving layer.

**Use Cases:**

1. **UC1 – Cross-Podcast Topic Research**: Query all episodes discussing a topic in milliseconds, filter by date/podcast/confidence, paginate results
2. **UC2 – Speaker-Centric Insight Mapping**: Build speaker profiles with top topics and claims via SQL joins
3. **UC3 – Claim Retrieval with Evidence**: Fetch claims with evidence pointers without opening JSON files
4. **UC4 – Semantic Question Answering**: Answer focused questions via deterministic SQL queries

## Goals

1. **Enable Fast Queries**: Transform file scanning into indexed SQL queries for UC1–UC4
2. **Support Notebook Research Workflows**: Enable power users to build topic dossiers, speaker profiles, and entity narratives
3. **Provide Stable Integration Interface**: Give downstream tools (CLI, notebooks, web UIs, agents) a consistent SQL interface
4. **Preserve Provenance**: Every database row is traceable back to `kg.json`, transcript evidence, and extraction metadata
5. **Enable Incremental Growth**: Support upserting new episodes without reprocessing historical data

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
- Upsert global nodes (topics, entities, speakers)
- Insert episode-scoped nodes (claims)
- Insert edge projections (episode_topics, episode_entities, claim_about)
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

**`entities`**
- `id` (PK) - Entity identifier (global, deduplicated)
- `name` - Entity name
- `entity_type` - Entity type (person, company, product, place)

**`claims`**
- `id` (PK) - Claim identifier (episode-scoped)
- `episode_id` (FK) - Reference to episodes
- `speaker_id` (FK) - Reference to speakers
- `text` - Claim text
- `confidence` - Extraction confidence (0.0-1.0)
- `char_start` - Character start in transcript
- `char_end` - Character end in transcript
- `timestamp_start_ms` - Timestamp start (milliseconds)
- `timestamp_end_ms` - Timestamp end (milliseconds)
- `transcript_ref` - Transcript file reference
- `model_version` - Model version used for extraction
- `ingestion_run_id` - Export run identifier

**Relationship Tables:**

**`episode_topics`** (DISCUSSES edge)
- `episode_id` (FK)
- `topic_id` (FK)
- `confidence` - Edge confidence (optional)

**`episode_entities`** (MENTIONS edge)
- `episode_id` (FK)
- `entity_id` (FK)
- `confidence` - Edge confidence (optional)

**`claim_about`** (ABOUT edge)
- `claim_id` (FK)
- `target_id` - Target node ID (topic or entity)
- `target_type` - 'topic' or 'entity'
- `confidence` - Edge confidence (optional)

### 3. Identity & Deduplication Rules

- **Episode ID**: Authoritative and immutable (derived from RSS GUID)
- **Claim IDs**: Episode-scoped (format: `claim:<episode_id>:<hash>`)
- **Speakers, Topics, Entities**: Use stable global IDs (slug-based normalization)
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

### 6. How Postgres Supports UC1–UC4

#### UC1 – Cross-Podcast Topic Research

**Question Examples:**
- "Show me all episodes discussing AI Regulation"
- "Which speakers talk about it most?"

**KG Terms:**
- Topic → Episode via DISCUSSES
- Episode → Speaker via SPOKE_IN

**Postgres Implementation:**
- `topics` + `episode_topics` + `episodes` + `speakers` (via join)

**Benefit:**
- Instant topic-wide views, fast filters by date/podcast/confidence
- Easy pagination ("show next 50 episodes")

**Example SQL:**
```sql
SELECT e.id, e.title, e.publish_date, e.podcast_id
FROM episodes e
JOIN episode_topics et ON e.id = et.episode_id
JOIN topics t ON et.topic_id = t.id
WHERE t.label = 'AI Regulation'
  AND e.publish_date >= '2026-01-01'
ORDER BY e.publish_date DESC;
```

#### UC2 – Speaker-Centric Insight Mapping

**Question Examples:**
- "What topics does Speaker X cover?"
- "What are their most confident claims?"

**KG Terms:**
- Speaker → Claim via ASSERTS
- Claim → Topic/Entity via ABOUT

**Postgres Implementation:**
- `speakers` → `claims` → `claim_about` (+ topics/entities)

**Benefit:**
- Speaker profiles become one query
- Easy ranking: "top claims by confidence", "latest claims", "claims about X"

**Example SQL:**
```sql
SELECT t.label, COUNT(DISTINCT c.id) as claim_count, AVG(c.confidence) as avg_confidence
FROM speakers s
JOIN claims c ON s.id = c.speaker_id
JOIN claim_about ca ON c.id = ca.claim_id
JOIN topics t ON ca.target_id = t.id AND ca.target_type = 'topic'
WHERE s.name = 'Sam Altman'
GROUP BY t.label
ORDER BY claim_count DESC
LIMIT 10;
```

#### UC3 – Claim Retrieval with Evidence

**Question Examples:**
- "Give me the exact quote + timestamp for this claim"
- "Show me context around it"

**KG Terms:**
- Claim has evidence pointing into transcript

**Postgres Implementation:**
- `claims` table stores:
  - `char_start`/`char_end` - Character spans
  - `timestamp_start_ms`/`timestamp_end_ms` - Temporal spans
  - `transcript_ref` and/or transcript path
  - `episode_id`

**Benefit:**
- Fetch claims without opening JSON
- Build claim viewer UI/CLI that resolves evidence reliably
- Audit extraction quality faster

**Example SQL:**
```sql
SELECT c.text, c.timestamp_start_ms, c.timestamp_end_ms,
       c.transcript_ref, c.char_start, c.char_end,
       e.kg_path, e.transcript_path
FROM claims c
JOIN episodes e ON c.episode_id = e.id
WHERE c.id = 'claim:episode:abc123:...';
```

#### UC4 – Semantic Question Answering (v1-Scoped)

**Question Examples:**
- "What claims mention OpenAI?"
- "Which claims are about AI Regulation since Jan 2026?"

**KG Terms:**
- Entity/Topic selection → ABOUT edges → Claims
- Filter by time/confidence

**Postgres Implementation:**
- `entities`/`topics` + `claim_about` + `claims` + `episodes` (for date)

**Benefit:**
- Deterministic "semantic" queries without LLM reasoning
- Fast enough to support interactive tools and future UI

**Example SQL:**
```sql
SELECT c.text, c.confidence, s.name, e.title
FROM entities ent
JOIN claim_about ca ON ent.id = ca.target_id AND ca.target_type = 'entity'
JOIN claims c ON ca.claim_id = c.id
JOIN speakers s ON c.speaker_id = s.id
JOIN episodes e ON c.episode_id = e.id
WHERE ent.name = 'OpenAI'
ORDER BY c.confidence DESC;
```

### 7. Notebook Research Workflows

Once data is in Postgres, power users can build research workflows:

**Topic Dossier:**
```python
import pandas as pd
import sqlalchemy

engine = sqlalchemy.create_engine("postgresql://...")

query = """
SELECT e.title, e.publish_date, c.text, c.confidence, s.name
FROM topics t
JOIN episode_topics et ON t.id = et.topic_id
JOIN episodes e ON et.episode_id = e.id
JOIN claims c ON e.id = c.episode_id
JOIN speakers s ON c.speaker_id = s.id
WHERE t.label = 'AI Regulation'
ORDER BY c.confidence DESC
LIMIT 50;
"""

df = pd.read_sql(query, engine)
# Build topic dossier with claims, speakers, evidence
```

**Speaker Profile:**
```python
query = """
SELECT t.label, COUNT(*) as claim_count, AVG(c.confidence) as avg_confidence
FROM speakers s
JOIN claims c ON s.id = c.speaker_id
JOIN claim_about ca ON c.id = ca.claim_id
JOIN topics t ON ca.target_id = t.id AND ca.target_type = 'topic'
WHERE s.name = 'Sam Altman'
GROUP BY t.label
ORDER BY claim_count DESC;
"""
```

**Quality Audit:**
```python
query = """
SELECT c.id, c.text, c.confidence, e.title
FROM claims c
JOIN episodes e ON c.episode_id = e.id
WHERE c.confidence < 0.5
   OR c.char_start IS NULL
   OR c.timestamp_start_ms IS NULL
ORDER BY c.confidence ASC;
"""
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

2. **Relational Projection Over Graph Database**
   - **Decision**: Use Postgres relational tables, not graph database (Neo4j, ArangoDB)
   - **Rationale**: Simpler ops, covers UC1–UC4, no new infrastructure, SQL is familiar

3. **Pointers Only for Transcript Text (v1)**
   - **Decision**: Store transcript path + spans, not full text
   - **Rationale**: Simpler, smaller DB, canonical stays on disk, easier migration

4. **Incremental Updates by Default**
   - **Decision**: Default to incremental mode, support rebuild flag
   - **Rationale**: Fast updates for new episodes, full control for schema changes

5. **Provenance Tracking Required**
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
- Validate UC1–UC4 queries return expected results
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

1. ✅ Episode + KG data can be exported to Postgres
2. ✅ UC1–UC4 queries run faster via DB than file scan
3. ✅ Data store can be rebuilt from disk with no data loss
4. ✅ All database rows are traceable to source `kg.json` files
5. ✅ Notebook workflows can query Postgres successfully

## Relationship to Other RFCs

This RFC (RFC-051) is part of the Knowledge Graph initiative that includes:

1. **RFC-049: Core KG Concepts & Data Model** - Defines ontology, storage, and schema
2. **RFC-050: KG Use Cases & Consumption** - Defines query patterns and integration
3. **PRD-017: Podcast Knowledge Graph** - Defines product requirements
4. **PRD-018: Database Export** - Defines product requirements for database export

**Key Distinction:**

- **RFC-049**: Focuses on *how* knowledge is extracted and stored (file-based)
- **RFC-050**: Focuses on *how* knowledge is consumed (file-based queries)
- **RFC-051 (This RFC)**: Focuses on *how* knowledge is projected into database for fast queries

Together, these RFCs provide:
- Complete technical design for Knowledge Graph implementation
- Clear separation between file-based storage (RFC-049, RFC-050) and database projection (RFC-051)
- Foundation for fast queries and downstream tooling

## Benefits

1. **Fast Queries**: UC1–UC4 queries complete in milliseconds instead of seconds/minutes
2. **Notebook Workflows**: Power users can build research workflows without file scanning
3. **Stable Interface**: Downstream tools (CLI, notebooks, web UIs, agents) get consistent SQL interface
4. **Incremental Growth**: New episodes can be ingested without reprocessing historical data
5. **Provenance Preserved**: Every database row is traceable to source files and evidence
6. **Tool Transformation**: Project becomes a tool, not just a pipeline

## Migration Path

N/A - This is a new feature, not a migration from an existing system.

## Open Questions

1. **Transcript Text Storage**: Should transcript text be stored verbatim or referenced only?
   - **Current Decision**: Pointers only for v1 (Option A)
   - **Open**: When does full text storage become necessary?

2. **Entity/Topic Normalization**: How aggressively should entity/topic normalization occur before export?
   - **Current Decision**: Basic normalization (slug-based IDs)
   - **Open**: External entity linking (Wikidata, etc.) in v1?

3. **Graph-Native Store**: When does a graph-native store (Neo4j, ArangoDB) become justified?
   - **Current Decision**: Deferred to post-v1
   - **Open**: Performance thresholds that trigger graph database adoption?

## References

- **Related PRD**: `docs/prd/PRD-017-podcast-knowledge-graph.md`
- **Related PRD**: `docs/prd/PRD-018-database-export-knowledge-graph.md`
- **Related RFC**: `docs/rfc/RFC-049-podcast-knowledge-graph-core.md`
- **Related RFC**: `docs/rfc/RFC-050-podcast-knowledge-graph-use-cases.md`
- **Ontology Specification**: `docs/kg/ontology.md`
- **Schema Specification**: `docs/kg/kg.schema.json`
- **Architecture**: `docs/ARCHITECTURE.md`
- **Related Issues**: #31, #40, #50
