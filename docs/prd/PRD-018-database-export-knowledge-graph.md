# PRD-018: Database Export for Knowledge Graph

- **Status**: 📋 Draft
- **Authors**: Podcast Scraper Team
- **Related RFCs**: RFC-049 (Core KG Concepts), RFC-050 (KG Use Cases), RFC-051 (DB Export Design)
- **Related Issues**: #31, #40, #50
- **Related Documents**:
  - `docs/kg/ontology.md` - Human-readable ontology specification
  - `docs/kg/kg.schema.json` - Machine-readable schema

## Summary

The **Database Export for Knowledge Graph** enables fast, queryable access to episode and KG data by projecting file-based artifacts into a relational database (Postgres). This transforms the project from a pipeline into a tool, enabling instant topic/speaker/entity exploration, notebook-based research workflows, and downstream integrations without changing the canonical file-based KG model.

The database serves as a "projection layer" that makes UC1–UC4 queries cheap and ergonomic, while files remain the canonical, auditable record. This enables power users to build research workflows, analysts to explore claims with evidence, and developers to integrate podcast data into applications.

## Background & Context

The Knowledge Graph (PRD-017) produces structured `kg.json` files per episode, enabling semantic querying and evidence-backed exploration. However, file-based access has limitations:

- **Scalability**: Scanning N folders and parsing N JSON files becomes slow at scale (50 episodes is fine; 5,000 is painful)
- **Query Performance**: No indexing, filtering, or pagination for global queries
- **Integration Friction**: Downstream tools (CLI, notebooks, web UIs, agents) must implement file scanning logic
- **Iteration Speed**: Re-running extraction or comparing runs requires manual file operations

**The Core Problem**: Files are canonical and auditable, but they're not optimized for fast queries and tooling.

**The Solution**: Export to Postgres as a "projection layer" that preserves provenance while enabling SQL-based queries, indexing, and incremental updates. Files remain the source of truth; the database becomes the serving layer.

**How it relates to existing features:**

- **Knowledge Graph (PRD-017)**: Database export projects KG data from `kg.json` files into tables
- **Metadata Generation (PRD-004)**: Episode metadata is also exported for complete episode context
- **Transcript Pipeline (PRD-001)**: Transcript paths are stored for evidence resolution
- **KG Use Cases (RFC-050)**: Database export makes UC1–UC4 queries fast and ergonomic

## Goals

1. **Enable Fast Queries**: Transform file scanning into indexed SQL queries for UC1–UC4
2. **Support Notebook Research Workflows**: Enable power users to build topic dossiers, speaker profiles, and entity narratives
3. **Provide Stable Integration Interface**: Give downstream tools (CLI, notebooks, web UIs, agents) a consistent SQL interface
4. **Preserve Provenance**: Every database row is traceable back to `kg.json`, transcript evidence, and extraction metadata
5. **Enable Incremental Growth**: Support upserting new episodes without reprocessing historical data

## Non-Goals (v1)

- Replacing file-based outputs as the source of truth (files remain canonical)
- Graph database as a hard dependency (relational projection is sufficient)
- End-user UI or dashboards (deferred to post-v1)
- Real-time or streaming ingestion (batch export only)
- Storing full transcript text in database (pointers only for v1)
- Advanced analytics (trends, sentiment) - deferred to post-v1
- Cross-run KG merging logic (episode-scoped in v1)

## Personas

- **Power Users (Researchers, Analysts, Founders)**: Need instant topic/speaker/entity exploration, claim retrieval with evidence, and the ability to build research workflows in notebooks
- **Builders (Developers, Agent Builders)**: Need a stable SQL interface for integrating podcast data into CLI tools, web UIs, and RAG pipelines
- **Iterators (ML Engineers, Pipeline Developers)**: Need to re-run extraction, re-export, and compare runs without manual file operations

## User Stories

- _As a researcher, I can query all episodes discussing a topic in milliseconds so that I can build topic dossiers without waiting for file scans._
- _As an analyst, I can retrieve claims with evidence using SQL so that I can cite statements with confidence._
- _As a developer, I can build a CLI tool on top of Postgres so that I can provide instant topic/speaker/entity queries._
- _As a power user, I can build notebook workflows that query Postgres so that I can create speaker profiles, entity narratives, and quality audits._
- _As an ML engineer, I can re-export after model changes so that I can compare extraction quality across runs._

## Functional Requirements

### FR1: Export Command & Interface

- **FR1.1**: Provide `kg export` CLI command that scans output directory and exports to Postgres
- **FR1.2**: Support `--target postgres` with `--dsn` connection string
- **FR1.3**: Support `--output-dir` to specify root output directory
- **FR1.4**: Support `--rebuild` flag to drop and recreate all tables
- **FR1.5**: Support incremental mode (default) that only ingests new episodes

### FR2: Episode Metadata Export

- **FR2.1**: Export episode-level records (id, title, publish_date, podcast_id, file paths)
- **FR2.2**: Store paths to `metadata.json`, `transcript.json`, `summary.json`, `kg.json`
- **FR2.3**: Include `schema_version` and `ingestion_run_id` for lineage tracking

### FR3: Knowledge Graph Node Export

- **FR3.1**: Export global nodes (speakers, topics, entities) with deduplicated IDs
- **FR3.2**: Export episode-scoped nodes (claims) with episode_id foreign key
- **FR3.3**: Preserve all KG node properties (name, label, entity_type, etc.)
- **FR3.4**: Include confidence scores and evidence metadata for ML-derived nodes

### FR4: Knowledge Graph Edge Export

- **FR4.1**: Export relationship tables (episode_topics, episode_entities, claim_about)
- **FR4.2**: Preserve edge properties (confidence, evidence) where applicable
- **FR4.3**: Support filtering by confidence thresholds in queries

### FR5: Evidence & Provenance Tracking

- **FR5.1**: Store transcript references (path, char_start, char_end) for claims
- **FR5.2**: Store timestamp ranges (timestamp_start_ms, timestamp_end_ms) for claims
- **FR5.3**: Include `model_version` and `extraction_method` for ML-derived data
- **FR5.4**: Ensure every row is traceable back to `kg.json` path and episode_id

### FR6: Query Support for UC1–UC4

- **FR6.1**: Support UC1 (Cross-Podcast Topic Research) via `topics` + `episode_topics` + `episodes` joins
- **FR6.2**: Support UC2 (Speaker-Centric Insight Mapping) via `speakers` → `claims` → `claim_about` joins
- **FR6.3**: Support UC3 (Claim Retrieval with Evidence) via `claims` table with evidence pointers
- **FR6.4**: Support UC4 (Semantic Question Answering) via entity/topic → claim filtering

### FR7: Incremental Updates & Rebuilds

- **FR7.1**: Support incremental ingestion (only new episodes by default)
- **FR7.2**: Support full rebuild (drop and recreate all tables)
- **FR7.3**: Handle schema version changes gracefully (warn or fail on mismatch)
- **FR7.4**: Support run comparison via `ingestion_run_id` tracking

## Success Metrics

- **Query Performance**: UC1–UC4 queries complete in < 100ms for datasets up to 5,000 episodes
- **Export Speed**: Export completes in < 5 minutes for 1,000 episodes
- **Provenance Accuracy**: 100% of database rows are traceable to source `kg.json` files
- **Notebook Workflow Adoption**: Positive feedback from power users on research workflow speed
- **Integration Success**: CLI tools and notebooks can query Postgres without file scanning

## Dependencies

- **PRD-017**: Podcast Knowledge Graph (KG data to export)
- **RFC-049**: Core KG Concepts & Data Model (defines ontology and storage)
- **RFC-050**: KG Use Cases & Consumption (defines UC1–UC4 query patterns)
- **RFC-051**: Database Export Design (technical implementation details)

## Constraints & Assumptions

**Constraints:**

- Must not replace file-based outputs as source of truth
- Must preserve full provenance (traceable to `kg.json` and transcript evidence)
- Must support incremental updates without reprocessing historical data
- Must be idempotent and rebuildable from disk
- Must not require graph database as hard dependency

**Assumptions:**

- Users have access to Postgres instance (local or remote)
- Database schema can be created/managed by export command
- Transcript text can remain on disk (pointers only in v1)
- Users understand SQL or can use provided query examples

## Design Considerations

### Transcript Text Storage

- **Option A**: Store only pointers (transcript path + spans) - **RECOMMENDED v1**
  - **Pros**: Simpler, smaller DB, canonical stays on disk, easier migration
  - **Cons**: Claim viewer must read disk files (acceptable for v1)
  - **Decision**: Option A for v1

- **Option B**: Store transcript text in database
  - **Pros**: Evidence resolution is a single query, fully DB-only serving
  - **Cons**: Big database, duplication, more migration pain
  - **Decision**: Deferred to post-v1 if needed

### Database Target Priority

- **Option A**: Postgres primary, ClickHouse/Elasticsearch future
  - **Pros**: Structured joins, strong constraints, simple ops, covers UC1–UC4
  - **Cons**: May need columnar store for aggregations later
  - **Decision**: Option A (Postgres v1, others deferred)

### Incremental vs Full Rebuild

- **Option A**: Support both modes (incremental default, rebuild flag)
  - **Pros**: Fast updates for new episodes, full control for schema changes
  - **Cons**: More complex export logic
  - **Decision**: Option A (both modes supported)

## Integration with Existing Features

Database export enhances the Knowledge Graph by:

- **KG File Output**: Projects `kg.json` files into queryable tables without changing file format
- **Episode Metadata**: Combines KG data with episode metadata for complete context
- **Transcript Evidence**: Stores pointers to transcript files for evidence resolution
- **Use Case Queries**: Makes UC1–UC4 queries fast and ergonomic via SQL

## Example Use Cases

### UC1: Cross-Podcast Topic Research (Postgres)

**Question**: "Show me all episodes discussing AI Regulation since Jan 2026"

**SQL Query**:
```sql
SELECT e.id, e.title, e.publish_date, e.podcast_id
FROM episodes e
JOIN episode_topics et ON e.id = et.episode_id
JOIN topics t ON et.topic_id = t.id
WHERE t.label = 'AI Regulation'
  AND e.publish_date >= '2026-01-01'
ORDER BY e.publish_date DESC;
```

**Benefit**: Instant topic-wide views, fast filters by date/podcast/confidence, easy pagination

### UC2: Speaker-Centric Insight Mapping (Postgres)

**Question**: "What topics does Sam Altman cover, and what are his top claims?"

**SQL Query**:
```sql
SELECT t.label, COUNT(DISTINCT c.id) as claim_count
FROM speakers s
JOIN claims c ON s.id = c.speaker_id
JOIN claim_about ca ON c.id = ca.claim_id
JOIN topics t ON ca.target_id = t.id AND ca.target_type = 'topic'
WHERE s.name = 'Sam Altman'
GROUP BY t.label
ORDER BY claim_count DESC
LIMIT 10;
```

**Benefit**: Speaker profiles become one query, easy ranking by confidence or recency

### UC3: Claim Retrieval with Evidence (Postgres)

**Question**: "Give me the exact quote + timestamp for claim X"

**SQL Query**:
```sql
SELECT c.text, c.timestamp_start_ms, c.timestamp_end_ms,
       c.transcript_ref, c.char_start, c.char_end,
       e.kg_path, e.transcript_path
FROM claims c
JOIN episodes e ON c.episode_id = e.id
WHERE c.id = 'claim:episode:abc123:...';
```

**Benefit**: Fetch claims without opening JSON, build claim viewer UI/CLI that resolves evidence reliably

### Notebook Research Workflows

**Topic Dossier**:
```python
import pandas as pd
import sqlalchemy

# Connect to Postgres
engine = sqlalchemy.create_engine("postgresql://...")

# Get all episodes + top claims + top speakers about a topic
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

**Speaker Profile**:
```python
# What does a speaker claim most, grouped by topic/entity
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

**Quality Audit**:
```python
# Show me low-confidence claims or claims with missing evidence
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

## Related Work

- **Issue #31**: Metadata persistence & structure
- **Issue #40**: Data storage / DB integration
- **Issue #50**: Querying & downstream usage
- **RFC-049**: Core KG Concepts & Data Model
- **RFC-050**: KG Use Cases & Consumption
- **RFC-051**: Database Export Design (technical implementation)

## Release Checklist

- [ ] PRD reviewed and approved
- [ ] RFC-051 created with technical design
- [ ] Database schema designed and documented
- [ ] Export command implemented
- [ ] Tests cover export, incremental updates, and rebuilds
- [ ] UC1–UC4 queries validated against Postgres
- [ ] Notebook examples created and documented
- [ ] CLI integration verified (kg query, kg speaker, etc.)
- [ ] Documentation updated (README, query examples)
- [ ] Integration with KG pipeline verified
