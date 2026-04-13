# ADR-054: Relational Postgres Projection for GIL and KG

- **Status**: Accepted
- **Date**: 2026-04-03
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-051](../rfc/RFC-051-database-projection-gil-kg.md)
- **Related PRDs**: [PRD-018](../prd/PRD-018-database-projection-gil-kg.md)

## Context & Problem Statement

The Grounded Insight Layer (GIL) produces `gi.json` per episode; the Knowledge Graph
(KG) produces per-episode KG artifacts. At scale (hundreds to thousands of episodes),
scanning N directories and parsing N JSON files becomes impractical for cross-episode
queries like the Insight Explorer (UC5), speaker profiles, and KG traversals.

A serving layer is needed to make GIL and KG data queryable without replacing the
file-based canonical source of truth. The key decision is: what kind of data store,
and how do GIL and KG coexist in it?

## Decision

We project GIL and KG data into **relational Postgres tables** as a derived, rebuildable
view. Files remain the canonical source of truth.

1. **Files canonical, database is projection**: On-disk `gi.json` and KG artifacts are
   the semantic truth. Postgres is a serving layer that can be rebuilt from disk at any
   time with zero data loss.
2. **Relational over graph database**: Use Postgres relational tables (normalized joins)
   rather than a graph database (Neo4j, ArangoDB). SQL covers UC1–UC5 for GIL and
   basic KG traversals.
3. **Separate GIL and KG projections**: GIL tables (`insights`, `quotes`,
   `insight_support`) and KG tables (`kg_nodes`, `kg_edges`) are distinct. The shared
   `episodes` table links both but grounding edges (`SUPPORTED_BY`) never mix with KG
   linking edges (`MENTIONS`, `RELATED_TO`).
4. **Pointers only for transcript text (v1)**: Store transcript path + character spans,
   not full transcript text. Evidence is resolved at query time by reading the file.
5. **Provenance on every row**: Every database row carries `ingestion_run_id`,
   `model_version`, and schema version fields. All rows are traceable to their source
   artifact and extraction metadata.
6. **Incremental by default, rebuild on demand**: Default mode upserts new episodes.
   `--rebuild` flag drops and recreates tables from disk.

## Rationale

- **Files-first**: Preserves auditability, co-location, and the ability to reprocess
  without database dependency. The database is additive, not required.
- **Relational**: Postgres is simpler to operate than a graph database, covers the v1
  query patterns (insight → quote, topic → insight, KG node → edges) with standard
  JOINs, and requires no new infrastructure. Graph databases become worth evaluating
  at ~1000 episodes or when multi-hop path queries are needed.
- **Separate projections**: GIL's `SUPPORTED_BY` semantics (grounding) are
  fundamentally different from KG's `MENTIONS`/`RELATED_TO` semantics (linking).
  Mixing them in one table conflates two different product contracts (PRD-017 vs
  PRD-019).
- **Pointers**: Keeps the database lean (~1 KB/quote vs ~50 KB for embedded text).
  Full text can be added as a denormalized cache in v1.1 if query latency warrants it.
- **Provenance**: Enables quality audits, regression detection, run comparison, and
  rollback.

## Alternatives Considered

1. **Graph database (Neo4j, ArangoDB)**: Rejected for v1; adds infrastructure
   complexity, harder to debug, breaks co-location pattern. Revisit at ~1000 episodes
   or when multi-hop traversals are needed (Apache AGE as a Postgres extension is a
   low-migration option).
2. **Store full transcript text in database**: Rejected; duplicates canonical content,
   inflates database size, creates migration headaches. Pointers are sufficient for v1.
3. **Global graph index file (JSON)**: Rejected; still requires file scanning, provides
   no indexing or SQL interface for downstream tools.
4. **Unified GIL + KG tables**: Rejected; conflates grounding semantics with linking
   semantics. Separate tables match the separate product contracts and allow independent
   evolution.

## Consequences

- **Positive**: Millisecond Insight Explorer queries. Notebook research workflows via
  SQL. KG traversals without file scanning. Stable SQL interface for CLI, notebooks,
  web UIs, and agents.
- **Negative**: Adds Postgres as an optional dependency. Export step required after
  extraction. Two-step query for full evidence (DB + file read).
- **Neutral**: `gi export` and `kg export` CLI commands are added. The `episodes` table
  carries both `gi_path` and `kg_path` columns.

## Implementation Notes

- **Module**: Database export module under `src/podcast_scraper/`
- **CLI**: `podcast gi export --target postgres --dsn ...` and
  `podcast kg export --target postgres --dsn ...`
- **Tables**: `episodes`, `speakers`, `topics`, `insights`, `quotes`,
  `insight_support`, `insight_topics` (GIL); `kg_nodes`, `kg_edges` (KG)
- **Identity**: Episode IDs are authoritative (RSS GUID). Insights and quotes use
  episode-scoped hashed IDs. Topics and speakers use slug-based global IDs.
- **Relationship to ADR-051**: ADR-051 defines per-episode JSON artifacts and optional
  materialization at the concept level; this ADR specifies the concrete Postgres
  projection design.

## References

- [RFC-051: Database Projection GIL & KG](../rfc/RFC-051-database-projection-gil-kg.md)
- [ADR-051: Per-Episode JSON Artifacts](ADR-051-per-episode-json-artifacts-with-logical-union.md)
- [ADR-052: Separate GIL and KG Layers](ADR-052-separate-gil-and-kg-artifact-layers.md)
- [PRD-018: Database Projection](../prd/PRD-018-database-projection-gil-kg.md)
