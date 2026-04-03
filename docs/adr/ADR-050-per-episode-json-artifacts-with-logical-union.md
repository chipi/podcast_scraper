# ADR-050: Per-Episode JSON Artifacts with Logical Union

- **Status**: Accepted
- **Date**: 2026-04-03
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-049](../rfc/RFC-049-grounded-insight-layer-core.md), [RFC-055](../rfc/RFC-055-knowledge-graph-layer-core.md), [RFC-061](../rfc/RFC-061-semantic-corpus-search.md)
- **Related PRDs**: [PRD-017](../prd/PRD-017-grounded-insight-layer.md), [PRD-019](../prd/PRD-019-knowledge-graph-layer.md)

## Context & Problem Statement

GIL and KG each produce structured graph data (nodes, edges, metadata) per episode.
The system needs a storage model that supports per-episode reprocessing, debugging,
sharding, and cross-episode queries — without requiring a global database for the
default CLI path.

## Decision

We adopt **per-episode JSON artifacts with logical union**:

1. **One JSON file per episode per layer**: `*.gi.json` for GIL, `*.kg.json` for KG.
   Files live in the episode's output directory alongside metadata, transcript, and
   summary artifacts.
2. **Logical union at query time**: Cross-episode views (e.g. `gi explore`, viewer
   merge, search index) are constructed by reading and merging per-episode files. There
   is no pre-built global artifact.
3. **Optional materialization**: RFC-051 (Postgres projection) and RFC-061 (FAISS
   vector index) provide pre-built global views for scale. These are optional
   accelerators, not replacements for the canonical per-episode files.

## Rationale

- **Debugging**: One file per episode is inspectable, diffable, and self-contained.
- **Reprocessing**: Re-running GIL or KG for a single episode replaces one file without
  touching others.
- **No global state**: CLI mode requires no database, no server, no coordination.
- **Proven pattern**: Aligns with ADR-004 (flat filesystem) and ADR-008
  (database-agnostic metadata).
- **Scale path exists**: RFC-051 and RFC-061 provide materialized views when file scan
  becomes too slow (~100+ episodes).

## Alternatives Considered

1. **Global graph database (Neo4j, SQLite)**: Rejected for v1; adds server dependency
   for CLI users. Deferred to platform mode.
2. **Single merged JSON file**: Rejected; loses per-episode reprocessing, creates
   merge conflicts, grows unboundedly.
3. **Append-only log**: Rejected; harder to query, harder to debug, no random access
   by episode.

## Consequences

- **Positive**: Simple, inspectable, no infrastructure dependency. Each layer (GIL, KG)
  follows the same pattern independently. Consumers can read a single file or scan all.
- **Negative**: Cross-episode queries require scanning all files (O(n) in episodes).
  Mitigated by optional materialization (RFC-051, RFC-061).
- **Neutral**: File naming conventions (`.gi.json`, `.kg.json`) must be consistent
  across all producers and consumers.

## Implementation Notes

- **GIL**: `metadata/<basename>.gi.json` — produced by GIL extraction stage
- **KG**: `metadata/<basename>.kg.json` — produced by KG extraction stage
- **Consumers**: `gi explore`, `gi query`, viewer, search indexer all scan episode
  output directories to build cross-episode views
- **Pattern**: Same co-location principle as ADR-004 (flat filesystem)

## References

- [ADR-004: Flat Filesystem Archive Layout](ADR-004-flat-filesystem-archive-layout.md)
- [ADR-008: Database-Agnostic Metadata Schema](ADR-008-database-agnostic-metadata-schema.md)
- [RFC-049: Grounded Insight Layer — Core](../rfc/RFC-049-grounded-insight-layer-core.md)
- [RFC-055: Knowledge Graph Layer — Core](../rfc/RFC-055-knowledge-graph-layer-core.md)
- [RFC-051: Database Projection](../rfc/RFC-051-database-projection-gil-kg.md)
