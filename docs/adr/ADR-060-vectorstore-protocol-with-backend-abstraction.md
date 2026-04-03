# ADR-060: VectorStore Protocol with Backend Abstraction

- **Status**: Accepted
- **Date**: 2026-04-03
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-061](../rfc/RFC-061-semantic-corpus-search.md)
- **Related PRDs**: [PRD-021](../prd/PRD-021-semantic-corpus-search.md)

## Context & Problem Statement

Semantic Corpus Search (RFC-061) requires a vector index over GIL insights, quotes,
summary bullets, and transcript chunks. The project needs to support FAISS for CLI/local
use (Phase 1) and Qdrant for platform/service mode (Phase 2). Consumers of the search
API (CLI, server, viewer) should not be coupled to a specific vector database
implementation.

## Decision

We define a **`VectorStore` protocol** (PEP 544) with the following interface:

1. **Core methods**: `upsert()`, `batch_upsert()`, `search()`, `delete()`,
   `persist()`, `stats()`.
2. **Standard result types**: `SearchResult` (doc_id, score, metadata) and
   `IndexStats` (total_vectors, doc_type_counts, feeds_indexed, embedding_model, etc.).
3. **Backend implementations**: `FaissVectorStore` (Phase 1), `QdrantVectorStore`
   (Phase 2). Both implement the same protocol.
4. **Metadata as flat dict**: Known keys (`doc_type`, `episode_id`, `feed_id`,
   `publish_date`, `speaker_id`, `grounded`, `char_start`, `char_end`,
   `timestamp_start_ms`). Filtering is backend-specific (post-filter for FAISS, native
   for Qdrant).

## Rationale

- **Decoupling**: CLI, server, viewer, and future digest all call `VectorStore.search()`
  without knowing which backend is active.
- **Migration path**: Switching from FAISS to Qdrant is a config change, not a code
  rewrite. The protocol is ~20 lines.
- **Testability**: Unit tests mock `VectorStore` protocol; integration tests test
  specific backends.
- **Consistency with project patterns**: Follows ADR-020 (Protocol-Based Provider
  Discovery) — same PEP 544 approach used for transcription/summarization providers.

## Alternatives Considered

1. **Raw FAISS API directly**: Rejected; locks in FAISS, no migration path to Qdrant
   or platform mode.
2. **ChromaDB as all-in-one**: Rejected; heavier than FAISS, SQLite-based storage adds
   fragility, less mature at scale.
3. **Postgres pgvector (via RFC-051)**: Rejected for Phase 1; requires Postgres server,
   violates CLI-first constraint. Good for Phase 3 (platform).
4. **LangChain/LlamaIndex abstractions**: Rejected; too heavy, too opinionated, pulls
   in large dependency tree for a thin protocol.

## Consequences

- **Positive**: Clean backend swap. Consumers write to one interface. Testable with
  mocks. Aligns with existing protocol patterns.
- **Negative**: Slight abstraction overhead (~20 lines of protocol code). Metadata
  filtering differs between backends (post-filter for FAISS, native for Qdrant).
- **Neutral**: New `faiss-cpu` dependency (~20 MB) for Phase 1.

## Implementation Notes

- **Module**: `src/podcast_scraper/search/protocol.py` — `VectorStore`, `SearchResult`,
  `IndexStats`
- **FAISS**: `src/podcast_scraper/search/faiss_store.py` — `FaissVectorStore`
- **Pattern**: PEP 544 Protocol (same as ADR-020 provider protocols)
- **Config**: `vector_backend: Literal["faiss", "qdrant"] = "faiss"`

## References

- [ADR-020: Protocol-Based Provider Discovery](ADR-020-protocol-based-provider-discovery.md)
- [RFC-061: Semantic Corpus Search](../rfc/RFC-061-semantic-corpus-search.md)
