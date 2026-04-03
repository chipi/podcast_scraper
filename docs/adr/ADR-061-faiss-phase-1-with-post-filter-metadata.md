# ADR-061: FAISS Phase 1 with Post-Filter Metadata Strategy

- **Status**: Accepted
- **Date**: 2026-04-03
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-061](../rfc/RFC-061-semantic-corpus-search.md)
- **Related PRDs**: [PRD-021](../prd/PRD-021-semantic-corpus-search.md)

## Context & Problem Statement

The `VectorStore` protocol (ADR-060) needs a Phase 1 implementation. FAISS provides
fast in-process vector search but has no built-in metadata filtering. The system needs
to filter search results by type, feed, date, speaker, and grounding status. The
question is how to handle filtering without native support.

## Decision

We adopt **FAISS with post-filter metadata** for Phase 1:

1. **FAISS as the default backend**: `faiss-cpu` in `[project.dependencies]`.
   `faiss-gpu` as optional extra for CUDA users.
2. **Over-fetch then post-filter**: FAISS returns `top_k * 3` candidates. A Python
   post-filter applies metadata predicates (type, feed, date, speaker, grounded).
   The top `k` from the filtered set are returned.
3. **Metadata sidecar**: `metadata.json` alongside `vectors.faiss` maps doc_id to
   metadata dict. Start with JSON; add SQLite option when corpora exceed ~50K vectors.
4. **Auto index type selection**: `IndexFlatIP` + `IndexIDMap` for < 100K vectors;
   `IndexIVFFlat` for 100K–1M; `IndexIVFPQ` for > 1M. Auto-selected at persist time.
5. **On-disk layout**: `<output_dir>/search/vectors.faiss`, `metadata.json`,
   `index_meta.json`.

## Rationale

- **CLI-first**: FAISS is in-process, zero server overhead, ~20 MB dependency. No
  Docker, no external database for the default path.
- **Sufficient at CLI scale**: Post-filtering on ~1K candidates is sub-millisecond.
  The over-fetch ratio (3x) handles typical filter selectivity.
- **Known upgrade path**: Qdrant Phase 2 replaces post-filter with native payload
  filtering. The `VectorStore` protocol (ADR-060) makes the switch transparent.
- **Simplicity**: Post-filter is ~15 lines of Python. No custom FAISS index or
  external filtering library.

## Alternatives Considered

1. **Qdrant local mode for Phase 1**: Rejected; heavier binary dependency, "for demos"
   per docs, Rust binary in Python package.
2. **FAISS with pre-filter (separate indexes per type)**: Rejected; multiplies index
   count, complicates incremental updates, marginal benefit at CLI scale.
3. **SQLite FTS5 for metadata + FAISS for vectors**: Rejected; adds complexity without
   matching the simplicity of JSON sidecar at < 50K vectors.

## Consequences

- **Positive**: Simple, fast, minimal dependencies. Works offline. Co-located with
  corpus outputs.
- **Negative**: Post-filter may return fewer than `k` results if filter is highly
  selective. Warning message when this occurs. Over-fetch ratio is a tunable.
- **Neutral**: Requires `faiss-cpu` as new dependency. Auto index type selection adds
  minor complexity but is transparent to callers.

## Implementation Notes

- **Module**: `src/podcast_scraper/search/faiss_store.py`
- **Index files**: `<output_dir>/search/vectors.faiss`, `metadata.json`,
  `index_meta.json`
- **Config**: `vector_backend: "faiss"` (default), `vector_index_path` (optional
  override)
- **Upgrade trigger**: When platform mode ships, switch to `vector_backend: "qdrant"`

## References

- [ADR-060: VectorStore Protocol](ADR-060-vectorstore-protocol-with-backend-abstraction.md)
- [RFC-061: Semantic Corpus Search](../rfc/RFC-061-semantic-corpus-search.md)
