# RFC-070: Semantic Corpus Search — Platform & Future Backends

- **Status**: Draft (not implemented; captures deferred scope split from [RFC-061](RFC-061-semantic-corpus-search.md))
- **Authors**: Podcast Scraper Team
- **Stakeholders**: Core team, platform owners, operators needing multi-tenant or remote vector search
- **Related PRDs**:
  - [PRD-021: Semantic Corpus Search](../prd/PRD-021-semantic-corpus-search.md) — product intent spans local + platform
- **Related ADRs**:
  - [ADR-060: VectorStore protocol with backend abstraction](../adr/ADR-060-vectorstore-protocol-with-backend-abstraction.md) —
    **`QdrantVectorStore`** named as Phase 2; implementation tracked here
- **Related RFCs**:
  - [RFC-061: Semantic corpus search (FAISS Phase 1)](RFC-061-semantic-corpus-search.md) — **shipped** indexer, CLI,
    `FaissVectorStore`, post-filter metadata, viewer **`/api/search`**
  - [RFC-051: Database projection (GIL & KG)](RFC-051-database-projection-gil-kg.md) — optional **pgvector** alignment
  - [RFC-068: Corpus digest API & viewer](RFC-068-corpus-digest-api-viewer.md) — digest may consume clustered /
    ranked vectors later
  - [RFC-062: GI/KG viewer v2](RFC-062-gi-kg-viewer-v2.md) — already consumes **`VectorStore.search()`** via HTTP;
    backend swap is transparent if protocol is honored
- **Related Documents**:
  - [GitHub #484](https://github.com/chipi/podcast_scraper/issues/484) — semantic search tracking
  - [GitHub #466](https://github.com/chipi/podcast_scraper/issues/466) — GI + KG depth roadmap
  - [Semantic search operator guide](../guides/SEMANTIC_SEARCH_GUIDE.md)
- **Updated**: 2026-04-11 (split from RFC-061)

## Abstract

[RFC-061](RFC-061-semantic-corpus-search.md) **completed** local semantic search with **FAISS**, the
**`VectorStore`** protocol, **`podcast search` / `podcast index`**, pipeline embed-and-index, and
semantic **`gi explore --topic`** when an index exists. This RFC holds **everything we considered for
later**: alternate **`VectorStore`** backends (**Qdrant**), **native metadata filtering**, tighter
integration with **digest clustering** and **re-ranking**, **pgvector** alongside [RFC-051](RFC-051-database-projection-gil-kg.md),
and **metadata storage** at very large corpus scale. None of this is required for the shipped v2.6
CLI + viewer experience; it becomes relevant for **platform / multi-tenant / remote** deployments.

## Problem Statement

**FAISS + JSON sidecars** ([RFC-061](RFC-061-semantic-corpus-search.md)) are the right default for a
**CLI-first**, single-machine corpus: in-process, no extra service, post-filter metadata is acceptable
up to roughly **~1M vectors** with the auto **IVF / IVFPQ** upgrade path already in code.

That design **stops being enough** when:

- **A vector service must run separately** (containers, horizontal scale, shared index for many
  clients).
- **Filtering is complex and frequent** — over-fetch + post-filter becomes wasteful compared to
  payload-native filters (**Qdrant**, **pgvector** with SQL predicates).
- **Digest or recommendations** need **online clustering**, **re-ranking** (e.g. cross-encoder), or
  **hybrid** sparse+dense retrieval — beyond a single bi-encoder pass.
- **Metadata sidecars** become a bottleneck (very large **`metadata.json`**); **SQLite** or **columnar**
  stores may be preferable.

This RFC does **not** reopen the Phase 1 design decisions in RFC-061; it defines **candidate Phase 2+
work** for when product priorities justify it.

## Goals (future)

1. **`QdrantVectorStore`**: Implement **`VectorStore`** against Qdrant (local or server); wire
   **`vector_backend: qdrant`** in [config](../api/CONFIGURATION.md) to select it (today only **`faiss`**
   is functional).
2. **Native filtering**: Map search filters to Qdrant payload (or SQL for pgvector) to reduce
   over-fetch and simplify high-selectivity queries.
3. **Operational story**: Document Docker / credentials / index lifecycle for a remote vector DB;
   align with [platform blueprint](../architecture/PLATFORM_ARCHITECTURE_BLUEPRINT.md).
4. **Digest + search**: Optional integration points for [RFC-068](RFC-068-corpus-digest-api-viewer.md)
   (e.g. theme clustering, dedupe at query time vs index time — product decision).
5. **Quality ladder**: Optional **cross-encoder re-ranking** on top-k bi-encoder hits; **hybrid** search
   if PRD-021 expands.
6. **Structured + vector SQL path**: Evaluate **pgvector** as part of [RFC-051](RFC-051-database-projection-gil-kg.md)
   for deployments that already run Postgres (single DB for projections + vectors).

## Non-Goals

- Replacing or re-specifying **FAISS Phase 1** (see [RFC-061](RFC-061-semantic-corpus-search.md)).
- Changing **`gi.json` / `kg.json`** shapes or pipeline artifact contracts.
- Authentication / multi-tenant **product** design (platform issues **#50**, **#347** — this RFC only
  covers **vector backend** options).

## Design Directions (not frozen)

### 1. Qdrant backend

- Same **`VectorStore`** methods as **`FaissVectorStore`** ([ADR-060](../adr/ADR-060-vectorstore-protocol-with-backend-abstraction.md)).
- **Upserts** and **payload** mirror flat **`metadata`** keys from RFC-061.
- **Config**: reuse **`vector_index_path`** or introduce **URL + collection** fields (TBD when
  implemented).

### 2. Metadata at scale

- RFC-061 uses **`metadata.json`** (+ **`id_map.json`**, **`index_meta.json`**) — appropriate for
  Phase 1.
- For **very large** corpora, consider **SQLite** (or backend-native storage) for doc_id → metadata;
  benchmark before mandating.

### 3. Re-ranking and clustering

- **Re-ranking**: run cross-encoder on top **N** results from **`VectorStore.search()`**; keep outside
  the protocol or add an optional **post-processor** hook (TBD).
- **Near-dedup**: RFC-061 **open question** — index stores all rows; clustering/dedupe may stay in
  **digest** or **post-filter** layers.

### 4. Alternatives (recap)

The following were **rejected for Phase 1** but remain **valid for platform evaluation** (detail was
formerly in RFC-061):

| Option | Why not Phase 1 | When to revisit |
| ------ | ----------------- | --------------- |
| **Qdrant-only** | Heavier ops than FAISS for local CLI | Remote/shared index, native filters |
| **ChromaDB** | Heavier embedded stack | If team standardizes on Chroma elsewhere |
| **pgvector** | Needs Postgres for CLI-first | RFC-051 platform deployments |
| **Raw FAISS only** | No swap path | N/A — protocol shipped |

## Testing Strategy (when implemented)

- Contract tests: **`VectorStore`** conformance suite shared by **FAISS** and **Qdrant** mocks.
- Integration: spin Qdrant test container (or local mode) in CI optional job.
- Parity: same **`podcast search`** and **`/api/search`** JSON shapes as RFC-061.

## Relationship to RFC-061

```text
RFC-061 (Completed)     FaissVectorStore, CLI, pipeline stage, viewer API
    ↓
RFC-070 (this RFC)       Qdrant / pgvector / scale / quality extras — Draft until prioritized
```

## Open Questions

1. **Single vs dual index**: One Qdrant collection per corpus root vs per-tenant collections?
2. **Migration**: Re-index from artifacts only, or FAISS → Qdrant vector copy (if formats allow)?
3. **CI**: Mandatory Qdrant job vs optional nightly?

## References

- [RFC-061](RFC-061-semantic-corpus-search.md)
- [ADR-060](../adr/ADR-060-vectorstore-protocol-with-backend-abstraction.md)
- [PRD-021](../prd/PRD-021-semantic-corpus-search.md)
