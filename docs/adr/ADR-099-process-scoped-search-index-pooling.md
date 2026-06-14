# ADR-099: Process-scoped search-index pooling — build the index handle once, reuse it

- **Status**: Proposed
- **Date**: 2026-06-14
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-090](../rfc/RFC-090-hybrid-retrieval.md) (two-tier LanceDB hybrid retrieval — the design this fixes the lifecycle of), [RFC-094](../rfc/RFC-094-search-powered-surfaces-query-layer.md) (search-powered surfaces consume this path)
- **Related ADRs**: [ADR-098](ADR-098-embedding-provider-profile-axis.md) (embedding provider axis — the embedding-model cache is the prior art for this pattern)
- **Related issues**: [#995](https://github.com/chipi/podcast_scraper/issues/995) (implementation)

## Context & Problem Statement

The serving search path (`run_corpus_search` → `hybrid_search.hybrid_candidates`)
opens the vector store **on every single query**. `hybrid_search.py` constructs a
brand-new `LanceDBBackend(index_dir)` per call, which connects to the LanceDB
database, opens its three tables (`segments`, `insights`, `aux`), and loads the
IVF-PQ (vector) and FTS (BM25) index readers into memory — then runs a ~7 ms query
and **discards the whole handle**. The FAISS fallback (`FaissVectorStore.load`)
has the same shape.

Measured on the re-diarized prod-v2 corpus (99 eps, ~9.8k chunks), single warm
serve-api:

| What | Time |
| --- | --- |
| LanceDB vector search, **warm reused** table | ~0.007 s |
| LanceDB BM25 search, **warm reused** table | ~0.007 s |
| One `run_corpus_search` (fresh backend per call) | **~0.8–1.0 s** |
| FAISS flat store load (per call) | ~0.03 s |
| Flat FAISS search | ~0.006 s |

So **~0.8 s of every lance query is pure lifecycle overhead** (open + index load),
and **~7 ms is the actual search**. This is a textbook anti-pattern: a long-lived,
expensive-to-build, stateful resource (a database connection + loaded indices) is
treated as a throwaway per-request value. "Open a database per query" is the
canonical mistake.

Blast radius — this is a **search-layer** problem, not a dashboard one:

- Every `/api/search` query is ~0.8 s.
- The Corpus Digest runs 3–4 band searches **sequentially** → ~4 s, which is what
  surfaced this (Dashboard Intelligence tab / validation V4 timing out; see
  `docs/wip/DASHBOARD-PERF-ANALYSIS-digest-99ep.md`).
- A concurrent first-load burst (dashboard mounts ~6 calls) **SIGSEGV'd serve-api**
  — N threads each doing a *cold* LanceDB connect at once.

The FAISS-only path masked this for the project's whole life because FAISS's
per-call load is 30 ms; building the LanceDB two-tier index (RFC-090) is what lit
the slow path up. Earlier mitigations considered (parallelise the band searches,
cache the digest result, lazy-load bands, route the digest through flat FAISS) are
all **workarounds for the lifecycle bug** — they become unnecessary once a query
is 20 ms.

## Decision

**Introduce a process-scoped, lazily-built, invalidation-aware pool of search
index handles, and have the query path borrow from it instead of constructing a
backend per call.** Concretely, three layers, each owning one concern:

1. **Embedding model (already correct).** One cached `SentenceTransformer` per
   `(model_id, device)` in `embedding_loader`. Unchanged — it is the prior art for
   this ADR's pattern.

2. **Index handle (the new layer).** A process-scoped cache of open backends keyed
   by `index_dir`, holding the *open* LanceDB connection + opened tables (IVF-PQ +
   FTS readers resident) — or the loaded FAISS store + sidecars. Built once under a
   lock, then read-shared. Invalidated and atomically rebuilt when the on-disk
   index changes.

3. **Query path (stateless).** `run_corpus_search` / `hybrid_candidates` become:
   embed (layer 1) → borrow the cached handle (layer 2) → search + fuse + enrich.
   No construction, no index I/O per request.

Cross-cutting requirements the pool must satisfy:

- **Invalidation / freshness.** The only reason to rebuild a handle is the on-disk
  index changing (reprocess / `cli index-two-tier`). Key the cache by
  `(index_dir, freshness-token)` where the token is the lance dir mtime and/or the
  schema/version stamp already exposed by `lance_index_is_stale()` /
  `stored_schema_version()`. On change, atomically swap; never serve a stale
  handle past a rebuild.
- **Concurrency.** One warm shared handle; LanceDB supports concurrent reads via
  its background runtime. The cache map and the first-build are lock-guarded; the
  handle is pre-warmed (tables opened) before publication so concurrent borrowers
  never trigger a cold open. **This also removes the concurrent-cold-init segfault**
  — there is exactly one cold build, serialized.
- **Lifecycle scope.** Handles live for the serve process. A bounded LRU (server
  normally serves one corpus → one entry) caps memory; closing evicted handles
  releases the LanceDB connection.

`run_corpus_search` keeps its existing **lance-preferred, FAISS-fallback** contract
(ADR-098 invariants intact); pooling applies to whichever backend is selected.

## Rationale

- The fix targets the actual cost (lifecycle, ~0.8 s) rather than its symptoms.
  Expected: lance query ~0.8 s → ~0.02 s (~40×); Digest ~4 s → ~0.1 s; search box
  ~0.8 s → ~0.02 s; concurrent-cold-init segfault eliminated.
- It is the standard **connection-pool / resource-manager** pattern every
  DB-backed service uses, and it mirrors the embedding-model cache we already
  trust (ADR-098).
- It is a single seam: one place to get right, every search surface benefits, and
  the relevance design (RFC-090 two-tier + RRF + intent router) is untouched.
- It removes the need for the band-parallelism / result-cache / lazy-load
  workarounds, keeping the digest code simple.

## Alternatives Considered

1. **Route the Digest through flat FAISS (skip hybrid).** ~20× on the dashboard
   only; leaves `/api/search` at 0.8 s and ignores the real bug. Acceptable as a
   one-line *stopgap*, not the fix.
2. **Parallelise the Digest band searches.** 4 s → ~1 s, but each query is still
   0.8 s and it re-raises concurrent cold-init (the segfault). Unnecessary once
   handles are pooled.
3. **Cache the digest *response* (per corpus+window).** Helps repeat navigation,
   not first load, and doesn't touch the search box. A possible later optimization
   on top of pooling, not a substitute.
4. **Open the backend once at startup and pass it everywhere (explicit DI).**
   Cleaner in theory but threads a handle through many call sites and fights the
   per-request `corpus_path` parameter (the server can target different corpora).
   A keyed pool keeps the call sites stateless and supports multiple corpora.

## Consequences

- **Positive**: ~40× on every lance query; Digest/Dashboard and the search box both
  fixed in one change; the concurrent-cold-init segfault removed; digest code
  stays simple (no parallelism/result-cache scaffolding); FAISS-fallback also stops
  reloading per call.
- **Negative**: introduces shared mutable state (a connection pool) → needs careful
  locking and an invalidation contract; a concurrency-safety test becomes mandatory.
- **Neutral**: first query after boot (or after a reindex) pays the one-time warm
  (~0.8 s); optionally hidden behind an eager startup warm for the configured
  corpus.

## Implementation Notes

- **Module**: `podcast_scraper/search/` — the pool sits between
  `run_corpus_search` / `hybrid_search.hybrid_candidates` and the backends
  (`backends/lancedb_backend.py`, `faiss_store.py`). The construction site to
  replace is `hybrid_search.py:~219` (`LanceDBBackend(str(index_dir))`).
- **Pattern**: keyed resource pool / lazy singleton with mtime-or-version
  invalidation + a lock; analogous to `embedding_loader._embedding_models`.
- **Invalidation hooks**: reuse `lance_index_is_stale()` / `stored_schema_version()`
  and the lance dir mtime as the freshness token.
- **Tests**: (1) a warm-reuse micro-benchmark asserting the second query is ≪ the
  first; (2) a concurrent-search test (N threads) for safety + no segfault;
  (3) an invalidation test (rebuild index → pool serves the new handle); existing
  search/digest e2e remain green.

## References

- [RFC-090: Hybrid retrieval (two-tier LanceDB)](../rfc/RFC-090-hybrid-retrieval.md)
- [ADR-098: Embedding provider as a profile axis](ADR-098-embedding-provider-profile-axis.md)
- `docs/wip/DASHBOARD-PERF-ANALYSIS-digest-99ep.md` — the measurement trail that found this.
