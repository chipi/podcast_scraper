# ADR-099: LanceDB-first single-index search; retire FAISS

- **Status**: Accepted — **Stage 2 (native hybrid) reverted, see [#1205](https://github.com/chipi/podcast_scraper/issues/1205)**
- **Date**: 2026-06-14
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-090](../rfc/RFC-090-hybrid-retrieval.md) (two-tier hybrid retrieval — this ADR completes its vision and retires the FAISS transitional path), [RFC-094](../rfc/RFC-094-search-powered-surfaces-query-layer.md) (search-powered surfaces consume this path)
- **Related ADRs**: [ADR-098](ADR-098-embedding-provider-profile-axis.md) (embedding provider axis — unchanged; the query still embeds in the index's recorded space)
- **Related issues**: [#995](https://github.com/chipi/podcast_scraper/issues/995) (implementation), [#1010](https://github.com/chipi/podcast_scraper/issues/1010) (shipped — FAISS retired, `faiss_store.py` deleted)

> **History.** This ADR was first drafted as "process-scoped search-index pooling"
> — a narrow fix to "we open a new `LanceDBBackend` per query." Investigation
> showed that was one symptom of a deeper issue: **the serving layer treats LanceDB
> like FAISS** — a load-per-call in-memory library — instead of a database. The
> decision was widened accordingly.

## Context & Problem Statement

Two indexes exist side by side:

- **FAISS** — a flat in-memory vector *library*: `load()` a binary blob, search an
  array. ~30 ms load, ~6 ms search. No persistence model, no text/BM25, no metadata
  (a JSON sidecar maps vector-id → doc).
- **LanceDB** — an embedded *database* (RFC-090): persistent columnar tables with
  native vector (IVF-PQ), full-text (BM25/FTS), and SQL filters; opened once, then
  queried in ~7 ms. Three tables: `segments`, `insights`, `aux`.

The serving path (`run_corpus_search` → `hybrid_search` → `LanceDBBackend`) was
built with **FAISS habits** and they do not transfer to a database:

1. **Open-per-query.** `hybrid_search` constructs a fresh `LanceDBBackend(index_dir)`
   *inside each request* — reconnect + reopen tables + reload IVF/FTS readers — then
   discards it. Cheap for a FAISS array (~30 ms); ~0.5 s for a database. The
   concurrent cold-init of this is what **SIGSEGV'd serve-api**.
2. **Over-fetch + ship the vectors back.** `retrieve` pulls `top_k × 25` rows per
   tier and materializes each **with its 384-float embedding column**, then
   `pop("embedding")`. Free from a FAISS array; ~1 s of Arrow→Python from a
   database. This is the dominant warm cost.
3. **FAISS as a runtime fallback.** `hybrid_search` is gated behind
   `serving.hybrid_enabled` and falls back to FAISS, ostensibly because LanceDB
   "covers only segment/insight." **That is stale** — the `aux` table already holds
   **quotes (2,301) + kg_entities (1,396) + kg_topics (990) + summary bullets
   (591)**. On the prod-v2 corpus LanceDB has 10,199 rows vs FAISS's 9,802 of the
   **same content**. FAISS is a redundant second copy of the same vectors.

Measured (prod-v2, 99 eps): `/api/search` ~0.8 s/query; Corpus Digest ~4 s (4
sequential band searches); warm lance table search on a reused handle ~7 ms.

So the problem is not "which index" — it is that we run a database like a library,
keep a duplicate index, and fork between them at runtime.

## Decision

**LanceDB is the single search index, run the way a database is meant to be run.
FAISS is removed entirely — build, serve, and config.**

1. **One database, opened once.** A process-scoped, lazily-built handle (the
   connection and the three tables) reused for the process lifetime, rebuilt only
   when the on-disk index changes. No per-request construction.
2. **All doc types served from LanceDB** — `segments`, `insights`, and `aux`
   (quote / kg_entity / kg_topic / bullet). The data is already there; finishing the
   routing removes the only reason FAISS was consulted.
3. **Idiomatic queries** — `select()` only the returned columns (never materialize
   the `embedding`), and a fetch size that is a small multiple of `top_k`, not ×25.
4. **Native hybrid** — use LanceDB's built-in hybrid (vector + FTS) with a reranker
   in-engine, replacing the hand-rolled `RetrievalLayer` vector+BM25+RRF fan-out.
5. **No runtime fallback.** A missing or stale index is a **build-time** condition
   ("(re)build via `cli index-two-tier`"), surfaced as `no_index` — not a second
   code path. `serving.hybrid_enabled` is retired (LanceDB is always the path).
6. **No back-compat / migration.** Pre-LanceDB (FAISS-only) corpora are **rebuilt**,
   not migrated. `cli index` stops producing FAISS artifacts; `cli index-two-tier`
   (or a renamed `cli index`) is the one builder.

## Rationale

- It removes accidental complexity by *deleting* code (FaissVectorStore, the
  fallback branch, the per-query construct, the over-fetch, the stale routing, the
  enable flag) rather than adding more. The architecture gets smaller.
- It fixes every symptom at once: cold-start, the segfault, the ~0.5 s/query open,
  and the ~1 s/query over-fetch → `/api/search` and Digest drop to tens of ms.
- It honours how LanceDB is designed to run (open once; `search().select().limit()`;
  native hybrid), so we stop fighting the tool.
- LanceDB is already the complete index; FAISS is pure redundancy. Keeping two
  copies + a runtime fork has no upside once routing is finished.

## Alternatives Considered

1. **Pool the handle but keep FAISS + the fallback** (the original ADR-099). Fixes
   cold-start/segfault but leaves the duplicate index, the runtime fork, and the
   over-fetch. Rejected: treats the symptom, keeps the FAISS-shaped design.
2. **Keep FAISS as a runtime fallback for missing/airgapped indices.** Rejected:
   re-introduces the dual path this ADR exists to delete. A missing index is a
   build concern; airgapped builds still build LanceDB (no network needed).
3. **Migrate old FAISS corpora.** Rejected by decision — rebuild instead; there is
   no corpus we must preserve without reprocessing.

## Consequences

- **Positive**: one index, one path; ~20–40× per query; cold-start + segfault gone;
  large net code deletion; serving matches LanceDB idiom.
- **Negative**: a hard cutover — every served corpus must have a LanceDB index
  (rebuild required); `serving.hybrid_enabled` and FAISS config/flags are removed
  (a breaking config change, acceptable pre-prod).
- **Neutral**: embedding-model selection (ADR-098) and the index *content* are
  unchanged; this is a serving-and-build-shape change, not a relevance change
  (except the deliberate native-hybrid swap, validated separately).

## Implementation Notes — staged, full solution on completion

- **Module**: `podcast_scraper/search/` — `hybrid_search.py`, `corpus_search.py`,
  `backends/lancedb_backend.py`, `retrieval.py`; build in `cli` (`index` /
  `index-two-tier`); config (`serving.hybrid_enabled` + FAISS keys).
- **Stage 1 — LanceDB-first serving + query hygiene.** Open-once handle; route all
  doc types to LanceDB; `select()` projection; sane fetch size. FAISS files remain
  on disk but are not read at serve time. Re-measure `/api/search` + digest.
- **Stage 2 — Native hybrid.** Replace the hand-rolled RRF fan-out with LanceDB's
  hybrid + reranker; validate top-k relevance vs Stage 1 on a query set.
  > **REVERTED (#1205).** LanceDB's native in-engine hybrid combine
  > (`_combine_hybrid_results` → `_normalize_scores` → native `pyarrow.compute`)
  > hard-SIGSEGVs the api worker under the digest route's search fan-out. Stage 2 was
  > backed out: `LanceDBBackend.search_hybrid` is removed and the default `hybrid`
  > signal fuses `search_bm25` + `search_vector` with the Python-side `fusion.rrf_fuse`
  > (the Stage-1 fan-out), which also restores the router's per-intent tier/signal
  > weighting the in-engine reranker dropped. The rest of this ADR (LanceDB-first,
  > FAISS retired) stands.
- **Stage 3 — Retire FAISS.** Delete `FaissVectorStore`, the fallback branch, the
  FAISS build path, `serving.hybrid_enabled` and FAISS config; update profiles +
  tests. Pure deletion once Stage 1–2 are proven.
- **Tests**: open-once reuse (2nd query ≪ 1st); concurrent-search safety (no
  segfault); invalidation on reindex; `no_index` when unbuilt; relevance check for
  the native-hybrid swap; existing search/digest unit + e2e green.

## References

- [RFC-090: Hybrid retrieval (two-tier LanceDB)](../rfc/RFC-090-hybrid-retrieval.md)
- [ADR-098: Embedding provider as a profile axis](ADR-098-embedding-provider-profile-axis.md)
- `docs/wip/DASHBOARD-PERF-ANALYSIS-digest-99ep.md` — the measurement trail.
