# Dashboard performance analysis — digest topic bands on a 99-episode corpus

> **RESOLVED → [ADR-099](../adr/ADR-099-lancedb-first-single-index-search.md).**
> Deeper profiling refined the root cause below: the dominant cost is **not** the
> number of topic bands — it is that **every lance query rebuilds the
> `LanceDBBackend` from scratch** (open DB + load IVF/FTS indices, ~0.8 s) while
> the actual search on a warm table is ~7 ms. This slows *all* lance search
> (`/api/search` ~0.8 s too), and the concurrent cold-init is what SIGSEGV'd
> serve-api. The fix is process-scoped index-handle pooling — see ADR-099. The
> original band-level analysis is kept below as the measurement trail.

Surfaced while running the Tier-3 validation walk against the re-diarized prod-v2
corpus (99 eps). Validation V4 (Dashboard topic-cluster chip) timed out at 30s;
manual load of the Intelligence tab took ~12s; and an earlier concurrent
first-load burst SIGSEGV'd serve-api. This is **pre-existing digest behaviour
exposed by corpus scale — not a graph-migration regression** (the #967/#974/#876
changes don't touch the digest path).

## Measurements (prod-v2, 99 eps, single serve-api on :8000)

Per-endpoint, **isolated** (one request at a time):

| Endpoint | Latency |
| --- | --- |
| `corpus/coverage` | 0.08s |
| `corpus/feeds` | 0.09s |
| `corpus/persons/top` | 0.10s |
| `corpus/query-activity` | 0.001s |
| `corpus/topic-clusters` | 0.001s |
| **`corpus/digest`** (full) | **4.7s** |
| `corpus/digest?compact=true` (no topic bands) | **0.08s** |

- **The entire digest cost is the topic bands**: 4.7s full vs 0.08s compact →
  ~4.6s is the topic-band semantic searches. Everything else is <0.1s.
- **Concurrent burst** (all 6 fired at once): **3.7s wall-clock ≈ the slowest
  single endpoint** (digest), NOT the sum (~5s). So FastAPI's threadpool already
  runs the sync handlers concurrently — **the problem is not serialization**, it's
  one slow endpoint.
- digest is **not cached**: repeated calls are a steady 3.4–3.7s each.

## Root cause

`GET /api/corpus/digest` (when `include topic bands`, the dashboard default) runs
several **semantic searches** per request, **sequentially**, with **no caching**:

- A `_topic_band_for_query` call per configured digest topic (3 defaults: Science,
  Technology, Business — `DEFAULT_DIGEST_TOPICS`), each running `run_corpus_search`
  (embed the query + vector-search the FAISS/LanceDB index), capped at
  `DIGEST_TOPIC_SEARCH_TIMEOUT_SEC = 0.8s` via a per-band `ThreadPoolExecutor(max_workers=1)`.
- Plus `build_cil_digest_topics_for_row` per picked row (`max_rows`, default 3).

The bands/rows are iterated in a plain `for` loop, so the searches run one after
another. On a 99-ep index each search is ~0.7–0.8s, and they sum to ~4.6s. Every
dashboard/digest load re-pays this in full.

## Why it *looked* like serialization / a stuck "Loading…"

The dashboard fires ~6 calls on mount. Five return in <0.1s; the digest holds for
~4.6s, so the Digest surface (and anything gated on the dashboard's combined
"ready" state) waits on it. The Intelligence **topic-landscape** itself fetches
`topic-clusters` (fast, 0.001s server-side) — its longer stall is partly the
shared load cascade and partly client-side work (mapping 86 clusters +
`graphCompoundParentIdFromCluster`). The dominant, clearly-actionable lever is the
digest.

## Secondary finding — serve-api SIGSEGV under the concurrent cold-start burst

On the very first dashboard load against a freshly-started serve-api, the worker
exited 139 (SIGSEGV). The digest's parallel-ish band searches each touch the
vector store (FAISS/LanceDB C-extension) while it is still cold-initialising; the
native layer is likely not safe under concurrent first-init. After a warm start
(any prior single query) it is stable. Separate robustness item — and a caveat for
Option A below.

## Options (ranked)

**A. Parallelise the topic-band searches (server).** Wrap the band/row loop in one
`ThreadPoolExecutor(max_workers=N)` instead of running them sequentially. Wall-clock
drops from the sum (~4.6s) to ~the slowest single band (~0.8s) — **~5–6× faster**.
Low risk, contained to the digest route. **Caveat:** warm the vector store once
(a single throwaway query) *before* fanning out, or this worsens the cold-init
segfault.

**B. Cache the digest result (server).** Memoise per `(corpus_root, window,
max_rows, topics)`, invalidated by the corpus dir mtime (or a short TTL). First
load still pays ~4.6s; every repeat navigation is instant. Low risk, big win for
the common "click around the dashboard" flow. Pairs well with A.

**C. Lazy-load the topic bands (client + server).** Return the base digest (0.08s)
immediately and fetch topic bands in a second request that fills them in. Best
*perceived* latency — the dashboard paints instantly and bands stream in. Needs a
small client change (the Digest panel already has a compact mode to build on).

**D. Precompute digest topic bands at index/digest-build time (offline).** Zero
runtime search cost; the dashboard just reads a file. Most work, best result, but
bands go stale between builds (acceptable — the corpus only changes on reprocess).

**E. Cost knobs (quick, lossy).** Fewer default topics, lower
`DIGEST_TOPIC_SEARCH_TOP_K` (24), tighter timeout. Fast to ship but degrades the
feature; treat as a stopgap, not a fix.

## Recommendation

- **A + B** (parallelise + cache) is the high-value, low-risk combination — ~5–6×
  on first load, instant on repeat — and keeps the feature intact.
- **C** (lazy-load) on top if we want the dashboard to *feel* instant regardless.
- Fix the **cold-init segfault** independently (warm-then-fan-out, or a
  process-wide store-init lock) — it gates A and is a correctness issue on its own.

All measured on a single laptop serve-api (DPR-1). Numbers scale with episode
count; a larger corpus widens the gap, so this is worth doing before the corpus
grows past ~100 eps in the default dashboard view.
