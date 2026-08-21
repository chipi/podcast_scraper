# Operator-viewer perf consistency with the player — analysis + rollout

**Date:** 2026-08-21
**Author:** Claude (with Marko)
**Status:** analysis / proposal — NOT implemented
**Branch of origin:** `feat/player-mobile-fidelity` (the player perf work being mirrored)

## Why this doc

On `feat/player-mobile-fidelity` we made the **consumer player** API scale from
700 → 10k episodes. This asks: what is the **minimal** set of the same changes
the **operator viewer** (`web/gi-kg-viewer` + its `/api/corpus/*`, `/api/artifacts/*`
routes) needs to reach the same consistency, and when to roll it out.

Evidence base: a read-only audit of the viewer's backend routes and frontend
fetch patterns (three parallel Explore passes, 2026-08-21). Payload sizes below
are read from code, **not measured** on a real 10k corpus (same caveat the
player numbers carry — they are linear projections).

## The player seams we are mirroring

All are **generic** (they take `root: Path`); only the module names are `app_`-prefixed:

| Seam | What it does |
|---|---|
| `app_catalog_cache.cached_catalog(root)` | caches `build_catalog_rows_cumulative` by `corpus_mtime` |
| `app_corpus_access.cached_json_artifact(root, relpath)` | path-safe JSON artifact read, cached by `corpus_mtime` |
| `app_kg_index.get_kg_index(root)` | inverted KG entity index — O(matches) not O(corpus) |
| `perf_cache.get_or_compute(ns, key, corpus_mtime, fn)` | the generic seam (already used by `corpus_digest`, `corpus_library`, `ops`) |
| `asyncio.to_thread(...)` / plain `def` | keep blocking O(corpus) work off the event loop |
| lean endpoint | server pre-filters a corpus-wide envelope to the rows a view needs |

**The seam is already shared** — `perf_cache` is imported by operator routes today
(`corpus_digest.py`, `corpus_library.py`, `ops.py`). `corpus_digest` is the
gold-standard operator route: catalog cached by `corpus_mtime`, topic bands cached
by `lance_mtime`, bounded output. The work is making the *other* routes match it.

## What the viewer already has (do NOT redo)

- **Frontend in-flight-promise dedup** already exists: `useEnrichmentEnvelopeCache.ts`
  stores the Promise (not the value) keyed by `corpusPath::enricherId` — the exact
  player-side dedup pattern. Per-enricher fetches are already lean (one enricher by id,
  not the whole envelope).
- `corpus_digest`, `corpus_library:/corpus/feeds`, and `app_corpus:/corpus*`
  (via `experienced_episode_set` → `cached_catalog`) are already cached.

So the consumer-frontend "stop downloading the whole envelope" rewrite is **mostly
already true** on the viewer. The gap is concentrated in the **backend**.

---

## The gaps (what is inconsistent)

### A. Uncached O(corpus) catalog scans — drop-in `cached_catalog`

Same scan the player cached, still bare here. All are `async def` blocking the loop:

| Route | File:line | Fix |
|---|---|---|
| `GET /corpus/episodes` (primary operator list) | `corpus_library.py:360` | `cached_catalog` + `to_thread`/`def` |
| `GET /corpus/stats` (dashboard widget) | `corpus_metrics.py:159` | `cached_catalog` + `to_thread` |
| `GET /corpus/coverage` (3 passes) | `corpus_coverage.py:56` | `cached_catalog` + `to_thread` |
| `GET /corpus/episodes/detail` | `corpus_library.py:457` | `cached_catalog` + `to_thread` |
| `GET /corpus/episodes/similar` | `corpus_library.py:617,621` | `cached_catalog` + `to_thread` |
| `POST /corpus/resolve-episode-artifacts` | `corpus_library.py:177` | `cached_catalog` + `to_thread` |

Note: `corpus_library.py:269` already caches one route (`/corpus/feeds`) — proving
the pattern; the file is internally inconsistent.

### B. Uncached whole-artifact reads — drop-in `cached_json_artifact`

The exact analog of the player's `/corpus/enrichment` anti-pattern; the
`cached_json_artifact` seam already does what these need, none use it:

| Route | File:line | Payload | Fix |
|---|---|---|---|
| `GET /corpus/enrichments/{enricher_id}` | `corpus_enrichments.py:174` | **whole envelope, multi-MB** for `topic_cooccurrence_corpus` | `cached_json_artifact` (+ optional lean scoping, §D) |
| `GET /corpus/theme-clusters` | `corpus_theme_clusters.py:36` | whole artifact | `cached_json_artifact` |
| `GET /corpus/topic-clusters` | `corpus_topic_clusters.py:29` | whole artifact | `cached_json_artifact` |
| `GET /corpus/enrichments` (list) | `corpus_enrichments.py:78` | compact, but re-parses **every** envelope per request | `perf_cache` |

### C. Heavy scans with no drop-in seam — small new projection

| Route | File:line | Problem | Fix |
|---|---|---|---|
| `GET /corpus/persons/top` | `corpus_persons.py:104` + loop | **worst spot**: catalog scan + parse **every** `*.gi.json` | `perf_cache.get_or_compute("persons_top", …, corpus_mtime)` + `to_thread`; or a cached GI-projection analogous to `get_kg_index` |
| `POST /corpus/node-episodes` | `corpus_library.py:218` → `cil_queries.py:223` | `os.walk` reading every `*.bridge.json` | `perf_cache` on the bridge index + `to_thread` |
| `GET /corpus/runs/summary` | `corpus_metrics.py:254` | `os.walk` whole tree for `run.json` | `perf_cache` + `to_thread` |

`persons/top` reads **GI** dicts, not KG, so it can't reuse `get_kg_index` directly —
it needs its own tiny cached projection (or just a `perf_cache` wrap of the result).

### D. One frontend anti-pattern + polish

| Item | File:line | Fix | Weight |
|---|---|---|---|
| `NodeEnrichmentSection` downloads the **full** `topic_cooccurrence_corpus` / `guest_coappearance` array, filters client-side to one entity | `NodeEnrichmentSection.vue:94-112` (also `GraphCanvas.vue:1127`) | lean per-entity endpoint (pairs with B row 1); frontend drops the `.find()` | medium — but client-cache already makes it first-open-only |
| `EnrichmentPanel` sits behind `v-show` (always mounts) → fires **5 fetches on every page load**, even when the tab is never opened | `EnrichmentPanel.vue:138`, `StatusBar.vue:1204` | `v-show` → `v-if` (or `onActivated`) | **trivial, high value** |
| `useRelationalCache` / `artifacts.ts` topic-clusters sentinel store the value, not the Promise → concurrent callers double-fetch | `useRelationalCache.ts:143`, `artifacts.ts:204` | store the in-flight Promise (the `useEnrichmentEnvelopeCache` pattern) | minor |
| `ensureTopicClusterCompoundVisible` N+1 wave loop (≤16 × resolve+GETs) | `artifacts.ts:768-815` | bounded already; batch or server-merge | low priority, harder |

---

## The MINIMAL set for consistency

**Tier 1 — pure reuse of existing seams, no new endpoints, no schema change.**
This is the true minimal set and captures most of the benefit:

1. Route every §A scan through `cached_catalog`.
2. Route every §B whole-artifact read through `cached_json_artifact` (+ `perf_cache` for the list route).
3. Make the §A/§B `async def` routes non-blocking (`to_thread` the scan, or declare `def` so FastAPI threadpools it — `/corpus/feeds` already does this).
4. Frontend one-liner: `EnrichmentPanel` `v-show` → `v-if` (§D row 2).

Tier 1 is low-risk because the seams are proven on the player + `corpus_digest`,
and it needs no contract changes. **Caveat that must be honored:** these caches
return **shared, read-only** objects (the player caches copy before callers sort
in place). Any viewer route that sorts/mutates the returned rows must copy first —
this is the one correctness trap.

**Tier 2 — small new cached projections (§C):** `persons/top`, `node-episodes`,
`runs/summary`. Each needs a `perf_cache` wrap (and, for `persons/top`, a GI
projection), not just a call swap. Higher value per route (`persons/top` is the
single worst spot) but a bit more code + a test each.

**Tier 3 — lean per-entity enrichment endpoint (§D row 1 + §B row 1):** the direct
mirror of the player's `entity-signals`. Real work (new endpoint + frontend swap),
and partly pre-mitigated by the viewer's existing client cache, so it is the
**lowest urgency** despite being the most "player-like."

**Tier 4 — polish:** promise-dedup in `useRelationalCache` / `artifacts.ts`; the
N+1 wave loop. Nice-to-have.

## NOT in scope / NOT verified (honesty section)

- **Not measured.** No 10k-episode viewer benchmark was run; payload/latency claims
  are read from code and projected, exactly like the player's numbers.
- **Concurrency matters less here.** The player's headline win (20 concurrent
  `/discover` 2.7s → 0.03s) came from `to_thread` under **consumer** concurrency.
  The viewer has **few** concurrent operators, so the `to_thread` part is lower-value
  for the viewer than the **caching** part (per-request latency at 10k). If forced
  to cut, keep the caching, defer `to_thread` — but they land in the same edit so
  there is little reason to split them.
- **Cache stats + warming:** because these routes use the same `perf_cache`, the
  obs-MCP `prod_cache_stats` counters cover them **for free** once they call
  `get_or_compute`. The startup warmer would need the new namespaces added to warm
  them; small follow-up, not required for correctness.
- **`get_kg_index` reuse:** does NOT drop into `persons/top` (GI vs KG). Stated so
  no one wires it wrong.
- Tests: every touched route has existing tests; each change needs its cache
  hit/invalidation contract asserted (the player added catalog-cache + KG-index
  tests — mirror that).

## Suggested rollout / sequencing (the "when")

Ordered by value ÷ risk; each tier is independently shippable and bisectable.

1. **Tier 1 as one PR** (`perf(viewer): cache corpus catalog + artifact reads on the
   operator surface`) — biggest consistency gain, lowest risk, reuses proven seams.
   Do this first, after `feat/player-mobile-fidelity` (PR #1803) merges so the seams
   are on `main`.
2. **Tier 2 as a second PR** — the three heavy scans, `persons/top` first.
3. **Tier 3** — only if operator drill-downs on `topic_cooccurrence_corpus` feel
   heavy in practice; otherwise defer (client cache masks it).
4. **Tier 4** — fold into whichever PR touches those files.

**Recommendation:** ship Tier 1 + Tier 2 together or back-to-back before the next
corpus growth step; treat Tier 3/4 as demand-driven. **Decision needed from Marko:**
do we bundle Tier 1+2 into one viewer-perf branch, and does this wait for #1803 to
merge or branch off it now?

## Appendix — player → viewer mapping at a glance

| Player optimization | Viewer analog | Tier |
|---|---|---|
| catalog cache | §A routes → `cached_catalog` | 1 |
| artifact cache | §B routes → `cached_json_artifact` | 1 |
| `to_thread` on blocking async | §A/§B/§C async routes | 1–2 |
| inverted KG index | `persons/top` GI projection (NOT `get_kg_index`) | 2 |
| lean endpoint (entity-signals) | per-entity cooccurrence/coappearance endpoint | 3 |
| client in-flight dedup | already present (`useEnrichmentEnvelopeCache`); extend to relational/artifacts | 4 |
| defer big fetch off first paint | `EnrichmentPanel` `v-show`→`v-if` | 1 |
| cache stats + warming → obs MCP | free once on `perf_cache`; add namespaces to warmer | follow-up |
</content>
</invoke>
