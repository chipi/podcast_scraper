# Player performance + scalability audit — 2026-08-21

> **Status update (implemented).** All findings below are now fixed and merged on
> `feat/player-mobile-fidelity`: the catalog cache + O(1) slug index (commit a2fb4207e),
> the shared artifact cache / filename-first enrichment / `to_thread` / client tidy-ups
> (same commit), and — the "remaining structural follow-up" this doc flagged — the
> **inverted KG entity index** (`app_kg_index.py`) that makes the person/topic/entity-search
> cards O(matches) instead of O(corpus). Measured after: `topics/{id}` 60–80 ms → **2.8 ms**
> warm; 20-concurrent `/discover` 2.7 s/req → **0.03 s**; 20-concurrent `topics/{id}` **0.04 s**.
> Every per-request O(corpus) scan is now paid once per ingest, so steady-state latency is flat
> from 700 to 10k episodes. The one remaining note is the cold build after each ingest (the first
> card/discover request rebuilds the cache) — optionally warm at startup; not yet done.

Focused client + backend performance run on the consumer player (`web/learning-player`
+ `/api/app/*`). Goal: measure current per-route performance, validate the recent
lean-endpoint work, and assess **scaling risk from ~700 episodes today → ~10,000
episodes and concurrent users**.

Method: local API (`make serve-api`) against the real enriched `prod-v2` corpus
(`.test_outputs/manual/prod-v2/corpus`, **100 distinct episodes** after cumulative
dedup, all corpus-scope enrichments present), Vite app on `:5174`, driven with
chrome-devtools (Home + Player traces + network waterfalls) plus a scripted
`curl` timing/size sweep of every `/api/app/*` endpoint, a direct micro-benchmark of
the catalog scan, and a 20-way concurrency test. Backend code audited read-only.

> Caveat up front: the local corpus is 100 eps (prod is ~700); local numbers are a
> **floor**. Where it matters I project linearly and say so. Dev-mode client timings
> (unminified Vite) are not comparable to prod paint times — I rely on the **network
> waterfall + backend timings**, which are representative, for the client story.

---

## TL;DR — the headline

1. **The lean-endpoint work lands and is live.** Home now fetches
   `/api/app/corpus/trending-topics` (**4.9 KB**) instead of `/api/app/corpus/enrichment`
   (**2.12 MB**) — a **436× payload cut**, measured. `entity-signals` is 7.9 KB vs 2.12 MB
   (269×). At prod scale the old payload is ~25 MB and the pair artifacts grow O(N²), so
   the win widens with size.

2. **The dominant scaling risk is NOT payload — it is an uncached full-corpus scan
   (`build_catalog_rows_cumulative`) run on nearly every request.** Measured **494 µs/episode**:

   | corpus | catalog scan / call | notes |
   |---|---|---|
   | 100 eps (local) | 49 ms | measured, warm FS cache |
   | 700 eps (today) | ~346 ms | projected |
   | 10,000 eps (target) | **~4.9 s** | projected — **per request, uncached** |

   Nearly every consumer endpoint pays this, some multiple times.

3. **The app does not parallelize under load.** Single `/discover` = 54 ms; **20 concurrent
   `/discover` = 2.7 s each — a 50× latency collapse at only 100 episodes.** Sync catalog
   scans hold the GIL and serialize. At 10k eps this is minutes, not seconds.

4. **Per-page fan-out multiplies the tax.** Home fires **4 endpoints that each independently
   scan the catalog**; the Player fires **~9 episode-scoped calls that each independently
   re-resolve the slug** (also an O(catalog) scan). So one Home load ≈ 4 scans; one Player
   load ≈ 9 scans.

**One fix removes most of the risk:** an mtime-keyed process cache of the catalog (+ a
`slug → path` map) on `app.state`. Everything else is secondary.

---

## Validated wins (measured, live)

| Endpoint | Payload | vs old |
|---|---|---|
| `/api/app/corpus/enrichment` (OLD, still present) | **2,118,921 B** | — |
| `/api/app/corpus/trending-topics` (NEW) | **4,862 B** | **436× smaller** |
| `/api/app/corpus/entity-signals?kind=topic` (NEW) | **7,876 B** | **269× smaller** |

Home's live waterfall confirms the client calls `trending-topics`, not `enrichment`.
The Player render-gating fix is confirmed: render waits only on
`episode` + `audio-source` + `playback`; `segments`/`insights`/`entities`/`enrichment`/`stats`
load in the background.

---

## Endpoint timing + size sweep (100-ep local corpus, 2 passes)

Two passes per endpoint; **pass-2 is never faster than pass-1 → no caching anywhere.**

| Endpoint | size | p1 | p2 |
|---|---|---|---|
| `podcasts` | 7.9 KB | 82 ms | 52 ms |
| `episodes?page=1&page_size=20` | 52 KB | 56 ms | 57 ms |
| `episodes?...&status=ready` | 52 KB | 61 ms | 60 ms |
| `discover` | 22 KB | 51 ms | 52 ms |
| `trending?kind=topic` | 68 B | 109 ms | 105 ms |
| `clusters` | 775 B | **1.8 ms** | 1.8 ms |
| `corpus/enrichment` (OLD) | **2.12 MB** | 17 ms | 16 ms |
| `corpus/trending-topics` (NEW) | 4.9 KB | 12 ms | 12 ms |
| `entity-signals?kind=topic` (NEW) | 7.9 KB | 15 ms | 14 ms |
| `topics/{id}` | 8.4 KB | 79 ms | 61 ms |
| `episode detail` | 2.2 KB | 52 ms | 50 ms |
| `episode entities` | 3.5 KB | 51 ms | 50 ms |
| `episode insights` | 7.7 KB | 50 ms | 49 ms |
| `episode segments` | 49 KB | 52 ms | 53 ms |
| `episode enrichment` | 9.8 KB | 52 ms | 52 ms |
| `search?q=…` | 17 KB | 232 ms | 191 ms |

**Reading it:**
- `episode detail` (2.2 KB) and `episode segments` (49 KB) both take ~50 ms — a 24× payload
  gap with identical latency. The time is **not payload-bound; it is the catalog scan**
  (`resolve_slug`). This is the empirical proof of finding #2.
- `clusters` = 1.8 ms is the only route that skips the catalog scan (reads one precomputed
  file) — the "what good looks like" baseline.
- `corpus/enrichment` is server-fast (16 ms) despite 2.12 MB — its cost is network + client
  parse, which is exactly what the lean endpoints removed.
- `search` is the slowest (~200 ms): vector index + query embedding. Acceptable now; watch it.

---

## Ranked scaling hot spots (audit + empirical)

Worst first. All file:line in `src/podcast_scraper/server/`.

### 1. `build_catalog_rows_cumulative` — uncached full-corpus scan, every request
`corpus_catalog.py:353`. Walks every `*.metadata.json`, `json.loads` each, **3 `os.path.isfile`
per episode** (gi/kg/bridge sidecar checks), dedups across runs. **Zero caching.**
Callers (all uncached): `discover` (`app_discover.py:159`), `episodes` (`app_episodes.py:132,245`),
`search` (`app_search.py:49`), `resolve_slug` (`app_slugs.py:69` — every per-episode route),
all three relational cards (`app_relational_view.py:92,179,236`), momentum
(`app_momentum.py:172,293`), feed signals, user-corpus. Module comment literally says
"cache later if the corpus grows large enough to matter." It does now.
- **Growth:** O(N) opens+parses + 3N stats. **~5 s/request @ 10k eps.**
- **Concurrency:** sync `def` → threadpool + GIL-bound JSON parse → serializes (see the 50× test).
- **Fix:** mtime-keyed `app.state` cache of the catalog **and** a `slug → metadata_path` map.
  Single change; collapses #1, #2 (partly), #4, #10.

### 2. `resolve_slug` — O(catalog) linear scan per per-episode call
`app_slugs.py:60`. Iterates the whole catalog computing a SHA-256 per row until match. Every
`/episodes/{slug}/*` route calls it; the **Player fires ~9 of these per load** (detail,
audio-source, playback, segments, insights, entities, enrichment, stats, related). At 10k eps
each is ~5 s. **Fix:** the `slug→path` map from #1 → O(1).

### 3. CIL topic routes — full `os.walk` + 3 JSON reads/episode, on the event loop
`cil_queries.py:162` (`iter_cil_episode_bundles`), used by `/topics/{id}/perspectives` and
`/conversation-arc`. `async def` routes call the **blocking** walk directly → blocks the whole
uvicorn event loop for the full duration. **Fix:** `asyncio.to_thread` now; precomputed
per-topic index later.

### 4. Relational cards — full catalog + every KG JSON per card
`app_relational_view.py` `build_person_card:179` / `build_topic_card:236` / `resolve_entity:92`.
Full scan + `_iter_kg_entities` loads every episode's `.kg.json` (~10 KB each → ~100 MB parsed
per request @ 10k). `async def` calling sync work on the loop. **Fix:** `to_thread` + inverted
KG index (person/topic → episodes).

### 5. `/discover` — catalog scan + up to 400 KG loads + uncached cluster/velocity maps
`app_discover.py:124`, `app_discover_view.py:479`. Has a partial cache
(`_INTEREST_INDEX_CACHE` on `search/metadata.json`) but the catalog scan, `consumer_topic_cluster_map`,
`consumer_theme_cluster_map`, and `temporal_velocity` are re-read every request. **Fix:** cache
those read-only artifacts on `app.state`; make the route `async` + `to_thread`.

### 6. `/corpus/enrichment` (OLD) — loads + returns all enrichment JSONs
`app_enrichment.py:87`. Pair artifacts (`topic_cooccurrence_corpus`, `topic_similarity`,
`guest_coappearance`) grow **O(N²)**. Now superseded by the lean endpoints for the app, but
still mounted and uncached. **Fix:** cache parsed envelopes on `app.state`; consider deprecating
for the consumer plane.

### 7. `/corpus/trending-topics` + `/corpus/entity-signals` (NEW) — parse-all-to-pick-one
`app_enrichment.py:121` (`_corpus_signals`). Correct + tiny output, but the loader **globs and
JSON-parses every `enrichments/*.json` to read `enricher_id`** before discarding the ones not
wanted — so `trending-topics` still parses the 1.54 MB cooccurrence file each call just to skip
it. **Fix (cheap, mine to make):** match by **filename** first (filename == enricher_id for all
current enrichers); parse only the wanted files. Optionally cache parsed envelopes on `app.state`.

### 8. `/podcasts/{id}/signals` — catalog scan + feed KGs + 4 enrichment files/request
`feed_signals.py:305`. **Fix:** shared enrichment cache (#6/#7) + catalog cache (#1).

### 9. `/trending` — loads `temporal_velocity.json` **twice** + catalog **twice** per call
`app_momentum.py:338,341`. Measured ~105 ms locally (2nd-slowest after search). **Fix:**
pass loaded data between `_content_weekly_by_entity` and `_entity_labels`; cache per corpus mtime.

**Multi-worker note:** every existing cache (`_INTEREST_INDEX_CACHE`, `_lance_pool`,
`_episode_reach_cache`) is **per-process**. Under multi-worker uvicorn/gunicorn each worker
warms independently and invalidation is uncoordinated. Plan the catalog cache with that in mind
(mtime key is safe; a shared store is not needed yet).

---

## Client-side waterfall findings

**Home (signed out)** — 6 API calls; the lean trending endpoint confirmed. But **4 of them each
scan the catalog**: `trending?kind=topic`, `trending?kind=show`, `discover`, `podcasts`. Plus
`/api/app/preferences` returns **401 while signed out** — a wasted round-trip that shouldn't fire.
At 10k eps Home ≈ 4 concurrent ~5 s scans → threadpool contention.

**Player** — render-gating fix validated. Two issues: **`related?top_k=6` is fetched twice**
(reqid 354 + 364 — redundant duplicate); and the ~9 episode-scoped calls each re-run
`resolve_slug` (finding #2). Fixing #1/#2 makes every one of these O(1).

**Not captured in-browser:** catalog / search / library / person / topic *page* paints — the
chrome run covered Home + Player (the two the user flagged). Their **backend** cost is in the
sweep table above; their fan-out would repeat the catalog-scan story.

---

## Recommendations (ranked by impact ÷ effort)

1. **[High impact / Medium effort] Catalog cache + slug map on `app.state`.** mtime-keyed. Fixes
   findings #1, #2, most of #4/#5/#8, and the concurrency collapse in one change. This is *the* fix.
2. **[High / Low] `to_thread` the blocking sync work inside the `async def` routes** (CIL topic
   routes #3, relational cards #4). Stops one slow request from freezing the whole event loop —
   directly improves concurrent-user behavior. Cheap.
3. **[Med / Low] Filename-first enrichment selection in `_corpus_signals`** (#7) — stop parsing
   the 1.5 MB cooccurrence file on every `trending-topics` call. Small, and it's my own new code.
4. **[Med / Low] Cache the read-only corpus artifacts** (`temporal_velocity`, cluster maps,
   enrichment envelopes) on `app.state`; dedupe the double-load in `/trending` (#9).
5. **[Low / Low] Client tidy-ups:** don't call `/preferences` when signed out; fix the duplicate
   `related` fetch on the Player.
6. **[Later] Precomputed inverted indexes** (topic→episodes, person→episodes, slug→path) built at
   pipeline time — turns the card/CIL routes from O(corpus) into O(matches). The structural answer
   for 10k+; the caches above are the bridge.

**Concurrency sizing:** until #1+#2 land, treat the consumer API as ~1 catalog-scan-bound request
at a time per worker. At 700 eps (~350 ms/scan) a single worker tops out around a handful of
req/s before latency climbs; the 20-way test shows tail latency exploding well before that. Scale
= workers × (1 / scan-time). After the catalog cache, the scan is amortized to ~0 and the ceiling
moves to real per-request work (KG loads, search embedding).

---

## NOT covered / NOT verified

- **Not tested at real scale.** All 10k-episode numbers are **linear projections** from a 100-ep
  measurement, not a 10k-ep run. `build_catalog_rows_cumulative` dedup cost could be super-linear
  in run count; I did not synthesize a 10k corpus to confirm.
- **Prod paint times not measured.** Local dev-mode client timings (LCP 628 ms) are unminified
  Vite and not comparable to prod; I did not run against `closelistening.app`.
- **catalog / search / library / person / topic pages not driven in-browser** — backend-only via
  the sweep; their client paint/main-thread cost is unmeasured.
- **No fix implemented.** This is analysis only. Nothing in this doc has been changed in code
  (the lean endpoints were the prior commit `ec3296fe5`).
- **Concurrency test was 20-way on a 100-ep corpus on a laptop**, not a load-test rig; absolute
  numbers will differ on prod hardware, but the *serialization behavior* (no parallelism) is
  structural and will hold.
- **Signed-in routes** (library, favorites, interests, resurfacing) measured only where they
  overlap Home/Player; a full signed-in sweep was not run.
