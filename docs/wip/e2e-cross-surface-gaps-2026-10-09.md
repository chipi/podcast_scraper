# E2E gaps behind the 2026-10-09 stale-surface fixes

Ten bugs were fixed on 2026-10-09 (commits 7d82835c8, 813377239, e44ba8020, 569218b0b, b078fba0d,
6480cffbf). All ten are one shape: **a write on one surface, and another surface that keeps showing
what it had before.** None was caught by a test. This note says why, and what to add so the next
one is.

## The ten, and the layer each lives in

| # | Bug | Writer | Stale reader | Layer |
|---|-----|--------|--------------|-------|
| 1 | Board made / added to from the Save sheet missing in Library | Save sheet | Library › Boards (kept alive) | player, keep-alive + private copy |
| 2 | Revisit link consumed in the player still counted | Player `?revisit` | Revisit count, Library › Revisit | player, store not told |
| 3 | Library › Revisit list frozen | anything that reviews | Library › Revisit (kept alive) | player, keep-alive |
| 4 | Your Week frozen on Home | listening elsewhere | Home › Your Week (kept alive) | player, keep-alive |
| 5 | Clear history leaves played marks | Profile | every episode row (store) | player, store not told |
| 6 | Recently played shows a cleared history | Profile | Queue › Recently played (cache) | player, `[]` read as failure |
| 7 | Board delete / remove / add-link not on Home | Boards view | Home boards teaser (store) | player, store not told |
| 8 | "Your trends" frozen on Discover | listen / save / follow | Discover › Trends (kept alive) | player, keep-alive |
| 9 | Viewer Library / Digest frozen after a pipeline job | Dashboard jobs | Library, Digest (kept alive) | viewer, keep-alive |
| 10 | Search resolves old storyline / theme names after re-enrichment | enrichment run | `resolve_entity` (lru cache) | server, cache without a token |

## Why the suite did not catch them

1. **E2E navigates by reloading.** 284 `page.goto` calls across the player specs; the in-app
   helper `navTo` (e2e/helpers.ts — whose own comment says `goto` "cannot test anything about
   client-side navigation") is used 10 times in 5 files. A full load rebuilds every store and
   remounts every kept-alive tab, so the stale copy can never be observed. **The case of record:
   `collections.spec.ts` already does bug #1's flow** — create a board from the sheet, then check
   Library — and checks with `page.goto('/library?tab=collections')` (line 83). It passed for
   exactly the reason the bug was invisible.
2. **No spec opens the reader first.** Keep-alive bugs need the reading tab to be ALREADY MOUNTED
   when the write happens (visit it, leave, write, come back). No spec has that shape.
3. **Unit tests mount one component.** Each surface was tested alone; the contract "the store
   Library reads hears about this write" lives between two files and no test owned it. Only the
   2026-10-09 tests mount a real `<KeepAlive>`.
4. **Nothing enumerates the writes.** There are 47 write functions in the player's `api.ts`, ~20
   called outside a store. Nothing says, per write, which surfaces must reflect it — so nothing can
   say a pair is untested.
5. **Server caches have no invalidation contract.** Most tokens on an mtime; two `lru_cache`s keyed
   on the corpus root alone did not, and no test rewrites an artifact under a live process.

## Proposal

Four pieces, in the order they pay off.

### A. A cross-surface journey spec, warm-tab shaped (player e2e, real API)

One file, `e2e/cross-surface.spec.ts`, one test per row of the matrix below. Every test has the same
shape and uses **no `page.goto` after sign-in**:

1. sign in, open the READER surface (it mounts and stays alive),
2. `navTo` elsewhere and do the WRITE the way a user does,
3. `navTo` back to the reader and assert the write shows.

A helper `warmThenReturn(page, reader, write)` keeps each test to a few lines. First rows are the
ten bugs above (the viewer one in the viewer suite). Also: fix `collections.spec.ts` lines 83 and
124 to `navTo` — the existing test then covers #1 by itself.

Cost: about 8 tests on the existing app-e2e stack, ~1–2 min of CI.

### B. A write → reader matrix, with a guard

`e2e/CONSISTENCY_MATRIX.md` (or a `.ts` table): every exported write in `src/services/api.ts`, the
surfaces that must reflect it, and the test that proves it (e2e row, or a unit test for stores).
A unit guard — same idea as `test_ios_uitest_suites_are_wired.py` — parses `api.ts` and fails
when a write function has no row. A new write then cannot ship without someone deciding where it
shows and how that is tested.

### C. Unit layer: a shared keep-alive harness and a rule

Move the `keptAlive()` helper (now copied in 3 test files) to `src/test/keptAlive.ts`. Rule for
every component under a kept-alive tab (`KEEP_ALIVE_TABS` in App.vue) that fetches user data in
`onMounted`: it must either read a store or re-read on `onActivated`, and have a keep-alive test
that says so. A static guard can list the `onMounted` fetches under those views for review.

### D. Server: a cache invalidation contract

A guard test that finds every module-level cache in `src/podcast_scraper/server`
(`lru_cache`, module dicts named `*_cache` / `*_CACHE`, `perf_cache.get_or_compute`) and requires
each to be in an allowlist with its invalidation token (artifact mtime / corpus mtime / TTL). Plus,
per artifact-keyed cache, one "rewrite the artifact, call the endpoint, see the change" test — the
shape of `test_resolve_entity_sees_a_re_enriched_storyline_without_a_restart`.

### Viewer

Same shape as A, in `web/gi-kg-viewer/e2e` (28 of its specs already stub the API with `page.route`): open Library, go to Dashboard, a job (stubbed
`/api/jobs` going running → succeeded) finishes, return to Library, the new episode is there.

## Not covered by this proposal

- Device tiers (iOS / Android UI tests) also navigate in-app but assert per screen; adding
  cross-surface rows there is possible but slow (minutes per row) — the web e2e row covers the
  same JS.
- Multi-device staleness (a write on the phone, the laptop tab open) is a different problem
  (no push channel); out of scope here.
- The matrix guard proves a row EXISTS, not that the test asserts the right thing; review still
  owns that.

## Built (2026-10-09)

| Piece | What landed |
|-------|-------------|
| A | `web/learning-player/e2e/cross-surface.spec.ts` (5 journeys × phone + desktop). `navTo` now waits for the destination URL. `collections.spec.ts` returns to Library in-app instead of `page.goto`. |
| B | `src/__checks__/consistencyMatrix.ts` (every write in `api.ts` → readers → proof) and `consistency-matrix.test.ts` (a write with no row fails). |
| C | `src/test/keptAlive.ts` (a real `<KeepAlive>`, with a `late` mode) and `src/__checks__/kept-alive-refresh.test.ts` (a per-user read with no `onActivated` must be exempt, with a reason). |
| D | `tests/unit/podcast_scraper/server/test_cache_invalidation_contract.py` (every cache under `server/` and `search/` states its invalidation; a token cache must take its token). |
| Viewer | `web/gi-kg-viewer/e2e/cross-surface-pipeline-job.spec.ts` (Library re-reads when a job it did not watch finishes). |

Each new test was checked against the bug it is for: with the fix reverted it fails, with the fix in place it passes.

### Bugs the new tests found while being written

1. **The `onActivated` refreshes from 813377239 never ran for a lazily-mounted component.** They skipped
   the "first activation" as if it were the mount, but a component that mounts after its tab is already
   showing (Library › Revisit on the tab's first visit, Discover's trends list) gets no activation for
   its mount — its first activation IS the first return. Now: re-read only after the tab was left once
   (`onDeactivated`). Found by the e2e trends journey; pinned by the `late` keep-alive unit tests.
2. **`getTrending` shared every answer for the whole session**, including `scope: 'mine'` — so "Your
   trends" never moved after the first load, and (no user in the key) a second account on the same
   device got the first account's. Now mine is shared only while in flight; corpus for 5 minutes.
3. **Home's Revisit rail** read its "kept · reviewed" line and the items due once per session. Found by
   the kept-alive refresh guard.
4. **`navTo` returned before arriving**, so a test could "leave" a tab without leaving it — the reason
   the trends journey first looked like a server problem.
