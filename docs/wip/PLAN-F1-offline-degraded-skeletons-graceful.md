# Plan — F1.2/F1.3/F1.4: offline degraded-mode, skeletons, graceful "can't load"

Status: IMPLEMENTED (F1.2/F1.3/F1.4 core) · 2026-09-09 · branch `feat/player-ux-overhaul`

Done: Phase A (useOnline + OfflineBanner + fail-fast reads), Phase B/C (reserved
skeletons + retry on Podcast, Search, Catalog, Show/Topic/Person browse views).
PlayerView + LibraryView + QueueView already carried the contract (cached-paint,
loadError+retry, page-level stale notice) from the #1905/#1909 offline arc.
Remaining polish (not blocking): a player-shaped skeleton for a cold UNCACHED
episode; broader per-page offline e2e (Phase E) beyond the banner + shell specs.
Scope: consumer learning player (`web/learning-player/`). Backlog ref:
`docs/wip/PLAYER-UX-BACKLOG-2026-09-09.md` §F1.

## The three slices

- **F1.2** App-level online/offline awareness → an explicit offline mode that changes *what* and
  *how* each page renders (show / episode / topic / storyline / person). Define the degraded-render
  contract ONCE.
- **F1.3** Layout stability: reserve space / skeletons so data loading causes **no layout jumps**,
  every page.
- **F1.4** Graceful "can't load this" per page — never blank, never crash.

## What already exists (REUSE, do not rebuild)

- `composables/useSectionState.ts` — per-section `loading | ready | error` + cache-read-before-fetch
  + `stale`; module-scope `anyStale` tally. Rule: only 401/403 destroys cache; transport errors keep
  it and mark stale.
- `components/SectionStatus.vue` — renders skeleton rows (`rows` prop) on `loading`, and an error
  message + `@retry` on `error`. Already on Home rails, Trending, YourWeek, Catalog, Collections,
  Topic/Person/Storyline browse, Highlights, PodcastSignalsBand, TopicConversationArc.
- `components/StaleNotice.vue` — ONE page-level notice + retry when any section shows cached-stale.
- `services/contentCache.ts` (+ `deviceStore.ts`) — per-account cache, `CACHE_KEYS`, hydrate-then-
  revalidate. `services/api.ts` `getMe` has an 8s `AbortSignal.timeout`; other calls have none.
- Stores expose `loaded / stale / unavailable` (library, favorites, queue, completed, capture,
  collections, auth). Auth hydrates from device snapshot first (F1.1 fix).

## Gaps this plan closes

1. **No online/offline awareness** — no `navigator.onLine`, no `useOnline`, no offline indicator.
   Failure is only discovered by a request hanging/erroring.
2. **Skeleton gaps (F1.3)** — PlayerView (none), PodcastView (heading only), SearchView (none),
   LibraryView per-tab (none).
3. **Error-state gaps (F1.4)** — PlayerView / PodcastView / SearchView swallow with `.catch(()=>[])`
   → blank or silent-empty on failure instead of a graceful retry.
4. **Fetch hang offline (F1.1a)** — data calls have no timeout; offline they wait on the OS.

## Approach — phased, each phase independently green + committed

### Phase A — Foundation (the contract + one primitive)

- `composables/useOnline.ts`: reactive `isOnline` from `navigator.onLine` + `online`/`offline`
  events (singleton listener, SSR/happy-dom safe). Unit-tested.
- **Degraded-render contract** (documented in UXS-014 "states", enforced by useSectionState):
  1. Have cache → paint it; if `isOnline` is false, mark `stale` and show the offline signal.
  2. No cache + loading → skeleton that reserves the final layout (no jump).
  3. No cache + failed/offline → graceful "can't load this — you're offline" + retry. Never blank.
  4. Network-only sub-sections (e.g. live search) → show "unavailable offline" in place, not empty.
- Fetch-hang fix: short-circuit at the data layer when `!isOnline` (fail fast to cache/error) — see
  DECISION 2.

### Phase B — Skeletons (F1.3)
Retrofit PlayerView, PodcastView, SearchView, LibraryView tabs onto `useSectionState` +
`SectionStatus` (or a reserved-height wrapper) so first paint reserves the final box. Verify no
layout jump (measured, not eyeballed).

### Phase C — Graceful errors (F1.4)
Replace `.catch(()=>[])` swallow in PlayerView / PodcastView / SearchView with `useSectionState`
error + retry. A failed load renders a retry card, never blank, never a crash.

### Phase D — Offline mode UX (F1.2)
Wire the offline signal (DECISION 1) to `useOnline`. Apply the contract per entity page
(show/episode/topic/storyline/person). Controls needing network dim/disable with a one-line reason
when offline (mirrors the queue's existing "showing saved queue" pattern).

### Phase E — Tests
- Unit: `useOnline`, the retrofitted views' loading/error, contract in useSectionState.
- e2e: extend `offline.spec.ts` — each key page (home/show/episode/topic/storyline/person/search)
  renders cached-or-graceful offline, never blank; a reserved skeleton on cold load.

## Decisions — RESOLVED (operator, 2026-09-09)

1. **Offline indicator UX** — **global top banner + keep the per-page `StaleNotice`.** A slim
   app-level bar when `!isOnline`, alongside the existing per-section stale notices.
2. **Fetch-hang strategy** — **`useOnline` short-circuit** (skip the call when known-offline → cache
   or graceful error), **plus a generous safety timeout backstop** for the online-but-unreachable
   case.
3. **How aggressive is "offline mode"** — **minimal**: cached content + the offline indicator +
   graceful "can't load" where there is no cache. No per-page layout re-arrangement; network-only
   sub-sections say "unavailable offline" in place.

## Non-goals
- Re-architecting the SW/cache (works; audio never cached, per-user never cached — keep).
- Making every network-only feature work offline (search embeddings, etc. stay "unavailable
  offline").
