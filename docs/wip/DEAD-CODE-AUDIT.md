# Dead-code audit — `web/learning-player`

**Date:** 2026-09-05 · **Scope:** `web/learning-player/src`, `e2e`, `index.html`
**Purpose:** evidence base for cutting the app back to a minimal working subset.
**Status:** read-only audit. Nothing was edited, deleted, or committed.

---

## Headline

**The app is not full of dead code.** Every one of the 46 components, 19 views, 10
stores and 6 composables is reachable from `src/main.ts`. Of ~85 exports in
`src/services/api.ts`, exactly **two** have no caller. The tools' long
"unused" lists are almost entirely internal-use-only exports, not deletable code.

The real waste is **duplication and unreachable routes**, not orphans. If the goal
is "cut back to a minimal subset", the lever is consolidating four
near-identical trending/browse surfaces — not deleting files.

---

## Method

| Tool | Command | Result |
|---|---|---|
| knip (default) | `npx knip --no-config-hints` | ran; 2 unused files, 15 unused exports, 16 unused types |
| knip (prod entries) | `npx knip --config /tmp/…/knip.prod.json` | ran; **0 unused files**, 47 unused exports, 19 unused types |
| ts-prune | `npx ts-prune -p tsconfig.app.json` | ran; 227 lines, subsumed by knip |

Knip's default config treats every `*.test.ts` as an entry point, which hides a
component whose only importer is its own test. I re-ran it with
`entry: ["src/main.ts","index.html"]` and tests ignored. **It still reported zero
unused files** — that is the strongest single result in this audit.

Every tool candidate below was then re-checked by hand with `grep` across `src/`,
`e2e/` and `index.html`, including same-file internal use (which both tools
ignore, and which flipped most candidates from "dead" to "alive").

---

## Summary table

| # | Category | Candidates | Survivors after manual check |
|---|---|---|---|
| 1 | Vue components never imported | 3 | **0** |
| 2 | Views not reachable from router | 3 | **0** |
| 3 | Routes defined but never linked | 16 checked | **4** |
| 4 | `api.ts` exports with no caller | ~85 checked | **2** |
| 5 | Other `src` exports with no caller | 47 | **1** |
| 6 | Unused exported *types* | 19 | **2** |
| 7 | Pinia stores never used by a component | 10 checked | **0** |
| 8 | Store getters/actions with no external caller | 20 | **2** dead + 9 export-narrowing |
| 9 | Composables with no caller | 6 checked | **0** |
| 10 | i18n keys never referenced | 26 | **23** |
| 11 | CSS classes in `style.css` never used | 10 defined | **0** |
| 12 | Feature flags permanently off | 1 | **1 (partial)** |
| 13 | Unused files (knip) | 2 | **0** |
| 14 | Unused devDependencies (knip) | 1 | **0** |
| — | Duplicated functionality | — | **7 clusters confirmed, 3 refuted** |

Total genuinely-dead symbols: **7 code symbols + 23 i18n keys.** That is the whole
harvest of classic dead code in this app.

---

## 1. Vue components never imported — 0 survivors

All 46 `src/components/*.vue` have at least one non-test importer.

Evidence — for every `.vue` file, references excluding the file itself and its own
`*.test.ts`:

```
for f in $(ls src/components/*.vue src/views/*.vue); do b=$(basename $f .vue)
  grep -rn "\b$b\b" src --include='*.ts' --include='*.vue' \
    | grep -v "^$f:" | grep -v '\.test\.ts:' | grep -v '__checks__' | wc -l
done
```

Minimum result across all 65 files: **1** (`LoginView`, `PersonView`,
`SettingsView` — each the router's lazy `import()`). No zeros. Corroborated by
knip's prod-entry run reporting no unused files.

**Nothing to delete here.**

---

## 2. Views not reachable from `src/router/index.ts` — 0 survivors

Three of the 19 views have no route. All three are **embedded as components** in
`LibraryView` — they are views by filename, components by role.

- `src/views/CollectionsView.vue` — `src/views/LibraryView.vue:25` (import), `:247` (`<CollectionsView />`)
- `src/views/HighlightsView.vue` — `src/views/LibraryView.vue:23` (import), `:222` (`<HighlightsView />`)
- `src/views/ResurfacingInbox.vue` — `src/views/LibraryView.vue:24` (import), `:252` (`<ResurfacingInbox />`)

**RISK: HIGH** to delete any of them — they are the Library tab bodies. See
"DO NOT DELETE".

---

## 3. Routes defined but never linked — 4 survivors

Method: for each of the 16 named routes, grep production code (excluding
`src/router/index.ts` and all `*.test.ts`) for `name: '<route>'` and for the
literal path string.

Four routes have **zero** links and zero `router.push` in production code. They
resolve only if the URL is typed or an e2e spec `goto()`s them.

### 3.1 `browse-shows` — `src/router/index.ts:92-96`
Renders `ShowBrowseView`, which is already embedded in the Browse hub at
`src/views/BrowseView.vue:77` (`<ShowBrowseView embedded />`).

- `grep -rn "name: 'browse-shows'" src | grep -v router/index.ts` → **0**
- `grep -rn -- "/browse/shows" src | grep -v router/index.ts` → **0**
- `grep -rn -- "/browse/shows" e2e` → **0 spec hits** (1 hit in `e2e/E2E_SURFACE_MAP.md`, docs only)

**RISK: LOW.** No link, no spec, and the view itself survives via the hub.
*Deletion note:* `browse-shows` is also listed in `OWNED_ROUTES` at
`src/components/BottomNav.vue:78` — remove it there in the same change.

### 3.2 `browse-people` — `src/router/index.ts:102-106`
- `grep -rn "name: 'browse-people'" src | grep -v router/index.ts` → **0**
- `grep -rn -- "/browse/people" src` → 1 hit, and it is a **comment**: `src/views/HomeView.vue:551`
- `grep -rn -- "/browse/people" e2e/*.spec.ts` → **0**

**RISK: LOW.** Same as above; `PersonBrowseView` survives via
`src/views/BrowseView.vue:79`. Also in `BottomNav.vue:78`.

### 3.3 `browse-topics` — `src/router/index.ts:97-101`
- `grep -rn "name: 'browse-topics'" src | grep -v router/index.ts` → **0**
- `grep -rn -- "/browse/topics" src` → 1 hit, a **comment** at `src/views/HomeView.vue:551`
- `grep -rn -- "/browse/topics" e2e` → **1 real hit**: `e2e/browse-and-topic-pages.spec.ts:55` — `await page.goto('/browse/topics')`

**RISK: MEDIUM.** Deleting the route breaks `browse-and-topic-pages.spec.ts:55`;
the spec must be repointed to `/browse?tab=topics` in the same change. Also in
`BottomNav.vue:78`.

> The comment at `src/views/HomeView.vue:550-552` says these routes would be "dead
> code" without Home's browse-nav strip — but the two links it added
> (`HomeView.vue:559`, `:565`) point at `{ name: 'browse', query: { tab: … } }`,
> the **hub**, not the standalone routes. The comment is stale; the routes it
> claims to rescue are still unreachable.

### 3.4 `queue` (`/queue`) — `src/router/index.ts:45-49`
- `grep -rn "name: 'queue'" src | grep -v router/index.ts` → **0**
- No `<RouterLink to="/queue">` anywhere in production code.
- `QueueView` itself is very much alive — reused as a component at
  `src/components/QueuePanel.vue:14` (import) and `:107` (`<QueueView hide-title />`),
  which `MiniPlayer.vue:113` and `PlayerView.vue:797` both render.
- `grep -rhoE "goto\('[^']*'\)" e2e` → `goto('/queue')` appears **4 times**.

**RISK: MEDIUM.** The *route* is unreachable in the UI, but four e2e specs
navigate to it directly and it is `meta: { requiresAuth: true }`. Deleting the
route is safe for users and breaks 4 specs. Deleting `QueueView.vue` would break
the queue panel — **do not**.

---

## 4. `api.ts` exports with no caller — 2 survivors

I enumerated every `export (async )?(function|const)` in `src/services/api.ts`
and counted references outside that file. Exactly two came back zero, and both
have no same-file internal use either.

### 4.1 `getAuthToken` — `src/services/api.ts:90`
```
$ grep -rn "getAuthToken" src e2e
src/services/api.ts:90:export function getAuthToken(): string | null {
```
One hit: its own definition. Its sibling `setAuthToken` (`api.ts:87`) has 3
production callers, so the setter is live and only the getter is orphaned.

**RISK: LOW.** Pure read accessor, zero references anywhere including tests.

### 4.2 `getCorpusEnrichment` — `src/services/api.ts:259`
```
$ grep -rn "getCorpusEnrichment" src e2e
src/services/api.ts:259:export function getCorpusEnrichment(): Promise<CorpusEnrichmentSignals> {
src/services/api.ts:290:/** Corpus enrichment signals filtered to one entity (same shape as getCorpusEnrichment). */
```
Two hits: the definition and a doc-comment mention. The per-entity variant
`getEntitySignals` (`api.ts:291`) is the one actually used.

**RISK: LOW.** *Note:* the return type `CorpusEnrichmentSignals` is still used by
`getEntitySignals`, so delete the function only, not the type.

> **The other ~83 exports all have callers.** The largest API surface in the app is
> essentially fully used. Five are used in production but have **no test and no
> e2e coverage** — see "NEEDS USAGE DATA".

---

## 5. Other `src` exports with no caller — 1 survivor

knip's prod-entry run listed 47 unused exports. I re-checked each for **same-file
internal use**, which knip does not consider. 46 of 47 are used inside their own
module; the export keyword is redundant but the code runs.

Only one is unreferenced even within its own file:

### 5.1 `APP_SCHEME` — `src/services/deepLinks.ts:48`
```
$ grep -n "APP_SCHEME" src/services/deepLinks.ts
48:export const APP_SCHEME = 'closelistening'
$ grep -rn "APP_SCHEME" src e2e | grep -v deepLinks.ts
(no output)
```
A custom URL scheme constant that nothing reads.

**RISK: MEDIUM.** The *string* `closelistening` may be referenced by native config
(`capacitor.config.ts`, iOS `Info.plist`, Android manifest). Deleting the TS
constant is safe; do not assume the scheme itself is unused.

### 5.2 Export-narrowing only (no deletion value) — 46 symbols
These are internal helpers/constants exported for unit tests. Examples with their
same-file use:
`pushSupported` (`usePushSubscription.ts:10`, used `:32`/`:54`), `renderCard`
(`useShareCard.ts:47`, used `:86`), `isAuthFailure` (`outbox.ts:219`, used `:265`),
`MATCHED_FIELD_ORDER` (`matchedFields.ts:34`, used `:87`), `prefersReducedMotion`
(`motion.ts:17`, used `:32`), `extractAccentFromImage` (`accent.ts:76`, used `:106`),
all five `downloads.ts` folder constants (used throughout that file),
`REGISTRY_KEY_PREFIX` (`stores/downloads.ts:92`, used `:96`).

**RISK: LOW to un-export, ZERO VALUE.** Removing `export` deletes no code and
breaks the tests that import them. Not recommended.

---

## 6. Unused exported types — 2 survivors

Same treatment: 19 candidates, 17 are referenced inside their own file by another
type. Two are referenced nowhere at all.

- `src/services/types.ts:349` — `interface HighlightsResponse`. `grep -c "HighlightsResponse" src/services/types.ts` → **1** (the declaration); cross-file refs → **0**.
- `src/services/types.ts:379` — `interface NotesResponse`. Same result: self=1, external=0.

**RISK: LOW** for both — types are erased at build time and nothing imports them.
They look like response envelopes for endpoints whose client functions return a
narrower shape.

---

## 7. Pinia stores never used by a component — 0 survivors

All 10 stores have at least two `.vue` consumers:

| Store | `useXStore` | `.vue` consumers |
|---|---|---|
| `auth.ts` | `useAuthStore` | 14 |
| `capture.ts` | `useCaptureStore` | 5 |
| `downloads.ts` | `useDownloadsStore` | 5 |
| `favorites.ts` | `useFavoritesStore` | 3 |
| `interests.ts` | `useInterestsStore` | 7 |
| `library.ts` | `useLibraryStore` | 4 |
| `player.ts` | `usePlayerStore` | 3 |
| `queue.ts` | `useQueueStore` | 6 |
| `savedQueries.ts` | `useSavedQueriesStore` | 2 |
| `userPreferences.ts` | `useUserPreferencesStore` | 6 |

**Nothing to delete.**

---

## 8. Store getters/actions with no external caller — 2 dead + 9 export-narrowing

I brace-matched each store's `getters`/`actions` block (options stores) or its
`return {}` (setup stores) and grepped each member for external use.

> My first pass here used a `sed`/`tr` pipeline that silently produced **zero
> members** for `queue.ts` and `library.ts` and dropped the first and last member
> of every other store. The "no dead members" result it produced was an artifact of
> a broken scan, not a finding. Rewritten with a brace-matching Node script.

### Genuinely dead (zero references anywhere, including the store's own file)

**8.1 `forEpisode` getter — `src/stores/capture.ts:44`**
```
$ grep -rn "\bforEpisode\b" src --include='*.vue' --include='*.ts' \
    | grep -v '\.test\.ts:' | grep -v '__checks__' | grep -v '^src/stores/capture.ts:'
(no output)
```
A per-episode highlight filter (`s.highlights.filter(h => h.episode_slug === slug)`)
that nothing calls. **RISK: LOW.**

**8.2 `feedIds` getter — `src/stores/library.ts:32`**
```
$ grep -rn "\bfeedIds\b" src --include='*.vue' --include='*.ts' \
    | grep -v '\.test\.ts:' | grep -v '^src/stores/library.ts:'
(no output)
```
`(s): string[] => s.items.map(i => i.feed_id)`. Its sibling getter (an
`isFollowed`-style `some()` check on the line above) is the one in use.
**RISK: LOW.**

### Not dead — internal or private (do not delete)

- `capture._sync` `:75`, `capture._capture` `:90`, `capture._uncapture` `:124`,
  `downloads._put` `:229`, `queue._sendItem` `:118` — underscore-prefixed private
  actions called via `this._x` **inside their own store**. My scan excluded the
  store's own file, so they registered as zero by construction. **False positives.**
- `player.onPlay` `:138`, `onPause` `:150`, `onTimeUpdate` `:158`,
  `onDurationChange` `:163`, `onError` `:167`, `savePosition` `:300`,
  `setRate` `:337` — all bound internally:
  `src/stores/player.ts:70-76` (`audio.addEventListener('play', onPlay)` …),
  `:82` (`pagehide` → `savePosition()`), `:344` (`cycleRate` → `setRate`).
  Only their **export** from the store's return object (`:431-441`) is
  unnecessary. **RISK: LOW to un-export, zero value.**
- `userPreferences.preferences` `:6`, `hydrated` `:45`, `available` `:15` — state
  refs no consumer reads. Every consumer uses `get` / `set` / `hydrate` /
  `$reset` / `resetAvailability` only (verified across `App.vue:300-302`,
  `ProfileView.vue:89-90`, `PlayerView.vue:93-104`, `HomeView.vue:121,262`,
  `YourWeek.vue:80-86`, `savedQueries.ts:55-100`). **RISK: MEDIUM** — Pinia
  devtools and `$reset` semantics depend on state being returned; narrowing the
  return object can change store behaviour. Not worth it.

---

## 9. Composables with no caller — 0 survivors

All six are imported by production code:

| Composable | Consumer |
|---|---|
| `useFollowedShows.ts` | `src/views/LibraryView.vue:15` |
| `usePushSubscription.ts` | `src/views/ProfileView.vue:12` |
| `usePwaUpdate.ts` | `src/components/PwaUpdateToast.vue:11` |
| `useSectionState.ts` | 12 files (`HomeView.vue:27`, `MomentumRail.vue:11`, …) |
| `useShareCard.ts` | `src/views/HighlightsView.vue:25` |
| `useSignInGate.ts` | 10 files (`FavoriteButton.vue:11`, `PlayerView.vue:19`, …) |

**Nothing to delete.**

---

## 10. i18n keys never referenced — 23 survivors of 26

`src/i18n/locales/en.json` has **524 leaf keys**; 453 appear as a literal
`t('…')` / `$t('…')` argument. I resolved the remainder against (a) any quoted
key-shaped string anywhere in `src`/`e2e` (catches `label: 'nav.home'` passed to
`t(item.label)`), and (b) template-literal dynamic bases.

Detected dynamic bases: `recap.`, `home.yourWeekSection.`, `home.yourWeekFirstRun.`,
`cache.`, `search.foldedKind.`, `search.kind.`, `collections.kind.` — 26 keys under
these are **live via dynamic construction** and must not be deleted.

**23 keys have zero references of any kind** (`grep -rn "<key>" src e2e` excluding
`locales/` → 0 for each):

| Key | Location | Value |
|---|---|---|
| `nav.catalog` | `en.json:7` | "Catalog" |
| `home.askTagline` | `en.json:26` | "Search across every episode — …" |
| `home.trendingChip` | `en.json:53` | "{topic} — {factor}× its own 6-month average" |
| `card.insights` | `en.json:85` | "Insights" |
| `browse.hubSubtitle` | `en.json:152` | "Explore the corpus by episode, topic, or person." |
| `browse.episodesDesc` | `en.json:154` | "The full catalogue, newest first." |
| `browse.showsCount` | `en.json:159` | "1 show \| {count} shows" |
| `browse.topicsDesc` | `en.json:162` | "Trending topics and storylines." |
| `browse.peopleDesc` | `en.json:164` | "Hosts, guests, and mentioned figures." |
| `interests.cardBody` | `en.json:314` | "Tell us what you're into for sharper picks." |
| `ec.sigGrounding` | `en.json:359` | "Grounding" |
| `ec.sigGroundedLine` | `en.json:360` | "{grounded} of {total} claims backed by quotes…" |
| `ec.sigSimilar` | `en.json:367` | "Similar topics" |
| `ec.sigAlongside` | `en.json:368` | "Often discussed alongside" |
| `collections.moment` | `en.json:406` | "Moment" |
| `library.shows` | `en.json:482` | "Shows" |
| `library.queue` | `en.json:494` | "Queue" |
| `library.recent` | `en.json:495` | "Recent" |
| `library.recentEmpty` | `en.json:503` | "Nothing played yet." |
| `highlights.obsidianDelta` | `en.json:511` | "Exported {written} changed, {removed} removed…" |
| `downloads.empty` | `en.json:582` | "Nothing downloaded yet…" |
| `downloads.cancel` | `en.json:590` | "Cancel download" |
| `downloads.remove` | `en.json:591` | "Remove download" |

**RISK: LOW** for all 23. `en.json` is the only locale (`SUPPORTED_LOCALES = ['en']`
at `src/i18n/index.ts:10`), so there is no second file to fall out of sync.

*Caveat:* `browse.*Desc` and `ec.sig*` read like copy that was written for a UI
revision that landed differently. They are dead **strings**, but they may be the
only surviving record of intended copy. Cheap to keep, cheap to delete.

**Watch out — `library.shows` looks referenced but is not.** `grep "library.shows"`
returns 3 hits, all substring matches on *different* keys:
`src/views/LibraryView.vue:120` uses `t('library.showsEmpty')` and `:127` uses
`t('library.showsBrowse')`. Do not let that fool a future grep.

---

## 11. CSS classes in `src/style.css` never used — 0 survivors

`src/style.css` is 175 lines and defines **10** class selectors, all under
`@layer components` (`:44`); the rest is `@layer base` element styling. Every one
is used in templates:

| Class | `src/style.css` | Uses in `src`/`e2e` |
|---|---|---|
| `.lp-kicker` | `:46` | 55 |
| `.lp-kicker--muted` | `:62` | 3 |
| `.lp-section` | `:69` | 42 |
| `.lp-speaker` | `:81` | 4 |
| `.lp-nav` | `:91` | 3 |
| `.lp-segment` | `:118` | 5 |
| `.lp-segment-option` | `:127` | 2 |
| `.lp-theme-chip` | `:148` | 9 |
| `.lp-fav` | `:156` | 5 |
| `.lp-fav--on` | `:172` | 1 |

`src/theme/tokens.css` and `src/theme/directions.css` define **no** class
selectors (CSS custom properties only), so there is nothing to audit there.

**Nothing to delete.** Styling in this app is Tailwind-first; there is no
accumulated dead stylesheet.

---

## 12. Feature flags / env-gated code — 1 partial survivor

Every `import.meta.env.VITE_*` read in `src`:

| Flag | Read at | Declared in `env.d.ts`? | In `.env.mobile`? |
|---|---|---|---|
| `VITE_API_BASE_URL` | `tier.ts:90` | yes `:10` | yes `:23` |
| `VITE_DEV_API_BASE` | `tier.ts:77` | yes `:14` | yes `:38` |
| `VITE_PREVIEW_BASIC_AUTH` | `tier.ts:126` | **no** | yes `:47` |
| `VITE_PREVIEW_COOKIE` | `tier.ts:144` | **no** | yes `:55` |
| `VITE_SENTRY_DSN_PLAYER` | `main.ts:75` | yes `:18` | yes `:15` |
| `VITE_SENTRY_DSN_PLAYER_DEV` | `main.ts:67` | yes `:23` | yes `:64` |
| `VITE_UMAMI_WEBSITE_ID` | `main.ts:127` | yes `:28` | yes `:18` |
| `VITE_UMAMI_SRC` / `_DEV` | `main.ts:124,128` | yes `:29,:32` | yes `:19,:65` |
| **`VITE_ANALYTICS_OFF`** | `main.ts:68` | **no** | **no** |

### 12.1 `VITE_ANALYTICS_OFF` — `src/main.ts:68`
```js
const devDefault = import.meta.env.DEV && import.meta.env.VITE_ANALYTICS_OFF !== '1'
```
```
$ grep -rn "VITE_ANALYTICS_OFF" src env.d.ts .env.mobile .env.mobile.example vite.config.ts
src/main.ts:58:// `VITE_ANALYTICS_OFF=1` disables the default.
src/main.ts:68:const devDefault = import.meta.env.DEV && ...
```
Declared nowhere, set nowhere. It is always `undefined`, so the comparison is
always true and the branch is **permanently ON** (dev analytics always enabled in
dev builds). Not dead code — an escape hatch that nobody can reach without
knowing it exists.

**RISK: LOW** to remove the check (behaviour is unchanged today);
**MEDIUM** if anyone relies on it as a local opt-out. Better fix: declare it in
`env.d.ts` and `.env.mobile.example` so it is discoverable.

### 12.2 `__MOBILE_INTERNAL__` — not dead
`vite.config.ts:81` defines it as `process.env.MOBILE_RELEASE !== '1'`; consumed at
`src/services/tier.ts:26-28` to tree-shake the tier switch out of release builds.
Working as designed.

---

## 13–14. knip's unused files and devDependency — 0 survivors

Both are false positives. See "DO NOT DELETE".

---

## DUPLICATED FUNCTIONALITY

Confirmed by reading both sides of each pair. Line references verified.

### D1. `TrendingSparkChips` ≈ `MomentumRail` — **strongest finding**
Two components rendering a structurally identical row list (label + `Sparkline` +
follow toggle + show-more), fed by two different endpoints, displayed on Home one
tab apart (`src/views/HomeView.vue:604-614`).

Byte-identical class strings:
```
src/components/TrendingSparkChips.vue:91   class="flex items-center gap-1 rounded-lg transition hover:bg-overlay"
src/components/MomentumRail.vue:101        class="flex items-center gap-1 rounded-lg transition hover:bg-overlay"

src/components/TrendingSparkChips.vue:95   class="flex min-w-0 flex-1 items-center gap-2.5 rounded-lg px-2 py-1 text-left"
src/components/MomentumRail.vue:106        class="flex min-w-0 flex-1 items-center gap-2.5 rounded-lg px-2 py-1 text-left"

src/components/TrendingSparkChips.vue:129  class="shrink-0 rounded-full px-2 py-1 text-base leading-none transition"
src/components/MomentumRail.vue:128        class="shrink-0 rounded-full px-2 py-1 text-base leading-none transition"
```
Same `Sparkline :width="56" :height="20"`, same `✓` / `＋` follow glyphs.

### D2. Four endpoints answer "what's trending"
`MomentumRail.vue:60` → `getTrending()`; `TrendingTopics.vue:64` →
`getTrendingTopics()`; `TrendingShowsRail.vue:31` → `getTrending('show',…)`;
`Storylines.vue:42` → `getStorylines()`; plus a fourth per-show chip row at
`PodcastSignalsBand.vue:230-243`.

`src/components/TrendingTopics.vue:143-151` states in prose that the first two
**disagree on live data** (`systems thinking` = 0.86× vs 1.78×) and that both are
kept on screen "while both are being evaluated against real data." That is an
unresolved production A/B by the file's own admission — a decision, not a refactor.

### D3. Show-more toggle triplicated verbatim
Identical `expanded`/`visible`/`hiddenCount` trio and button markup at
`TrendingSparkChips.vue:30,79-82,142`, `MomentumRail.vue:67-70,142`,
`Storylines.vue:49-54,110` (the third has drifted: `mt-2 px-1` vs `mt-1 px-2`).
`Storylines.vue:48` documents the copy: *"same as the Rising/Trending rails."*

### D4. `titleOf()` / `vFmt()` copied between two files
`src/components/TrendingShowsRail.vue:53-60` and
`src/components/MomentumRail.vue:72-79` are byte-identical — and neither lives in
`src/components/trending.ts`, which already owns `trendDirection`/`trendColor`/
`trendArrow`.

### D5. `PersonBrowseView` ⊂ `TopicBrowseView`
`comm -12` on sorted unique lines: **88 shared lines** of 119 / 165.
`trendingTheme` is byte-identical (`PersonBrowseView.vue:51-60` vs
`TopicBrowseView.vue:53-62`); `trendingRows` differs by one line
(`role: e.role ?? null`, `PersonBrowseView.vue:45`); `loadTrending` differs by one
string (`'person'` vs `'topic'`). `PersonBrowseView` could be
`<TopicBrowseView kind="person">`. `ShowBrowseView` is genuinely different — not
duplicated.

### D6. Segmented control implemented four times, three geometries
`TrendWindowTabs.vue:25/:35` is the extracted component. Three sites ignore it and
re-type the markup: `HomeView.vue:583/:592`, `EntityCardBody.vue:244/:252`,
`ListeningRecap.vue:86/:91` (which also drops the track and uses a fourth active
style). Plus a fifth pill dialect in `InterestsPicker.vue:142/:163`.

**139 `rounded-full` sites across 47 files; there is no `Chip.vue`/`Pill.vue`.**
The only shared artifact is `.lp-theme-chip` (`style.css:148`), which carries
colour only — every caller still re-types the geometry.

### D7. Nav route list duplicated, and the two copies disagree
`BottomNav.vue:46-52` lists tabs declaratively; `App.vue:425-450` hand-writes the
same destinations as `<NavIconLink>` blocks with four byte-identical inline SVGs.
They disagree on one destination:
```
src/App.vue:425                 <NavIconLink :to="{ name: 'catalog' }" :label="t('nav.browse')">
src/components/BottomNav.vue:48   { name: 'browse', label: 'nav.browse' },
```
**Desktop "Browse" goes to `/catalog`; mobile "Browse" goes to `/browse`.** Same
label, two destinations. This is a bug, not just duplication.

### Refuted (investigated, genuinely not duplicated)
- **List/card components** — `EpisodeCard` (80px horizontal row), `ShowTile`
  (square tile), `EntityCard` (modal shell), `EntityCardBody` (452-line shared
  content), `StorylineCard` (bottom sheet), `DownloadedList` (40px offline row),
  `TranscriptList` (transcript) each have a distinct job. `EpisodeCard` really is
  the single episode row (`CatalogView.vue:118`, `QueueView.vue:68`,
  `QueuePanel.vue:116`).
- **Search entry points** — exactly **one** search implementation:
  `SearchView.vue:219` `searchCorpus(...)`. `HomeView.vue:248` and
  `EntityCardBody.vue:172` only `router.push({name:'search'})`.
  `KnowledgePanel.vue:231` calls a *different* endpoint (`searchEpisode`,
  episode-scoped). `ListToolbar`/`CatalogView` do client-side `includes()`
  filtering, which is not search.
- **Player controls** — implemented once, in `src/stores/player.ts`.
  `PlayerView.vue:626` says so; `:1018-1021` wires straight through;
  `PlayerControls.vue` is purely presentational. Only the play/pause SVG glyph is
  drawn twice with different path data (`MiniPlayer.vue:107-110` vs
  `PlayerControls.vue:132-138`) — cosmetic.
- **"Horizontal trending rails"** — false. Only `CardRail.vue:65` scrolls
  horizontally (`snap-x` appears once in the whole repo). `MomentumRail` and
  `TrendingShowsRail` are vertical stacks; the "Rail" names mislead.

---

## HALF-BUILT

### Zero TODO/FIXME markers
```
$ grep -rn "TODO\|FIXME\|XXX\|HACK" src --include='*.vue' --include='*.ts' | grep -v '\.test\.ts'
(no output)
```
No `v-if="false"`, no commented-out template blocks, no stub components.
**This codebase has no TODO debt.** Broadening to `WIP|stub|placeholder|coming soon`
yields 17 hits, all `:placeholder="t(...)"` bindings or prose using "stub" to
describe a chart bar.

### Untranslated strings shipped to users
In an app with a working 524-key i18n bundle:
```
src/components/TrendingSparkChips.vue:47   `${tp.label} — ${tp.v}× vs recent average · ${tp.total} mentions`
src/components/TrendingSparkChips.vue:98   `${tp.label}, trending at ${tp.v} times its recent average`
src/components/TrendingSparkChips.vue:133  isFollowed(tp.id) ? `Following ${tp.label}` : `Add ${tp.label} to my interests`
src/components/MomentumRail.vue:77         dir === 'up' ? 'rising' : dir === 'down' ? 'cooling' : 'steady'
src/components/TrendingShowsRail.vue:58    (identical to MomentumRail.vue:77)
src/views/LoginView.vue:84                 placeholder="or a custom name…"
```
`TrendingSparkChips.vue:133` is the sharpest — `Storylines.vue:99-101` does the
identical job correctly via `t('home.storylineFollowing', { label: s.label })`.
(`TierSwitch.vue:35-36` is excluded — self-labelled "internal build only".)

### Divergent de-slug regexes — a live label bug
Four copies, three different regexes:
```
src/components/FollowedInterests.vue:39  id.replace(/^(tc|thc|topic|person):/, '').replace(/[-_]+/g, ' ')
src/views/ProfileView.vue:71             id.replace(/^(tc|topic|person):/,     '').replace(/-/g,      ' ')
src/components/EntitySignals.vue:51      norm(id).replace(/^(?:person|topic|org):/, '').replace(/[-_]+/g, ' ')
src/components/TrendingTopics.vue:121    x.topic_id.replace(/^topic:/, '').replace(/[-_]+/g, ' ')
```
`ProfileView`'s alternation omits `thc:` (storylines) and its second replace omits
`_`. A followed storyline `thc:managing-risk` renders as **"Managing risk"** in
Library and **"thc:managing risk"** in Profile — same token, same session, two
labels. `FollowedInterests.vue:8-9` claims it matches ProfileView; it does not.

### Partially adopted error contract
`useSectionState` (the "an outage must not look like an empty account" contract)
is adopted by 12 files and ignored by 11. Notably `TopicBrowseView.vue:82` and
`PersonBrowseView.vue:69` swallow every failure into an empty state via
`.catch(() => [])` — exactly the failure mode the composable exists to prevent —
while the `MomentumRail` beside them handles it correctly.

---

## DO NOT DELETE — looks dead, isn't

Verified false positives. Please don't re-litigate these.

| Thing | Why it looks dead | Proof it is alive |
|---|---|---|
| `src/views/CollectionsView.vue` | no route | `LibraryView.vue:25` import, `:247` `<CollectionsView />` |
| `src/views/HighlightsView.vue` | no route | `LibraryView.vue:23` import, `:222` `<HighlightsView />` |
| `src/views/ResurfacingInbox.vue` | no route | `LibraryView.vue:24` import, `:252` `<ResurfacingInbox />` |
| `src/views/QueueView.vue` | `/queue` route unlinked | reused as a component: `QueuePanel.vue:14` import, `:107` `<QueueView hide-title />`, rendered by `MiniPlayer.vue:113` + `PlayerView.vue:797` |
| `src/views/ShowBrowseView.vue`, `TopicBrowseView.vue`, `PersonBrowseView.vue` | their routes are unlinked | embedded in the hub: `BrowseView.vue:77`, `:78`, `:79` (`embedded` prop) |
| `public/push-sw.js` | knip "unused file" | `vite.config.ts:143` — `importScripts: ['push-sw.js']` |
| `playwright.docker.config.ts` | knip "unused file"; no reference inside `web/learning-player` | monorepo `Makefile:1908` — `npx playwright test --config=playwright.docker.config.ts` |
| `@capacitor/assets` devDependency | knip "unused devDependency" | a CLI, documented at `docs/mobile-app-guide.md:217` and `docs/capacitor-build-runbook.md:326` |
| `kp.density_early` / `_mid` / `_late` i18n keys | no literal `t('kp.density_early')` | built dynamically: `EpisodeDensity.vue:93`, `:101`, `:106` — ``t(`kp.density_${seg}`)``. My first scan missed this because the base ends in `_`, not `.` |
| 26 keys under `recap.` / `home.yourWeekSection.` / `home.yourWeekFirstRun.` / `cache.` / `search.kind.` / `search.foldedKind.` / `collections.kind.` | no literal `t()` | template-literal dynamic bases exist for all seven prefixes |
| `capture._sync/_capture/_uncapture`, `downloads._put`, `queue._sendItem` | "no external caller" | private actions called via `this._x` **within their own store** |
| `player.onPlay/onPause/onTimeUpdate/onDurationChange/onError/savePosition/setRate` | "no external caller" | bound internally at `stores/player.ts:70-76`, `:82`, `:344` |
| 46 of 47 knip "unused exports" (`pushSupported`, `renderCard`, `isAuthFailure`, `MATCHED_FIELD_ORDER`, `prefersReducedMotion`, `DOWNLOAD_*`, …) | knip only sees cross-file imports | each is used **inside its own file**; un-exporting also breaks the unit tests that import them |
| 17 of 19 knip "unused types" (`EpisodeStatus`, `HighlightKind`, `NoteTarget`, `CommsDigest`, `FoldedKind`, `YearKey`, …) | knip only sees cross-file imports | referenced by other types in the same file |
| `library.showsEmpty` / `library.showsBrowse` | a grep for `library.shows` matches them | `LibraryView.vue:120`, `:127` — distinct live keys |

---

## NEEDS USAGE DATA — wired correctly, undecidable from code

These are fully implemented and reachable. Code analysis cannot tell whether a
human ever uses them. Each entry names the signal that would settle it.

| Surface | Where | Why code can't decide | Signal that would settle it |
|---|---|---|---|
| **Web push notifications** | `usePushSubscription.ts` → `ProfileView.vue:108-121`; API `getVapidKey`/`subscribePush`/`unsubscribePush` (`api.ts:967-985`) | fully wired, but zero unit tests and **zero e2e coverage** (`grep -rl "enablePush\|subscribePush" e2e` → 0) | count of rows in the server's push-subscription table; delivery/click-through on sent pushes |
| **MCP / Connected Agents** | `ConnectedAgents.vue` (only host: `ProfileView.vue`); API `getMcpConfig`/`getMcpTokens`/`createMcpToken`/`revokeMcpToken`/`revokeMcpConnection` (`api.ts:1059-1102`) | a complete token-management UI reachable only from deep inside Profile | count of MCP tokens ever minted; distinct users with ≥1 live connection |
| **Obsidian export** | `exportObsidian` (`api.ts:858`), `highlightsExportUrl` (`:829`), `fetchHighlightsExport` (`:837`) → `HighlightsView.vue:107` | works; no e2e coverage | `POST`/`GET` hit counts on the export endpoints |
| **Highlight share cards** | `useShareCard.ts:86` `shareHighlightCard` → `HighlightsView.vue:25` | canvas render + native share; no e2e coverage | native-share invocations; count of rendered cards |
| **Dev-user login** | `getDevUsers` (`api.ts:721`) → `LoginView.vue:42` | gated on a server `enabled` flag this repo can't read | whether `/auth/dev-users` returns `enabled: true` in production |
| **`/queue` route** | `router/index.ts:45` | unlinked in the UI, but a bookmarkable URL | pageview count for `/queue` vs `/episode/*` queue-panel opens |
| **`/browse/shows`, `/browse/topics`, `/browse/people`** | `router/index.ts:92-106` | unlinked, but external deep-links or bookmarks may exist | pageviews for those three paths over the last 90 days; inbound referrers |
| **Which trending surface earns its space** | `MomentumRail` vs `TrendingTopics` vs `TrendingShowsRail` vs `Storylines` on Home | `TrendingTopics.vue:143-151` says both are deliberately kept while being evaluated; the evaluation's outcome is not in the code | per-rail click-through rate on Home; tab-selection counts for `home` trending tabs |
| **Library sub-tabs** | `CollectionsView`, `HighlightsView`, `ResurfacingInbox` inside `LibraryView` | all three load on mount; all are reachable | per-tab engagement in Library |
| **Interests / follow graph** | `InterestsPicker`, `FollowedInterests`, `interests` store (7 `.vue` consumers) | heavily wired; value depends on whether users actually follow things | distinct users with ≥1 followed topic/person; effect on ranking CTR |

Umami is already wired (`src/main.ts:124-128`) — these questions are answerable
from existing analytics, not new instrumentation.

---

## RANKED RECOMMENDATION — best value ÷ risk

| # | Action | Value | Risk | Why |
|---|---|---|---|---|
| 1 | Delete route `browse-shows` (`router/index.ts:92-96`) + its entry in `BottomNav.vue:78` | Removes an unreachable route and a lazy chunk | **LOW** | Zero links, zero e2e `goto`, view survives via `BrowseView.vue:77` |
| 2 | Delete route `browse-people` (`router/index.ts:102-106`) + `BottomNav.vue:78` entry | Same | **LOW** | Only in-repo mention is a stale comment (`HomeView.vue:551`); zero e2e |
| 3 | Delete the 23 dead i18n keys (§10) | Shrinks the 524-key bundle by 4.4%; removes copy for UI that no longer exists | **LOW** | Zero references of any kind; `en` is the only locale |
| 4 | Delete `getAuthToken` (`api.ts:90`) and `getCorpusEnrichment` (`api.ts:259`) | Trims the API surface | **LOW** | Only self-references; keep the `CorpusEnrichmentSignals` type |
| 5 | Fix D7: point `App.vue:425` at `{ name: 'browse' }` | Fixes a real bug — desktop and mobile "Browse" land on different pages | **LOW** | One-line change; `/catalog` is still reachable as the hub's default tab |
| 6 | Delete `capture.forEpisode` (`capture.ts:44`) and `library.feedIds` (`library.ts:32`) | Two dead getters | **LOW** | Zero references including inside their own stores |
| 7 | Collapse `TrendingSparkChips` + `MomentumRail` into one component (D1) | Largest single code reduction available; removes a duplicated row/follow/expand implementation | **MEDIUM** | Both are on Home and user-visible; needs a product call on which data source wins (see NEEDS USAGE DATA) |
| 8 | Fix the de-slug divergence: extract one helper, use it in all four sites | Fixes a visible label inconsistency (`thc:managing risk` in Profile) | **LOW** | Pure extraction; `FollowedInterests.vue:39` already has the correct regex |
| 9 | Make `PersonBrowseView` a mode of `TopicBrowseView` (D5) | Deletes ~88 duplicated lines | **MEDIUM** | Both are live tab panels in the Browse hub; e2e covers `/browse?tab=topics` |
| 10 | Delete route `browse-topics` (`router/index.ts:97-101`) + `BottomNav.vue:78` entry | Same as #1/#2 | **MEDIUM** | Must repoint `e2e/browse-and-topic-pages.spec.ts:55` from `/browse/topics` to `/browse?tab=topics` in the same change |

**Deliberately not recommended:** removing `export` from the 46 internal-only
symbols (§5.2) or narrowing store return objects (§8) — both delete zero code and
break unit tests. And the `/queue` route (§3.4): unlinked, but it is a
bookmarkable authenticated URL with 4 e2e specs behind it; decide with pageview
data, not code.

---

## NOT COVERED / NOT VERIFIED

Stated with equal weight to the findings above.

- **Nothing here is runtime-verified.** I did not run the app, the unit suite
  (`npm run test:unit`), the type checker, or Playwright. Every claim is from
  static reading plus the greps quoted inline. A "LOW risk" rating means *no
  static reference exists* — not that a build was proven green after deletion.
- **Dynamic i18n detection is a heuristic.** It catches ``t(`base.${x}`)`` and
  `'base.' + x`, and any quoted key-shaped literal anywhere in `src`/`e2e`. It
  would **miss** a key assembled from two variables, or one arriving from the
  server. My first pass already missed `kp.density_${seg}` because the base ends
  in `_`. Treat the 23 keys as high-confidence, not certain.
- **`src/services/api.ts` export enumeration** relied on a regex over
  `^export (async )?(function|const)`. A re-exported symbol or an
  `export { x }` block would not be counted. I did not audit `api.test.ts` for
  exports it reaches that production code does not.
- **e2e specs were searched but not read.** I grepped them for route paths and
  symbol names; I did not read the specs to confirm what each assertion actually
  exercises.
- **Native (Capacitor) surfaces are unaudited.** `android/`, `ios/`,
  `capacitor.config.ts` and `Info.plist` were not searched. This specifically
  weakens the `APP_SCHEME` finding (§5.1) — the constant is dead in TS, but the
  scheme string may be live in native config.
- **`src/player/`, `src/utils/`, `src/theme/` were audited for dead exports but
  not for duplication.** `matchedFields.ts:5` cross-references logic in a
  *different repo* (`gi-kg-viewer`), which was not opened.
- **Test-file duplication is entirely unassessed** (40+ `*.test.ts`), as is
  whether each duplication above is pinned by a test — which changes the cost of
  every consolidation in the ranked list.
- **Tailwind `@apply` classes** in component `<style>` blocks were not audited;
  only 3 files carry a `<style>` block, but I did not read them. My "no shared
  chip component" claim is scoped to `src/components/`.
- **I could not determine** whether `TrendingTopics` + `MomentumRail` coexisting
  is duplication or a live experiment. `TrendingTopics.vue:143-151` argues it is
  deliberate and temporary; nothing in the code says whether that evaluation
  concluded. That needs a human decision.
