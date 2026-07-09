# RFC-104: Operator Shows Library (shows-first browse)

**Status:** Draft
**PRD:** [PRD-044](../prd/PRD-044-operator-shows-library.md) · **UXS:** [UXS-015](../uxs/UXS-015-operator-shows-library.md)
**Surface:** `web/gi-kg-viewer` — the `library` main tab · **Backend:** none (reuses existing endpoints)

---

## Abstract

Add a **shows-first** browse mode to the operator viewer's Library tab: a grid of the corpus's shows
(`GET /api/corpus/feeds`) that opens a per-show detail (cover, description, episode count, RSS) listing
that show's episodes (`GET /api/corpus/episodes?feed_id=…`), each episode cross-linked into the graph
via the existing `subject.focusEpisode` path. No backend change; reuses `PodcastCover`, the episode-row
shape, and the shared corpus lens. The existing episode-first `LibraryView` (UXS-003) is untouched and
becomes the "Episodes" mode of a two-mode Library tab.

## Problem

The `library` tab is episode-first only — a flat, paginated list with a Feed filter chip. There is no
way to browse the corpus *by show*: to see its shows, their identity (art/description/count), and drill
show → episodes. The data and endpoints already exist (see PRD-044 Background); only the navigation is
missing. The consumer player already ships the shows-first shape (Home "Your shows" → PodcastView).

## Design

No new endpoints. No new stores. Three new components + one toggle, wired into the existing library tab.

> **Amendment (operator feedback, 2026-07-09).** A show now opens **in the right subject rail**, not
> in-panel. The card-click calls `subject.focusShow(feed_id)` → a new `'show'` subject kind →
> **`ShowRailPanel`** (mirrors `EpisodeDetailPanel`'s header: cover + title + "N episodes · RSS", then
> the episode list). Clicking an episode in the show rail → `subject.focusEpisode` (which now pushes the
> show onto the subject Back stack, so the episode rail carries a ‹ Back to the show). The grid stays in
> the main library surface. The former in-panel `ShowDetailView` is retired from the path (file kept,
> dormant). §1 below describes the original in-panel plan; the rail wiring supersedes the grid⇄detail
> hop. **Phase 2 (shipped, 2026-07-09):** a richer show landing — a Signals band above the episode
> list shows the show's **top topics** + **key people**, from a new `GET /api/corpus/feed-signals`
> endpoint that counts Topic/Person nodes across the feed's per-episode KGs (ranked by episode count,
> diarization placeholders filtered). Chips open the node view in the same rail via
> `subject.focusTopic`/`focusPerson` (each pushes the show onto the Back stack). **Cross-show overlaps
> remain deferred** — they need an all-feeds pass, not a single-feed aggregate.

### §1 Component tree

```text
library tab (App.vue)
├─ LibraryModeToggle.vue      [new]  segmented control: Shows | Episodes (persisted per session)
├─ v-if mode==='shows'
│   └─ ShowsBrowse.vue        [new]  owns shows-first sub-state (selectedFeedId)
│        ├─ ShowsView.vue     [new]  grid of shows (fetchCorpusFeeds)
│        └─ ShowDetailView.vue[new]  one show: header + episode list (fetchCorpusEpisodes{feedId})
└─ v-else  (mode==='episodes')
    └─ LibraryView.vue        [existing, UNCHANGED]  the flat episode-first list
```

`ShowsBrowse` is the only stateful new piece: a `selectedFeedId: string | null` ref. `null` → render
`ShowsView` (grid); non-null → render `ShowDetailView` for that feed. Selecting a show sets it;
"Back to shows" clears it. This is **replace-in-panel** (house pattern, per UXS-014 / operator memory),
not a modal or stacked overlay.

### §2 Data flow (all client-side over existing endpoints)

- **ShowsView**: on mount / corpus change, `fetchCorpusFeeds(shell.corpusPath)` →
  `CorpusFeedItem[]`; sort `episode_count desc, display_title asc`; render one `ShowCard` per feed
  (`PodcastCover` + title + "N episodes" + clamped description). Emits `select(feed)` upward.
- **ShowDetailView**: props `{ feed: CorpusFeedItem, corpusPath }`. On `feed` change, reset + first
  page `fetchCorpusEpisodes(corpusPath, { feedId: feed.feed_id, limit })`; "Load more" appends via
  `next_cursor`. Header from the `feed` prop (title, count, `rss_url`, clamped `description`, large
  `PodcastCover`). Episode rows reuse the Library episode-row markup.
- **Episode open**: a row click emits `open-library-episode({ metadata_relative_path })`, which
  `App.vue` routes to `subject.focusEpisode(metadata_relative_path)` (the same handler the flat Library,
  Digest, and Search already use) → episode opens in graph / episode-detail. **No new cross-link
  policy** — it composes with the existing `graphNavigation` path (E2E_SURFACE_MAP §"automation
  contract"), deliberately not adding another band-aid load-source.

### §3 State, persistence, lens

- Library **mode** (`shows` | `episodes`) is a ref in the library-tab host, persisted to
  `localStorage` (`gikg.library.mode`, mirroring the theme/shell stores). **Default `episodes`**
  (status quo — the existing operator flow + the 9 `library.spec.ts` e2e are unchanged); **Shows is
  opt-in** via the toggle and remembered once chosen. Promoting Shows to the default is PRD-044 OQ1 —
  a deliberate operator decision, not baked in here.
- `selectedFeedId` is ephemeral sub-state of `ShowsBrowse` (not persisted across reloads for v1; a
  return from the graph to the Library tab preserves it because the component is kept alive under the
  tab, matching how `LibraryView` retains its scroll/selection).
- The shared date lens (`corpusLens.sinceYmd`) is **not** applied to the shows grid (a show's identity
  is lens-independent) but the ShowDetail episode list MAY honor it later (OQ, deferred); v1 lists all
  of a show's episodes newest-first.

### §4 Reuse (no duplication)

- `PodcastCover.vue` — cover resolution (episode art → feed art → initials), already used in Digest /
  EpisodeDetail / NodeDetail / Library. Used for both the grid card and the detail header.
- Episode-row shape (cover, title, recency dot, publish date, summary line, topic pills, GI/KG badges)
  — extracted from `LibraryView`'s row into a shared `EpisodeListRow.vue` **only if** the extraction is
  clean; otherwise `ShowDetailView` renders a row with the same classes/testids to avoid destabilizing
  the 35 KB `LibraryView`. (Decision at implementation; default: a small shared `EpisodeListRow` if it
  drops ≥ ~40 lines of duplication, else inline parity. Tracked in the PR description.)
- `corpusLibraryApi.ts` — `fetchCorpusFeeds`, `fetchCorpusEpisodes` (existing; typed).

### §5 Accessibility & states

- Grid cards + episode rows are `role="button"`, `tabindex="0"`, Enter/Space activate, visible
  focus ring (parity with `data-library-episode-row`).
- Every async surface has explicit loading / error / empty states (`shows-grid` empty when 0 feeds;
  `show-detail` empty when a show has 0 episodes — never a silent blank).
- Descriptions clamp (line-clamp) with an expand toggle when truncated (mirrors consumer PodcastView).

## API

No change. Contract already covered by existing integration tests for `/api/corpus/feeds` and
`/api/corpus/episodes` (`tests/integration/server/test_*corpus*`). This RFC adds **consumer-side**
assertions that the shapes carry the fields the surface needs (title, count, image, description, rss).

## Testing

Matches house tiers (roadmap rubric weak→good→excellent):

- **vitest unit** — `ShowsView` sort/empty/error; `ShowsBrowse` select↔back state; description clamp.
- **vitest mount** — `ShowsView.mount` (grid renders N cards from a mocked feeds payload, cover +
  count), `ShowDetailView.mount` (header from feed, episodes from mocked payload, "Load more",
  episode-click emits `open-library-episode`), `LibraryModeToggle` (toggle switches + persists).
- **server-contract integration** — assert `/api/corpus/feeds` + `/api/corpus/episodes?feed_id` return
  the fields the surface binds (already partly covered; add field-presence asserts on the v3 fixture).
- **operator e2e (mocked, fast PR gate)** — `web/gi-kg-viewer/e2e/shows-library.spec.ts`: Library tab →
  Shows grid → open a show → episode list → click episode → lands on graph. Uses `page.route` mocks
  (the harness pattern of the existing 50+ specs).
- **served-corpus stack-test** — `tests/stack-test/stack-shows-library.spec.ts`: same flow against the
  Docker-served v3/seeded corpus (mirrors `stack-viewer` / `stack-person-profile`). Exercises real
  `/api/corpus/feeds` + real `PodcastCover` art resolution.

DoD: every new component ≥ **good** (mount test asserting real payload); the end-to-end shows→episode
→graph flow is **excellent** (served-corpus stack-test).

## Phasing

1. **P1 — components + toggle** (ShowsView, ShowDetailView, ShowsBrowse, LibraryModeToggle) + vitest.
2. **P2 — wire into App.vue library tab** + `focusEpisode` cross-link + surface-map updates
   (VIEWER_IA, E2E_SURFACE_MAP, uxs/index).
3. **P3 — e2e** (mocked spec + stack-test spec) + server field-presence asserts.

Each phase is independently green (vue-tsc 0 + targeted vitest) and bisectable.

## Alternatives considered

- **A. New top-level "Shows" tab** (peer of Digest/Graph/Library). Rejected: shows + episodes are the
  same browse concern; a second tab fragments IA and duplicates the corpus/lens plumbing. A mode toggle
  inside Library keeps one home for "browse the corpus".
- **B. Fold shows-first into `LibraryView`** (add a `groupBy=show` to the existing view). Rejected for
  v1: `LibraryView` is 35 KB and heavily tested; a grid↔detail state machine inside it raises
  regression risk. A sibling `ShowsBrowse` isolates the new surface. (Unifying the Feed filter chip
  with show-detail is PRD-044 OQ3, deferred.)
- **C. Add a `/api/corpus/shows/{feed_id}` detail endpoint.** Rejected: `feeds` + `episodes?feed_id`
  already provide the header and the list; a detail endpoint is redundant server surface.

## No ADR needed

This reuses existing endpoints, stores, and component patterns; there is no durable architectural fork
(no new dependency, schema, or cross-cutting contract). The RFC + UXS are the record. If P2 extraction
of a shared `EpisodeListRow` grows into a cross-surface row contract, that graduates to its own note.

## References

- [PRD-044](../prd/PRD-044-operator-shows-library.md) · [UXS-015](../uxs/UXS-015-operator-shows-library.md)
  · [UXS-003](../uxs/UXS-003-corpus-library.md) (episode-first Library) · [VIEWER_IA](../uxs/VIEWER_IA.md)
- `web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md` (graph-navigation automation contract)
- `corpusLibraryApi.ts`, `PodcastCover.vue`, `App.vue` (`mainTab`, `focusEpisode`)
