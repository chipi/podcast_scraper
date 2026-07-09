# PRD-044: Operator Shows Library (shows-first browse)

**Status:** Draft
**Related:** PRD-042 (consumer Home "Your shows"), UXS-003 (operator Library, episode-first),
UXS-015 (this feature), RFC-104 (this feature), [VIEWER_IA](../uxs/VIEWER_IA.md)
**Surfaces:** `web/gi-kg-viewer` (operator viewer) — the `library` main tab

---

## Summary

The operator viewer's `library` tab today is **episode-first**: one flat, paginated list of every
episode in the corpus, with a *Feed* filter chip to narrow to one show. There is no way to **browse
the corpus by show** — to see "which podcasts are in here", each show's identity (cover art,
description, episode count), and drill from a show into its episodes.

The consumer player already has this shows-first shape (Home "Your shows" grid → per-podcast detail
view). This PRD brings the same mental model to the **operator** viewer: a **Shows** browse mode in
the Library tab — a grid of the corpus's shows, each opening a **show detail** (cover, description,
episode count, RSS) with that show's episodes, every episode cross-linked into the existing
graph / episode-detail / node surfaces.

It is a **pure frontend surface over existing endpoints** (`GET /api/corpus/feeds`,
`GET /api/corpus/episodes?feed_id=…`) — no new server code, no new data.

## Background & Context

- `GET /api/corpus/feeds` already returns, per show: `feed_id`, `display_title`, `episode_count`,
  `image_url` / `image_local_relpath` (cover art), `rss_url`, `description`.
- `GET /api/corpus/episodes?feed_id=…` already returns that show's episodes with artwork, summary
  title/bullets, publish date, duration, and GI/KG artifact presence.
- The operator viewer already renders feed/episode cover art via the shared `PodcastCover` component
  and cross-links an episode into the graph via `subject.focusEpisode(metadata_relative_path)`.
- The `library` main tab exists (`App.vue` `mainTab` includes `'library'`) and hosts the episode-first
  `LibraryView` (UXS-003).

So every capability this feature needs already ships — it is unshipped **navigation/IA**, not missing
data or endpoints. This mirrors the PR-D finding that operator enricher surfaces were built but not
surfaced; here the *data path* is built but the shows-first *browse* is not.

## Goals

- **G1** — Browse the corpus by show: a grid of shows with cover, title, and episode count.
- **G2** — Open a show to its detail: cover, description, episode count, RSS, and its episode list.
- **G3** — Cross-link everywhere: show → episodes; episode → graph / episode-detail / node view
  (reusing the existing `focusEpisode` path so a Show-detail episode behaves like a Library episode).
- **G4** — Zero backend change; reuse `PodcastCover`, the episode-row pattern, and `corpusLens`.
- **G5** — Full test coverage matching house tiers: vitest (unit + mount), server-contract integration
  (existing endpoints), and operator e2e (mocked fast gate + served-corpus stack-test).

## Non-Goals

- **NG1** — No new server endpoints or data model changes (feeds/episodes already expose everything).
- **NG2** — Not replacing the episode-first Library (UXS-003) — the shows-first browse is an
  **additional mode** in the same tab; the flat list stays for cross-show/date scans.
- **NG3** — No per-user state (favourites/queue) — that's the consumer's personal Library (PRD-042);
  the operator browses the whole corpus.
- **NG4** — No editing of show metadata (title/art/description) — read-only browse. Feed *overrides*
  remain in the existing `FeedOverrideEditor` (admin), out of scope here.
- **NG5** — No audio playback — the operator viewer is a knowledge/graph tool, not a player.

## Personas

- **Operator / analyst** — loads a corpus, wants to orient by "what shows are here" before diving into
  the graph; needs a show's episode list as a launchpad into GI/KG.

## User Stories

- As an operator, I open the Library tab and **see the shows in my corpus** as a grid of covers with
  titles + episode counts, so I know the corpus's composition at a glance.
- As an operator, I **click a show** and see its cover, description, episode count, and RSS, then its
  episodes newest-first.
- As an operator, I **click an episode** in a show and land on that episode in the graph / episode
  detail, exactly as if I'd opened it from the flat Library or Digest.
- As an operator, I can **switch** between "Shows" (grouped) and "Episodes" (flat) in the Library tab
  without losing my corpus/date lens.

## Functional Requirements

### FR1: Library mode toggle (Shows ⇄ Episodes)
The `library` tab gains a segmented control: **Episodes** (existing `LibraryView`, the **default** —
status quo, so nothing regresses) and **Shows** (new, opt-in). The chosen mode persists per browser
(`localStorage`); the shared corpus/date lens (`corpusLens`) is unchanged across modes.
`data-testid="library-mode-shows"` / `library-mode-episodes`. (Promoting Shows to the default is OQ1.)

### FR2: Shows grid (ShowsView)
From `GET /api/corpus/feeds`: a responsive grid of show cards, each with `PodcastCover`, `display_title`
(fallback `feed_id`), and `episode_count` ("N episodes"). Sorted by episode_count desc then title.
Empty state when the corpus has no feeds. Loading + error states. `data-testid="shows-grid"`,
`shows-card-{feed_id}`.

### FR3: Show detail (ShowDetailView)
Clicking a show replaces the grid in-panel (no modal — house pattern) with: a header (large
`PodcastCover`, `display_title`, `episode_count`, RSS link if present, expandable `description`) and the
show's episodes from `GET /api/corpus/episodes?feed_id=…`, newest-first, paginated via the existing
cursor. A **Back to shows** control returns to the grid. `data-testid="show-detail"`,
`show-detail-back`, `show-detail-episode-{i}`.

### FR4: Episode rows are cross-linked
Each Show-detail episode row reuses the Library episode-row shape (cover, title, publish date, summary
line, topic pills, GI/KG badges) and, on click, calls `subject.focusEpisode(metadata_relative_path)`
→ the episode opens in the graph/episode-detail exactly like a flat-Library episode. Recency dot +
"has GI/KG" affordances match UXS-003.

### FR5: Descriptions & images
Show + episode descriptions render (clamped, expandable when long, mirroring consumer PodcastView's
180-char clamp). Cover art resolves via `PodcastCover` (episode art → feed art → initials fallback);
absent art degrades gracefully, never a broken image.

### FR6: Deep-link / return continuity
Selecting a show sets library sub-state (`selectedFeedId`); returning from the graph to the Library tab
restores the last show detail (or grid). No router (operator viewer is tab-state, not route-based) —
state lives in the library view / shell store.

## API summary

No new endpoints. Reuses:
- `GET /api/corpus/feeds?path=…` → shows (title, count, art, description, rss).
- `GET /api/corpus/episodes?path=…&feed_id=…&cursor=…` → a show's episodes (paginated).
- (Existing) `subject.focusEpisode` → graph/episode-detail cross-link.

## Success Metrics

- Operators can identify corpus composition (shows + counts) without scrolling a flat episode list.
- Time-to-first-episode-open from a cold Library ≤ 2 clicks (tab → show → episode).
- Zero server changes; no regression in the existing episode-first Library (UXS-003) tests.

## Open Questions

- **OQ1** — Default Library mode. **Implemented default = Episodes** (status quo — keeps the existing
  operator flow + all 9 `library.spec.ts` e2e unchanged; Shows is opt-in and remembered once chosen).
  Open: promote **Shows** to the default (more discoverable) once the operator has lived with it? A
  one-line change in `LibraryTab.readMode()`.
- **OQ2** — Show sort: episode_count desc (proposed) vs alphabetical vs recency of newest episode?
- **OQ3** — Should the Feed filter chip in the flat Episodes mode deep-link into a show detail (unify
  the two entry points)? Proposed: later; keep modes independent for v1.

## References

- PRD-042 (consumer Home "Your shows"), UXS-003 (operator Library), UXS-015 + RFC-104 (this feature)
- [VIEWER_IA](../uxs/VIEWER_IA.md) — operator shell IA
- `corpusLibraryApi.ts` (`fetchCorpusFeeds`, `fetchCorpusEpisodes`) · `PodcastCover.vue`
