# WIP: Episode detail rail (right column)

## Problem

Episode detail lived inside **Library** (`aria-label="Episode"`) while **Search** and **Explore** used the **right sidebar**. Two detail contexts competed for horizontal space and duplicated the “where do I read this episode?” mental model.

## Decision

- The **right sidebar** (`w-80`) is **one** surface: either **tools** (Search + Explore tabs) or **episode summary** (full episode detail previously in Library).
- **Mutual exclusion:** only one mode is visible at a time (`paneKind: 'tools' | 'episode'`).
- **Safe default (Search):** selecting or focusing a search hit does **not** switch to episode mode. An explicit **Episode summary** control on each result opens the episode rail when `source_metadata_relative_path` is available.
- **Library:** choosing an episode in the list still opens episode detail (now in the rail) — that selection is explicit.
- **Digest:** **Recent** / topic **hit** rows call `openEpisodePanel` only (no tab switch); toolbar **Open Library** still goes to **Library**. Search **L** still uses pending Library selection + **Library** tab. Digest rows use **`bg-overlay`** when `metadata_relative_path` matches the rail (parity with Library list).
- **Digest ↔ Library:** Library’s corpus/health watch clears episode context only when those values **change**, not on first mount (so Digest-selected episode survives opening **Library**). After `loadEpisodes`, if `metadataRelativePath` is not in the loaded list and `next_cursor` is null, clear. **Digest** `loadDigest`: if the current path is absent from **rows** and **topic hits**, clear (episode not in current digest window).

## State (`stores/episodeRail.ts`)

- `paneKind`: `'tools' | 'episode'`.
- `toolsTab`: `'search' | 'explore'` — restored when returning to tools.
- `metadataRelativePath`: corpus `metadata_relative_path` for `GET /api/corpus/episodes/detail`.
- Actions: `openEpisodePanel(path)`, `showTools(opts?)`, `resumeDetailPanel()` (graph node if stashed id, else episode if path set), `clearEpisodeContext()` (clear path + tools mode).

## Shell (`App.vue`)

- When `paneKind === 'episode'`: top nav shows **Episode** + **Search & Explore** (returns to tools); body is `EpisodeDetailPanel`.
- When `paneKind === 'tools'`: Search / Explore tab nav; **`Back to episode`** / **`Back to details`** when stashed (`resumeDetailPanel()`). Collapsed rail vertical strip uses the same back action when expanding from tools with stash.
- `watch(metadataRelativePath)`: expand `rightOpen` when a non-empty path is set (opening detail should reveal the rail).
- **`/`** keyboard: `showTools()` + Search tab + focus query (existing behavior, now ensures tools mode).

## Follow-ups (optional)

- ~~Graph node detail~~ — **Done:** double-tap / search focus opens ``NodeDetail`` in the App right rail; layout controls stay **top-left** of the canvas (no in-canvas strip).
- Mobile: stack or drawer instead of three columns.
- Reduce duplicate feeds/index fetches between Library and `EpisodeDetailPanel` (shared store or cache).
