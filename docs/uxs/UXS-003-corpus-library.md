# UXS-003: Corpus Library

- **Status**: Active
- **Authors**: Podcast Scraper Team
- **Parent UXS**: [UXS-001: GI/KG Viewer](UXS-001-gi-kg-viewer.md) -- shared tokens,
  typography, layout, states
- **Related PRDs**:
  - [PRD-022: Corpus Library & Episode Browser](../prd/PRD-022-corpus-library-episode-browser.md)
- **Related RFCs**:
  - [RFC-067: Corpus Library API & Viewer](../rfc/RFC-067-corpus-library-api-viewer.md)
  - [RFC-062: GI/KG viewer v2](../rfc/RFC-062-gi-kg-viewer-v2.md)
- **Implementation paths**:
  - `web/gi-kg-viewer/src/components/library/LibraryView.vue`
  - `web/gi-kg-viewer/src/components/episode/EpisodeDetailPanel.vue`
  - `web/gi-kg-viewer/src/stores/episodeRail.ts`

---

## Summary

The **Corpus Library** is a catalog view inside the same SPA: it answers "what feeds
and episodes were processed?" and connects to Graph and Semantic search without a
separate theme or product chrome. This UXS defines layout, density, and component
rules for the Library tab and the shared Episode rail. All tokens reference
[UXS-001](UXS-001-gi-kg-viewer.md).

---

## Information architecture

- **Entry:** A main tab labeled **Library**, placed after Digest and before
  Graph / Dashboard, with `aria-label` consistent with other main views.
- **Layout (desktop):** Library main column (`canvas` / `surface`) plus the right
  shell rail:

1. **Episode filters** -- one collapsible: **subtitle** row explains narrowing and holds
   primary **Apply** (reloads using title and summary filters; same as **Enter** in those
   fields). On wide viewports, a three-column row -- **published on or after** (date +
   presets, aligned with Search **since**; **same Pinia `corpusLens` state as Digest**) on
   the left (column heading **Dates**); **Title** and **summary / topic** in the center
   as **two rows**, each **label left** + **field right** (same idea as the date row);
   header **Title & summary** plus small **Clear text**; **feed** list on the right;
   stacks vertically on narrow widths. The **Feed** column uses a **scrollable**
   list with a capped height (taller on large viewports) so catalogs with many feeds
   (for example ~20) get a **vertical scrollbar** instead of stretching the panel.
   With no row selected, episodes include **all** feeds; choosing a feed narrows the list.
   **Clear feed filter** stays next to the **Feed** label at all times: **disabled** when
   no feed is selected, **enabled** when a feed is selected; click restores the all-feeds
   episode list.
   Feed rows use display title when present (stable `feed_id`, plus RSS and description in
   `title` hover when `GET /api/corpus/feeds` includes them), `border` dividers, and `overlay`
   for the selected feed row.

2. **Episode column** -- **`h2` Episodes** with a muted tabular count: **`(N)`** for the
   loaded page set, **`(N+)`** when **`next_cursor`** indicates more pages (native **`title`**
   on the count explains scroll / **Load more**), plus a **?** **HelpTip** (same control as
   Digest **Recent**) holding the short guide to filters, infinite scroll / **Load more**,
   and the right **Episode** rail. List **`region`** **`aria-label`** includes the count and
   whether more episodes are available. Scrollable list: cursor pagination from
   `GET /api/corpus/episodes` (`limit` ~20 per page + `next_cursor`); **Load more**
   at the bottom plus scroll-to-load when the user nears the end. Title row is
   episode title (left) and right column: feed display name (truncated, left) and
   publish date, E#, duration (inline, right, when each is present); native `title`
   hover on the feed name shows RSS URL, feed id, and feed description when the API
   provides them. Summary line under the title -- same recap rules as Digest, including
   topics fallback -- then topic pills; recap wraps fully (no line clamp); selected
   row uses `overlay`.

3. **Episode rail** (right shell sidebar) -- `role="region"`
   `aria-label="Episode"` (use `exact: true` in automation so it does not match
   "Episodes"). Shown when the user selects a Library episode or opens one from
   Digest; mutually exclusive with Search & Explore in that rail. Same content:
   `surface` card styling, episode title (heading scale); meta block: feed on the
   first line (full width, wrap), then publish date, E#, duration on the line below
   (left, `muted`, list-scale meta); optional summary title and summary prose; when
   bullets follow that recap, a `border` separator plus `h4` "Key points" (`muted`,
   small caps scale) precedes summary bullets as a semantic list (`ul` / `li`).
   From Search & Explore, **Back to episode** restores the Episode rail when a
   library episode path is stashed; **Back to details** restores Graph node rail.

---

## Graph integration (Episode rail)

- Double-tapping an Episode node opens the Episode rail when a corpus metadata path
  resolves (node properties, loaded artifact path, or catalog episode_id lookup).
- With the Episode rail already open from Library or Digest, switching the main tab
  to Graph highlights that episode's node and centers/zooms the canvas when the node
  is present in the merged graph.
- Below the scrollable episode body, a **Graph neighborhood and connections** strip
  (only when the main tab is Graph) shows a read-only Local neighborhood mini
  Cytoscape (1-hop ego around the node) then the Connections list with `G` per row.
- When RFC-072 bridge.json is available, **Appears in (bridge):** shows whether the
  node's canonical id appears in Grounded Insights, the Knowledge graph, or both.
- Technical ids: `E` uses the same Episode blue as the graph legend / Cytoscape
  (bold white glyph, strong border); `?` uses a separate neutral chip. Native
  tooltip on `E` carries the graph id.

---

## Visual and token rules

- Reuse `surface`, `border`, `elevated`, `muted`, `primary` for lists, filters, and
  actions -- no new palette for the library.
- GI / KG actions in the Episode rail use domain tokens (`gi`, `kg`) for buttons or
  badges that refer to artifact type.
- Search handoff control uses `primary` (it is a navigation affordance to an existing
  panel, not a domain-colored insight).
- **Vector index awareness (RFC-067 Phase 3):** Optional **Indexed** chip on feed rows
  when that `feed_id` appears in `GET /api/index/stats` -> `feeds_indexed`.
  Episode rail exposes Episode and feed diagnostics (help control -> tooltip) with
  paths, ids, Feed in vector index, and index stats when loaded; **E** and **`?`** sit
  in a vertical stack beside the episode title (**E** above **`?`**) to widen the title. **Similar episodes**
  loads automatically after episode detail succeeds and lists peer episodes; empty
  successful responses show a short no peers state; loading shows
  "Searching similar episodes..." (`aria-live="polite"`). **Prefill semantic search**
  opens Search with the Feed filter and the same field order as Similar episodes,
  with length caps so oversized title/bullets do not fill the query field.

---

## Accessibility

- Feed and episode columns are keyboard navigable (`Tab` / arrow patterns as
  implemented per RFC-067); each selectable row exposes a clear accessible name
  (episode title + feed).
- Topic pills are separate buttons with their own labels; they apply a topic filter
  and reload the list.
- Selected list row and primary actions use the same focus ring convention as the
  rest of the viewer (UXS-001 Key states).
- Summary bullets remain plain text list items; if any bullet is truncated, provide
  a visible or tooltip expansion only if it does not break contrast rules.

---

## Empty states

Use the same empty state language as the rest of the viewer (`muted`, short
instruction, e.g. set corpus path or confirm metadata exists).

---

## E2E contract

When the Library UI ships or changes, implementers must update the
[E2E surface map](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md)
first for any E2E-visible strings, roles, or hooks, then align Playwright specs per
the E2E Testing Guide.

---

## Revision history

| Date       | Change                                                         |
| ---------- | -------------------------------------------------------------- |
| 2026-04-10 | Initial content (in UXS-001)                                   |
| 2026-04-13 | Extracted from UXS-001 into standalone UXS-003                 |
