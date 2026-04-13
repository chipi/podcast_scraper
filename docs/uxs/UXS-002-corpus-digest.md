# UXS-002: Corpus Digest

- **Status**: Active
- **Authors**: Podcast Scraper Team
- **Parent UXS**: [UXS-001: GI/KG Viewer](UXS-001-gi-kg-viewer.md) -- shared tokens,
  typography, layout, states
- **Related PRDs**:
  - [PRD-023: Corpus Digest & Library Glance](../prd/PRD-023-corpus-digest-recap.md)
- **Related RFCs**:
  - [RFC-068: Corpus Digest API & Viewer](../rfc/RFC-068-corpus-digest-api-viewer.md)
  - [RFC-062: GI/KG viewer v2](../rfc/RFC-062-gi-kg-viewer-v2.md)
- **Implementation paths**:
  - `web/gi-kg-viewer/src/components/digest/DigestView.vue`
  - `web/gi-kg-viewer/src/utils/digestRowDisplay.ts`

---

## Summary

The **Digest** tab is the discovery surface for rolling-window recent episodes and
semantic topic bands. It is the default selected tab on first load. This UXS defines
the visual contract for the Digest tab layout, density, and component appearance.
All tokens reference [UXS-001](UXS-001-gi-kg-viewer.md).

---

## Information architecture

- **Main nav order:** Digest, Library, Graph, Dashboard (left to right).
  **Default selected tab:** Digest on first load.
- **Digest tab:** A main nav item labeled **Digest** (same control pattern as other
  main views). No duplicate in-column heading -- the nav label is the page title.
  Default time lens **7d** (behavioral default in RFC-068); user can switch
  **24h** / **7d** / **since** per RFC.
- **Toolbar:** One row -- rolling bounds (start -> end, row count, left when digest is
  loaded) and **Window** + **Open Library** (right); **Open Library** switches to the
  Library tab without clearing corpus path.

---

## Layout and density

- **Digest tab:** Single-column scroll on `canvas`; sections:

1. **Window** -- top toolbar row: `muted` rolling bounds (start -> end, row count)
   left (readable date + time, explicit UTC suffix, not raw ISO with fractional
   seconds; `<time datetime="...">` for machine-readable instants) when digest data
   is loaded; **Window** controls + **Open Library** right (`muted` labels, `surface`
   controls). When the digest is loading or unavailable, controls stay end-aligned
   on that row.

2. **Topic bands** -- outer `region` "Topic bands" with `max-height`
   (`min(50vh, 24rem)`) and vertical scroll for the whole grid (single scrollbar;
   per-topic lists do not scroll independently); inner responsive grid (`sm:2` /
   `xl:3` columns); compact card padding; per-topic hit row is one clickable control
   (opens the Episode rail while Digest stays selected); `summary_preview` under the
   title -- tight gap (`mt-0.5`), full wrap, no clamp; right column top to bottom:
   similarity score as a small mono pill with native tooltip, publish date + duration
   on one line when either is present, feed (truncated); hover on feed for RSS / id /
   description (same as Library episode rows). No per-hit Graph button -- use the
   topic title control for top-hit Graph or Open in graph on the Episode rail.
   **Search topic** prefills semantic search with the topic query and sets
   **Since (date)** to the digest window start calendar day. Selected hit row uses
   `bg-overlay` when its path matches the Episode rail.

3. **Recent (diverse)** -- `h3` "Recent" subtitle + `?` tooltip explaining
   diversification vs topic bands; `role="region"` `aria-label="Recent episodes"`;
   episode rows aligned with Library list: cover `h-9`, title + right column (feed
   left, publish date / E# / duration inline right on the same row), full-wrap
   `summary_preview` / recap; card click opens Episode rail (Digest remains the main
   tab). Selected row uses `bg-overlay` when that episode's path matches the Episode
   rail. Accessible name matches Library rows: episode title, feed.

---

## Actions and tokens

- **Recent episode cards:** Episode rail handoff (card click); Digest tab stays
  selected. **Open in graph** / **Prefill semantic search** follow
  [UXS-003](UXS-003-corpus-library.md) on the Episode rail, not on the Digest card.
- **Open GI / Open KG / Prefill semantic search (Episode rail):** Same domain and
  primary rules as [UXS-003](UXS-003-corpus-library.md) (GI/KG use `gi`/`kg`;
  search handoff uses `primary`).
- **Topic band:** **Search topic** uses `primary`; opens the Search panel with the
  topic query prefilled per RFC-068. **Topic title** opens the top hit in Graph;
  each hit row opens the Episode rail; **Open in graph** for a specific hit lives on
  the Episode rail.

---

## Digest and Episode rail interaction

Recent and topic band hit rows open the Episode rail on the right without switching
away from Digest (same detail as selecting an episode in Library), so Graph and
Prefill semantic search stay one click away on that rail. Switching Digest to Library
keeps the Episode rail selection when that episode is still in scope. Library no
longer embeds a New (24h) digest strip so the two tabs do not compete -- users open
Digest for "what's new," Library for catalog browse.

---

## Health discovery

- `GET /api/health` returns `corpus_digest_api` (RFC-068) and `corpus_library_api`
  (RFC-067) on current server builds.
- Digest tab is enabled when `corpus_digest_api` is true, or when it is omitted but
  `corpus_library_api` is true (viewer infers digest for health JSON from builds
  before the digest flag existed). If `corpus_digest_api` is explicitly false, the
  tab shows an upgrade message.
- If the API process is too old to mount `GET /api/corpus/digest`, the Digest tab
  fetch fails and shows a load error.

---

## Accessibility

- `aria-label` on main nav matches visible Digest text.
- Digest Recent rows use the same accessible name pattern as Library episode rows
  (episode title, feed display label). Topic band hit rows use the same pair plus a
  suffix (e.g. "-- topic band hit") so the name stays unique when the same episode
  appears under Recent and in a topic.

---

## E2E contract

Any new visible labels (e.g. Digest, Open Library, Search topic) require updates to
the [E2E surface map](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md)
before or with implementation. Dedicated Playwright coverage lives in
`e2e/digest.spec.ts` (mocked `GET /api/corpus/digest` + health).

---

## Revision history

| Date       | Change                                                         |
| ---------- | -------------------------------------------------------------- |
| 2026-04-10 | Initial content (in UXS-001)                                   |
| 2026-04-11 | Health discovery, glance gate, Search topic, E2E digest.spec   |
| 2026-04-11 | No in-column title; Window + Open Library toolbar row          |
| 2026-04-13 | Extracted from UXS-001 into standalone UXS-002                 |
