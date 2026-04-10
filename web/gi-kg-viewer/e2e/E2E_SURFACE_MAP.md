# GI / KG viewer — E2E surface map

This document is the **Playwright automation contract** for the GI/KG viewer (`web/gi-kg-viewer`).
It lists surfaces, entry paths, owning specs, and selectors tests rely on. It complements
[UXS-001](../../../docs/uxs/UXS-001-gi-kg-viewer.md) (visual and experience contract) and
[RFC-062](../../../docs/rfc/RFC-062-gi-kg-viewer-v2.md) (technical design); it does not replace them.

**Related:** [ADR-066](../../../docs/adr/ADR-066-playwright-for-ui-e2e-testing.md). Tracked in
[GitHub #509](https://github.com/chipi/podcast_scraper/issues/509).

## Runtime

| Item | Value |
| ---- | ----- |
| Config | [playwright.config.ts](../playwright.config.ts) |
| `baseURL` | `http://127.0.0.1:5174` |
| Dev server | Vite via Playwright `webServer` (dedicated port; avoids clashing with `npm run dev` on 5173) |
| Browser | Firefox (single project) |
| Specs | `e2e/*.spec.ts`, shared [fixtures.ts](fixtures.ts), [helpers.ts](helpers.ts) |

## Surfaces and owning specs

| Surface | Intent (short) | Typical entry | Spec files |
| ------- | -------------- | ------------- | ---------- |
| **Graph shell** | Graph canvas + toolbar: **Fit**, **Re-layout**, **Export PNG**, zoom **−** / **+** / **100%** (aria **Zoom out** / **Zoom in**), hint text (**Shift+dbl-click** / **Shift+drag** box zoom); **Sources** row on card (when applicable): merged **GI** / **KG**, **Hide ungrounded** (GI / both), **filters active**; **Layout** combobox (**Graph layout algorithm** → COSE / breadthfirst / circle / grid), **Degree** buckets, **Minimap**, **Clear degree filter**; **Edges** row; **Types** row (checkboxes + **all** / **none**, swatches, counts); optional minimap `aria-label` **Graph minimap**. **Selected node:** **Re-layout** and degree filter keep the selection anchored; **wheel** and **toolbar** zoom adjust pan so the selected node stays at the same on-screen position (incremental zoom steps); **Fit** reframes all visible elements and resets that anchor; wheel zoom-out does not auto-center while a node is selected. | After artifact load (healthy API: **Corpus path** auto **List**+merge all GI/KG; or manual **List** → **Load into graph**; offline: file picker) | `offline-graph.spec.ts`, `export-png.spec.ts`, `search-to-graph-mocks.spec.ts` (post-load), `keyboard-shortcuts.spec.ts` (**Esc** test only — after `loadGraphViaFilePicker`) |
| **Shell — semantic search** | **`/`** focuses `#search-q` when focus is not in an input | `goto('/')` with **`**/api/health`** fulfilled (200 JSON) — **no** graph load | `keyboard-shortcuts.spec.ts` (first test) |
| **Dashboard** | **Summary row** (when corpus path + healthy API): **`role="group"`** **`aria-label="Corpus summary counts"`** — **Feeds** / **Episodes** / **Topics** (digest bands from server config, `digest_topics_configured` on `GET /api/corpus/stats`) / **Insights** (GI JSON count from `GET /api/artifacts`); then **charts:** GI+KG **mtime** timeline, **publish month** histogram (`GET /api/corpus/stats`), **manifest** feed bars (Y labels from **`GET /api/corpus/feeds`** …) + **run duration** (`GET /api/corpus/runs/summary`), **cumulative growth** + **GI vs KG cumulative-by-day** (when under cap); **latest run** stage stack + **episode outcomes** doughnut; bottom row: node types + indexed doc types (snapshot); short **help** blurbs; external **tooltips**; blurb points to **API · Data** | **Dashboard** tab (often after graph load for bottom charts) | `dashboard.spec.ts` |
| **API · Data (left panel)** | **API** heading + elevated card: **Health** row (label + value), then **Yes/No** rows — **Artifacts (graph)**, **Semantic search**, **Graph explore**, **Index routes**, **Corpus metrics**, **Library API**, **Digest API**, **Binary (covers)** (`GET /api/health` flags); **Retry health**; offline **Choose files** when health fails; **Data** heading + blurb + **Corpus root:** / resolved text; two elevated cards — **Graph**, **Vector index** (#507, **Refresh** / **Update index** / **Full rebuild**, `GET /api/index/stats` + `POST /api/index/rebuild`) | Left rail **API · Data** tab | `dashboard.spec.ts` (overview headings); `dashboard-index-rebuild-mocks.spec.ts` (rebuild POST **202**) |
| **API · Data — offline load** | Load `.gi.json` via file picker when health fails | `loadGraphViaFilePicker` in [helpers.ts](helpers.ts) | `offline-graph.spec.ts`, `dashboard.spec.ts`, `export-png.spec.ts`, `keyboard-shortcuts.spec.ts` (**Esc** test) |
| **API panel — mocked corpus graph** | Healthy API + corpus path → auto `GET /api/artifacts` + fetch each GI/KG (merged graph); manual **List** / **Load into graph** still available | `goto('/')` + mocks in `beforeEach` | `search-to-graph-mocks.spec.ts` |
| **Corpus hints** | Banner when `GET /api/artifacts` returns `hints` | Mocked artifacts + **List** | `corpus-hints.spec.ts` |
| **Corpus Digest** | `GET /api/corpus/digest` + **`GET /api/corpus/feeds`**; digest **range** on **one line** (human-readable UTC bounds via `<time datetime="ISO">` · row count; not raw ISO microseconds); **feed name** + **publish date** share the **first** line of the episode card **right column** (feed **left**, date **right**); **E#** / **duration** below. **Topic band** cards: compact padding; each **hit row** is **`role="button"`** (same **`episode_title`, `feed`** pattern as **Recent**), **click** → **Library** episode panel; **right** meta column **top → bottom**: **publish date** (when API sends `publish_date` on topic hits), **duration**, **feed** (truncated, **native `title`** hover); **similarity score** pill at row end; **no** per-hit **Graph** button — **topic title** still opens **top** hit in **Graph**. **Truncated** feed + hover on **Recent** rows as above. **episode rows** — `h-9` cover, **full-wrap** recap; no **GI/KG** chips; **Open Library** (toolbar) | Default main tab on `goto('/')` | `digest.spec.ts` (mocked health + digest + **`feeds`** + optional `**/api/artifacts/metadata/...gi.json**`) |
| **Corpus Library** | **Feeds** sidebar: row **`title`** hover adds **RSS** + **description** when `GET /api/corpus/feeds` includes them; **Episodes** list — **right-column** meta aligned with Digest **Recent** (feed + date on **first** line, **E#** / duration below); **full-wrap** recap; **topic pills**; **Episode** panel heading; **Episode** panel meta: feed **left**, publish date **right** on first line; **Episode and feed diagnostics**; **Open in graph** / **Prefill semantic search**; **Indexed**; **Find similar episodes**; **Episode filters**: **`label` Published on or after** + **`#lib-filter-since-q`** on the **first** row, preset buttons (**All time** / **7d** / …) **below**. **No** embedded **24h digest** strip — use **Digest** tab for discovery. | **Library** tab + corpus path; mock corpus + optional `index/stats` / `similar` | `library.spec.ts` |
| **Theme tokens** | `--ps-canvas` matches asserted dark/light hex in [theme.spec.ts](theme.spec.ts) | `goto('/')` + `emulateMedia` and/or `localStorage` | `theme.spec.ts` |

### Offline graph load (shared helper)

[`loadGraphViaFilePicker`](helpers.ts):

1. Route `**/api/health` → **abort** (`failed`).
2. `goto('/')`, wait for heading **Podcast Intelligence Platform** (`SHELL_HEADING_RE` in [helpers.ts](helpers.ts)).
3. Click **Graph** (default app tab is **Digest**; graph canvas lives on **Graph**).
4. Left nav **API · Data** (`leftPanelTabs` in [helpers.ts](helpers.ts)), wait for **Choose files** (`/Choose files/i`).
5. Set files on `input[type="file"]` using [GI_SAMPLE_FIXTURE](fixtures.ts) (`e2e/fixtures/ci_sample.gi.json`).
6. Wait for **Fit** visible.

### Mocked API corpus path → graph → search (`search-to-graph-mocks.spec.ts`)

- Fulfill `**/api/health`, `**/api/artifacts?**`, per-file `GET /api/artifacts/{relPath}?**`, and `**/api/search?**` as in that spec. Artifact list JSON must include **`mtime_utc`** on each item (server schema; #507).
- If a spec drives **API · Data** index UI, also fulfill `**/api/index/stats?**` (200 JSON with `available`, `reindex_recommended`, `rebuild_in_progress`, etc.) and, when testing rebuild, `**/api/index/rebuild?**` POST → **202** (optional follow-up spec).
- Fill corpus placeholder (triggers auto **List** + load of all GI/KG when health is ok); open **Graph** tab, wait for **Fit** (no manual **List** / **Load into graph** required for the happy path).
- Fill `#search-q`, form **Search**, wait for stub result text, click `article` (first card), assert **Fit** still visible and `.graph-canvas` visible.

### Index rebuild mocks (`dashboard-index-rebuild-mocks.spec.ts`)

- Fulfill `**/api/health` (200), `**/api/index/stats**` (200 JSON with `available: true`, `rebuild_in_progress: false`, minimal `stats`).
- Fulfill `**/api/index/rebuild**` on **POST** with **202** + `IndexRebuildAccepted` JSON.
- `goto('/')` → left **API · Data** → assert **Update index** / **Full rebuild** enabled → click → `waitForRequest` on POST; incremental must not set `rebuild=true`; full rebuild must include `rebuild=true` in query.

### Keyboard shortcuts (`keyboard-shortcuts.spec.ts`)

Two isolated setups:

1. **Search focus:** Fulfill `**/api/health` → `goto('/')` → wait for `#search-q` enabled → click `body` (blur) → `keyboard.press('/')` → expect `#search-q` focused.
2. **Esc on graph:** `loadGraphViaFilePicker` → click `.graph-canvas` → `keyboard.press('Escape')` → expect **Fit** visible.

### Theme tokens (`theme.spec.ts`)

- **Dark:** `emulateMedia({ colorScheme: 'dark' })`, `goto('/')`, expect `--ps-canvas` trimmed lowercase **`#111418`**.
- **Light:** `emulateMedia({ colorScheme: 'light' })`, init script sets `localStorage` **`gi-kg-viewer-theme`** = **`light`**, `goto('/')`, expect **`#f6f7f9`**.

## Stable selectors and hooks (contract)

Prefer updating this section when Playwright assertions change.

**Roles / accessible names**

- `heading` **Podcast Intelligence Platform** (`SHELL_HEADING_RE` in [helpers.ts](helpers.ts)) — shell ready (v2 in child span).
- `navigation` **Main views** — header tabs (**Digest**, **Library**, **Graph**, **Dashboard**); scope tab clicks here so Playwright does not match **Open Library** or **Load into graph** (`mainViewsNav` in [helpers.ts](helpers.ts)).
- `navigation` **Left panel tabs** — **Corpus** vs **API · Data** (`leftPanelTabs` in [helpers.ts](helpers.ts)).
- `button` **Digest**, **Library**, **Graph**, **Dashboard**, **API · Data**, **Fit**, **Re-layout**, **Export PNG**, **100%**, **List**, **Load into graph**, **Search** (scoped: `locator('form').getByRole('button', { name: 'Search' })`).
- `button` **Zoom out**, **Zoom in** — `aria-label` on **−** / **+** zoom controls (visible label is the glyph only).
- `combobox` **Graph layout algorithm** — layout algorithm for **Re-layout** / initial layout.
- `checkbox` **Minimap** — toggles bird’s-eye navigator (`aria-label` **Graph minimap** on container when shown); container sits **bottom-left** inside the graph host, compact size.
- `button` **Clear degree filter** — visible only when a degree bucket filter is active.
- `button` matching degree bucket label, e.g. **`0 (3)`** — `aria-pressed` reflects active filter; bucket ids **`0`**, **`1`**, **`2-5`**, **`6-10`**, **`11+`**.
- `button` **all** (under **Edges**) — re-enables all edge-type toggles (substring-safe: scope under graph chrome).
- `button` **all** / **none** (under **Types** on the graph card) — node-type visibility; scope under graph chrome to avoid clashing with **Edges**.
- Visible text **Sources** (graph card chrome) — merged **GI** / **KG** checkboxes, **Hide ungrounded**; **filters active** when any graph filter applies.
- Visible hint substring **`Shift+drag canvas: box zoom`** or **`Shift+dbl-click`** — interaction help next to zoom controls.
- Search form label **Feed id (substring)** — `GET /api/search` feed filter; Library handoff fills catalog `feed_id` (substring match).
- `button` **Refresh**, **Update index**, **Full rebuild** — **API · Data** → **Data** section → vector index card (#507).
- `button` **Choose files** — `/Choose files/i`.
- `checkbox` name matching `/ci_sample\.gi\.json/`.
- `heading` **Data** (`<h2>`, exact) — left panel **Data** section title (same elevated-card pattern as **API**; **Graph** + **Vector index** inside).
- `heading` **Graph** (`<h3>`), **Vector index** (`<h3>`) — under **API · Data** (left panel; **Graph** is the metrics card title from `MetricsPanel`).
- Placeholder **`/path/to/output`** — corpus root field.
- Visible text **Corpus path hint**, **Unified semantic index** (substring) — corpus hints spec.
- Visible text **Summary insight (stub)** (non-exact) — mocked search result card.
- Visible text **Reindex recommended**, **Search / corpus note**, **Background index job running**, **Last rebuild error** (substring) — **API · Data** index health / rebuild (#507); exact copy may evolve.

**IDs and DOM hooks (semi-stable)**

- `#search-q` — semantic search query; fill + mocked search; **`/`** shortcut focus (after `body` click to blur).
- **`data-testid="digest-root"`** — Corpus Digest main column; visible on **Digest** tab (default).
- `button` matching **`Open top hit for topic`** … **`in graph`** — Digest topic band title → **Graph** tab + load + focus (top semantic hit).
- Topic band **hit rows** — **`role="button"`**, **`aria-label`** **`episode_title`, `feed` — topic band hit** (distinct from **Recent** when the same episode appears twice); **click** → **Library** tab + episode panel (no separate **Graph** control on the row).
- **`data-testid="library-root"`** — Corpus Library main column; visible on **Library** tab.
- **`data-testid="library-similar"`** — Similar-episodes region (episode panel column).
- **`data-testid="podcast-cover"`** — Optional artwork tile (Library feeds/episodes/detail, Digest cards/topic rows, similar list); shows ♪ placeholder when no URL or image error.
- `region` **Feeds**, **Episodes**, **Episode** — Library layout (ARIA `role="region"` + `aria-label` on the right-hand episode panel). In Playwright, use **`exact: true`** for **`role="region", name: "Episode"`** so it does not substring-match **Episodes**.
- `button` **Episode filters** (collapsible), **All time** / **7d** / **30d** / **90d** (since presets), **Apply**, **Clear text & date**, **All feeds**, **Load more**, **Open in graph**, **Prefill semantic search**, **Find similar episodes** — Library.
- **Strict names (avoid substring clashes):** Digest **Recent** episode card uses **`episode_title`, `feed`** (same pattern as Library episode rows), e.g. **`Digest Episode Alpha, Mock Feed Show`**. When the same episode also appears under **Topic bands**, Playwright needs **`exact: true`** on the **Recent** name (topic hit name is the same pair plus **`— topic band hit`**). Library feed row vs episode row: prefer **`Mock Show, feed id f1, 1 episodes`** vs **`Mock Episode Title, Mock Feed Show`** (episode name includes show), not a bare `/Mock Show/` regex.
- Visible text **Indexed** (feed chip); **Episode** panel — **first** meta line: feed **left**, publish date **right** (same **native `title`** hover as list rows: RSS / feed id / description); **E#** / duration on a **second** line when present; heading **Key points** (`h4`) when bullets follow summary title/text, with **`border-t`** separator; `button` **Episode and feed diagnostics** opens **role="tooltip"** with troubleshooting rows (**Feed in vector index**, metadata/GI/KG paths, ids, index stats when loaded) — Library Phase 3 (RFC-067).
- Status **From Library: query uses episode summary…** — Search panel after Library **Prefill semantic search** handoff (query text prefers **`summary_text`** when set; else title + bullets per `buildLibrarySearchHandoffQuery`).
- **`data-testid="library-similar-empty"`** — Similar-episodes ran successfully but returned no peer rows.
- `locator('body')` — click at a small offset to move focus away from the search field before **`/`** ([keyboard-shortcuts.spec.ts](keyboard-shortcuts.spec.ts)).
- `locator('article').first()` — first search result card in mocked search flow (click after **Search**).
- `.graph-canvas` — graph container; click before **Esc** test; visibility after mocked search path; changing the class requires updating specs and this map.

**Network routes (tests differ by intent)**

- **`**/api/health` — abort** (`failed`) — `loadGraphViaFilePicker` (offline file picker path).
- **`**/api/health` — fulfill** 200 + `{ status: 'ok', corpus_library_api: true, corpus_digest_api: true }` (extra health booleans optional; mocks may omit **Artifacts** / **search** / … flags → UI treats as **Yes**). Used by `keyboard-shortcuts`, `corpus-hints`, `search-to-graph-mocks`, `dashboard-index-rebuild-mocks`, `library.spec.ts`, `digest.spec.ts` (**Digest** needs **`corpus_digest_api`** or upgrade message; **Library** no longer calls digest).
- Library tab mocks (`library.spec.ts`): `**/api/corpus/feeds`, `**/api/corpus/episodes`, `**/api/corpus/episodes/detail`, `**/api/index/stats`, `**/api/corpus/episodes/similar` (when testing Phase 3 UI).
- **Dashboard** corpus charts (when online + corpus path): `**/api/corpus/stats?**`, `**/api/corpus/runs/summary?**`, `**/api/corpus/feeds?**` (for manifest bar labels; 404 OK if ignored), optional `**/api/corpus/documents/manifest?**` (404 OK); `**/api/artifacts?**` for GI+KG mtime timeline.
- **`**/api/index/stats`** (optional) — Dashboard index metrics + staleness + `rebuild_in_progress` / `rebuild_last_error` (#507). **`**/api/index/rebuild`** POST (optional) — background index job; respond **202** for happy path tests.

**Theme**

- `localStorage` key **`gi-kg-viewer-theme`** — set to **`light`** in light test (`addInitScript` before `goto`).
- CSS variable **`--ps-canvas`** on `document.documentElement` — asserted **`#111418`** (dark) / **`#f6f7f9`** (light); keep in sync with [theme.spec.ts](theme.spec.ts) and UXS-001.

## Maintenance

Use this order for **viewer UX** work (humans and agents); details also live in
[E2E Testing Guide — When you change viewer UX](../../../docs/guides/E2E_TESTING_GUIDE.md#when-you-change-viewer-ux-required-workflow).

1. **This file (`E2E_SURFACE_MAP.md`)** — When **user-visible labels**, **routes**, **E2E entry flows**,
   or **selectors** in specs change, update the map in the **same PR** (usually **before** or
   alongside test edits so the contract stays obvious).
2. **Playwright** — Update `e2e/*.spec.ts`, `helpers.ts`, or `fixtures.ts`; run **`make test-ui-e2e`**.
3. **UXS** — Update [UXS-001](../../../docs/uxs/UXS-001-gi-kg-viewer.md) when the **visual or experience
   contract** (tokens, density, documented patterns) changes, not only when tests fail.

- **Reviewers:** if a PR changes Playwright selectors or primary control labels, confirm
  `E2E_SURFACE_MAP.md` was updated when applicable.
- If copy is expected to churn often, consider adding `data-testid` in Vue and documenting it
  here (follow-up; not required for v1).

## Commands

From repo root: `make test-ui-e2e`. In package: `npm run test:e2e` (under `web/gi-kg-viewer`).
