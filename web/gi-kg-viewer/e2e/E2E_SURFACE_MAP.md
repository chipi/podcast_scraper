# GI / KG viewer — E2E surface map

This document is the **Playwright automation contract** for the GI/KG viewer (`web/gi-kg-viewer`).
It lists surfaces, entry paths, owning specs, and selectors tests rely on. It complements
the UXS specs -- [UXS-001](../../../docs/uxs/UXS-001-gi-kg-viewer.md) (shared design system)
plus feature UXS files ([UXS-002](../../../docs/uxs/UXS-002-corpus-digest.md) Digest,
[UXS-003](../../../docs/uxs/UXS-003-corpus-library.md) Library,
[UXS-004](../../../docs/uxs/UXS-004-graph-exploration.md) Graph,
[UXS-005](../../../docs/uxs/UXS-005-semantic-search.md) Search,
[UXS-006](../../../docs/uxs/UXS-006-dashboard.md) Dashboard) -- and
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
| **Graph shell** | Graph: **top toolbar** hint (**Shift+dbl-click** / **Shift+drag** box zoom); **bottom-right** canvas overlay **Fit** + zoom **−** / **+** / **100%** + **Export PNG** (rightmost; `role="toolbar"` **Graph fit, zoom, and export**; aria **Zoom out** / **Zoom in** on **−** / **+**); **upper-right** canvas overlay **`role="region"`** **Graph layout, re-layout, and degree filter** (`.graph-layout-controls`) — **Re-layout**, **Layout** combobox (**Graph layout algorithm** → COSE / breadthfirst / circle / grid), **Degree** buckets, **Clear degree filter**; **Sources** row on card (when applicable): merged **GI** / **KG**, **Hide ungrounded** (GI / both), **filters active**; **Minimap** checkbox row; **Edges** row; **Types** row (checkboxes + **all** / **none**, swatches, counts); minimap container `aria-label` **Graph minimap**. **Selected node:** **Re-layout** and degree filter keep the selection anchored; **wheel** and **Graph fit, zoom, and export** toolbar adjust pan so the selected node stays at the same on-screen position (incremental zoom steps); **Fit** reframes all visible elements and resets that anchor; wheel zoom-out does not auto-center while a node is selected. | After artifact load (healthy API: **Corpus path** auto **List**+merge all GI/KG; or manual **List** → **Load into graph**; offline: file picker) | `offline-graph.spec.ts`, `export-png.spec.ts`, `search-to-graph-mocks.spec.ts` (post-load), `keyboard-shortcuts.spec.ts` (**Esc** test only — after `loadGraphViaFilePicker`) |
| **Shell — semantic search** | **`/`** focuses `#search-q` when focus is not in an input | `goto('/')` with **`**/api/health`** fulfilled (200 JSON) — **no** graph load | `keyboard-shortcuts.spec.ts` (first test) |
| **Dashboard** | **Summary row** (when corpus path + healthy API): **`role="group"`** **`aria-label="Corpus summary counts"`** — **Feeds** / **Episodes** / **Topics** (digest bands from server config, `digest_topics_configured` on `GET /api/corpus/stats`) / **Insights** (GI JSON count from `GET /api/artifacts`); then **`role="tablist"`** **`aria-label="Dashboard sections"`** — **Pipeline** vs **Content intelligence** (`role="tab"`); **Pipeline:** **manifest** feed bars + **run duration** (`GET /api/corpus/runs/summary`), **cumulative growth** from `run.json`, **latest run** stage bars + **episode outcomes** horizontal bars; **Content intelligence:** **`role="region"`** **`aria-label="Vector index and digest glance"`** (heading **Vector index and digest glance**; `GET /api/index/stats` freshness + status + footprint **dl** + **Feeds in index vs catalog** bars + compact **`GET /api/corpus/digest`** one-liner when digest API on), then GI+KG **mtime** timeline, **publish month** histogram (insight when **catalog episode count** ≠ histogram sum), **GI vs KG cumulative-by-day** (when under cap), node types + indexed doc types (**bar-end** count + % of vectors when total known); short **help** blurbs; Chart.js **tooltips** (in-chart, not `document.body`); blurb points to **API · Data** | **Dashboard** tab (often after graph load for content-intelligence charts) | `dashboard.spec.ts` |
| **API · Data (left panel)** | **API** heading + elevated card: **Health** row (label + value), then **Yes/No** rows — **Artifacts (graph)**, **Semantic search**, **Graph explore**, **Index routes**, **Corpus metrics**, **Library API**, **Digest API**, **Binary (covers)** (`GET /api/health` flags); **Retry health**; offline **Choose files** when health fails; **Data** heading + blurb; elevated cards — **Corpus root** (**Path** / **Resolved**), **Corpus catalog** (`GET /api/corpus/stats` snapshot + **Refresh**; optional **GI/KG paths listed** when artifacts list loaded), **Graph** (merged GI/KG **MetricsPanel** rows + **Refresh** = re-list artifacts + reload selection), **Vector index** (#507, **Refresh** / **Update index** / **Full rebuild**, `GET /api/index/stats` + `POST /api/index/rebuild`) | Left rail **API · Data** tab | `dashboard.spec.ts` (overview headings); `dashboard-index-rebuild-mocks.spec.ts` (rebuild POST **202**) |
| **API · Data — offline load** | Load `.gi.json` via file picker when health fails | `loadGraphViaFilePicker` in [helpers.ts](helpers.ts) | `offline-graph.spec.ts`, `dashboard.spec.ts`, `export-png.spec.ts`, `keyboard-shortcuts.spec.ts` (**Esc** test) |
| **API panel — mocked corpus graph** | Healthy API + corpus path → auto `GET /api/artifacts` + fetch each GI/KG (merged graph); manual **List** / **Load into graph** still available | `goto('/')` + mocks in `beforeEach` | `search-to-graph-mocks.spec.ts` |
| **Corpus hints** | Banner when `GET /api/artifacts` returns `hints` | Mocked artifacts + **List** | `corpus-hints.spec.ts` |
| **Corpus Digest** | **No** in-column page title — **Digest** main nav tab is the panel title (parity with **Library**). Episode row / topic hit → **Episode** rail without leaving **Digest**; **Digest** ↔ **Library** retains rail selection when the episode is still in the digest payload and still listed under Library for current filters (see UXS). **Topic band** hit rows and **Recent** cards use **`bg-overlay`** on the clickable row when its `metadata_relative_path` matches the **Episode** rail (same visual selection cue as **Library**). Top toolbar row: rolling **range** **left** when loaded (human-readable UTC via `<time datetime="ISO">` · **N** **episodes** / **1 episode**; not raw ISO microseconds); **Window** + **Open Library** **right** (controls **end** when range hidden). `GET /api/corpus/digest` + **`GET /api/corpus/feeds`**; **`region` `Topic bands`** wraps the topic **grid** with **`max-h`** ~ **`min(50vh,24rem)`** + **`overflow-y-auto`** (one outer scroll, not per-card lists); bottom **`region` `Recent episodes`**: **`h3` Recent** + **`?`** (**About the Recent digest list**); episode card **right column**: **feed** (truncate, **left**) and **publish date**, **E#**, **duration** on **one** inline row (**right**). **Topic band** cards: compact padding; each **hit row** is **`role="button"`** (same **`episode_title`, `feed`** pattern as **Recent**), **click** → **Episode** rail while staying on **Digest** (same detail as Library selection); **right** meta column **top → bottom**: **similarity score** pill (when present), **publish date** + **duration** on **one** line (when either present), **feed** (truncated, **native `title`** hover); **no** per-hit **Graph** button — **topic title** still opens **top** hit in **Graph**; **Search topic** opens Search with the topic **`query`** and **Since (date)** prefilled from digest **`window_start_utc`** (UTC calendar day, same starting point as digest topic search). **Truncated** feed + hover on **Recent** rows as above. **episode rows** — `h-9` cover, **full-wrap** recap; no **GI/KG** chips | Default main tab on `goto('/')` | `digest.spec.ts` (mocked health + digest + **`feeds`** + optional `**/api/artifacts/metadata/...gi.json**`) |
| **Corpus Library** | **Feeds** sidebar: row **`title`** hover adds **RSS** + **description** when `GET /api/corpus/feeds` includes them; **Episodes** list (scroll container) — **cursor pagination** (`GET /api/corpus/episodes` `limit` ≈ **20** + `cursor`): **Load more** plus **scroll-to-load** when the sentinel nears the bottom of the list (same requests as **Load more**); per-row **right-column** meta aligned with Digest **Recent** (**feed** left; **publish date**, **E#**, **duration** inline **right** on the **same** row); **full-wrap** recap; **topic pills**. **Episode detail** (title, stacked feed + date meta, prose + key points, **Open in graph**, **Prefill semantic search**, **Indexed**, **Similar episodes**, diagnostics) lives in the **right shell rail** **`role="region"`** **`name: Episode`** **`exact: true`** — **mutually exclusive** with **Search & Explore** in that rail; selecting a Library episode opens the **Episode** rail. **Episode filters**: **`label` Published on or after** + **`#lib-filter-since-q`** on the **first** row, preset buttons (**All time** / **7d** / …) **below**. **No** embedded **24h digest** strip — use **Digest** tab for discovery. | **Library** tab + corpus path; mock corpus + optional `index/stats` / `similar` | `library.spec.ts` |
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
- Fill `#search-q`, **Search** in the **Semantic search** `section` (`locator('section').filter({ has: heading Semantic search }).getByRole('button', { name: 'Search', exact: true })` — avoids the **Search** right-panel tab), wait for stub result text; optional **`getByRole('button', { name: 'Search result insights' })`** → **`dialog` Search result insights** with **`region` Doc types** / **`region` Publish month**, then **Close**; then **`getByRole('button', { name: 'Show on graph' })`**, assert **Fit** still visible and `.graph-canvas` visible.

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
- `button` **Digest**, **Library**, **Graph**, **Dashboard**, **API · Data**, **Fit**, **Re-layout**, **Export PNG**, **100%**, **List**, **Load into graph**, **Search** (right-panel tab — same accessible name as semantic **Search** submit; scope semantic submit under `section` **Semantic search** + `exact: true`, or avoid tab by context).
- `toolbar` **Graph fit, zoom, and export** — bottom-right on graph canvas host; contains **Fit** (primary), **−** / **+** / **100%**, **Export PNG** (rightmost).
- `button` **Zoom out**, **Zoom in** — `aria-label` on **−** / **+** (visible label is the glyph only; inside that toolbar).
- **Graph tab + Episode rail** — Switching the main tab to **Graph** while **`region` `Episode`** is open (Library or Digest selection) applies the same **library highlight** (search-hit ring) and **pending focus** as **Open in graph**: the **Episode** node whose ``metadata_relative_path`` matches the rail is centered with a modest zoom when it exists in the **merged filtered** graph (no-op if the episode is not in the current load or **Episode** type is filtered off).
- `region` **Graph layout, re-layout, and degree filter** — upper-right on graph canvas host (`.graph-layout-controls`); contains **Re-layout**, **Layout** + **Degree** + **Clear degree filter**. **Double-tap** a node (without Shift): **Episode** nodes open the same **Episode** rail as Library (corpus detail + similar episodes) when metadata path resolves (node properties, loaded ``.gi.json``/``.kg.json`` stem → ``.metadata.json``, or corpus episode list by ``episode_id``); **`region` `Graph neighborhood and connections`** (**`data-testid="graph-connections-section"`**) includes a read-only **Local neighborhood** Cytoscape preview (**`data-testid="graph-neighborhood-mini"`**) — **1-hop ego** subgraph around the selected node (same as ``filterArtifactEgoOneHop``), then **Connections** with per-row **`G`**. Shown **only while the main tab is Graph** for episode context; **focus** neighbor via ``requestFocusNode`` + switch to **Graph**. Other node types open **graph node detail** in that rail (type-colored avatar tile + **Connections**). Optional visible line **Appears in (bridge):** (RFC-072) when sibling ``.bridge.json`` was loaded with the corpus selection (auto-fetched next to the selected ``.gi.json`` when the file exists). **Graph node id** / **Episode id**: **`E`** chip uses graph legend **Episode** hex (**`searchResultActionStyles`**, blue fill + white glyph, native ``title`` tooltip); **`?`** **`Graph node diagnostics`** sits after **`E`** on a high-contrast neutral chip (near-black glyph, light fill; dark theme inverted); property list omits redundant **`episode_id`** on **Insight** / **Quote** (and other non-**Episode** nodes). No inline ``g:…`` id. Graph node rail **embed** has no **✕** (use **Search & Explore**). **Single tap** on a node only updates graph selection — it does **not** change the right rail. **Tap** empty canvas still clears selection and returns the rail to Search & Explore.
- `combobox` **Graph layout algorithm** — inside that region (canvas overlay), not the top toolbar.
- `checkbox` **Minimap** — on graph **card** chrome below **Sources** (when shown); toggles bird’s-eye navigator (`aria-label` **Graph minimap** on container when shown); minimap sits **bottom-left** inside the graph canvas host (inset panel ~10.5rem × 7.5rem, capped by viewport fraction), not a floating viewport-fixed tile.
- `button` **Clear degree filter** — `aria-label` **Clear degree filter** (visible label often **Clear** only); shown when a degree bucket filter is active.
- `button` matching degree bucket label, e.g. **`0 (3)`** — `aria-pressed` reflects active filter; bucket ids **`0`**, **`1`**, **`2-5`**, **`6-10`**, **`11+`**.
- `button` **all** (under **Edges**) — re-enables all edge-type toggles (substring-safe: scope under graph chrome).
- `button` **all** / **none** (under **Types** on the graph card) — node-type visibility; scope under graph chrome to avoid clashing with **Edges**.
- Visible text **Sources** (graph card chrome) — merged **GI** / **KG** checkboxes, **Hide ungrounded**; **filters active** when any graph filter applies.
- Visible hint substring **`Shift+drag canvas: box zoom`** or **`Shift+dbl-click`** — interaction help on the **top** graph toolbar row (not in the bottom-right zoom cluster).
- Search: **`#semantic-search-form`** — **`#search-q`** (no visible **Query** label; `aria-label` **Search query**), then **Since (date)** and **Top‑k** (`#search-top-k`) on **one** compact row; **Advanced search** (`button`); **`region` Active advanced filters** (read-only summary when any advanced filter is non-default; non-default **Feed** appears as **`Feed: …`** not “Feed id”); **Search** / **Clear** below (submit uses `form="semantic-search-form"`). **Advanced search** opens **`dialog` Advanced search** — **Grounded insights only**, **Feed** (`#search-advanced-feed`, substring on catalog `feed_id` for `GET /api/search`; **Library prefill** shows **catalog display title** when known from `GET /api/corpus/feeds`, with native **`title`** hint for the id until the user edits), **Speaker**, **Embedding model**, **Doc types**; **Close** dismisses the dialog. When results exist: optional visible substring **`Lift:`** (ratio **linked / transcript** when `lift_stats.transcript_hits_returned` > 0 from `GET /api/search`); per **transcript** hit card, optional collapsible **`region` Lifted GI insight** (`aria-label` **Lifted GI insight**) when the hit JSON includes **`lifted`**; **`button` Search result insights** opens **`dialog` Search result insights** (`h2` + hit count + dominant-type insight line); **`region` Doc types** and **`region` Publish month** (side-by-side on wide viewports); **`region` Episodes** / **Feeds** / **Similarity scores** / **Terms**; **Close** dismisses.
- `button` **Refresh** — **API · Data** → **Data** → **Corpus catalog** card, **Graph** card (re-list + reload GI/KG), and **Vector index** card (#507); **Update index** / **Full rebuild** remain on **Vector index** only.
- `button` **Choose files** — `/Choose files/i`.
- `checkbox` name matching `/ci_sample\.gi\.json/`.
- `heading` **Data** (`<h2>`, exact) — left panel **Data** section title (same elevated-card pattern as **API**; **Corpus root**, **Corpus catalog**, **Graph**, **Vector index** cards).
- `heading` **Corpus root**, **Corpus catalog**, **Graph**, **Vector index** (`<h3>`) — under **API · Data** → **Data** (left panel); **Graph** / **Corpus catalog** card headers sit above metric rows (inner **Graph** title from `MetricsPanel` is hidden — card **h3** is the section label).
- `tablist` **Dashboard sections** — **Dashboard** tab: **Pipeline** / **Content intelligence** (`tab`, `aria-selected`).
- Placeholder **`/path/to/output`** — corpus root field.
- Visible text **Corpus path hint**, **Unified semantic index** (substring) — corpus hints spec.
- Visible text **Summary insight (stub)** (non-exact) — mocked search result card.
- Visible text **Reindex recommended**, **Search / corpus note**, **Background index job running**, **Last rebuild error** (substring) — **API · Data** index health / rebuild (#507); exact copy may evolve.

**IDs and DOM hooks (semi-stable)**

- `#search-q` — semantic search query; fill + mocked search; **`/`** shortcut focus (after `body` click to blur).
- `#semantic-search-form` — wraps **`#search-q`** + **Since (date)** + **`#search-top-k`**; **Search** submit button may be a sibling with `form="semantic-search-form"`.
- **Advanced search** modal — checkbox **Merge duplicate KG surfaces (kg_entity / kg_topic)** (default checked); when unchecked, search requests include **`dedupe_kg_surfaces=false`**.
- **`data-testid="digest-root"`** — Corpus Digest main column; visible on **Digest** tab (default).
- `region` **Topic bands** — Digest topic **grid**; bounded height + **vertical scroll** when content exceeds cap.
- `region` **Recent episodes** — Digest **Recent** list (below topic bands); **`?`** **About the Recent digest list** explains diversification vs topic bands.
- `button` matching **`Open top hit for topic`** … **`in graph`** — Digest topic band title → **Graph** tab + load + focus (top semantic hit).
- Topic band **hit rows** — **`role="button"`**, **`aria-label`** **`episode_title`, `feed` — topic band hit** (distinct from **Recent** when the same episode appears twice); **click** → **Episode** rail, **Digest** tab stays selected (no separate **Graph** control on the row).
- **`data-testid="library-root"`** — Corpus Library main column; visible on **Library** tab.
- **`data-testid="library-similar"`** — Similar-episodes region inside the **Episode** rail; **`?`** **`About similar episodes`** — tooltip with embedding explanation + **`query_used`** when present.
- **`data-testid="podcast-cover"`** — Optional artwork tile (Library feeds/episodes/detail, Digest cards/topic rows, similar list); shows ♪ placeholder when no URL or image error.
- `region` **Feeds**, **Episodes** — Library main column. **`region` `Episode`** — **shell right rail** (detail when an episode is selected or opened from Digest); **mutually exclusive** with Search/Explore in that rail. In Playwright, use **`exact: true`** for **`role="region", name: "Episode"`** so it does not substring-match **Episodes**.
- `button` **Episode filters** (collapsible), **All time** / **7d** / **30d** / **90d** (since presets), **Apply**, **Clear text & date**, **All feeds**, **Load more**, **Open in graph**, **Prefill semantic search** — Library.
- **`button` Back** (tools rail header) — visible when **Search** / **Explore** is active and there is stashed detail context: **`Back to episode`** if **`metadataRelativePath`** is set (no stashed graph node id), else **`Back to details`** when a graph node id was kept opening tools from **Graph node** rail (restores **`region` `Graph node: …`**). **`button` Search & Explore** — from **Episode** or **Graph node** rail header, switches to tools. Collapsed right rail: vertical **Details** expands the sidebar when **`paneKind === 'graph-node'`** (same as before); vertical back label matches the same **Back to episode** / **Back to details** logic when **`paneKind === 'tools'`** with stash.
- **Strict names (avoid substring clashes):** Digest **Recent** episode card uses **`episode_title`, `feed`** (same pattern as Library episode rows), e.g. **`Digest Episode Alpha, Mock Feed Show`**. When the same episode also appears under **Topic bands**, Playwright needs **`exact: true`** on the **Recent** name (topic hit name is the same pair plus **`— topic band hit`**). Library feed row vs episode row: prefer **`Mock Show, feed id f1, 1 episodes`** vs **`Mock Episode Title, Mock Feed Show`** (episode name includes show), not a bare `/Mock Show/` regex.
- Visible text **Indexed** (feed chip); **Episode** rail — meta block: **feed** on the **first** line (**full width**, **wrap**, same **native `title`** hover as list rows: RSS / feed id / description); **publish date**, **E#**, duration on the **second** line below, **left**-aligned (list-scale `muted`); heading **Key points** (`h4`) when bullets follow summary title/text, with **`border-t`** separator; `button` **Episode and feed diagnostics** (visible **`?`**, neutral high-contrast chip beside legend-colored **`E`**) opens **role="tooltip"** with troubleshooting rows (**Feed in vector index**, metadata/GI/KG paths, ids, index stats when loaded) — Library Phase 3 (RFC-067).
- **Prefill semantic search** (Library) / **Search topic** (Digest) open Search with query (and feed / **Since (date)** for Digest from **`window_start_utc`**) already filled; no separate handoff banner. Library prefill uses the same field order as **Similar episodes** (`build_similarity_query`) with **client-side length caps** (long recap in `summary.title` or one giant bullet does not fill the query box).
- **`data-testid="library-similar-empty"`** — Similar-episodes ran successfully but returned no peer rows.
- `locator('body')` — click at a small offset to move focus away from the search field before **`/`** ([keyboard-shortcuts.spec.ts](keyboard-shortcuts.spec.ts)).
- `button` **Show on graph** — search result row (**G**, GI token); mocked search → graph flow ([search-to-graph-mocks.spec.ts](search-to-graph-mocks.spec.ts)). **`Open episode in Library`** (**L**) when hit metadata includes **`source_metadata_relative_path`**, corpus path is set, health is OK, and the row is **not** a merged KG surface (**`kg_surface_match_count` ≤ 1** or absent). **`Episode summary in right panel`** (**S**) — same gating as **L**; opens the **Episode** rail without switching main tab to **Library** (focus/hit navigation does **not** auto-open episode mode). **E** hidden under the same merged condition; **G** remains with an explanatory tooltip.
- `.graph-canvas` — graph container; click before **Esc** test; visibility after mocked search path; changing the class requires updating specs and this map.
- `.graph-zoom-controls` — bottom-right **Fit** + zoom + **Export PNG** strip on the graph canvas host (optional scope vs page-wide **Zoom out** / **Zoom in** if names ever clash).
- `.graph-layout-controls` — upper-right **narrow (~6.75rem) vertical** strip: **Re-layout**, stacked **Layout** label + combobox, **Degree** in a **2×** grid, **Clear** (degree).

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
3. **UXS** — Update the relevant feature UXS file
   ([UXS-002](../../../docs/uxs/UXS-002-corpus-digest.md) through
   [UXS-006](../../../docs/uxs/UXS-006-dashboard.md)) when the **visual or experience contract**
   (layout, density, documented patterns) changes; update
   [UXS-001](../../../docs/uxs/UXS-001-gi-kg-viewer.md) only when **shared** tokens or design
   system primitives change. See [UXS index](../../../docs/uxs/index.md) for the full list.

- **Reviewers:** if a PR changes Playwright selectors or primary control labels, confirm
  `E2E_SURFACE_MAP.md` was updated when applicable.
- If copy is expected to churn often, consider adding `data-testid` in Vue and documenting it
  here (follow-up; not required for v1).

## Commands

From repo root: `make test-ui-e2e`. In package: `npm run test:e2e` (under `web/gi-kg-viewer`).
