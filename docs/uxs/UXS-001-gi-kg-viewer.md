# UXS-001: GI / KG viewer

- **Status**: Active
- **Authors**: Podcast Scraper Team
- **Related PRDs**:
  - [PRD-003: User Interfaces & Configuration](../prd/PRD-003-user-interface-config.md)
  - [PRD-017: Grounded Insight Layer](../prd/PRD-017-grounded-insight-layer.md)
  - [PRD-019: Knowledge Graph Layer](../prd/PRD-019-knowledge-graph-layer.md)
  - [PRD-021: Semantic Corpus Search](../prd/PRD-021-semantic-corpus-search.md) (search panel in viewer)
  - [PRD-022: Corpus Library & Episode Browser](../prd/PRD-022-corpus-library-episode-browser.md) (library tab / catalog)
  - [PRD-023: Corpus Digest & Library Glance](../prd/PRD-023-corpus-digest-recap.md) (Digest tab — discovery; PRD may still describe historical Library glance)
  - [PRD-024: Graph exploration toolkit](../prd/PRD-024-graph-exploration-toolkit.md) (graph chrome: zoom, layout, filters)
- **Related RFCs**:
  - [RFC-062: GI/KG viewer v2](../rfc/RFC-062-gi-kg-viewer-v2.md)
  - [RFC-069: Graph exploration toolkit](../rfc/RFC-069-graph-exploration-toolkit.md) (Cytoscape controls, minimap, degree filter)
  - [RFC-067: Corpus Library API & Viewer](../rfc/RFC-067-corpus-library-api-viewer.md) (catalog APIs, handoffs — behavioral detail in RFC)
  - [RFC-068: Corpus Digest API & Viewer](../rfc/RFC-068-corpus-digest-api-viewer.md) (digest API, **Digest** tab — behavioral detail in RFC)
- **Playwright / E2E**:
  - [E2E surface map](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md) — automation contract (surfaces, selectors); update with viewer UI/E2E changes ([#509](https://github.com/chipi/podcast_scraper/issues/509))
- **Related UX specs**: (none yet)
- **Related issues**:
  - [GitHub #489](https://github.com/chipi/podcast_scraper/issues/489) — Viewer v2 implementation (RFC-062)
- **Implementation paths**:
  - **v2:** Tailwind theme + shared tokens per RFC-062 (Vue 3 + Vite) in `web/gi-kg-viewer/`

## Summary

The GI/KG viewer is a **local, developer-oriented** surface for exploring grounded insights
and knowledge-graph artifacts alongside the graph. This UXS locks **semantic colors,
typography, and light/dark behavior** so the Vue/Tailwind v2 stays visually
coherent and easy to maintain.

## Principles

- **Data-first:** Maximize space for the graph and evidence; chrome stays quiet.
- **Respect system theme:** Support light and dark via `prefers-color-scheme` (or
  equivalent in v2) unless a future PRD mandates a manual toggle only.
- **Semantic coloring:** GIL vs KG cues use dedicated domain tokens (`gi`, `kg`), not
  ad hoc greens and purples in components. UI-level intents (`primary`, `success`,
  `warning`, `danger`) are separate from domain tokens.
- **Intelligence-tool aesthetic:** Feel like a serious analytics/intelligence platform
  (Palantir Blueprint, Grafana, Elastic) -- professional, data-dense, modern but
  relaxed. Not a marketing site, not a consumer app.

## Scope

**In scope:**

- Viewer shell: header (**Podcast Intelligence Platform** + **v2**), panels, file/folder pickers, banners
- **Corpus library module** (PRD-022 / RFC-067): feed and episode browsing surfaces that
  reuse the same tokens and density as the rest of the viewer — see [Corpus library module](#corpus-library-module-prd-022)
- **Corpus digest** (PRD-023 / RFC-068): **Digest** main tab for **recent + topic** discovery;
  **Library** stays catalog-first (no embedded 24h digest strip) — see [Corpus digest (PRD-023)](#corpus-digest-prd-023)
- Graph visualization chrome (legends, selection affordances) at the token level
- Chart.js panels that summarize graph or artifact stats
- **Dashboard** tab: **Chart.js** panels — **GI + KG artifact write-day** timeline (from artifact mtimes),
  **publish-month** histogram (`/api/corpus/stats`), **manifest** feed throughput and **run.json**
  duration trend + **latest-run** stage stack and **episode-outcome** doughnut when data exists;
  plus node-type and indexed doc-type **bars**; corpus path, graph metrics, vector index,
  **staleness / reindex** (#507), and rebuild actions live in **API · Data** → **Data** (elevated cards)

**Non-goals:**

- Brand marketing pages, email templates, or MkDocs theme
- CLI ANSI styling (covered only if a separate UXS is added)

**Boundary note:** This UXS covers the **static visual contract** (tokens, layout, component
appearance, accessibility targets). Behavioral rules (animation timing, debounce intervals,
resize/collapse logic, keyboard shortcuts) belong in
[RFC-062](../rfc/RFC-062-gi-kg-viewer-v2.md). Library-specific **behavior** (pagination
sizes, API query parameters, debounce for search-as-you-type, deep-link query names) belongs
in [RFC-067](../rfc/RFC-067-corpus-library-api-viewer.md). Digest-specific **behavior**
(windows, `compact` mode, topic timeouts, diversity caps) belongs in
[RFC-068](../rfc/RFC-068-corpus-digest-api-viewer.md). See the
[UXS vs RFC boundary](index.md#uxs-vs-rfc-boundary) guidance.

## Dashboard tab (charts)

- **Layout (top to bottom):** Timeline / corpus-shape charts (GI mtime line, publish-month vertical
  bars, optional manifest + run-duration lines), then **latest run** pipeline (stacked stage bar +
  outcomes doughnut), then graph **node-type** and **index doc-type** horizontal bars.
- **Copy:** Short blurb points to **API · Data** for corpus path and index tooling; optional
  “Loading corpus charts…” while dashboard fetches aggregate APIs.
- **E2E contract:** [E2E surface map](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md).

## API · Data (left panel)

- **API:** Neutral section title + muted blurb, then one **elevated** card: **Health** (label + value,
  e.g. OK from `/api/health`), then **Yes/No** rows for **Artifacts (graph)**, **Semantic search**,
  **Graph explore**, **Index routes**, **Corpus metrics**, **Library API**, **Digest API**, **Binary (covers)** —
  from the same health JSON (omit a flag on older servers → treated as advertised except catalog flags).
  **Retry health**; when health fails, the offline **Choose files** affordance stays here.
- **Data:** Neutral section title + blurb; **Corpus root:** / **Resolved:** lines sit as plain text
  under the blurb (not inside a card). Then two sibling **elevated** cards — **Graph**
  metrics and **Vector index** (`GET /api/index/stats` + rebuild actions) — same depth as the API
  card, without an extra nested metrics frame inside each card.
- **Density:** Same `surface` / `border` cards; sidebar uses slightly smaller type (`text-[10px]` /
  `text-xs`) so the panel stays scannable at `w-72`.
- **Intent colors:** **Reindex recommended** uses **warning** panels; informational notes use **muted**
  framing; **Last rebuild error** uses **danger** text (consistent with shell errors).
- **Actions:** **Update index** and **Full rebuild** sit next to **Refresh**; disabled while
  `rebuild_in_progress` or `faiss_unavailable`.
- **E2E contract:** [E2E surface map](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md) — **API · Data** tab, **Data** heading, vector index buttons.

## Corpus path (left **Corpus** tab)

- **Online (health ok):** Typing a corpus path triggers listing via `GET /api/artifacts` and loads
  **every** listed `.gi.json` / `.kg.json` into the **merged** graph automatically—aligned with
  Digest/Library refreshing from the same path without an extra click. **List** still refreshes the
  checkbox list; **Load into graph** still applies the current selection. Very large corpora may take
  noticeable time and memory.
- **Offline:** Unchanged: **Choose files…** on **API · Data** loads local GI/KG without the server.

## Corpus library module (PRD-022)

The **Corpus library** is a **catalog** view inside the same SPA: it answers “what feeds and
episodes were processed?” and connects to **Graph** and **Semantic search** without a
separate theme or product chrome.

### Information architecture

- **Entry:** A **main** tab labeled **Library**, placed after **Digest** and before
  **Graph** / **Dashboard**, with `aria-label` consistent with other main views.
- **Layout (desktop):** Three logical regions on `canvas` / `surface`:
  1. **Episode filters** — one collapsible: on wide viewports, a **three-column** row —
     **published on or after** (compact presets + custom `YYYY-MM-DD`, aligned with Search
     **since**) on the **left** — **`label` Published on or after** and the **YYYY-MM-DD** input share the **first**
     row; preset buttons (**All time**, **7d**, …) sit **below**; **title** and **summary / topic** plus **Apply** / **Clear text and
     date** in the **center**, **feed** list on the **right**; stacks vertically on narrow widths.
     Feed rows use **display title** when present (**stable `feed_id`**, plus **RSS** and
     **description** in **`title` hover** when `GET /api/corpus/feeds` includes them), `border`
     dividers, and `overlay` for the selected feed row.
  2. **Episode column** — scrollable list of episodes: **title row** is **episode title** (left) and **right**
     column: **first** line **feed display name** (truncated, **left**) and **publish date** (**right**);
     **second** line **E#** and **duration** when present; **native
     `title` hover** on the feed name shows **RSS URL**, **feed id**, and **feed description** when
     the API provides them (`feed_rss_url` / `feed_description` or feeds catalog). **Summary** line
     under the title — same recap rules as **Digest** / `digestRowSummaryPreview`, including
     **`topics`** fallback — then **topic pills**; recap **wraps fully** (no line clamp); selected
     row uses `overlay`.
  3. **Episode panel** — `surface` card with heading **Episode**, episode title (heading scale); **first** meta line:
     **feed** (**left**) and **publish date** (**right**, `muted`) with the **same native hover** on the feed as
     episode list rows (RSS, id, description when available); **E#** / **duration** on a **second** line when present;
     optional **summary title** and **summary prose**; when bullets follow that recap, a **`border`**
     separator plus **`h4` “Key points”** (muted, small caps scale) precedes summary bullets as a
     **semantic list** (`ul` / `li`), not wall-of-text paragraphs.
- **Empty states:** Use the same **empty state** language as the rest of the viewer
  (`muted`, short instruction, e.g. set corpus path or confirm metadata exists).

### Visual and token rules

- Reuse **surface**, **border**, **elevated**, **muted**, **primary** for lists, filters, and
  actions — no new palette for the library.
- **GI / KG actions** in the episode panel use **domain tokens** (`gi`, `kg`) for buttons or
  badges that refer to artifact type (consistent with **Domain tokens (GIL / KG identity)**
  under [Semantic color tokens](#semantic-color-tokens)).
- **Search handoff** control uses **primary** (it is a navigation affordance to an existing
  panel, not a domain-colored insight).
- **Vector index awareness (RFC-067 Phase 3):** Optional **Indexed** chip on feed rows when
  that `feed_id` appears in `GET /api/index/stats` → `feeds_indexed` (server-normalized ids);
  episode panel exposes **Episode and feed diagnostics** (help control → tooltip) with paths, ids,
  **Feed in vector index**, and index stats when loaded — no always-on troubleshooting strip.
  **Find similar episodes** runs
  `GET /api/corpus/episodes/similar` and lists peer episodes (scores + open-in-library when paths
  resolve); **empty** successful responses show a short **no peers** state. **Prefill semantic search**
  opens Search with **feed id** + **summary-based query** (full summary text when available); user runs **Search** for vector hits.

### Library accessibility targets

- **Lists:** Feed and episode columns are **keyboard navigable** (`Tab` / arrow patterns as
  implemented per RFC-067); each selectable row exposes a clear **accessible name** (episode
  title + feed). **Topic pills** (summary bullets, capped) are separate **buttons** with their
  own labels; they apply a **topic** filter (`topic_q`) and reload the list (same field as the
  filter panel’s summary/topic input).
- **Focus:** Selected list row and primary actions use the same **focus ring** convention
  as the rest of the viewer ([Key states](#key-states)).
- **Episode panel bullets:** Summary bullets remain **plain text list items**; if any bullet is
  truncated, provide a visible or tooltip expansion only if it does not break contrast rules.

### Implementation and E2E contract

When the Library UI ships, implementers **must** update the
[E2E surface map](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md)
first for any **E2E-visible** strings, roles, or hooks, then align Playwright specs per
[DEVELOPMENT_GUIDE.md](../guides/DEVELOPMENT_GUIDE.md) and [E2E_TESTING_GUIDE.md](../guides/E2E_TESTING_GUIDE.md).

## Corpus digest (PRD-023)

The **Digest** tab is the **discovery** surface for rolling-window **recent episodes** and **semantic topic** bands.
Row clicks send episodes to **Library** (episode panel) for **Graph** and **Prefill semantic search**; **topic bands**
still expose a **topic title → Graph** shortcut (top hit only) and **Search topic**. **Library** no longer embeds a
**New (24h)** digest strip so the two tabs do not compete — users open **Digest** for “what’s new,” **Library** for
catalog browse and episode detail. Same **surface** tokens; **orientation**, not a second theme.

### Digest information architecture

- **Main nav order:** **Digest**, **Library**, **Graph**, **Dashboard** (left to right).
  **Default selected tab:** **Digest** on first load.
- **Digest tab:** A **main** nav item labeled **Digest** (same control pattern as other
  main views). Default time lens **7d** (behavioral default in RFC-068); user can switch
  **24h** / **7d** / **`since`** per RFC.
- **Toolbar:** **Open Library** switches to the **Library** tab without clearing corpus path.

### Layout and density

- **Digest tab:** Single-column **scroll** on `canvas`; sections:
  1. **Window** — compact preset control row (`muted` labels, `surface` controls). A **muted** summary line shows
     rolling bounds (**start → end · row count**) with readable date + time and an explicit **UTC** suffix (not raw ISO
     with fractional seconds); `<time datetime="…">` keeps machine-readable instants.
  2. **Topic bands** — responsive **grid row** (`sm:2` / `xl:3` columns): compact card padding; per-topic **hit row**
     is one **clickable** control (same handoff as **Recent**): opens **Library** with that episode; **`summary_preview`**
     under the title — **tight** gap (`mt-0.5`), **full wrap**, no clamp; **right** column **top → bottom**:
     **publish date** (from API when present), **duration**, **feed** (truncated); **hover** on feed for RSS / id /
     description (same as Library episode rows); **similarity score** as a small mono pill with **native tooltip**
     explaining vector similarity (model-dependent). **No** per-hit **Graph** button — use the **topic title** control
     for **top-hit Graph** or open **Library** and use **Open in graph** there. **Search topic** prefills semantic search.
  3. **Recent (diverse)** — episode rows aligned with **Library** list: **cover** `h-9`, **title** +
     row: **right** column **first** line **feed** + **publish date**; **E#** / **duration** **below** (**hover** on feed = RSS, id,
     description when available), **full-wrap** **`summary_preview`** / recap;
     **card click** → **Library** episode panel (no **GI/KG** chips on the card — status lives on **Library**
     episode panel). **Hover** may include **feed id** when it differs from the display title.
     **Accessible name** matches Library rows: **`episode_title`, `feed`**. **Feed** text prefers
     **`GET /api/corpus/feeds`** ``display_title`` (Library sidebar source), then digest row fields.
     **Open in graph** and **Prefill semantic search** live on **Library** episode panel in one row
     (**equal-width** buttons); Prefill keeps a small **?** help tip beside them.

### Actions and tokens

- **Digest — recent episode cards:** Only **Library** handoff (card click). **Open in graph** /
  **Prefill semantic search** follow [Corpus library module](#corpus-library-module-prd-022) on
  **Library** episode panel, not on the Digest card.
- **Open GI / Open KG / Prefill semantic search (Library):** Same **domain** and **primary** rules
  as [Corpus library module](#corpus-library-module-prd-022) (GI/KG use `gi`/`kg`; search handoff
  uses `primary`).
- **Topic band:** **Search topic** uses `primary` (or `surface`+`border` per button hierarchy);
  opens the Search panel with the topic query prefilled per RFC-068. **Topic title** opens the **top** hit in **Graph**;
  each **hit row** opens **Library** (same as **Recent** cards); **Open in graph** for a specific hit lives on **Library**.

### Digest accessibility

- **Digest tab:** `aria-label` on main nav matches visible **Digest** text.
- **Lists:** Digest **Recent** rows use the same **accessible name** pattern as **Library** episode
  rows (**episode title**, **feed** display label). **Topic band** hit rows use the same pair plus a
  **suffix** (e.g. **— topic band hit**) so the name stays unique when the same episode appears under **Recent** and in a topic.

### Digest E2E contract

Any **new** visible labels (e.g. **Digest**, **Open Library**, **Search topic**) require
updates to the
[E2E surface map](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md)
before or with implementation. Dedicated Playwright coverage lives in **`e2e/digest.spec.ts`**
(mocked `GET /api/corpus/digest` + health).

### Health discovery (capability flags)

- **`GET /api/health`** returns **`corpus_library_api`** (RFC-067) and **`corpus_digest_api`**
  (RFC-068) on current server builds.
- **Digest tab** is enabled when **`corpus_digest_api`** is **true**, or when it is **omitted**
  but **`corpus_library_api`** is true (viewer infers digest for health JSON from builds before
  the digest flag existed). If **`corpus_digest_api`** is **explicitly false**, the tab shows an
  **upgrade** message.
- If the API process is too old to mount **`GET /api/corpus/digest`**, the **Digest** tab **fetch** fails and shows a
  load error — restart **`podcast serve` / `make serve-api`** from a checkout that includes RFC-068 after
  **`pip install -e ".[server]"`**.

## Theme support

- **Mode:** both (follows system via `prefers-color-scheme`)
- **Primary palette:** dark — the viewer is a data tool used in terminal-adjacent contexts;
  dark mode is the design baseline
- **Breakpoints:** desktop only (minimum 1024px assumed; no mobile breakpoints)

## Semantic color tokens

Token values are inspired by the **Palantir Blueprint** gray scale with the **shadcn/ui**
foreground-pairing convention. Every surface token has a matching `-foreground` text token
so contrast is validated at the token level. V2 maps the full set to Tailwind/CSS
variables.

**Design references:** Blueprint gray scale (`#111418` .. `#F6F7F9`), shadcn/ui
background/foreground pairs, Grafana layered surfaces, Elastic UI intent/vis separation.

### Surface tokens (gray scale)

Five semantic steps for surfaces, from deepest background to lightest border.

| Token                  | Dark      | Light     | Usage                                 |
| ---------------------- | --------- | --------- | ------------------------------------- |
| `canvas`               | `#111418` | `#F6F7F9` | Page/app background                   |
| `canvas-foreground`    | `#E5E8EB` | `#1C2127` | Primary text on canvas                |
| `surface`              | `#1C2127` | `#FFFFFF` | Panels, cards, `code` blocks          |
| `surface-foreground`   | `#E5E8EB` | `#1C2127` | Primary text on surface               |
| `elevated`             | `#252A31` | `#F6F7F9` | Popovers, dropdowns, modals           |
| `elevated-foreground`  | `#DCE0E5` | `#252A31` | Text on elevated surfaces             |
| `overlay`              | `#2F343C` | `#EDEFF2` | Hover overlays, active table rows     |
| `overlay-foreground`   | `#DCE0E5` | `#252A31` | Text on overlay surfaces              |
| `border`               | `#404854` | `#D3D8DE` | Dividers, input borders               |

### Text tokens

| Token      | Dark      | Light     | Usage                                    |
| ---------- | --------- | --------- | ---------------------------------------- |
| `muted`    | `#8F99A8` | `#5F6B7C` | Secondary labels, help text, timestamps  |
| `disabled` | `#5F6B7C` | `#ABB3BF` | Disabled controls and placeholder text   |
| `link`     | `#6cb3f7` | `#1a6fc4` | Inline text links (distinct from action) |

### Intent tokens (UI actions and feedback)

Separate from domain tokens; used for buttons, alerts, and status indicators.

| Token                  | Dark      | Light     | Usage                          |
| ---------------------- | --------- | --------- | ------------------------------ |
| `primary`              | `#4C90F0` | `#2D72D2` | Primary action buttons, links  |
| `primary-foreground`   | `#111418` | `#FFFFFF` | Text on primary surfaces       |
| `success`              | `#32A467` | `#238551` | Positive status, confirmations |
| `success-foreground`   | `#111418` | `#FFFFFF` | Text on success surfaces       |
| `warning`              | `#EC9A3C` | `#C87619` | Caution states, non-critical   |
| `warning-foreground`   | `#111418` | `#111418` | Text on warning surfaces       |
| `danger`               | `#E76A6E` | `#CD4246` | Errors, destructive actions    |
| `danger-foreground`    | `#111418` | `#FFFFFF` | Text on danger surfaces        |

### Domain tokens (GIL / KG identity)

These are visualization-level cues, not generic UI intents. They stay stable across
themes and distinguish GIL from KG content at a glance.

| Token                | Dark      | Light     | Usage                     |
| -------------------- | --------- | --------- | ------------------------- |
| `gi`                 | `#7dd3a0` | `#1e7a4a` | GIL / insight affordances |
| `gi-foreground`      | `#111418` | `#FFFFFF` | Text on gi surfaces       |
| `kg`                 | `#c4a3ff` | `#5c3d9e` | KG affordances            |
| `kg-foreground`      | `#111418` | `#FFFFFF` | Text on kg surfaces       |

### Chart series tokens

Used by Chart.js bar/line/pie charts when more than two series are plotted. Derived
from Blueprint extended palette colors that complement the domain and intent tokens.

| Token      | Dark      | Light     | Usage                    |
| ---------- | --------- | --------- | ------------------------ |
| `series-1` | `#4C90F0` | `#2D72D2` | First series (= primary) |
| `series-2` | `#7dd3a0` | `#1e7a4a` | Second series (= gi)     |
| `series-3` | `#c4a3ff` | `#5c3d9e` | Third series (= kg)      |
| `series-4` | `#EC9A3C` | `#C87619` | Fourth series (= warning)|
| `series-5` | `#3FA6DA` | `#147EB3` | Fifth series (teal)      |

Banners may use `color-mix` against `surface` / `border`; new variants should
still derive from the tokens above.

## Typography

- **UI font:** `Inter, system-ui, -apple-system, "Segoe UI", sans-serif` — Inter
  (variable, self-hosted or Google Fonts) provides tabular numerals for data tables and
  a tall x-height for dense layouts. Falls back to system fonts if Inter is not loaded.
- **Monospace font:** `"JetBrains Mono", ui-monospace, "Cascadia Code", "Fira Code",
  monospace` — used for transcript snippets, code blocks, JSON, and any fixed-width data.
  Clear glyph differentiation (0/O, 1/l/I) matters for data review.
- **Scale:** Prefer relative sizes (`rem` / Tailwind `text-sm` / `text-base`); keep body
  at **1rem** (16px) equivalent for long reading; tighten slightly for dense metadata
  rows. Heading scale: `text-xl` (h1), `text-lg` (h2), `text-base font-semibold` (h3).
- **Dense UI:** Tables and stat rows may use **one step smaller** (`text-sm`, 0.875rem)
  than body but not below **0.8125rem** without an explicit accessibility review.
- **Font weights:** 400 (regular body), 500 (medium emphasis), 600 (semibold headings
  and labels), 700 (bold, sparingly).
- **Font stack:** Inter is the primary UI font for v2.

## Layout and spacing

- **Base unit:** **4px** (0.25rem) — spacing and gaps should be multiples where practical.
- **Max content width:** **960px** for primary column content; graph area may be
  full viewport width.
- **Regions:** Header + lede + panel stack; graph pages use full-height canvas with
  overlays/panels that respect `surface` and `border`.

## Key states

Visual treatment for interactive elements. Behavioral timing (debounce, transition
duration) belongs in RFC-062.

- **Hover:** Show `overlay` background (one step lighter than `surface`); underline on
  links; graph nodes may show a tooltip or highlight ring using `primary` at reduced
  opacity
- **Active / pressed:** Show `elevated` background (one step darker than `overlay`)
- **Focus:** 2px solid `primary` ring with 2px offset; never remove native outline
  without an equivalent replacement
- **Disabled:** Use `disabled` text color, 40% opacity on controls, `cursor: not-allowed`
- **Loading:** Skeleton placeholder using `surface` / `border` stripes; graph area may
  show a centered spinner using `muted`
- **Empty state:** Centered `muted` text explaining what to load (e.g. "Select a file or
  folder to begin"); no decorative illustration required
- **Error state:** `danger` border on inputs; inline error message in `danger`; banner
  variant for page-level errors using `danger` / `danger-foreground`

## Components

- **Buttons:** Primary uses `primary` / `primary-foreground`; secondary uses `surface` /
  `border`; destructive uses `danger` / `danger-foreground`.
- **File inputs / pickers:** Match `surface` background; clear label weight (**600**).
- **Banners / alerts:** Success uses `success` background tint; warning uses `warning`
  tint; error uses `danger` tint; neutral info uses `primary` or `muted`. Domain-specific
  success messages (e.g. "insights loaded") may lean on `gi` instead of generic `success`.

## Charts and graph

- Chart.js (and future chart wrappers) **must** resolve axis/grid/text colors from the
  same light/dark logic as the page (CSS variables or a shared `theme` module).
- Multi-series charts use `series-1` through `series-5` in order. Single-series charts
  default to `primary`. Domain-specific charts (GIL vs KG breakdowns) use `gi` / `kg`.
- Cytoscape (v2) node/edge styling should consume the same semantic tokens (or a small
  `theme.ts` re-export) so the graph matches panels and charts.

### Graph exploration chrome (PRD-024 / RFC-069)

- **Toolbar (primary row):** **Fit**, **Re-layout** (runs layout on visible elements only),
  **Export PNG**, zoom **−** / **+** / **100%** (100% = `zoom(1)`, pan unchanged), and a short
  hint for **Shift+dbl-click** (1-hop / neighborhood) and **Shift+drag** box zoom.
- **Toolbar (chrome below zoom):** Optional **Sources** row first (merged **GI** / **KG**,
  **Hide ungrounded**, **filters active** when relevant); then Layout **select** (cose,
  breadthfirst, circle, grid), **Degree** buckets as **visibility filter**, **Minimap**,
  **Clear degree filter**; then **Edges** and **Types** (per-type checkboxes + **all** /
  **none**, swatches match node fills, counts). No separate panel above the graph.
- **Density:** Use existing `text-[10px]` / `border-border` patterns so the extra row stays
  compact and scannable; minimap is a small overlay in the **lower-left** of the graph viewport
  (clipped with the canvas), not over the app’s right rail.
- **E2E contract:** [E2E surface map](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md) — graph shell row.

## Accessibility

- **Focus:** Visible focus styles on all interactive elements (native or custom); do not
  remove outline without a replacement that meets contrast.
- **Contrast:** Aim for **WCAG 2.1 AA** for normal text against `canvas` / `surface`;
  verify `muted` on `surface` for small text. Foreground-pairing ensures each surface
  token has a pre-validated text color.
- **Motion:** Respect `prefers-reduced-motion` for non-essential animations.

## Tunable parameters

The token **architecture** (names, pairing convention, intent/domain split) is frozen.
The **values** below are deliberately open for experimentation during early v2 development.
Adjust them in `theme.css` (CSS custom properties) or via browser DevTools; do not
hard-code alternatives in component files.

| Parameter                 | Current value                              | Status | Notes                                          |
| ------------------------- | ------------------------------------------ | ------ | ---------------------------------------------- |
| UI font family            | Inter                                      | Open   | Could try Geist, IBM Plex Sans, system stack   |
| Monospace font family     | JetBrains Mono                             | Open   | Could try Fira Code, Source Code Pro           |
| Base font size            | 16px (1rem)                                | Open   | 14px more dense; 16px more readable            |
| Border radius             | 0.375rem (6px)                             | Open   | 0 = sharp/Palantir; 0.5rem = softer            |
| Spacing base unit         | 4px                                        | Open   | Could move to 6px or 8px for more air          |
| Shadow depth              | None (flat)                                | Open   | Subtle shadow = more depth/elevation cue       |
| Surface gray palette      | Blueprint-derived `#111418`..`#404854`     | Open   | Exact values may shift during contrast review  |
| Intent color hues         | Blueprint-derived blue/green/orange/red    | Open   | Hues stable; saturation/lightness may tune     |
| Domain color hues (gi/kg) | Green `#7dd3a0` / Purple `#c4a3ff`         | Frozen | Identity colors; do not change w/o UXS rev     |
| Token names               | `canvas`, `surface`, `primary`, `gi`, etc. | Frozen | Names are the API; values are the theme        |
| Pairing convention        | Every surface gets `-foreground`           | Frozen | Structural rule; not negotiable                |
| Intent/domain separation  | Intent for UI; domain for GIL/KG           | Frozen | Structural rule; not negotiable                |

### How to experiment

- **Browser DevTools:** Inspect `:root`, change any `--` variable live. Fastest loop.
- **`theme.css` presets:** Create alternate files (`theme-compact.css`,
  `theme-relaxed.css`) that override the open parameters. Import one at a time.
- **Pinia theme store (optional):** A small store that sets CSS variables at runtime,
  enabling a theme switcher in the UI for side-by-side comparison.

When an open parameter is finalized after experimentation, update its status to
**Frozen** in this table and record the decision in the revision history.

## Visual references

No wireframes checked in yet. As the v2 UI stabilizes, add annotated screenshots or Figma
frames here to lock layout expectations for the main panels, graph canvas, and search
overlay.

## Acceptance criteria

- [ ] New viewer UI uses semantic tokens only (no stray hex in Vue/SFC or JS style objects)
- [ ] Theme support matches declared mode (both, system-driven, dark baseline)
- [ ] Light and dark values match the token table (or the table is updated with rationale)
- [ ] Every surface token uses its matching `-foreground` for text (no ad hoc text colors)
- [ ] Intent tokens (`primary`/`success`/`warning`/`danger`) are used for UI feedback;
      domain tokens (`gi`/`kg`) are used only for GIL/KG identity
- [ ] Key interactive states match this spec (hover, focus, disabled, error, empty, loading)
- [ ] Focus states visible on buttons, inputs, and graph controls
- [ ] Chart.js series use `series-1` through `series-5`; graph colors derive from palette
- [ ] Inter and JetBrains Mono load correctly (or system fallbacks render acceptably)
- [ ] RFC-062 implementation checklist references this UXS for theme work
- [ ] Tunable parameters table reflects current status (open values finalized → frozen)
- [ ] Theme preset swap (e.g. `default.css` → `compact.css`) changes visuals without
      component edits

## Revision history

| Date       | Change                                                              |
| ---------- | ------------------------------------------------------------------- |
| 2026-04-03 | Initial draft                                                       |
| 2026-04-03 | Added theme support, key states, boundary note, visual refs section |
| 2026-04-03 | Blueprint gray scale, shadcn pairing, intent/domain split, Inter +  |
|            | JetBrains Mono, chart series tokens, design-reference citations     |
| 2026-04-03 | Added Tunable parameters section (frozen vs open knobs); RFC-062    |
|            | and PRD cross-references updated to document theme preset system    |
| 2026-04-06 | Status **Active** — viewer v2 implements this contract per RFC-062; |
|            | PRD-021 linked for search UI scope                                  |
| 2026-04-10 | Corpus library module (PRD-022 / RFC-067): IA, tokens, a11y;        |
|            | E2E surface map obligation for implementation                       |
| 2026-04-10 | Corpus digest (PRD-023 / RFC-068): Digest tab + Library 24h glance; |
|            | RFC boundary note; E2E obligation                                   |
| 2026-04-11 | Digest health `corpus_digest_api`; glance gate; **Search topic**    |
|            | E2E `digest.spec.ts`                                                |
