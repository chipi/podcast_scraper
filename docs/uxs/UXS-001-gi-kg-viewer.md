# UXS-001: GI / KG Viewer -- Shared Design System

- **Status**: Active
- **Authors**: Podcast Scraper Team
- **Related PRDs**:
  - [PRD-003: User Interfaces & Configuration](../prd/PRD-003-user-interface-config.md)
  - [PRD-017: Grounded Insight Layer](../prd/PRD-017-grounded-insight-layer.md)
  - [PRD-019: Knowledge Graph Layer](../prd/PRD-019-knowledge-graph-layer.md)
- **Related RFCs**:
  - [RFC-062: GI/KG viewer v2](../rfc/RFC-062-gi-kg-viewer-v2.md)
- **Shell information architecture (regions, axes, persistence):**
  [VIEWER_IA.md](VIEWER_IA.md)
- **Playwright / E2E**:
  - [E2E surface map](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md)
- **Feature UX specs** (each viewer surface has its own UXS):
  - [UXS-002: Corpus Digest](UXS-002-corpus-digest.md)
  - [UXS-003: Corpus Library](UXS-003-corpus-library.md)
  - [UXS-004: Graph Exploration](UXS-004-graph-exploration.md)
  - [UXS-005: Semantic Search](UXS-005-semantic-search.md)
  - [UXS-006: Dashboard](UXS-006-dashboard.md)
  - [UXS-007: Topic Entity View](UXS-007-topic-entity-view.md)
  - [UXS-008: Enriched Search](UXS-008-enriched-search.md)
- **Implementation paths**:
  - `web/gi-kg-viewer/src/theme/tokens.css` -- CSS custom properties (`--ps-*`)
  - `web/gi-kg-viewer/tailwind.config.js` -- Tailwind color mapping to CSS vars
  - `web/gi-kg-viewer/src/style.css` -- global styles, font stacks, focus ring
  - `web/gi-kg-viewer/src/utils/chartTheme.ts` -- Chart.js theme integration
  - `web/gi-kg-viewer/src/utils/cyGraphStylesheet.ts` -- Cytoscape node/edge styles

---

## Summary

This is the **shared design system** for the GI/KG viewer SPA (`web/gi-kg-viewer/`).
It defines semantic color tokens, typography, layout primitives, key interactive states,
accessibility targets, and component conventions that all feature UXS files reference.

Individual viewer surfaces (search, library, digest, graph, dashboard, topic entity
view, enriched search) have their own UXS files for layout, density, and
surface-specific rules. This file is the foundation they all build on.

**Shell layout (information architecture):** The three-column shell (left
query column, main tabs, right subject rail, bottom status bar) is defined in
[VIEWER_IA.md](VIEWER_IA.md) — regions, navigation axes, subject persistence and
clearing, status bar contents, and left-panel collapse. This UXS-001 file
anchors **tokens, typography, tunables, and shared components** for those
regions; do not duplicate the full IA narrative here.

**InsightCard:** The reusable **InsightCard** layout for GIL insight rows (slots,
tokens, accessibility) is specified below under **[Components → InsightCard (shared
component)](#insightcard-shared-component)**. UXS-007, UXS-009, and UXS-010 reference
that subsection instead of duplicating card rules.

---

## Principles

- **Data-first:** Maximize space for the graph and evidence; chrome stays quiet.
- **Respect system theme:** Support light and dark via `prefers-color-scheme` (or
  equivalent) unless a future PRD mandates a manual toggle only.
- **Semantic coloring:** GIL vs KG cues use dedicated domain tokens (`gi`, `kg`), not
  ad hoc greens and purples in components. UI-level intents (`primary`, `success`,
  `warning`, `danger`) are separate from domain tokens.
- **Intelligence-tool aesthetic:** Feel like a serious analytics/intelligence platform
  (Palantir Blueprint, Grafana, Elastic) -- professional, data-dense, modern but
  relaxed. Not a marketing site, not a consumer app.

---

## Scope

**In scope:**

- Viewer shell: header (**Podcast Intelligence Platform** + **v2**), panels,
  file/folder pickers, banners
- Semantic color tokens (surface, text, intent, domain, chart series)
- Typography (font stacks, scale, weights)
- Layout primitives (base unit, max width, regions)
- Key interactive states (hover, focus, disabled, loading, empty, error)
- Component conventions (buttons, inputs, banners)
- Chart and graph color rules
- Accessibility targets

**Non-goals:**

- Brand marketing pages, email templates, or MkDocs theme
- CLI ANSI styling (covered only if a separate UXS is added)
- Feature-specific layout and density (see feature UXS files)
- Full **shell information architecture** (regions, the three navigation axes, subject persistence and clearing rules, first-run policy, cross-surface pseudocode) — specified only in **[VIEWER_IA.md](VIEWER_IA.md)**; UXS-001 does not replace it

**Historical note:** The pre–shell-restructure left-panel **Corpus | API·Data** tabs are obsolete. Current chrome follows **VIEWER_IA** (status bar corpus path + left Search/Explore); see [GitHub #606](https://github.com/chipi/podcast_scraper/issues/606).

**Boundary note:** This UXS covers the **static visual contract** (tokens, layout,
component appearance, accessibility targets). Behavioral rules (animation timing,
debounce intervals, resize/collapse logic, keyboard shortcuts) belong in
[RFC-062](../rfc/RFC-062-gi-kg-viewer-v2.md). See the
[UXS vs RFC boundary](index.md#uxs-vs-rfc-boundary) guidance.

---

## Theme support

- **Mode:** both (follows system via `prefers-color-scheme`; manual override via
  `data-theme` attribute on `:root`)
- **Primary palette:** dark -- the viewer is a data tool used in terminal-adjacent
  contexts; dark mode is the design baseline
- **Breakpoints:** desktop only (minimum 1024px assumed; no mobile breakpoints).
  Approximate three-column widths at 1024px are recorded in
  [VIEWER_IA.md — Viewport](VIEWER_IA.md#viewport--three-column-widths-1024px-baseline).

---

## Semantic color tokens

Token values are inspired by the **Palantir Blueprint** gray scale with the
**shadcn/ui** foreground-pairing convention. Every surface token has a matching
`-foreground` text token so contrast is validated at the token level.

CSS custom properties use the `--ps-` prefix (defined in
`src/theme/tokens.css`). Tailwind maps these via `tailwind.config.js`.

**Design references:** Blueprint gray scale (`#111418` .. `#F6F7F9`), shadcn/ui
background/foreground pairs, Grafana layered surfaces, Elastic UI intent/vis
separation.

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

**`warning` token vs clusters:** After CIL digest pill alignment (UXS-002),
**`warning`** fill/border for **search-hit emphasis** on graph nodes (e.g.
Quote / search-highlighted nodes) only. **Topic cluster** membership and topic
discovery UI use the **`kg`** token everywhere — Digest CIL pills, graph
`TopicCluster` compounds, Dashboard Intelligence cluster cards. Do **not** use
`warning` fill for cluster membership or topic grouping.

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

---

## Typography

- **UI font:** `Inter, system-ui, -apple-system, "Segoe UI", sans-serif` -- Inter
  (variable, self-hosted or Google Fonts) provides tabular numerals for data tables and
  a tall x-height for dense layouts. Falls back to system fonts if Inter is not loaded.
- **Monospace font:** `"JetBrains Mono", ui-monospace, "Cascadia Code", "Fira Code",
  monospace` -- used for transcript snippets, code blocks, JSON, and any fixed-width
  data. Clear glyph differentiation (0/O, 1/l/I) matters for data review.
- **Scale:** Prefer relative sizes (`rem` / Tailwind `text-sm` / `text-base`); keep
  body at **1rem** (16px) equivalent for long reading; tighten slightly for dense
  metadata rows. Heading scale: `text-xl` (h1), `text-lg` (h2),
  `text-base font-semibold` (h3).
- **Dense UI:** Tables and stat rows may use **one step smaller** (`text-sm`, 0.875rem)
  than body but not below **0.8125rem** without an explicit accessibility review.
- **Font weights:** 400 (regular body), 500 (medium emphasis), 600 (semibold headings
  and labels), 700 (bold, sparingly).

---

## Layout and spacing

- **Base unit:** **4px** (0.25rem) -- spacing and gaps should be multiples where
  practical.
- **Max content width:** **960px** for primary column content; graph area may be
  full viewport width.
- **Left query column** (**Semantic search** default; **Explore** as a slide-in mode from
  **`LeftPanel.vue`**): collapsible **`w-72`** rail in `App.vue` when expanded.
- **Right subject column** (Episode detail, graph node detail, future topic/person): fixed
  width **`w-96`** (**24rem** / **384px** at **16px** root) when expanded. To tune further,
  change that class on the same element (desktop-only layout).
- **Status bar:** full-width footer under the three-column row — visual contract in **[Status bar](#status-bar)** below (`StatusBar.vue`, **`data-testid="app-status-bar"`**).
- **Regions:** Header + lede + panel stack; graph pages use full-height canvas with
  overlays/panels that respect `surface` and `border`.

---

## Key states

Visual treatment for interactive elements. Behavioral timing (debounce, transition
duration) belongs in RFC-062.

- **Hover:** Show `overlay` background (one step lighter than `surface`); underline on
  links; graph nodes may show a tooltip or highlight ring using `primary` at reduced
  opacity
- **Active / pressed:** Show `elevated` background (one step darker than `overlay`)
- **Focus:** 2px solid `primary` ring with 2px offset; never remove native outline
  without an equivalent replacement
- **Disabled:** Use `disabled` text color, 40% opacity on controls,
  `cursor: not-allowed`
- **Loading:** Skeleton placeholder using `surface` / `border` stripes; graph area may
  show a centered spinner using `muted`
- **Empty state:** Centered `muted` text explaining what to load (e.g. "Select a file
  or folder to begin"); no decorative illustration required
- **Error state:** `danger` border on inputs; inline error message in `danger`; banner
  variant for page-level errors using `danger` / `danger-foreground`

---

## Components

- **Buttons:** Primary uses `primary` / `primary-foreground`; secondary uses
  `surface` / `border`; destructive uses `danger` / `danger-foreground`.
- **File inputs / pickers:** Match `surface` background; clear label weight (**600**).
- **Banners / alerts:** Success uses `success` background tint; warning uses `warning`
  tint; error uses `danger` tint; neutral info uses `primary` or `muted`.
  Domain-specific success messages (e.g. "insights loaded") may lean on `gi` instead
  of generic `success`.

### InsightCard (shared component) {#insightcard-shared-component}

The **InsightCard** is a reusable card component used across multiple feature UXSs
(UXS-007, UXS-009, UXS-010) to display a GIL Insight with consistent visual
treatment. Feature UXSs declare which slots they use; the component renders only
populated slots.

**Required slots** (always present):

- **Insight text**: the Insight's text content (`text-sm`, `canvas-foreground`).
- **Grounding badge**: "grounded" (`success` token, `text-xs`) or "ungrounded"
  (`warning` token, `text-xs`). Visual indicator of data quality.

**Optional slots** (feature UXSs opt in):

| Slot                 | Token / style                                         | Used by          |
| -------------------- | ----------------------------------------------------- | ---------------- |
| `insight_type` badge | Small pill, `text-xs`. See token mapping below        | UXS-009, UXS-010 |
| `position_hint` bar  | 40px bar, `primary` on `border`; UXS-009 tooltip      | UXS-009, UXS-010 |
| Confidence score     | `muted`, `text-xs`. Hidden when unavailable           | UXS-009, UXS-010 |
| Speaker chip         | `muted`, `text-sm`. Clickable to Person Landing       | UXS-007          |
| Episode attribution  | Episode title + publish date, `muted`, `text-xs`      | UXS-007, UXS-010 |
| Supporting quote     | Blockquote, `border` left 3px, italic. With timestamp | UXS-007, UXS-009 |

**`insight_type` token mapping**: `claim` = `primary`, `recommendation` = `success`,
`observation` = `muted`, `question` = `link`, other = `muted`.

**"Strongest" variant** (UXS-010 only): When a card represents the
`strongest_insight` for a topic group, it receives a `primary` left border (3px)
and a "Strongest" label (`primary`, `text-xs`).

**Accessibility**: InsightCard uses `role="article"` with `aria-label` constructed
from the insight type and first ~50 characters of text. Badge tokens are
supplemented by text labels (colour is never the sole differentiator).

---

## Charts and graph

- Chart.js (and future chart wrappers) **must** resolve axis/grid/text colors from the
  same light/dark logic as the page (CSS variables or the shared `chartTheme.ts`
  module).
- Multi-series charts use `series-1` through `series-5` in order. Single-series charts
  default to `primary`. Domain-specific charts (GIL vs KG breakdowns) use `gi` / `kg`.
- Cytoscape (v2) node/edge styling consumes the same semantic tokens via
  `cyGraphStylesheet.ts` so the graph matches panels and charts.
- **Dashboard charts (UXS-006):** Every chart on the Dashboard **must** expose a
  **written takeaway** (Tufte-style insight): either a dedicated line below the chart
  **or** the chart title itself when it states the quantitative conclusion (e.g. stage
  timings headline). Parents **must not** render a chart with data and no takeaway.
  If the data does not support a clear, honest takeaway, **that chart does not belong**
  on the Dashboard surface.

---

## Status bar {#status-bar}

Visual contract for the permanent bottom row (**`StatusBar.vue`**, **`data-testid="app-status-bar"`**). **What belongs in the bar** (corpus path, folder picker, health dot, rebuild indicator, persistence keys) is defined in [VIEWER_IA.md — Status bar](VIEWER_IA.md#status-bar). Here: **tokens and layout** only.

- **Height:** ~**36px** single row; background **`canvas`**; top border **`border`**
- **Corpus path field:** `surface` background, **`border`**; placeholder copy **Set corpus path…** (`muted`); full remaining width left of trailing affordances
- **Health dot:** **8px** filled circle; **`success`** / **`warning`** / **`danger`** intent tokens
- **Rebuild indicator (⚡):** **`warning`** when index stats recommend reindex (see VIEWER_IA)
- **Folder picker** (offline): icon control per shipped chrome; match **`surface`** / **`border`**

### Corpus path field — listing and artifacts

**Subject clearing and path-change cascade** (what invalidates the subject rail, tab handoffs): [VIEWER_IA.md — Corpus path change](VIEWER_IA.md#corpus-path-change).

- **Online (health ok):** Typing a corpus path in the **status bar** field triggers listing via
  `GET /api/artifacts` and loads every listed `.gi.json` / `.kg.json` into the merged
  graph automatically. **List** opens the **Corpus artifacts** dialog (`data-testid="artifact-list-dialog"`)
  with **All** / **None** / **Load into graph**; **Load into graph** applies the current selection
  and switches to **Graph** when ready.
- **Offline:** **Files** on the **status bar** (or **Choose files…** inside the **Health** dialog)
  loads local GI/KG without the server. The value persists in **`localStorage`** key **`ps_corpus_path`**.

### Corpus sources dialog (Feeds | Operator YAML) {#corpus-sources-dialog}

Modal **`data-testid="status-bar-sources-dialog"`** (title **Corpus sources**). Opens from status bar **Feeds** / **Operator YAML** when the matching capability is true on **`GET /api/health`**.

- **Tabs:** **Feeds** (`data-testid="sources-dialog-tab-feeds"`) | **Operator YAML** (`data-testid="sources-dialog-tab-operator"`). Switching tabs **lazy-loads** only that tab’s API so a broken operator file does not block the Feeds editor.
- **Feeds tab:** Internal sub-tabs **Manage** (`sources-dialog-feeds-panel-list`) vs **Raw JSON** (`sources-dialog-feeds-panel-json`). **Manage:** **`sources-dialog-feeds-add-url`** + **`sources-dialog-feeds-add-btn`** (**Add feed**) appends one URL at a time (deduped); per-row **Edit** / **Delete**; each change persists via **`PUT /api/feeds`**. **Raw JSON:** **`sources-dialog-feeds-textarea`** + **Apply JSON**. Persisted as corpus **`feeds.spec.yaml`** (same as CLI **`--feeds-spec`**, RFC-077 / #626). Help copy states feeds are **not** edited in Operator YAML.
- **Operator YAML tab:**
  - Shows resolved file path from **`GET /api/operator-config`** (`operator_config_path`) in small muted text above the controls.
  - **Profile** row: native **`<select>`** **`data-testid="sources-dialog-profile-select"`**. Options: **`None`** (empty value — no `profile:` line persisted), then every name in **`available_profiles`** from **GET** (sorted server list). If the on-disk file references a **`profile:`** name **not** in that list (custom / stale preset), show an extra option **`{name} (custom)`** so the value round-trips until the operator changes it. When **`viewer_operator.yaml`** is **missing or whitespace-only**, **GET** **`/api/operator-config`** seeds **`profile: cloud_balanced`** on disk if that stem exists under packaged **`config/profiles`** (otherwise content stays empty until the user saves).
  - **Overrides** editor: monospace **`sources-dialog-operator-textarea`** — YAML for keys **other than** top-level **`profile:`** (the **`<select>`** is the sole source of truth for `profile:` on **Save**). Any `profile:` line pasted into the textarea is **stripped on save** when resolving body text; **None** in the menu therefore always persists **without** a `profile:` key even if the textarea still contained a pasted `profile:` line. On load, the client splits file content into select + body (see `operatorYamlProfile.ts`).
  - **Save YAML** persists via **`PUT /api/operator-config`** with JSON `{ "content": "..." }`. Errors: **422** invalid YAML; **400** with `detail.keys` for forbidden secrets or forbidden feed keys (`rss`, `rss_url`, `rss_urls`, `feeds` at root). Server validation is **shallow** (top-level keys only), not a full duplicate of `Config` validation.
  - Help copy: secrets only via environment; packaged preset defaults merge first (#593), explicit keys in the file override; feeds belong in the Feeds tab / list file.

**Playwright map:** update [E2E surface map](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md) when `data-testid`s here change.

---

## Shell keyboard shortcuts

Implemented in `useViewerKeyboard.ts` (global handlers, not per-component ad hoc):

- **Slash** — When focus is not in an editable control: expand the left query column if
  needed, switch to **Search** mode, then focus `#search-q`.
- **Escape** — On the **Graph** main tab, when focus is not in an editable control: clear
  graph interaction / transient selection state.
- **G** / **L** on semantic search hits — Bound as **click targets on each result row**
  (Show on graph / open episode in subject panel), **not** as document-level shortcuts, so
  they do not fight the browser’s find shortcut or OS bindings.

---

## Accessibility

- **Focus:** Visible focus styles on all interactive elements (native or custom); do
  not remove outline without a replacement that meets contrast.
- **Contrast:** Aim for **WCAG 2.1 AA** for normal text against `canvas` / `surface`;
  verify `muted` on `surface` for small text. Foreground-pairing ensures each surface
  token has a pre-validated text color.
- **Motion:** Respect `prefers-reduced-motion` for non-essential animations.

---

## Tunable parameters

The token **architecture** (names, pairing convention, intent/domain split) is frozen.
The **values** below are deliberately open for experimentation during early v2
development. Adjust them in `tokens.css` (CSS custom properties) or via browser
DevTools; do not hard-code alternatives in component files.

| Parameter                               | Current value                              | Status | Notes                                                    |
| --------------------------------------- | ------------------------------------------ | ------ | -------------------------------------------------------- |
| UI font family                          | Inter                                      | Open   | Could try Geist, IBM Plex Sans, system stack             |
| Monospace font family                   | JetBrains Mono                             | Open   | Could try Fira Code, Source Code Pro                     |
| Base font size                          | 16px (1rem)                                | Open   | 14px more dense; 16px more readable                      |
| Border radius                           | 0.375rem (6px)                             | Open   | 0 = sharp/Palantir; 0.5rem = softer                      |
| Spacing base unit                       | 4px                                        | Open   | Could move to 6px or 8px for more air                    |
| Shadow depth                            | None (flat)                                | Open   | Subtle shadow = more depth/elevation cue                 |
| Surface gray palette                    | Blueprint-derived `#111418`..`#404854`     | Open   | Exact values may shift during contrast review            |
| Intent color hues                       | Blueprint-derived blue/green/orange/red    | Open   | Hues stable; saturation/lightness may tune               |
| Domain color hues (gi/kg)               | Green `#7dd3a0` / Purple `#c4a3ff`         | Frozen | Identity colors; do not change w/o UXS rev               |
| Token names                             | `canvas`, `surface`, `primary`, `gi`, etc. | Frozen | Names are the API; values are the theme                  |
| Pairing convention                      | Every surface gets `-foreground`           | Frozen | Structural rule; not negotiable                          |
| Intent/domain separation                | Intent for UI; domain for GIL/KG           | Frozen | Structural rule; not negotiable                          |
| Digest similarity strong floor          | **0.85**                                   | Open   | Strong vs good; constant DIGEST_SIMILARITY_STRONG_MIN    |
| Digest similarity good floor            | **0.70**                                   | Open   | Good vs weak; constant DIGEST_SIMILARITY_GOOD_MIN        |
| Recency dot rolling window              | **24h** from local **YYYY-MM-DD** midnight | Open   | Digest + Library episode rows; date-only API             |
| Library feed search threshold           | **15** feeds                               | Open   | Constant in LibraryView.vue; default 15 feeds trigger    |
| Graph default episode cap (initial)     | **15** episodes                            | Open   | GRAPH cap in graphEpisodeSelection.ts; graph-only lens   |
| Graph episode load recency score floor  | **0.2**                                    | Open   | ``GRAPH_SCORE_RECENCY_MIN``; linear mix with max **1.0** |
| Graph episode load topic cluster bonus  | **0.4**                                    | Open   | TOPIC_CLUSTER_BONUS; additive (graphEpisodeSelection.ts) |
| Graph episode load all-time decay days  | **90**                                     | Open   | ALL_TIME_DECAY_DAYS; trailing window from newest publish |
| Graph episode load GI density max weight| **0.4**                                    | Open   | GI_DENSITY_MAX; future per-episode coverage API cap      |
| Graph recency seed default              | **7d**                                     | Open   | Initial graph window; VIEWER_GRAPH_SPEC (init load)      |
| Dashboard GI coverage warn threshold    | **50%**                                    | Open   | Briefing + Coverage; UXS-006 §3.4                        |
| Dashboard index coverage warn threshold | **60%**                                    | Open   | Briefing + Coverage; UXS-006 §3.4                        |
| Dashboard action items max              | **3**                                      | Open   | Briefing card triage cap                                 |
| Dashboard Top voices limit              | **5**                                      | Open   | persons/top default limit                                |
| COSE ABOUT edge ideal length            | **80px**                                   | Open   | cyCoseLayoutOptions.ts; VIEWER_GRAPH_SPEC (styling)      |
| COSE MENTIONS edge ideal length         | **150px**                                  | Open   | cyCoseLayoutOptions.ts; VIEWER_GRAPH_SPEC (styling)      |
| Graph recency decay (opacity)           | **90 days**                                | Open   | VIEWER_GRAPH_SPEC §4.1 node recency vs episode pick      |
| Graph recency minimum opacity           | **0.4**                                    | Open   | VIEWER_GRAPH_SPEC §4.1                                   |
| Graph degree heat max degree            | **30**                                     | Open   | GraphCanvas.vue; VIEWER_GRAPH_SPEC §4.3                  |
| Graph label zoom (hide all labels)      | **0.5**                                    | Open   | Below: no labels (GraphCanvas.vue)                       |
| Graph label zoom (full labels)          | **1.0**                                    | Open   | Above: full node labels                                  |
| Graph compound fill opacity             | **0.06**                                   | Open   | TopicCluster compound; VIEWER_GRAPH_SPEC §3.6            |

**Code map:** Similarity floors live in `web/gi-kg-viewer/src/utils/digestRowDisplay.ts`; recency parsing and the rolling window live in `web/gi-kg-viewer/src/utils/digestRecency.ts`; the feed-count threshold constant lives in `LibraryView.vue`. Corpus graph episode pick (cap, recency + topic-cluster score, tie-break) lives in `web/gi-kg-viewer/src/utils/graphEpisodeSelection.ts`.

**Graph canvas / layout (open, not in table above):** They ship in code today but are
intentionally easy to tune without a UXS revision — see
[Viewer graph spec — Graph visual styling](../architecture/VIEWER_GRAPH_SPEC.md#graph-visual-styling) and [UXS-004](UXS-004-graph-exploration.md).

- **COSE semantic `idealEdgeLength`:** per `edgeType` in `cyCoseLayoutOptions.ts`, applied only
  after the intra-`tc:` topic-cluster branch (that branch stays highest priority).
- **COSE semantic `edgeElasticity`:** same file; inside `tc:` clusters the profile default applies.
- **Label zoom thresholds:** `GraphCanvas.vue` — no labels when zoom is strictly below **0.5**;
  short labels from **0.5** through **1.0**; full label above **1.0** (event-driven node classes,
  not Cytoscape min-zoom stylesheet selectors).
- **Topic degree heat `maxDegree`:** **30** in `GraphCanvas.vue` (WIP §4.3 cap).
- **Rail minimap (`GraphNeighborhoodMiniMap.vue`):** Reuses **`buildGiKgCyStylesheet`**
  **compact** + **`prefersReducedMotionQuery()`**, and **`syncGraphLabelTierClasses`**
  after **`layoutstop`** / **fit** so label-tier rules match that instance’s zoom.

### How to experiment

- **Browser DevTools:** Inspect `:root`, change any `--ps-*` variable live.
- **`tokens.css` presets:** Create alternate files (`theme-compact.css`,
  `theme-relaxed.css`) that override the open parameters. Import one at a time.
- **Pinia theme store (`stores/theme.ts`):** Sets CSS variables at runtime, enabling
  a theme switcher in the UI for side-by-side comparison.

When an open parameter is finalized after experimentation, update its status to
**Frozen** in this table and record the decision in the revision history.

---

## Visual references

No wireframes checked in yet. As the v2 UI stabilizes, add annotated screenshots or
Figma frames here to lock layout expectations for the main panels, graph canvas, and
search overlay.

---

## Acceptance criteria

- [ ] New viewer UI uses semantic tokens only (no stray hex in Vue/SFC or JS style
  objects)
- [ ] Theme support matches declared mode (both, system-driven, dark baseline)
- [ ] Light and dark values match the token table (or the table is updated with
  rationale)
- [ ] Every surface token uses its matching `-foreground` for text (no ad hoc text
  colors)
- [ ] Intent tokens (`primary`/`success`/`warning`/`danger`) are used for UI feedback;
  domain tokens (`gi`/`kg`) are used only for GIL/KG identity
- [ ] Key interactive states match this spec (hover, focus, disabled, error, empty,
  loading)
- [ ] Focus states visible on buttons, inputs, and graph controls
- [ ] Chart.js series use `series-1` through `series-5`; graph colors derive from
  palette
- [ ] Inter and JetBrains Mono load correctly (or system fallbacks render acceptably)
- [ ] RFC-062 implementation checklist references this UXS for theme work
- [ ] Tunable parameters table reflects current status (open values finalized -> frozen)

---

## Revision history

| Date | Change |
| --- | --- |
| 2026-04-20 | Tunables: graph initial-load **episode scoring** (recency floor/max, topic-cluster bonus, 90d all-time decay, future GI density cap) in `graphEpisodeSelection.ts` + table rows |
| 2026-04-19 | Tunable: **Graph default episode cap** (15) for corpus graph initial load (`VIEWER_GRAPH_SPEC` initial load) |
| 2026-04-19 | Open tunables for digest similarity tiers, recency dot window, Library feed-search threshold (#610) |
| 2026-04-03 | Initial draft |
| 2026-04-03 | Added theme support, key states, boundary note, visual refs section |
| 2026-04-03 | Blueprint gray scale, shadcn pairing, intent/domain split, Inter + JetBrains Mono, chart series tokens, design-reference citations |
| 2026-04-03 | Added Tunable parameters section (frozen vs open knobs); RFC-062 and PRD cross-references updated to document theme preset system |
| 2026-04-06 | Status Active — viewer v2 implements this contract per RFC-062 |
| 2026-04-13 | UXS-001 hub; UXS-002 through UXS-008 for feature specs |
| 2026-04-13 | Added shared InsightCard component (UXS-007/009/010 alignment) |
| 2026-04-19 | Summary pointer to InsightCard subsection for discoverability |
| 2026-04-17 | Open tunables for graph COSE semantics, label zoom tiers, Topic degree heat cap (`VIEWER_GRAPH_SPEC`) |
| 2026-04-17 | Graph bullets: rail minimap shares reduced-motion + `syncGraphLabelTierClasses` with main canvas |
| 2026-04-19 | Corpus path **List** opens status-bar **artifact-list-dialog** (not Dashboard workspace) |
| 2026-04-19 | Corpus path List/Load wording; shell keyboard shortcuts (slash, Escape, row G/L) |
| 2026-04-19 | **Status bar** section with `#status-bar`; scope non-goal + historical note for shell IA; corpus path subsection links VIEWER_IA flows |
| 2026-04-20 | **Corpus sources dialog** (`#corpus-sources-dialog`): Feeds vs Operator YAML; profile `<select>` + overrides textarea; `sources-dialog-profile-select`; RFC-077 / #593 merge semantics |
| 2026-04-21 | Operator save: profile `<select>` sole source of truth; None strips pasted `profile:` in textarea; shallow validation called out |
| 2026-04-21 | Corpus sources Feeds: **Add feed** one URL at a time + **Manage** / **Raw JSON** sub-tabs; operator **GET** seeds `profile: cloud_balanced` when `viewer_operator.yaml` is missing/whitespace-only and preset exists; client parses FastAPI JSON `detail` for clearer errors |
| 2026-04-19 | Shell IA pointer to VIEWER_IA; `warning` vs `kg` clusters; Dashboard charts require a **written takeaway** (dedicated line or chart title); tunables: graph 7d seed, COSE lengths, recency decay, label tiers, compound opacity, Dashboard thresholds / caps |
