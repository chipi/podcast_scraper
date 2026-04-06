# UXS-001: GI / KG viewer

- **Status**: Draft
- **Authors**: Podcast Scraper Team
- **Related PRDs**:
  - [PRD-003: User Interfaces & Configuration](../prd/PRD-003-user-interface-config.md)
  - [PRD-017: Grounded Insight Layer](../prd/PRD-017-grounded-insight-layer.md)
  - [PRD-019: Knowledge Graph Layer](../prd/PRD-019-knowledge-graph-layer.md)
- **Related RFCs**:
  - [RFC-062: GI/KG viewer v2](../rfc/RFC-062-gi-kg-viewer-v2.md)
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

- Viewer shell: header, panels, file/folder pickers, banners
- Graph visualization chrome (legends, selection affordances) at the token level
- Chart.js panels that summarize graph or artifact stats

**Non-goals:**

- Brand marketing pages, email templates, or MkDocs theme
- CLI ANSI styling (covered only if a separate UXS is added)

**Boundary note:** This UXS covers the **static visual contract** (tokens, layout, component
appearance, accessibility targets). Behavioral rules (animation timing, debounce intervals,
resize/collapse logic, keyboard shortcuts) belong in
[RFC-062](../rfc/RFC-062-gi-kg-viewer-v2.md). See the
[UXS vs RFC boundary](index.md#uxs-vs-rfc-boundary) guidance.

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

No wireframes yet. When v2 implementation begins, attach annotated screenshots or Figma
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
