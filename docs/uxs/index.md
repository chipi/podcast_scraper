# UX Specifications (UXS)

## Purpose

UX Specifications define the **experience and visual contract** for user-facing surfaces:
layout density, **semantic design tokens** (color, type, spacing), key interaction patterns,
accessibility expectations, and checklists for review.

They complement:

- **PRDs** -- what and why (requirements, success metrics)
- **RFCs** -- technical how (APIs, architecture, implementation)
- **ADRs** -- durable architectural *decisions* (token detail stays in UXS, not ADRs)

## When to create a UXS

Create a UXS when a feature introduces or significantly changes **UI** (web viewer, local
server UI, dashboards maintained in-repo). Skip for pure CLI behavior unless you explicitly
standardize terminal presentation.

## How UXS fits the lifecycle

1. PRD may list **Related UX specs** for features with a meaningful UI.
2. RFC references the same UXS so implementation maps tokens to code (`tailwind.config`,
   `:root` CSS variables, Cytoscape/Chart.js themes).
3. GitHub issues can link PRD + RFC + UXS for a single source of truth.

See [Engineering Process](../guides/ENGINEERING_PROCESS.md) for the full PRD / RFC / ADR flow.

## UXS vs RFC boundary

A UXS owns the **static visual contract**: tokens, type scale, layout grid, component
appearance, and accessibility targets. The RFC owns **behavioral rules**: animation
timing, debounce intervals, resize/collapse logic, data-fetching strategies, and keyboard
shortcut maps.

**Rule of thumb:** if the specification changes a CSS variable or a Tailwind class, it
belongs in the UXS. If it changes a `setTimeout`, an event handler, or a state machine
transition, it belongs in the RFC. When a section could live in either document, prefer
the RFC and reference the UXS for the visual aspect only.

This boundary will sharpen over time. When reviewing a UXS, flag any bullet that describes
*when* or *how often* something happens (rather than *how it looks*) as a candidate to
move into the RFC.

**GI/KG viewer — IA vs visuals vs behavior:** For **`web/gi-kg-viewer/`**, **[VIEWER_IA.md](VIEWER_IA.md)** is the canonical spec for **shell information architecture** (regions, the three navigation axes, persistence and clearing, first-run behavior, and cross-surface flows). **UXS** (**[UXS-001](UXS-001-gi-kg-viewer.md)** and feature **UXS-002+**) define the **static visual contract** (tokens, typography, density, component styling) for those regions and surfaces. **[RFC-062: GI/KG viewer v2](../rfc/RFC-062-gi-kg-viewer-v2.md)** (and feature RFCs) own **behavioral** rules: animation timing, debounce, resize/collapse logic, and keyboard shortcut maps. Do not duplicate full shell IA in UXS-001 — link **VIEWER_IA** instead.

## Status lifecycle

| Status         | Meaning                                                                |
| -------------- | ---------------------------------------------------------------------- |
| **Draft**      | Under review; tokens and rules may change before implementation begins |
| **Active**     | Authoritative; implementations should conform to this spec             |
| **Superseded** | Replaced by a newer UXS (link the replacement in revision history)     |

## Living documents and ship boundary

UX specifications are **living documents** for product UI: they are created or extended
**when building new surfaces**, then **updated whenever shipped behavior changes** so
**Active** UXS stays aligned with what users see in the built app.

**Draft versus Active:** **Draft** UXS may trail or lead `main` while work is in motion;
temporary gaps between a branch and the index are normal. When a feature **merges**,
resolve Draft work (promote, fold into an existing Active UXS, or supersede) so
**Active** UXS does **not** stay authoritative for chrome that never shipped.

**RFCs:** RFCs record technical design and often keep a longer historical arc (for example
**Completed** viewer RFCs). UXS tracks the **current** visible and interaction contract
for reviewers and E2E more tightly; keep both linked from issues and PRs.

**Same delivery:** Prefer updating UXS in the **same PR** as viewer changes (or an
immediate docs follow-up in the same release train). See [Conventions](#conventions)
for the E2E surface map and UXS order.

## UXS architecture

**Shell information architecture** (regions, axes, persistence, first-run states)
is centralized in **[VIEWER_IA.md](VIEWER_IA.md)**. UXS-001 holds tokens and
shared components; VIEWER_IA holds *where* things live in the app chrome.

The viewer UXS is split into a **shared design system hub** (UXS-001) and
**per-feature specs** (UXS-002+). Each feature UXS references UXS-001
for tokens, typography, and shared conventions. This keeps individual specs short
(2-4 pages) while the shared foundation stays in one place.

Draft UXS documents (UXS-007 through UXS-010) exist in this directory but are **not
indexed** until promoted to Active. They are discoverable by filename.

## Conventions

- **Browser E2E + UXS (GI/KG viewer):** Anyone changing **viewer UX** (including AI agents)
  should treat **docs + tests** as part of the same change, in order:
  **(1)** [E2E surface map](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md)
  (Playwright contract: `getByRole` strings, hooks, entry flows),
  **(2)** `web/gi-kg-viewer/e2e/*.spec.ts` and helpers -- run **`make test-ui-e2e`**,
  **(3)** the relevant feature UXS when **tokens, layout density, or stated experience rules**
  change (not only when tests fail). See
  [E2E Testing Guide](../guides/E2E_TESTING_GUIDE.md#when-you-change-viewer-ux-required-workflow)
  and [GitHub #509](https://github.com/chipi/podcast_scraper/issues/509). Meta / rollout for **VIEWER_IA inside UXS**: [GitHub #623](https://github.com/chipi/podcast_scraper/issues/623).
- **IDs:** `UXS-NNN` with three digits (see table below for the next number).
- **Files:** `docs/uxs/UXS-NNN-kebab-case-slug.md`
- **Length:** Keep specs short (about two to four printed pages). Split or move narrative
  to an RFC if a doc grows too large; keep UXS as the token table and principles.

## Active UX specifications

Authoritative specs; current implementations should conform
(see [status lifecycle](#status-lifecycle)).

| UXS | Title | Related PRDs / RFCs | Description |
| --- | ----- | ------------------- | ----------- |
| [UXS-001](UXS-001-gi-kg-viewer.md) | GI/KG Viewer (Shared Design System) | PRD-003, PRD-017, PRD-019; RFC-062 | Shared tokens, typography, layout, states, accessibility, components; shell regions / navigation → [VIEWER_IA](VIEWER_IA.md) |
| [UXS-002](UXS-002-corpus-digest.md) | Corpus Digest | PRD-023; RFC-068 | Digest tab: topic bands, recent episodes, rolling window |
| [UXS-003](UXS-003-corpus-library.md) | Corpus Library | PRD-022; RFC-067 | Library tab: feed/episode catalog, Episode subject rail, filters |
| [UXS-004](UXS-004-graph-exploration.md) | Graph Exploration | PRD-024; RFC-069, RFC-076 | Graph chrome: toolbar, gesture discovery overlay, minimap, degree filter, node detail |
| [UXS-005](UXS-005-semantic-search.md) | Semantic Search | PRD-021; RFC-061 | Search panel: query, advanced filters, result cards, insights modal |
| [UXS-006](UXS-006-dashboard.md) | Dashboard | PRD-025; RFC-071 | Dashboard tab: briefing + Coverage / Intelligence / Pipeline (normative spec in UXS-006) |

## Templates

- [UXS template](UXS_TEMPLATE.md) -- copy when adding `UXS-NNN-*.md`
