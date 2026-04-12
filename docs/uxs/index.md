# UX Specifications (UXS)

## Purpose

UX Specifications define the **experience and visual contract** for user-facing surfaces:
layout density, **semantic design tokens** (color, type, spacing), key interaction patterns,
accessibility expectations, and checklists for review.

They complement:

- **PRDs** — what and why (requirements, success metrics)
- **RFCs** — technical how (APIs, architecture, implementation)
- **ADRs** — durable architectural *decisions* (token detail stays in UXS, not ADRs)

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

## Status lifecycle

| Status         | Meaning                                                                |
| -------------- | ---------------------------------------------------------------------- |
| **Draft**      | Under review; tokens and rules may change before implementation begins |
| **Active**     | Authoritative; implementations should conform to this spec             |
| **Superseded** | Replaced by a newer UXS (link the replacement in revision history)     |

## Conventions

- **Browser E2E + UXS (GI/KG viewer):** Anyone changing **viewer UX** (including AI agents) should
  treat **docs + tests** as part of the same change, in order:
  **(1)** [E2E surface map](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md)
  (Playwright contract: `getByRole` strings, hooks, entry flows),
  **(2)** `web/gi-kg-viewer/e2e/*.spec.ts` and helpers — run **`make test-ui-e2e`**,
  **(3)** [UXS-001](UXS-001-gi-kg-viewer.md) when **tokens, layout density, or stated experience rules**
  change (not only when tests fail). See [E2E Testing Guide — workflow](../guides/E2E_TESTING_GUIDE.md#when-you-change-viewer-ux-required-workflow)
  and [GitHub #509](https://github.com/chipi/podcast_scraper/issues/509).
- **IDs:** `UXS-NNN` with three digits (see table below for the next number).
- **Files:** `docs/uxs/UXS-NNN-kebab-case-slug.md`
- **Length:** Keep specs short (about two to four printed pages). Split or move narrative
  to an RFC if a doc grows too large; keep UXS as the token table and principles.

## Active UX specifications

Authoritative specs; current implementations should conform (see [status lifecycle](#status-lifecycle)).

| UXS | Title | Related PRDs / RFCs | Description |
| --- | ----- | ------------------- | ----------- |
| [UXS-001](UXS-001-gi-kg-viewer.md) | GI / KG viewer | PRD-003, PRD-017, PRD-019, PRD-021, PRD-022, PRD-023, PRD-025; [RFC-062](../rfc/RFC-062-gi-kg-viewer-v2.md), [RFC-071](../rfc/RFC-071-corpus-intelligence-dashboard-viewer.md), [RFC-067](../rfc/RFC-067-corpus-library-api-viewer.md), [RFC-068](../rfc/RFC-068-corpus-digest-api-viewer.md) | Tokens + **Dashboard** (PRD-025 / RFC-071) + **Library** + **Digest** (PRD-023 / RFC-068); digest discovery on **Digest** only |

## Draft UX specifications

None. Add new specs here while status is **Draft**; move rows to **Active** when the spec is frozen and implementation has landed (or is in progress against a locked token set).

## Templates

- [UXS template](UXS_TEMPLATE.md) — copy when adding `UXS-NNN-*.md`
