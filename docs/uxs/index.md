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

- **IDs:** `UXS-NNN` with three digits (see table below for the next number).
- **Files:** `docs/uxs/UXS-NNN-kebab-case-slug.md`
- **Length:** Keep specs short (about two to four printed pages). Split or move narrative
  to an RFC if a doc grows too large; keep UXS as the token table and principles.

## Active and draft UX specs

| UXS | Title | Status | Related PRDs / RFCs | Description |
| --- | ----- | ------ | ------------------- | ----------- |
| [UXS-001](UXS-001-gi-kg-viewer.md) | GI / KG viewer | Draft | PRD-003, PRD-017, PRD-019; RFC-062 | Visual and token contract for the local GI/KG visualization surface |

## Templates

- [UXS template](UXS_TEMPLATE.md) — copy when adding `UXS-NNN-*.md`
