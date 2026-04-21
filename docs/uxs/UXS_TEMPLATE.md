# UXS-NNN: [Feature or Surface Name]

- **Status**: Draft | Active | Superseded
- **Authors**: [Names]
- **Related PRDs**:
  - `docs/prd/PRD-XXX-feature.md` (if applicable)
- **Related RFCs**:
  - `docs/rfc/RFC-XXX-feature.md` (if applicable)
- **Related UX specs**:
  - `docs/uxs/UXS-XXX-other.md` (if this doc extends or conflicts with another UXS)
- **Shell IA (GI/KG viewer only):**
  - [VIEWER_IA.md](VIEWER_IA.md) — canonical shell regions, navigation axes, persistence; link when this UXS describes a surface inside the viewer chrome
- **Related issues**:
  - GitHub #NNN — [short label]
- **Implementation paths** (where tokens and styles should land):
  - [e.g. `web/.../styles.css`, `tailwind.config.ts`, Cytoscape theme module]

## Summary

[2-4 sentences: what UI surface this spec covers and what problem the visual/UX contract
solves.]

## Principles

- [Principle 1 — e.g. data-dense, minimal chrome]
- [Principle 2 — e.g. keyboard-first for power users]
- [Principle 3]

## Scope

**In scope:**

- [Screen or panel 1]
- [Screen or panel 2]

**Non-goals:**

- [Explicitly out of scope 1]
- [Explicitly out of scope 2]

**Boundary note:** This UXS covers the **static visual contract** (tokens, layout, component
appearance, accessibility targets). Behavioral rules (animation timing, debounce intervals,
resize/collapse logic, keyboard shortcuts) belong in the related RFC. See the
[UXS vs RFC boundary](index.md#uxs-vs-rfc-boundary) guidance.

## Theme support

[Declare which mode is the source of truth and whether both are required.]

- **Mode:** dark only | light only | both (follows system) | both (manual toggle)
- **Primary palette:** [dark | light] — the mode used as the design baseline
- **Breakpoints:** [desktop only | responsive — list breakpoints if applicable]

## Semantic color tokens

Use **semantic names** in code (CSS custom properties, Tailwind theme keys, or shared
constants). Do not scatter raw hex in components except in the single token definition
layer. Every surface token should have a matching `-foreground` text token so contrast is
validated at the token level.

### Surface tokens

| Token                | Dark (example) | Light (example) | Usage                |
| -------------------- | -------------- | --------------- | -------------------- |
| `canvas`             | `#111418`      | `#F6F7F9`       | Page background      |
| `canvas-foreground`  | `#E5E8EB`      | `#1C2127`       | Text on canvas       |
| `surface`            | `#1C2127`      | `#FFFFFF`       | Panels, cards        |
| `surface-foreground` | `#E5E8EB`      | `#1C2127`       | Text on surface      |
| `elevated`           | ...            | ...             | Popovers, modals     |
| `overlay`            | ...            | ...             | Hover, active rows   |
| `border`             | ...            | ...             | Dividers, inputs     |

### Text tokens

| Token      | Dark (example) | Light (example) | Usage                   |
| ---------- | -------------- | --------------- | ----------------------- |
| `muted`    | ...            | ...             | Secondary labels        |
| `disabled` | ...            | ...             | Disabled controls       |
| `link`     | ...            | ...             | Inline links            |

### Intent tokens (UI actions and feedback)

Separate from domain tokens. Used for buttons, alerts, and status indicators.

| Token                | Dark (example) | Light (example) | Usage             |
| -------------------- | -------------- | --------------- | ----------------- |
| `primary`            | ...            | ...             | Primary actions   |
| `primary-foreground` | ...            | ...             | Text on primary   |
| `success`            | ...            | ...             | Positive feedback |
| `warning`            | ...            | ...             | Caution states    |
| `danger`             | ...            | ...             | Errors            |

### Domain tokens (optional -- for feature-specific identity)

Domain tokens are visualization-level cues that distinguish feature areas (e.g. GIL vs KG).
They are separate from generic UI intents.

| Token | Dark (example) | Light (example) | Usage               |
| ----- | -------------- | --------------- | ------------------- |
| ...   | ...            | ...             | [feature-X cue]     |

### Chart series tokens (optional)

For multi-series charts. Derive from the palette above; avoid introducing unrelated colors.

| Token      | Dark (example) | Light (example) | Usage         |
| ---------- | -------------- | --------------- | ------------- |
| `series-1` | ...            | ...             | First series  |
| `series-2` | ...            | ...             | Second series |
| ...        | ...            | ...             | ...           |

## Typography

- **UI font:** [e.g. Inter, system-ui, sans-serif]
- **Monospace font:** [e.g. JetBrains Mono, ui-monospace, monospace]
- **Scale:** [e.g. xs / sm / base / lg / xl — map to rem or Tailwind sizes]
- **Font weights:** [e.g. 400 regular, 500 medium, 600 semibold, 700 bold]
- **Dense UI** (tables, metadata): [rules]

## Layout and spacing

- **Base unit:** [e.g. 4px]
- **Max content width:** [value]
- **Major regions:** [header / sidebar / main — behavior]

## Key states

Define the visual treatment for interactive states that are most likely to drift
during implementation. Behavioral timing (debounce, animation duration) belongs in the
RFC.

- **Hover:** [e.g. show `overlay` background, underline links]
- **Active / pressed:** [e.g. show `elevated` background]
- **Focus:** [e.g. 2px solid `primary` ring, 2px offset]
- **Disabled:** [e.g. `disabled` text color, 40% opacity]
- **Loading:** [e.g. skeleton placeholder using `surface`/`border`]
- **Empty state:** [e.g. centered `muted` text, optional illustration]
- **Error state:** [e.g. `danger` border + inline message]

## Components (standardize only what matters now)

- **Buttons:** [primary / secondary / danger — token mapping]
- **Inputs:** [focus, error state]
- **Banners / alerts:** [intent-based variants]
- [Other]

## Charts and graph

- Chart.js (or equivalent) and graph libraries **must** use colors derived from the same
  semantic palette as the rest of the UI (e.g. read CSS variables or a shared `theme`
  module).
- Multi-series charts use `series-*` tokens in order.
- [Node/edge color rules for graph, if applicable]

## Accessibility

- **Focus:** [visible focus ring requirement]
- **Contrast:** [target, e.g. WCAG AA for text; foreground pairing ensures pre-validated
  contrast]
- **Motion:** [reduced motion policy]

## Tunable parameters (optional)

If the design is still being explored, list which token **values** are open for
experimentation vs which token **names and conventions** are frozen. This helps
implementers know what they can safely adjust in DevTools or preset files and what
requires a UXS revision.

| Parameter | Current value | Status | Notes |
| --------- | ------------- | ------ | ----- |
| [e.g. UI font family] | [e.g. Inter] | Open | [alternatives considered] |
| [e.g. Token names] | [e.g. `canvas`, `primary`] | Frozen | [API — do not rename] |

### How to experiment

[Brief instructions — e.g. swap CSS preset files, use DevTools, optional Pinia store.]

When an open parameter is finalized, update its status to **Frozen** and note the
decision in the revision history.

## Visual references (optional)

Attach annotated screenshots, wireframes, or mockups here when available. Visual
references are not mandatory but reduce ambiguity for layout and component appearance.

## Acceptance criteria (for issues / review)

Copy into GitHub issues or PR checklists as needed.

- [ ] New UI uses semantic tokens only (no one-off hex in components)
- [ ] Every surface uses its matching `-foreground` for text
- [ ] Intent tokens used for UI feedback; domain tokens for feature identity only
- [ ] Theme support matches declared mode (see Theme support section)
- [ ] Key interactive states match this spec (hover, focus, disabled, error)
- [ ] Focus states visible on interactive elements
- [ ] Charts/graph colors use series tokens or palette colors
- [ ] Tunable parameters table reflects current status (open → frozen as decisions land)

## Revision history

| Date       | Change        |
| ---------- | ------------- |
| YYYY-MM-DD | Initial draft |
