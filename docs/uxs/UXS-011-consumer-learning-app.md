# UXS-011: Consumer Learning App — Design System + Player

- **Status**: Draft
- **Authors**: Marko
- **Related PRDs**:
  - `docs/prd/PRD-035-learning-platform.md` (parent vision)
  - `docs/prd/PRD-038-catalog.md`, `docs/prd/PRD-039-player.md` (the first surfaces this skins)
- **Related RFCs**:
  - `docs/rfc/RFC-099-learning-platform-consumer-client.md` (the app this spec dresses — owns behaviour)
  - `docs/rfc/RFC-098-learning-platform-foundation.md` (the `/api/app/*` data this UI renders)
  - `docs/rfc/RFC-100-audio-bridge-subsystem.md` (origin-audio source for the Player)
- **Related UX specs**:
  - `docs/uxs/UXS-001-gi-kg-viewer.md` — the **operator** viewer design system. This is a **separate**
    design system for the **consumer** app (PRD-035 D3); tokens here are independent and must not be
    assumed to match UXS-001.
- **Related issues**:
  - GitHub #911 (Epic 1 — foundation), Epic 2 (consumer app — to be opened)
- **Implementation paths** (where tokens and styles should land):
  - `app/` (new top-level Vue 3 project — RFC-099 §1)
  - `app/src/styles/tokens.css` (`:root` CSS custom properties — the single token layer)
  - `app/tailwind.config.ts` (theme keys mapped to the CSS variables, if Tailwind is adopted)

## Summary

This is the **shared visual contract for the new consumer Learning App** (`app/`, RFC-099) — a
separate design system from the operator GI/KG viewer (UXS-001). The chosen identity is **Editorial
Bold**: a dark-primary, high-contrast, type-led aesthetic that signals "a serious thing you learn
from," not a jukebox. This doc holds the design-system foundation (tokens, typography, layout,
states, components, accessibility, i18n) **and** the first concrete surface — the **Player** (PRD-039).
Catalog / Discovery / Capture surfaces are added as `UXS-012+` when those surfaces are built.

## Principles

- **Editorial, not consumerist** — big expressive display type and a per-show colour field do the
  heavy lifting; chrome is minimal. The UI should read like a publication, reinforcing
  learning-over-consumption (PRD-035 thesis).
- **The artwork zone is a live intelligence surface, not decoration** — speaking-now, a grounding
  badge, and the insight surfacing *at this moment* live on the show colour field. This is where the
  Player borrows an **adaptive, immersive treatment** (colour derived from the episode artwork) from
  the "cinematic" exploration — concentrated on the one screen that earns it.
- **Transcript is a first-class citizen, balanced with playback** — balanced split: artwork +
  controls above, synced transcript below (single column on mobile; two columns on desktop). The
  synced segment is always legible and in view.
- **Degrade with dignity** — every intelligence cue (insight, grounding, topics, persons) is optional
  and disappears cleanly when its artifact is absent; the core listen + read experience never breaks.
- **Accessible and localisable from line one** — WCAG 2.1 AA, full keyboard operability, reduced-motion
  respect, and no hard-coded copy (RFC-099 §6). These are acceptance criteria, not follow-ups.

## Scope

**In scope:**

- The consumer-app **design-system foundation** (tokens, type, layout grid, states, core components).
- The **Player** surface visual contract (PRD-039): masthead, intelligence artwork zone, scrubber +
  controls, synced transcript list, knowledge dock.

**Non-goals:**

- Catalog, Discovery, Capture, and the consumer Corpus surface — separate `UXS-012+` specs (this doc
  defines the tokens they will inherit).
- The operator GI/KG viewer (UXS-001 / VIEWER_IA) — different audience, different design system.
- Behavioural rules: transcript-sync timing, autoscroll re-enable delay, scrape-progress polling,
  keyboard-shortcut maps, animation durations — all owned by **RFC-099** (see boundary below).

**Boundary note:** This UXS covers the **static visual contract** (tokens, layout, component
appearance, accessibility targets). Behavioural rules (animation timing, debounce intervals,
autoscroll/seek logic, keyboard shortcuts) belong in **RFC-099**. See the
[UXS vs RFC boundary](index.md#uxs-vs-rfc-boundary).

## Theme support

- **Mode:** dark only for the MVP (the Editorial Bold baseline is designed dark-first). Token **names**
  are structured so a light theme can be added later without renaming — light is a post-MVP fast-follow,
  not a v2.7 commitment.
- **Primary palette:** dark — the design baseline.
- **Breakpoints:** responsive, mobile-first. `sm` 0–599 (single column, full-viewport player),
  `md` 600–1023 (single column, wider gutters), `lg` 1024+ (two-column player: transcript main +
  Knowledge Panel rail).

## Semantic color tokens

Use **semantic names** in code (CSS custom properties / Tailwind theme keys). No raw hex in
components except in the single token layer (`app/src/styles/tokens.css`). Every surface token has a
matching `-foreground` so contrast is validated at the token level.

The **accent is per-show adaptive**: `--accent` is a runtime variable set from the episode/show
artwork (with a brand default — "Ember"). Components reference `--accent`; they never hard-code the
show colour. A guardrail clamps the derived accent to a minimum contrast against `surface` (see
Accessibility).

### Surface tokens

| Token                | Dark                    | Usage                                   |
| -------------------- | ----------------------- | --------------------------------------- |
| `canvas`             | `#0E0D10`               | Page background                         |
| `canvas-foreground`  | `#F4F1EA`               | Text on canvas (warm off-white "paper") |
| `surface`            | `#161419`               | Cards, panels                           |
| `surface-foreground` | `#F4F1EA`               | Text on surface                         |
| `elevated`           | `#1F1B24`               | Popovers, sheets, dock                  |
| `overlay`            | `rgba(244,241,234,.06)` | Hover / active rows                     |
| `border`             | `#272430`               | Dividers, hairline rules, inputs        |

### Text tokens

| Token      | Dark            | Usage                                 |
| ---------- | --------------- | ------------------------------------- |
| `muted`    | `#9C97A6`       | Secondary labels, inactive transcript |
| `disabled` | `#6E6A78`       | Disabled controls, faint meta         |
| `link`     | `var(--accent)` | Inline links                          |

### Intent tokens (UI actions and feedback)

| Token                | Dark            | Usage                                         |
| -------------------- | --------------- | --------------------------------------------- |
| `primary`            | `var(--accent)` | Primary actions (per-show)                    |
| `primary-foreground` | `#1A0E08`       | Text/icon on primary fill                     |
| `brand-default`      | `#FF6A3D`       | "Ember" — accent fallback when no show colour |
| `success`            | `#3FB984`       | Positive feedback                             |
| `warning`            | `#E8B339`       | Caution (pending, partial)                    |
| `danger`             | `#F0533F`       | Errors                                        |

### Domain tokens (knowledge-layer identity)

Domain cues that mark intelligence provenance — separate from generic UI intents. They keep the GIL /
KG / grounding semantics visually consistent with the operator stack's meaning without copying its hues.

| Token      | Dark            | Usage                                           |
| ---------- | --------------- | ----------------------------------------------- |
| `grounded` | `#7BE6B0`       | "N% grounded" badge, grounded-quote affordances |
| `insight`  | `var(--accent)` | GIL insight markers / "insight surfacing now"   |
| `topic`    | `#C9B6FF`       | KG topic chips                                  |
| `person`   | `#FFB37A`       | Person chips / speaker emphasis                 |

## Typography

- **Display font (editorial headline):** a heavy grotesque used for episode/show titles and section
  mastheads. Recommended: **Inter** at weight 800, tight tracking (`-0.025em`) for the MVP (already
  ubiquitous, variable, free); upgrade to a licensed display face (e.g. a Söhne/Geist-class grotesque)
  is an **Open** tunable. Title case, never all-caps for the headline itself.
- **UI / body font:** `Inter, system-ui, sans-serif`.
- **Monospace:** `ui-monospace, "SF Mono", monospace` — timestamps and tabular numerics only.
- **Scale (rem):** `xs .6875` · `sm .8125` · `base .9375` · `lg 1.125` · `xl 1.375` · `display-1 1.875`
  · `display-2 2.5` (clamped responsively).
- **Weights:** 400 regular, 500 medium, 700 bold, 800 display.
- **Kickers / eyebrows / dock labels:** `xs`, weight 800, `letter-spacing .16em`, uppercase, in
  `--accent` or `muted`. This is the editorial signature — use sparingly and consistently.
- **Tabular numerics:** timestamps, durations, and the scrubber readout use `font-variant-numeric:
  tabular-nums` so digits don't jitter.

## Layout and spacing

- **Base unit:** 4px (`space-1`); the editorial rhythm leans on `space-4`/`space-5` gutters
  (16/20px) for breathing room.
- **Max content width:** 1200px on `lg`; the Player two-column splits transcript (≈60%) + Knowledge
  rail (≈40%).
- **Major regions (Player):** masthead → intelligence artwork zone → scrubber + controls → synced
  transcript list → knowledge dock. On `lg` the artwork zone + controls sit in the left rail head and
  the transcript scrolls beside the Knowledge Panel.
- **Hairline rules** (`border`) separate regions instead of heavy cards — part of the editorial feel.

## Key states

- **Hover:** `overlay` background on rows / cards; links gain an `--accent` underline.
- **Active / pressed:** `elevated` background; controls scale to ~0.97 (timing in RFC-099).
- **Focus:** 2px solid `--accent` ring, 2px offset, on every interactive element — always visible,
  never removed for mouse users.
- **Disabled:** `disabled` text colour, 45% opacity, no focus ring.
- **Loading:** skeleton blocks using `surface`/`border`; the transcript shows shimmer lines, the
  artwork zone shows the show colour field with a muted pulse.
- **Empty / degraded:** absent intelligence sections are **omitted**, not shown empty. All-absent
  Knowledge Panel shows one `muted` line: "Insights appear once this episode is processed."
- **Error:** `danger` hairline + inline `muted` message; audio failure surfaces a retry affordance,
  never a dead player.

### Player-specific states

- **Active transcript segment:** `surface-foreground` text at weight 600, a 3px `--accent` left rule,
  and a subtle `overlay` background. Inactive segments are `muted`.
- **Grounding badge:** `grounded` text on a translucent field; hidden when no grounding signal.
- **Insight surfacing "now":** the artwork-zone insight card swaps content as playback crosses an
  insight's anchor; the "now" kicker uses `insight`. (Swap timing → RFC-099.)
- **Scrape-pending episode (queued):** `warning` progress affordance inline; flips to playable on Ready.

## Components (standardize only what matters now)

- **Buttons:** primary (fill `--accent`, text `primary-foreground`), secondary (outline `border`,
  text `canvas-foreground`), ghost (text only). Pill radius for dock actions; circular for transport.
- **Transport controls:** play/pause as a circular outline button (editorial), skip-back 15 /
  skip-forward 30 as type-led glyphs, speed as a text pill in `--accent`.
- **Scrubber:** a 2px editorial rule (not a fat bar); progress in `--accent`; a small round thumb.
- **Chips:** topic (`topic`), person (`person`), grounding (`grounded`) — `xs`, rounded, low-fill.
- **Knowledge dock:** two equal cells split by a `border` hairline — "N insights" (`--accent`) and
  "Ask this episode". On `lg` this becomes the Knowledge Panel rail header.
- **Cards (Catalog, future):** hairline-separated rows over heavy boxes; artwork as a small editorial
  block with the episode number as an oversized faint numeral.

## Charts and graph

- No charts in the Player. The optional consumer KG browser (RFC-099 §8, P2+) reuses the RFC-069 graph
  toolkit but **must** read these tokens (e.g. `topic`, `person`, `--accent`) rather than the operator
  viewer palette, so the consumer aesthetic holds.

## Accessibility

- **Focus:** visible `--accent` focus ring on all interactive elements; logical tab order through
  masthead → controls → transcript → dock.
- **Contrast:** WCAG 2.1 AA for text. The per-show `--accent` is **clamped at runtime** to maintain
  ≥4.5:1 against `surface` for text uses and ≥3:1 for large/graphical uses; if a derived show colour
  fails, fall back toward `brand-default`. Every surface pairs with its `-foreground`.
- **Transcript sync & motion:** the now-playing segment is announced via an ARIA live region (polite);
  autoscroll respects `prefers-reduced-motion` (jump instead of smooth-scroll). Tap/seek targets are
  ≥44px.
- **Audio:** native controls remain keyboard-operable; speed and skip have accessible names and are
  reachable without a pointer.
- **i18n:** all copy via `vue-i18n`; layout is RTL-ready; dates/numbers locale-aware. The display face
  must cover required glyph ranges (a constraint on the font tunable).

## Tunable parameters (optional)

| Parameter                        | Current value                                | Status                                | Notes                                                              |
| -------------------------------- | -------------------------------------------- | ------------------------------------- | ------------------------------------------------------------------ |
| Display font family              | Inter 800, tight                             | Open                                  | Upgrade to a licensed grotesque considered; must cover i18n glyphs |
| `brand-default` accent ("Ember") | `#FF6A3D`                                    | Open                                  | Brand colour pending; used only when no show colour                |
| Per-show accent derivation       | from artwork, contrast-clamped               | Frozen (mechanism) / Open (algorithm) | The *clamp contract* is frozen; the extraction algorithm is open   |
| Token names                      | `canvas`, `surface`, `accent`, domain tokens | Frozen                                | API — do not rename                                                |
| Dark-only (MVP)                  | dark baseline                                | Open                                  | Light theme is a post-MVP fast-follow                              |

### How to experiment

Swap values in `app/src/styles/tokens.css` (`:root`) or via DevTools; the per-show `--accent` is set
on the player root element at runtime. Token **names and the contrast-clamp contract are frozen**;
values and the extraction algorithm are open until promoted.

## Visual references

Annotated phone mockups of the three explored directions live in
`docs/wip/player/mockups/` (HTML + PNG). **Direction B (Editorial Bold)** is the adopted baseline;
the Player's artwork zone additionally borrows the adaptive/immersive now-playing treatment explored in
Direction C. These are design aids (WIP), not shipped assets.

## Acceptance criteria (for issues / review)

- [ ] New UI uses semantic tokens only (no one-off hex in components; single token layer)
- [ ] Every surface uses its matching `-foreground` for text
- [ ] Per-show `--accent` is contrast-clamped; falls back to `brand-default` when it fails AA
- [ ] Intent tokens for UI feedback; domain tokens (`grounded`/`topic`/`person`/`insight`) for
      knowledge-layer identity only
- [ ] Dark baseline matches this spec; token names allow a future light theme without renames
- [ ] Active transcript segment uses the `--accent` left rule + weight-600 treatment and stays in view
- [ ] Key interactive states match (hover, active, focus ring, disabled, loading, error)
- [ ] Focus ring visible on all interactive elements; listen→capture path fully keyboard operable
- [ ] Autoscroll + insight-swap respect `prefers-reduced-motion`; now-playing segment in an ARIA live region
- [ ] All copy via `vue-i18n` (no hard-coded strings); layout RTL-ready
- [ ] Absent intelligence sections omit cleanly (no empty panels)
- [ ] Tunable parameters table reflects current status (open → frozen as decisions land)

## Revision history

| Date       | Change                                                                 |
| ---------- | ---------------------------------------------------------------------- |
| 2026-06-24 | Initial draft — Editorial Bold baseline (Direction B) + Player surface |
