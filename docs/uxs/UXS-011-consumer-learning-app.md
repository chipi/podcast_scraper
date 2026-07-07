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
  - `web/learning-player/` (new top-level Vue 3 project — RFC-099 §1)
  - `web/learning-player/src/styles/tokens.css` (`:root` CSS custom properties — the single token layer)
  - `web/learning-player/tailwind.config.ts` (theme keys mapped to the CSS variables, if Tailwind is adopted)

## Summary

This is the **shared visual contract for the new consumer Learning App** (`web/learning-player/`, RFC-099) — a
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
components except in the single token layer (`web/learning-player/src/styles/tokens.css`). Every surface token has a
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
- **Insights dock:** two cells — "N insights" (`--accent`) + "Ask this episode" — that open the
  **Insights** panel (titled "Insights" in the UI; shipped #1091). The panel is a single
  vertical column: Ask · Summary · **Topics & People (one merged, expandable row; chips → corpus
  search)** · Insights (grounded cards, `●` grounded marker) · More like this.
- **Episode card (Catalog + search):** hairline-separated row — artwork block + clean **lede**
  (summary title) + `date · duration` + a grounded **✦ insights icon** that reveals the full
  summary bullets on hover/tap. *No topic pills on the card.* (The oversized faint **numeral** is the
  Home **What's-new** ranked hero/rows, not the Catalog card — see UXS-012.)

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

Swap values in `web/learning-player/src/styles/tokens.css` (`:root`) or via DevTools; the per-show `--accent` is set
on the player root element at runtime. Token **names and the contrast-clamp contract are frozen**;
values and the extraction algorithm are open until promoted.

## Capture & Consolidation surfaces (P2 + P3 — shipped)

The "Remember" half of the app (PRD-040 Capture, PRD-041 Consolidation). All affordances below are
**auth-gated** — signed-out users see the app exactly as before (no capture controls, no scope
toggles). Everything is grounded (slug + timestamp) and extractive (**no request-time LLM**).

### Capture affordances (inline actions, never overlays)

Capture is a one-tap **inline** action on the surface you're already on — it never opens a modal
(contrast the EntityCard replace-in-panel pattern, UXS-014). Three entry points, one shared
bookmark glyph filled-when-saved:

- **Mark this moment** — a bookmark control in the Player hero (`PlayerView`). One tap captures the
  current content-time as a `moment` highlight (tagging the active speaker); a brief accent flash +
  a polite SR live-region announcement ("Moment saved") confirm. Idempotent monotonic add.
- **Save a transcript line / phrase** — a quiet per-line bookmark in `TranscriptList` (revealed on
  row hover/focus; `focus-visible` keeps it keyboard-reachable). With **no selection** it saves the
  whole line (toggles off on re-tap, `aria-pressed`); with an active **text selection inside the
  line** it captures that exact phrase (char offsets + verbatim quote) and always adds.
- **Save an insight** — a bookmark on each Knowledge-panel insight card, visually distinct from the
  favorites heart (favorites = a saved list; highlights = the personal-corpus material that feeds
  recall/resurfacing).

**Colour** — a fixed 5-token palette (amber · rose · sky · emerald · violet; `utils/highlightColors.ts`).
Set/cleared via a swatch row in the Highlights view; the chosen colour paints the highlight card's
left border. Colour names are exposed via `aria-label` (never colour-only meaning).

### Library tabs (the per-user hub)

The Library is tabbed; P2/P3 add two tabs alongside Saved · Knowledge · Queue · Recent:

- **Highlights** (`HighlightsView`) — captured moments/spans/insights grouped by episode (titles
  hydrated, slug fallback), each with jump-to-moment (`?t=`), a drift badge when the timestamp
  re-anchored on re-scrape, inline notes (add/edit/remove), a per-highlight colour swatch picker,
  a header **colour filter**, and a **Export Markdown** link.
- **Revisit** (`ResurfacingInbox`) — the spaced-resurfacing inbox (see below).

### Recall scope lens (Search) + your-corpus lens (entity cards)

A **stateful segmented toggle** (`role="tablist"`, the selected tab `aria-selected`) — a filter
state, not a new view:

- **Search** gains **Everything / My corpus**. "My corpus" runs grounded recall over the user's
  heard∪captured set (`scope=mine`); results still group by episode with jump-to-moment. Honest
  zero-coverage copy ("Nothing in your corpus on this yet — listen to or capture episodes to build
  it"), never a global fallback.
- **EntityCardBody** gains **All / My corpus**. "My corpus" refetches the person/topic card scoped
  to the episodes you've heard — the "you also heard them in …" connection.

### Resurfacing inbox (Revisit tab)

Past highlights resurfaced on a spaced ladder (2d/1w/1mo/3mo, computed on read). Each card shows a
deterministic **reflection prompt** (no LLM), the highlight, a one-tap **jump-to-moment**, and a
**"Got it"** dismiss (advances the ladder). A header **Pause/Resume** control governs pacing;
paused or nothing-due shows an honest empty state.

### States & a11y for these surfaces

- Every capture/scope control is a real `<button>` with a clear `aria-label`/`aria-pressed`/
  `aria-selected`; the transcript save is reachable via keyboard despite its hover-quiet styling.
- Capture confirmations also reach screen readers via a polite live region (visual flash alone is
  not announced).
- axe (no serious/critical) is asserted in e2e on the signed-in Player **and** the Library
  Highlights review surface.

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

| Date       | Change                                                                                                                                       |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-06-24 | Initial draft — Editorial Bold baseline (Direction B) + Player surface                                                                       |
| 2026-06-28 | Add Capture & Consolidation surfaces (P2/P3): capture, Library Highlights/Revisit tabs, Recall + your-corpus scope lenses, resurfacing inbox |
