# UXS-011: Consumer Learning App — Design System + Player

- **Status**: Active — foundations + Player + Home + Library +
  Capture all shipped (v2.7). See
  [#1261](https://github.com/chipi/podcast_scraper/issues/1261) for
  the listener-search enhancements layered on top: enriched
  related-topic chips, transcript-cluster dedupe, related-episodes
  rail, matched-field kicker, standalone Topic/Person pages,
  year-header grouping on Search, saved queries.
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

> **Not universally true (#1604).** `canvas` and `surface` have `-foreground` pairs; **`elevated`
> and `overlay` do not** — they are backgrounds for text that inherits from `canvas`. The prose
> above overclaims, and the token table below never listed pairs for them either. Corrected here
> rather than pretending: a claim that contrast is "validated at the token level" is only as true
> as the pairs that exist.

The **accent is per-show adaptive**: `--lp-accent` is a runtime variable derived from the current
episode's artwork and contrast-clamped, falling back to the brand constant "Ember"
(`--lp-brand-default`) when there is no artwork or extraction fails. Components reference `--accent`
and never hard-code the colour.

> **SHIPPED — per-show adaptive accent (#1598).** `src/theme/accent.ts` samples the artwork
> (`extractAccentFromImage` → `vibrantColorFromPixels`), clamps the result to ≥4.5:1 against
> `--lp-surface` (`src/theme/contrast.ts`, `clampToContrast`), and applies it via `setShowAccent`.
> `App.vue` calls it off `player.currentArtwork`, so the accent tracks the episode in focus. Any
> failure (image error, cross-origin canvas taint, or an artwork with no vivid colour) falls back
> to the brand default — a missing accent is never an error. Covered by `contrast.test.ts` (the
> clamp math) and `accent.test.ts` (extraction, fallback, per-show change).
>
> This was retracted-as-not-shipped for six weeks (`setShowAccent` existed with zero call sites and
> #1083 never landed the wiring). #1598 built the extraction + real clamp + wiring, so the spec is
> true again rather than aspirational.

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

| Parameter                        | Current value                                | Status            | Notes                                                                               |
| -------------------------------- | -------------------------------------------- | ----------------- | ----------------------------------------------------------------------------------- |
| Display font family              | Inter 800, tight                             | Open              | Upgrade to a licensed grotesque considered; must cover i18n glyphs                  |
| `brand-default` accent ("Ember") | `#FF6A3D`                                    | Open              | Brand colour pending; used only when no show colour                                 |
| Per-show accent derivation       | artwork to vibrant colour                    | Frozen            | Built #1598: `theme/accent.ts` + `theme/contrast.ts`, >=4.5:1 clamp                 |
| Token names                      | `canvas`, `surface`, `accent`, domain tokens | Frozen            | API — do not rename                                                                 |
| Dark-only (MVP)                  | dark baseline                                | Open              | Light theme is a post-MVP fast-follow                                               |

### How to experiment

Swap values in `web/learning-player/src/styles/tokens.css` (`:root`) or via DevTools; the per-show `--accent` is set
on the player root element at runtime. Token **names and the contrast-clamp contract are frozen**;
values and the extraction algorithm are open until promoted.

## Capture & Consolidation surfaces (P2 + P3 — shipped)

The "Remember" half of the app (PRD-040 Capture, PRD-041 Consolidation). All affordances below are
**auth-gated**, and since #1590 "gated" means *deferred, not hidden*: the control **renders for
signed-out visitors** and its tap routes to sign-in with a redirect back to where they were.

> **Amended 2026-08-13 (#1590).** This paragraph previously read "signed-out users see the app
> exactly as before (no capture controls, no scope toggles)". That was the shipped behaviour and it
> was wrong: it hid the capabilities that differentiate this product from precisely the visitors who
> had not yet decided to sign up, and the only prompts left were the two header buttons. Hiding is
> also the wrong shape of honesty — the control is not unavailable, it is deferred.
>
> The rule now: **a gated control renders, states its requirement in its accessible name
> (`auth.signInTo*`), claims no toggle state (`aria-pressed` is omitted, since nothing is toggled),
> and routes to sign-in on tap.** It must never call the API — the stores swallow write failures, so
> an ungated click flips optimistically, takes a 401, and silently reverts, which reads to the user
> as their own action failing.
>
> Enforced by `src/__checks__/auth-gate.test.ts`, which fails on any component performing a per-user
> write without the gate. That guard exists because I wired two call sites and missed four.

Everything is grounded (slug + timestamp) and extractive (**no request-time LLM**).

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

The Library is tabbed. As of the 2026-08 mobile pass the tabs are **Saved · Following ·
Collections · Revisit** (Highlights folded into Saved as a section; **Queue and Recent moved OUT to
the player surface** — see "Player-surface Queue & Recent" below):

- **Saved** — per-kind sections (**Episodes** / **Insights**) plus a folded **Highlights** section
  (captured moments/spans/insights grouped by episode, titles hydrated with slug fallback), each with
  jump-to-moment (`?t=`), a drift badge when the timestamp re-anchored on re-scrape, inline notes
  (add/edit/remove), a per-highlight colour swatch picker, a header **colour filter**, and an
  **Export Markdown** link.
- **Following** — the shows and interest tokens (`topic:`/`person:`/`thc:`) the user follows.
- **Collections** — its own first-class tab (was nested under Saved); see "Collections" below.
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

## Mobile navigation & surfaces (2026-08 pass — shipped)

The mobile walkthrough reworked how the app's breadth is reached. Baseline for the e2e surface map
(`web/learning-player/e2e/E2E_SURFACE_MAP.md`), which bridges this spec to the Playwright suite.

### Browse hub (#14)

The three standalone corpus indexes (episodes, topics, people) plus shows fold into ONE tabbed hub
(`BrowseView`, route `/browse`), tabs **Episodes · Shows · Topics · People** (`browse-tab-{tab}`,
Episodes default — mirrors the Library tab pattern). Each panel embeds the standalone index view in
`embedded` mode (drops its page heading + back-to-Home button) and is `v-show`-mounted so switching
tabs never refetches. `?tab=` deep-links a tab; because the hub is kept-alive, a later in-app
navigation to a new `?tab=` **re-syncs the active tab live** (a `watch` on the query — without it the
kept-alive instance kept the stale tab). Home's browse chips (`home-browse-nav`) deep-link in
(`?tab=topics` / `?tab=people`). The standalone `/browse/topics` · `/browse/people` routes still
resolve for direct links. A trending topic chip (`trend-spark-row`) opens the standalone `/topic/:id`
page (it is a `router.push` button, not an anchor).

Trending rows (topics + people) are **sparkline chips** coloured by storyline, sorted **velocity
first, then total volume**, collapsed to the top 20 with a show-more (#11/#12). **People** rows carry
a **role badge** — the person's strongest KG role across the corpus (**host > guest > mentioned**,
`trend-spark-role`) — so a trending person reads with *why* they trend (a frequent host vs a
much-discussed guest); topics never carry a role.

### Player-surface Queue & Recent (#1838)

**Up next** and **Recently played** are no longer Library tabs — they open FROM the transport as a
bottom-sheet modal (`QueuePanel`, `queue-panel`, close `queue-panel-close`; mirrors the EntityCard
modal shell — teleport, focus trap, Esc/backdrop dismiss). Triggers: `player-queue` on the full
player (next to the speed pill, reachable while playing) and `mini-player-queue` on the mini-player.
Two sections — **Up next** (the play queue, reused `QueueView`) and **Recently played**
(`queue-panel-recent`; tapping a row **resumes**, it does not re-queue — the intent is find/resume,
not build a queue).

### Collections (RFC-119 / #1839)

Holistic pinboards — "a Pinterest for listening". A collection holds **typed** items
`{kind: highlight | episode | show | search | topic | person | link}` (identity `(kind, ref)`; bare
legacy strings migrate to `{kind: highlight}`). Create / open / delete; **add an external link**
(URL-only, for an article found while researching); **Play-all** queues the collection's episodes
oldest-pinned-first and opens the first. An **Add to collection** button rides episode cards, entity
cards, search results and the podcast page. Empty state: "No collections yet". Auth-gated (empty when
signed out). Own first-class Library tab.

### Settings (#8)

A **Settings** surface (`SettingsView`) surfaces the app version / build info and the client control
plane, reachable from Profile — the "is this broken or just empty?" affordance the first-run states
also serve.

## Offline, downloads and the look-back (2026-09 arc — shipped)

The "listen anywhere, and be told the truth about it" half of the app (#1905 downloads, #1906
offline mode, #1909 content cache, #1910 offline writes, #1914 recaps, #1924 listen events).

Three rules govern every surface below, and they are worth stating before the components because
each one exists to prevent a specific way of lying to the listener:

1. **Only a 401/403 may destroy cached state. A transport error never may.** A failed refresh must
   not sign anyone out; a failed GET must not blank a populated view. A moment of no signal is not
   data loss.
2. **A refusal is shown, never silent.** A control that cannot act says so — disabled with a reason
   in its `title`, or an inline notice. A control that silently does nothing reads as broken, and
   that is how the queue's reorder arrows read before this arc.
3. **A number states its own coverage.** Where the app reports a measurement it cannot fully
   support yet, it says how much of the window it actually has, in the same breath.

### Downloads (native only)

Behind `isNative()` — the web build has **no offline-audio story at all** (Capacitor's web
Filesystem would put third-party audio in IndexedDB, which PRD-035 Principle 4 exists to prevent),
so these render nothing in a browser.

- **`DownloadButton`** — the ONE mark-for-offline control, identical on every surface (UXS-014:
  define once, use everywhere). Auth-gated in the #1590 sense: it renders for signed-out visitors
  and routes to sign-in. Five visible states, because a download is not binary:
  `Download for offline` · `Waiting for Wi-Fi` / `Waiting for a connection` · `Downloading {pct}%`
  (indeterminate when the host sends no `Content-Length` — "0%" forever would be a lie) ·
  `Downloaded — tap to remove` · a failure that names its KIND (`tap to retry`, `Not enough space`,
  `No longer available`). `queued` is a first-class state on purpose: under the L1 design a flagged
  episode legitimately waits for an allowed connection, and without a distinct label that is
  indistinguishable from a broken button.
- **`DownloadedList`** — the Downloaded section in Library. Renders from the DEVICE registry with
  no API call, so it is the one list that is fully itself offline.
- **`DeviceSettings`** — in Profile, and deliberately at the BOTTOM: these settings belong to the
  phone, not the account, and are shared by everyone who signs in on it. Wi-Fi-only vs
  Wi-Fi-and-cellular, and a storage cap whose help text promises that only FINISHED episodes are
  reclaimed — nothing unplayed is ever deleted to make room.

### The queue, offline

Adding and removing are item-level and replay safely, so they stay available with no connection.
**Reordering does not**: it sends the whole list, and writing that from a copy we never revalidated
would delete whatever the server actually holds.

So the Queue view states what it is showing (`Offline — showing your saved queue. You can add and
remove; reordering needs a connection.`) and the `↑`/`↓` chevrons **disable with a reason** rather
than doing nothing. An earlier revision of this arc disabled the queue TOGGLE too; that was an
over-correction and was walked back — it made a working control read as dead.

### The look-back: recap panel and Home prompt

- **`ListeningRecap`** (Profile → "Your listening") — a window toggle (Week · Month · This year),
  per-day BARS (discrete days read better as bars than as a line), what kept coming up with its
  movement (`↑2`, `↓3`, `new` — a delta of zero renders nothing, since an "unchanged" marker is
  noise on every chip that did not move), and the line the listener saved, linking to the MOMENT it
  came from.
  **It always states its coverage while the window is partial** ("Recorded 3 of 7 days"), and the
  line disappears on its own as coverage fills in.
- **`RecapPrompt`** (Home, under Your Week) — deliberately NOT the recap: one line, one tap,
  pointing at Profile. Profile is the permanent home you can always go to; this is the periodic
  reminder that it is there. It self-hides when signed out, when the request fails, and when
  nothing was listened to — a row reading "0h · 0 episodes" spends space telling the user what they
  already know.

**A number that was removed.** Profile's "Your listening" previously showed a headline `Xh` fed by
`sum(position_seconds)` — a lifetime snapshot of furthest-position-reached. It rose when you seeked
forward without hearing anything and did not move when you re-listened. It is gone; the panel below
is retitled **"Your activity"**, because opens-over-time is a different question from time listened
and conflating them is how the first number went unexamined for so long.

### Deep links

`closelistening://episode/<slug>` (also `podcast`/`show`, `topic`, `person`), and the same shapes as
`https://` URLs. `?t=<seconds>` opens an episode AT that moment, overriding the remembered resume
position **for that load only** — a recap's saved line, a shared quote or an agent's citation points
INTO an episode, and opening at the resume point would silently drop the only reason the link
existed. A malformed `t` still opens the episode: losing the moment is a shame, losing the episode
is a broken link.

### Where each surface is verified

| Surface | Spec |
| ------- | ---- |
| Downloads, Downloaded list, Device settings | device tier only (`make test-app-ios-journey`) — `isNative()`, no browser coverage possible |
| Queue offline behaviour | `queue-offline-surface.spec.ts`, `recap-and-offline-writes-real-corpus.spec.ts` (Tier-3) |
| Recap panel + Home prompt | `recap-and-deep-links.spec.ts`, `recap-and-offline-writes-real-corpus.spec.ts` (Tier-3) |
| `?t=` deep links | `recap-and-deep-links.spec.ts`, `recap-and-offline-writes-real-corpus.spec.ts` (Tier-3) |
| Offline shell / SW | `offline.spec.ts`, `offline-shell-real-corpus.spec.ts` (Tier-3) |

## The view inventory (documented 2026-09-03)

Eleven views had Playwright automation and no design spec — the app's whole navigational surface
was undocumented while individual widgets were specified in detail. Each entry below states what
the view is FOR and the one rule that governs it, which is what a spec has to carry for someone to
review or rebuild it.

| View | What it is for | The rule that governs it |
| ---- | -------------- | ------------------------ |
| `LoginView` | The sign-in surface, including the dev identity picker | Signing in must return the visitor **where they were**, never to Home — a gated tap is deferred, not restarted (#1590) |
| `CatalogView` (Browse) | The corpus by show / topic / person | It is a HUB, not a list: it routes onward and holds no state of its own |
| `PodcastView` | One show: its episodes, its signals band, follow | Following is the primary action and must respond instantly (optimistic), reverting only on a server REFUSAL |
| `TopicView` | One topic: perspectives, arc, episodes | Every claim carries its source; ungrounded content is omitted, not shown greyed |
| `PersonView` | One person: positions, topics, episodes | Same grounding rule as `TopicView` |
| `ShowBrowseView` · `TopicBrowseView` · `PersonBrowseView` | The three browse indexes behind Catalog | Consistent card + heading treatment across all three; they differ in content, never in shape |
| `ProfileView` | Identity, activity, interests, connected agents, device settings | Ordered account-first, device-LAST: device settings belong to the phone and are shared by everyone who signs in on it |
| `CollectionsView` | User-made collections of episodes | Per-item additive; a collection can never destroy the queue or another collection |
| `HighlightsView` | Everything captured, with export | The listener's own words — export must be lossless and must never require a network round trip to read |

### Remaining shared components

- **`KnowledgePanel`** — the insights surface on the player. Opens **in place** on mobile as a
  modal dialog (focus-trapped, ESC closes) and as a column on desktop; it never stacks over another
  dimmed layer (UXS-014's core rule).
- **`QueueButton`** — the ONE add/remove-to-queue control, everywhere. Renders for signed-out
  visitors (#1590). Since the 2026-09 offline arc it stays ENABLED with no connection: adding and
  removing are item-level and replay safely; only reordering needs a live list.
- **`InterestsPicker`** — the modal over the corpus's top clusters. Modal a11y matches the entity
  card exactly; "Not now" must be as easy to reach as "Save", because a picker that traps someone
  into choosing is a picker they will dismiss by leaving.
- **`PwaUpdateToast`** — announces a new build is available. It **offers**, never forces: a reload
  mid-listen would cost the listener their place, so the toast waits for a deliberate tap.

## Visual references

Annotated phone mockups of the three explored directions live in
`docs/wip/player/mockups/` (HTML + PNG). **Direction B (Editorial Bold)** is the adopted baseline;
the Player's artwork zone additionally borrows the adaptive/immersive now-playing treatment explored in
Direction C. These are design aids (WIP), not shipped assets.

## Acceptance criteria (for issues / review)

- [ ] New UI uses semantic tokens only (no one-off hex in components; single token layer)
- [ ] Every surface that HAS a `-foreground` uses it for text (`canvas`, `surface`; `elevated` and `overlay` inherit — see the token section)
- [x] Per-show `--accent` is derived from artwork and contrast-clamped to ≥4.5:1 against `surface`, falling back to `brand-default` on failure (#1598; `theme/accent.ts`, `theme/contrast.ts`).
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

| Date       | Change                                                                                                                                                                                                                               |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 2026-06-24 | Initial draft — Editorial Bold baseline (Direction B) + Player surface                                                                                                                                                               |
| 2026-06-28 | Add Capture & Consolidation surfaces (P2/P3): capture, Library Highlights/Revisit tabs, Recall + your-corpus scope lenses, resurfacing inbox                                                                                         |
| 2026-08-26 | Mobile pass: Browse hub (#14), player-surface Queue & Recent (#1838), holistic Collections (RFC-119/#1839), Following + Settings tabs; Library tabs now Saved · Following · Collections · Revisit (Queue/Recent moved to the player) |
| 2026-09-03 | Offline arc: downloads + device settings (native), queue offline behaviour, the listening recap + Home prompt, `?t=` deep links, and the removal of the fabricated "Hours" tile                                                      |
