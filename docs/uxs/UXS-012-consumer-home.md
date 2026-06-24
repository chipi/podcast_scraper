# UXS-012: Consumer Home (Learning Hub)

- **Status**: Draft
- **Authors**: Marko
- **Related PRDs**:
  - `docs/prd/PRD-042-home.md` (this surface) · `docs/prd/PRD-038-catalog.md` · `docs/prd/PRD-039-player.md`
- **Related RFCs**:
  - `docs/rfc/RFC-099-learning-platform-consumer-client.md` (§Home & corpus search — behaviour)
  - `docs/rfc/RFC-090-*` (hybrid search backing the corpus-wide search)
- **Related UX specs**:
  - `docs/uxs/UXS-011-consumer-learning-app.md` — **the design-system hub**: this surface
    inherits all tokens, typography, and components from UXS-011 (Editorial Bold, dark-primary).
- **Related issue**: GitHub #1090
- **Implementation paths**: `app/src/views/HomeView.vue`, `app/src/views/SearchView.vue`,
  `app/src/components/*` (reuses `EpisodeCard`)

## Summary

Home is the app's **launch surface** — a learning hub, not a list. This spec defines its
visual + information-architecture contract: an **adaptive hero** (resume-first when there's
history, search/featured otherwise) with the **"Ask your library"** corpus search always
prominent, plus the supporting sections and the **corpus-wide search results** surface.
Behaviour (the adaptive switch logic, debounce, endpoints) lives in RFC-099.

## Principles

- **Orient and resume, don't dump a list.** The first glance answers "where was I / what's new",
  not "here are all episodes" (that's `/catalog`).
- **The corpus is queryable — make that visible.** "Ask your library" is always one glance away;
  it is the consumer face of the moat (a growing, searchable knowledge corpus).
- **Adaptive, graceful.** The hero adapts to state; every section hides cleanly when empty,
  signed-out, or its index/artifact is absent. No empty panels.
- **Inherits UXS-011.** No new tokens or type scale — Editorial Bold, dark-primary, per-show
  adaptive accent (the resume hero borrows the player's artwork-derived accent).

## Scope

**In scope:** the Home surface (adaptive hero + sections) and the corpus-wide **search results**
surface (`/search`).
**Non-goals:** the full catalog (`/catalog`, UXS-011/PRD-038), the Player (UXS-011/PRD-039),
Discovery (PRD-037), the recommendation engine (PRD-041 — Home only renders its output).

**Boundary note:** static visual contract here; behavioural rules (when the hero switches
state, search debounce, data fetching, phasing) live in **RFC-099**.

## Theme support

Inherits UXS-011: dark-primary (MVP), responsive mobile-first (`sm`/`md`/`lg` per UXS-011).

## Layout & regions

Mobile-first single column; on `lg` the rails widen and Home uses the app's max content width.
Region order, top to bottom:

1. **Masthead** — app identity kicker + title; account/sign-in affordance (per UXS-011 shell).
2. **Adaptive hero** (one of two states — see below).
3. **Continue listening** — *only when not already the hero* (auth; hidden otherwise).
4. **What's new** — horizontal rail of newest episodes; "Browse all →" to `/catalog`.
5. **Recommended for you** — rail; hidden when no signal.
6. **Your shows** — grid of followed podcasts → that show's catalog.
7. **Featured / spotlight** — *only when not already the hero*.

### Adaptive hero — the two states

- **Resume state** (signed-in **and** has in-progress history): the hero is a large **Continue**
  card — artwork-derived background (per-show adaptive accent, contrast-clamped per UXS-011),
  episode title, show, a progress rule (`12:04 / 48:00 · 36 min left`), and a primary resume
  control. The **"Ask your library" search bar sits prominently directly below the hero.**
- **Discover state** (signed-out **or** no history): the hero leads with **"Ask your library"**
  (kicker + a short value line + a large search input + a few example query chips) and a
  **Featured spotlight** episode. No empty "Continue" card is ever shown.

In both states the search entry is visually prominent (in or immediately under the hero).

## Corpus-wide search results (`/search`)

- A query field (carries the Home query) + a results list of **grounded passages** across the
  whole library: verbatim passage text, **source episode + show + speaker**, and a `▶ mm:ss`
  control that **opens the Player at that moment**. Editorial cards, hairline-separated.
- **No generated prose** (D6) — passages are extractive; no disclaimer needed.
- **Empty / no-index**: a single `muted` line ("Search needs the library index") — never a
  broken panel. **No results**: "No grounded passages found."

## Key states

- **Hero (resume):** artwork-derived bg, `--lp-accent` progress + resume button (`accent-foreground`).
- **Hero (discover):** `surface` panel, `topic`-toned kicker, large search input (UXS-011 input).
- **Rails:** horizontal scroll, momentum; cards are `EpisodeCard`-derived (UXS-011). Hover →
  `overlay`. Snap optional (timing → RFC-099).
- **Loading:** skeleton hero + skeleton rail cards (`surface`/`border`).
- **Empty/degraded:** sections with no data are omitted; a fully-empty signed-out Home still
  shows the discover hero (search) + What's new.
- **Search result active/jump:** the `▶ mm:ss` uses `--lp-accent`; focus ring per UXS-011.

## Components

- Reuse **`EpisodeCard`** (UXS-011) for rails and Continue (compact variant).
- **Search bar:** pill input (UXS-011 input tokens), search icon, example chips (`topic` toned).
- **Continue hero card:** artwork bg + progress rule + circular resume button (player transport
  styling, UXS-011).
- **Search result card:** passage text (`surface-foreground`), source line (`muted` + `accent`
  show link), `▶ mm:ss` (`accent`, `font-mono` tabular).

## Accessibility

- Search input has a visible/programmatic label; example chips are buttons with names.
- Rails are keyboard-scrollable and not focus-traps; each card is a link with an accessible name.
- One `h1` (Home), section `h2`/headings in order; the adaptive hero swap preserves heading order.
- `▶ mm:ss` controls have accessible names ("Play from 12:04 in <episode>").
- Respects `prefers-reduced-motion` (no rail auto-advance; instant scroll). WCAG 2.1 AA contrast
  (inherits UXS-011 tokens; per-show accent contrast-clamped).

## Tunable parameters

| Parameter                              | Current                                | Status | Notes                                   |
| -------------------------------------- | -------------------------------------- | ------ | --------------------------------------- |
| Hero switch rule                       | resume when in-progress history exists | Open   | exact "in-progress" threshold → RFC-099 |
| Rail length (What's new / Recommended) | ~6                                     | Open   | perf vs richness                        |
| Example search chips                   | derived/static                         | Open   | could be topic-driven later             |
| Tokens / type                          | inherit UXS-011                        | Frozen | do not fork the design system           |

## Acceptance criteria

- [ ] Home uses UXS-011 tokens only (no new hex/scale; no design-system fork)
- [ ] Adaptive hero: resume-state when history exists, discover-state otherwise; **never** an
      empty Continue card; search prominent in **both** states
- [ ] Every section hides cleanly when empty / signed-out / no index (no broken panels)
- [ ] Corpus search results show source episode + speaker + working jump-to-moment; extractive
      (no generated prose); graceful empty/no-index states
- [ ] Rails keyboard-operable; one `h1`; headings ordered; visible focus; reduced-motion honoured
- [ ] All copy via `vue-i18n` (no hard-coded strings); RTL-ready
- [ ] Mobile-first; perf budget on the worst common device (per UXS-011)

## Visual references

`docs/wip/player/mockups/home-{a-search-first,b-resume-first}.{html,png}` — the two explored
directions. **Decision: the adaptive hero** (resume-state borrows A's prominent search; both
states keep "Ask your library" one glance away). WIP aids, not shipped assets.

## Revision history

| Date       | Change                                                                  |
| ---------- | ----------------------------------------------------------------------- |
| 2026-06-24 | Initial draft — adaptive hero (resume/discover) + corpus search surface |
