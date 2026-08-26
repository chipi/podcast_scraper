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
- **Implementation paths**: `web/learning-player/src/views/HomeView.vue`, `web/learning-player/src/views/SearchView.vue`,
  `web/learning-player/src/components/*` (reuses `EpisodeCard`)

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
- **Adaptive, graceful.** The hero adapts to state. Sections hide when signed-out or when their
  index/artifact is absent — but see the state contract below: "hides cleanly when empty" was too
  blunt a rule and is superseded (#1591).
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
4. **What's new** — *shipped (#1091) as an editorial **ranked** layout, **not** a horizontal rail*:
   a featured **#01** hero (artwork + gradient + oversized faint numeral) over compact numbered rows
   (02–06, each with artwork), all on screen; "Browse all →" now opens the **Browse hub**
   (`/browse?tab=episodes`, UXS-011), not the standalone catalog.
5. **Discover — tabbed (#4)** — **Rising now** (momentum) / **Trending** / **Storylines** were three
   stacked rails that made Home very tall; they fold into ONE tabbed switcher (`home-discovery`,
   `discovery-tab-{key}`, `rising` default). The **active tab's label IS the section heading** — the
   rails no longer render a duplicate `<h2>`. `v-show` keeps each panel mounted (no refetch on
   switch).
6. **New in topics & people you follow (#1836)** — recent UNHEARD episodes about a followed topic or
   featuring a followed person (deterministic; no ranking score). Also a **Your-Week** digest section.
7. **Browse chips** — a compact `home-browse-nav` strip (**Browse topics** / **Browse people**) that
   deep-links into the Browse hub (`?tab=topics` / `?tab=people`).
8. **Recommended for you** — *shipped as a no-scroll responsive **grid***; hidden when no signal.
9. **Your shows** — grid of followed podcasts → that show's catalog.
10. **Featured / spotlight** — *folded into What's-new as the #01 hero (no separate block).*

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

- A query field (carries the Home query) + results across the whole library. *Shipped (#1091):
  results are **grouped by source episode** (ranked by best hit); each episode header shows an
  **artwork thumbnail** + title + show + match count, and each passage is **labelled by kind**
  (Insight / Transcript / Topic). A `▶ "Play from m:ss"` control appears **only when the passage
  carries a real timestamp** (opens the Player there) — otherwise the header opens the episode.
  Bare topic-term matches are de-emphasised (muted italic).*
- **No generated prose** (D6) — passages are extractive; no disclaimer needed.
- **Empty / no-index**: a single `muted` line ("Search needs the library index") — never a
  broken panel. **No results**: "No grounded passages found."

## Key states

- **Hero (resume):** artwork-derived bg, `--lp-accent` progress + resume button (`accent-foreground`).
- **Hero (discover):** `surface` panel, `topic`-toned kicker, large search input (UXS-011 input).
- **What's new / Recommended:** *shipped as no-scroll layouts — What's-new is the ranked
  hero+rows, Recommended is a responsive grid* (the earlier horizontal-rail/`CardRail` direction
  was dropped on Home; `CardRail` remains available for future Catalog use). Hover → `overlay`.
- **Loading:** skeleton hero + skeleton rail cards (`surface`/`border`).
- **Empty/degraded:** see the state contract below (#1591) — this previously said "sections with no
  data are omitted", which conflated three different situations. A fully-empty signed-out Home
  still shows the discover hero (search) + What's new.

### Section state contract (#1591)

Every data-backed section distinguishes **three** states. Collapsing them is what made a cold
corpus, a brand-new account and **a total API outage** render the same page.

| State | Behaviour |
| ----- | --------- |
| **loading** | Skeleton, in the section's own shape. The header renders — it is what tells the user this content exists before it arrives. |
| **error** | A message plus a **retry**. Never silently equal to empty. Styled once, via `SectionStatus.vue`, so the same class of failure stops looking different in different views. |
| **ready + empty** | Depends on *why* it is empty — see below. |

**The rule for empty: hide when the SYSTEM is empty, render when the USER is.**

- **System-empty** — nothing to show because the corpus or the user's history has nothing yet, and
  there is no action available. Hide; an empty shell is noise. *Storylines, Trending topics,
  Trending shows, Momentum, Recommended.*
- **User-empty** — empty because of an action the user has not taken yet. **Render, and the empty
  state must carry that action** — not a description of it, the action itself. *"Your shows" shows
  followable suggestions; "Your Week" shows a first-run row per digest section — as of #1836 there
  are **four** rows (`new_in_follows`, `new_in_interests`, `revisit`, `trending_in_your_corpus`), the
  two actionable ones (`new_in_follows`, `new_in_interests`) linking out because they are fixable
  today.*

A section that merely *describes* what the user could do is the failure mode this rule exists to
prevent: it makes the reader go and find the control it is telling them about.

**Known gap:** user-empty states currently render indefinitely, so someone who deliberately follows
nothing sees the prompt forever. The intended fix is to stop after the first success (first follow,
first capture), which needs a per-user preference flag. Not yet built.

- **Search result active/jump:** the `▶ mm:ss` uses `--lp-accent`; focus ring per UXS-011.

## Components

- **`EpisodeCard`** (UXS-011) is reused on **Catalog + search-result episodes**, not Home; *Home's
  What's-new hero (#01) + numbered rows and the Continue card are bespoke layouts* (the EpisodeCard
  is the clean-lede + ✦ insights-popover card).
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

| Date       | Change                                                                                                                                                                                                               |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-06-24 | Initial draft — adaptive hero (resume/discover) + corpus search surface                                                                                                                                              |
| 2026-08-26 | Mobile pass: Discover folded into tabs (#4, `discovery-tab-{key}`); "New in topics & people you follow" (#1836) + 4th Your-Week first-run row; What's-new "Browse all" + browse chips deep-link the Browse hub (#14) |
