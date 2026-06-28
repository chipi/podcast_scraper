# UXS-014: Interaction patterns (consumer)

- **Status**: Active (foundational — applies to every consumer surface)
- **PRD**: `docs/prd/PRD-043-knowledge-layer.md` (and all consumer PRDs)
- **RFC**: `docs/rfc/RFC-102-knowledge-clusters-entity-cards.md`
- **Inherits**: UXS-011 (Editorial Bold tokens, `--lp-*`), UXS-012 (Home), UXS-013 (knowledge).

This UXS is the **shared contract for how the consumer app behaves and is styled across surfaces**.
It exists because the operator demanded strict consistency: *define patterns and styles once, apply
them app-wide; never restyle the same affordance per page.* New UXS docs and components **conform to
this spec** — they do not re-invent navigation, layering, or saving.

## Surfaces — the four kinds and when to use each

| Surface | What | Examples |
| ------- | ---- | -------- |
| **Page** | A route; URL-addressable destination | Home, Search, Catalog, Player, **Library** |
| **Panel** | Persistent, in-layout region; not modal | Insights panel beside the Player |
| **Modal** | One dimmed backdrop; teleported to `<body>` | Interests picker, entity card **from Search** |
| **Sheet** | The mobile form of a panel/modal (bottom, drag-handle) | Insights on mobile |

## Core rule — never stack two dimmed layers

- **Drilling deeper *inside a panel* replaces the panel's content in place** with a `‹ Back`
  (a back-stack), never a new overlay. Example: tapping a person/topic chip in Insights swaps the
  panel to the entity card; `‹ Back` returns to the insight list.
- **A modal opens only from a page-level surface**, never on top of a panel/sheet. So the entity
  card is *in-panel* from Insights but a *modal* from Search (a page).
- **At most one backdrop on screen.** If you would dim a second layer, use replace-in-place instead.

### Entry-point → surface map

| You tap … | … here | Result |
| --------- | ------ | ------ |
| Person/topic chip | Insights **panel** | Replace-in-panel (entity card, `‹ Back`) |
| Entity match | Search **page** | Modal entity card |
| "Set interests" | Home **page** | Modal picker |
| A "see all" link | any | Navigate to a **page** |

## Layering mechanics

- Every overlay **`Teleport`s to `<body>`** so it covers the viewport and escapes any clipped or
  transformed ancestor (a panel sheet uses `overflow-hidden`/offsets that otherwise clip a nested
  `position: fixed`).
- **z-scale:** panel `z-40`, modal `z-50`, mobile backdrop `z-30`.

## Headers & navigation

- **Header order is `‹ Back` (own row) → kicker → title.** The entity card header mirrors the
  episode-detail masthead exactly — back never crammed beside the kicker/name.
- **Navigation reads differently from content labels.** A `‹ Back` control is muted (`.lp-nav`),
  never the accent content eyebrow (`.lp-kicker`); the two clashed when both used the kicker style.

## Shared style classes — define once, use everywhere

Recurring affordances are **single classes in `app/src/style.css`**, not per-element Tailwind
hand-rolled on each page. Adding a one-off `class="text-muted …"` for one of these is a regression.

| Class | Role |
| ----- | ---- |
| `.lp-kicker` | Editorial eyebrow / content label (accent, uppercase) — e.g. a show name |
| `.lp-section` | Section/region heading (calm white display heading) — **never** the kicker, so a section title can't be mistaken for a show name |
| `.lp-speaker` | Speaker attribution in transcript / quotes (muted, normal-case) — distinct from the kicker |
| `.lp-nav` | Back / navigation control (muted; distinct from content) |
| `.lp-fav` | Favorite (heart) toggle; `.lp-fav--on` = saved |

When a new recurring treatment appears, add one class and reuse it — do not copy styles between
pages.

**Show names never truncate.** A podcast/show name is never `truncate`d to an ellipsis — when it's
too long it **wraps to the next line**. (Episode titles may still clamp; show names do not.)

## Drill-in navigation

- Drilling deeper is **replace-in-place with a `‹ Back` stack**; closing returns to the prior view
  in the **same** surface (no layer added or removed).
- The shared body (e.g. `EntityCardBody`) is rendered **inline** in a panel and **wrapped in the
  modal** from a page — one component, two presentations (`variant`), so they cannot drift.

## Dismissal & accessibility (every modal)

- Dismiss via **ESC**, **backdrop tap**, and an explicit control — all three.
- `role="dialog"` + `aria-modal`, a **focus trap**, **initial focus**, and **restore focus on
  close**. In-panel replacements move focus to the new heading instead of trapping.

## Player hero (artwork zone)

The Player masthead is a **hero**: a fixed-square artwork carrying overlays, so layout height is
constant regardless of content length.

- **Summary** is revealed on demand: hidden by default (clean artwork), it **slides up + fades in on
  hover/focus** over a darker legibility gradient (`from-black/95 via-black/85 to-black/40`, white
  text) so it stays readable even over bright artwork. Always shown on touch (no hover). Full text,
  never clamped — the fixed-square hero stabilises height regardless.
- **Live intelligence** ("Insight now / Speaking now") sits top-left; the **per-episode reach**
  cluster (listeners · opens · Insights + a tiny opens-over-time `Sparkline`) sits top-right. The
  Insights score opens the panel — no duplicate "Ask" entry (Ask lives inside the panel).
- The **Grounded** chip sits up by the date/meta line, **not** floating over the image.

## Saved & Library

- Per-user collections live in **one "Library" hub** (page) with tabs **Saved · Knowledge · Queue ·
  Recent**. ("Shows" returns only once subscriptions are user-curated — we don't show the whole
  corpus as "your shows".) **Recent** is playback history (newest-played); the player auto-resumes
  from the saved position, so the card needs no separate "resume at" affordance.
- **One card, every surface.** Catalog, Saved, Queue and Recent all showcase an episode through the
  shared `EpisodeCard` (Queue keeps a slim ↑/↓ reorder rail beside it; the card's own queue toggle
  removes). Hydrated `EpisodeDetail`s are adapted via `summaryFromDetail` so they never drift.
- **Saved** holds favorited **episodes**; **Knowledge** holds saved **insights** (snapshot text +
  jump-to-moment) — its own tab, no longer an "Insights" group inside Saved. The backend favorites
  bucket is still polymorphic (grouped by kind in `AppFavoritesResponse`: `episodes`, `insights`),
  but the UI splits the two kinds across tabs rather than showing "Episodes"/"Insights" group
  headings. Saving is the shared `.lp-fav` heart on the item (episode cards, the player masthead,
  each insight).
- Favorites / queue / interests / playback are **per-user files** (no DB). Interests are viewable +
  editable on the **Profile** page (header → user icon).
- **Following an interest** is a one-tap toggle on a person/topic **entity card** (`Follow` /
  `Following`), in addition to the Home cluster picker. The interest list is a mixed token set —
  clusters (`tc:`), topics (`topic:`) and people (`person:`) — and re-ranks "Recommended for you"
  by how many followed tokens an episode matches (flag-gated personalized discovery, PRD-043).

## Listening analytics

Listening stats are computed from per-user files — **no LLM, no DB** — and surface in two places:

- **Profile "Your listening"** (own data): single scores — day streak, episodes, shows, hours — plus
  an opens-over-time `Sparkline`. Derived from the user's playback + listen log.
- **Player per-episode reach** (cross-user, anonymous): distinct listeners, total opens, an
  opens-over-time `Sparkline`, and the grounded-insight count. Aggregated by scanning every user's
  listen log; counts only, never identities. Public (no auth).
- The **listen-events log** (`<data_dir>/users/<id>/listen_events.jsonl`, append-only) is the only
  per-listen history we keep — playback stays last-position-only. The player appends one "open"
  event on mount. `Sparkline` is the single shared mini-chart (`currentColor`) for both surfaces.

## Header navigation

The header uses **icon links with hover/focus tooltips** (`NavIconLink`) — Browse (compass),
Library (book-spines), Profile (user) — never bare emoji; one shared component, labelled by
tooltip. Lists use the shared collapsible **`ListToolbar`** (search · sort · filter, incl.
filter-by-show), not stock inputs.

## Conformance checklist

- [ ] No second backdrop; drilling inside a panel is replace-in-place with `‹ Back`.
- [ ] Overlays `Teleport` to `<body>`; correct z-scale.
- [ ] Back/nav uses `.lp-nav`; content labels use `.lp-kicker`; section headings use `.lp-section`;
      speakers use `.lp-speaker`; saving uses `.lp-fav`.
- [ ] Episodes showcase through the shared `EpisodeCard` on every collection surface.
- [ ] Library tabs are **Saved · Knowledge · Queue · Recent**; saved insights live under
      **Knowledge**, not as an "Insights" group inside Saved.
- [ ] Following an interest is the one-tap `Follow` / `Following` toggle on a person/topic entity
      card (mixed-token interests: `tc:` / `topic:` / `person:`), alongside the Home cluster picker.
- [ ] The Player hero summary is hover/focus-revealed (slide-up + fade over the legibility
      gradient), always shown on touch; never clamped.
- [ ] Listening analytics surface through the single shared `Sparkline` — Profile "Your listening"
      (own data) and the Player per-episode reach cluster (cross-user, anonymous, public).
- [ ] Header order: `‹ Back` row → kicker → title.
- [ ] Modal a11y: dialog/aria-modal, focus trap, restore focus, ESC + backdrop + control.
- [ ] No per-page restyle of a shared affordance.
