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
| `.lp-kicker` | Editorial eyebrow / content label (accent, uppercase) |
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

- **Summary** sits over a bottom legibility gradient (`from-black/90`), shown in full (scrolls in
  place only if unusually long) — never clamped, since the fixed hero already stabilises height.
- **Live intelligence** ("Insight now / Speaking now") sits top-left; the **Insights** action is a
  floating pill top-right. One action only — no duplicate "Ask" entry (Ask lives inside the panel).
- The **Grounded** chip sits up by the date/meta line, **not** floating over the image.

## Saved & Library

- Per-user collections live in **one "Library" hub** (page) with tabs **Saved · Queue · Recent**.
  ("Shows" returns only once subscriptions are user-curated — we don't show the whole corpus as
  "your shows".) **Recent** is playback history (newest-played, with resume position).
- **Saved** is the polymorphic favorites bucket, **grouped by kind** (Episodes, Insights, … later
  People/Topics). Saving is the shared `.lp-fav` heart on the item (episode cards, the player
  masthead, each insight); insights snapshot their text + jump target (no global detail route).
- Favorites / queue / interests / playback are **per-user files** (no new persistence), mirroring
  each other. Interests are viewable + editable on the **Profile** page (header → user icon).

## Header navigation

The header uses **icon links with hover/focus tooltips** (`NavIconLink`) — Browse (compass),
Library (book-spines), Profile (user) — never bare emoji; one shared component, labelled by
tooltip. Lists use the shared collapsible **`ListToolbar`** (search · sort · filter, incl.
filter-by-show), not stock inputs.

## Conformance checklist

- [ ] No second backdrop; drilling inside a panel is replace-in-place with `‹ Back`.
- [ ] Overlays `Teleport` to `<body>`; correct z-scale.
- [ ] Back/nav uses `.lp-nav`; content labels use `.lp-kicker`; saving uses `.lp-fav`.
- [ ] Header order: `‹ Back` row → kicker → title.
- [ ] Modal a11y: dialog/aria-modal, focus trap, restore focus, ESC + backdrop + control.
- [ ] No per-page restyle of a shared affordance.
