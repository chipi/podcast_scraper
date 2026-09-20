# Plan — fold Search into Discovery

**Operator, 2026-09-20.** Search stops being a top-level destination and becomes part of Discovery.
The `/search` results page stays as a standalone full page; what changes is where you enter it from
and which tab reads as "you are here".

## What the operator asked for, verbatim in effect

1. The search box moves onto Discovery, **between "Trending shows" and "Trends"**.
2. Tapping search lands on the full `/search` results page, as today.
3. **Discovery stays highlighted** on both Discovery and the search results page.
4. The Search entry is **removed from the main menu**.
5. A masthead search icon is acceptable.

## What the code actually looks like today (measured, not assumed)

| Surface | Class | Who sees it |
| --- | --- | --- |
| `BottomNav.vue` | `sm:hidden` | **phone only** — Home · Discovery · Search · Library |
| Masthead nav icons in `App.vue` | `hidden … sm:flex` | **desktop only** — Browse + Search icons |

They are complementary, not duplicates. That changes the shape of this work:

- **The masthead search icon already exists** (`App.vue:544`, added by #1588). It is not new work;
  it is currently invisible below the `sm` breakpoint.
- Removing the Search tab therefore only strips search from the **phone** bar. Desktop is unaffected.
- So "add a masthead search icon" reduces to: **let the existing icon show at all widths**.

`#1588` is the reason it exists, and its comment is directly relevant:

> Search is the differentiator … It had exactly ONE entry point (the Home search box), so from the
> catalogue, player, library or a show page there was no way to reach it at all (#1588).

Removing a nav entry for search is exactly the condition #1588 was fixing. Keeping the masthead icon
visible everywhere is what stops this plan from re-opening that bug.

## The one principle this knowingly overrules

`BottomNav.vue` states:

> The player owns NOTHING, deliberately. You can reach an episode from any tab, so lighting one up
> would assert a path the user may not have taken — and a wrong "you are here" is worse than none.

Discovery lighting up on `/search` asserts a path a user who searched from Home's "Ask" box did not
take. **The operator has decided to overrule this** ("Discovery leads anyway once you get to search.
Search is really part of discovery"). The comment gets updated to record the decision rather than
being left contradicting the code.

Supporting precedent: this split was made deliberately in #14, and its reasoning points the same way
as the operator's instinct —

> Browse now has its own tab: the hub plus the corpus indexes and show pages belong to it, **since it
> is the destination that gathers them**. Search owns only the search route again.

Before #14, `search` owned `['search', 'catalog', 'podcast']`. #14 inverted the parent. This plan
finishes that inversion by moving the last discovery surface under the gatherer.

## Stages

### S1 — navigation

- `BottomNav.vue`: drop `search` from `TABS` (4 tabs → 3: Home · Discovery · Library).
- `BottomNav.vue`: `OWNED_ROUTES.browse` gains `'search'`; drop the `search` key.
- `BottomNav.vue`: rewrite the ownership comment to record the overruled principle.
- `App.vue`: the existing search `NavIconLink` moves OUT of the `hidden … sm:flex` span so it shows
  at every width. Browse stays desktop-only (it has a phone tab).

### S2 — the search box on Discovery

`BrowseView.vue`, between "Trending shows" (`home.trendingShows`) and "Trends"
(`browse.trendsTitle`). Reuses Home's existing `lp-search` form markup so the two entry points are
the same control, not two dialects of one.

### S3 — tests and specs

Known referrers, from a grep for the search nav: `BottomNav.test.ts`,
`__checks__/touch-affordances.test.ts`, `HomeView.test.ts`, `PlayerView.test.ts`,
`LibraryView.test.ts`, `LibraryView.vue`, plus e2e specs that navigate by tab.

New assertions worth having, because each pins a decision rather than an implementation:

- the phone bar has exactly three tabs and none of them is Search;
- `/search` lights **Discovery** (the overruled principle, pinned so it cannot regress silently);
- the masthead search icon is reachable at phone width (the #1588 guard);
- Discovery's search box submits to `/search` with the query.

## Explicitly NOT in this plan

- **Deleting the `/search` route.** It stays: deep links, the results page, and Home's Ask box all
  target it. This changes navigation, not routing.
- **Removing Home's "Ask" box.** It is a second entry point by design (#1588).
- **Changing search behaviour, ranking, or the results page itself.**

## Gates

Per stage, not at the end:

1. `npx vitest run` (player) — full suite, not just the touched files.
2. `npm run build` — `vue-tsc -b` strict. The `--noEmit` typecheck does NOT catch what this does.
3. `npx playwright test` — full player e2e, 0 failed AND 0 flaky.
4. Screenshots of the phone bar and Discovery, against a **freshly built** bundle — Playwright's
   `reuseExistingServer` has served a stale build twice on this branch and shown "no change" after
   real edits. Kill :4174 before every shot.

## Risks

| Risk | Mitigation |
| --- | --- |
| Search becomes hard to find on a phone | Masthead icon visible at all widths (#1588's own fix), plus Home's Ask box, plus the new Discovery box — three entry points, up from two |
| A tab bar that silently lost an item looks like a bug | The three-tab count is asserted, so the change is deliberate and pinned |
| e2e specs navigate by the Search tab and would fail obscurely | Grep-listed above; each updated to the masthead icon or a direct `/search` visit |
| "Discovery lit on /search" reads as wrong to a user who came from Home | Accepted by the operator; recorded in the code comment so the next reader sees a decision, not a bug |
