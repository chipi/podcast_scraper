/**
 * Which nav destination OWNS a given route — shared by BOTH navs.
 *
 * The phone tab bar (`BottomNav`, `sm:hidden`) and the desktop masthead icons (`NavIconLink` in
 * `App.vue`, `hidden … sm:flex`) are different components showing the same information architecture.
 * They disagreed: the bar computed ownership from its own map, while the masthead relied on
 * `RouterLink`'s default exact-active. On `/search` that lit **Discovery** on a phone and **Search**
 * on a desktop — the same URL answering "where am I" two different ways depending on window width.
 *
 * Ownership lives here so there is one answer. A destination owns the routes it GATHERS, not just
 * the one it links to: highlighting on exact match alone left both navs blank on `player`,
 * `podcast` and `catalog`, the routes users spend the most time on.
 *
 * `browse` owns `search` (operator 2026-09-20). #14 moved every other discovery surface under
 * Browse "since it is the destination that gathers them" and left search outside; this finishes it.
 * The cost is deliberate: someone who searched from Home's Ask box lands on `/search` with Discovery
 * lit, a path they did not take. Accepted because search has a canonical parent — an episode does
 * not, which is why `player` appears nowhere below and no nav item lights up on it.
 *
 * `profile` is absent for a different reason: it is reached from the masthead AVATAR, not from a
 * `NavIconLink`, so nothing would consume an entry for it. An unused key here reads as a wired
 * destination and invites someone to "fix" the avatar to match.
 */
export const OWNED_ROUTES: Record<string, readonly string[]> = {
  home: ['home'],
  browse: [
    'browse',
    'search',
    'catalog',
    'podcast',
    'browse-shows',
    'browse-topics',
    'browse-people',
  ],
  library: ['library'],
}

/** True when `routeName` belongs to the nav destination `owner`. */
export function ownsRoute(owner: string, routeName: string | null | undefined): boolean {
  if (!routeName) return false
  return OWNED_ROUTES[owner]?.includes(routeName) ?? false
}
