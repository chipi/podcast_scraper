/**
 * Where a navigation came from (#2267, epic #2263).
 *
 * `source` is the most important property in the beta analytics spec, because it is the only thing
 * that answers its core question: *do people move across the corpus — topic → person → another
 * show's episode — or just play shows they already follow?* It is the input to Discovery share,
 * Pivot rate and the derived cross-show hop, and the beta's per-participant reports show each
 * person's top three sources.
 *
 * ── Why this is plumbing and not a tracking call ─────────────────────────────
 * The value is known by whatever the listener TAPPED, and nothing downstream can recover it. By
 * the time an episode page has mounted, "they got here from the momentum rail" is gone. Measured
 * scale: 25 navigations to entity/episode routes across 47 files, and 92 `EntityCard` /
 * `openEntity` references, against a 20-value enum.
 *
 * ── The two halves ───────────────────────────────────────────────────────────
 * 1. **Taps carry their own source.** A rail knows which rail it is; only the component rendering
 *    the handler can distinguish `home_momentum` from `home_trending_topics`. Those pass it
 *    explicitly, and the typed `Source` union makes a typo a compile error rather than a new
 *    category that quietly appears in a dashboard.
 *
 * 2. **Surfaces have a default.** Everything else — a tap inside the player, the library, the queue
 *    — can be answered by "which screen am I on", which the router already knows. `sourceForRoute`
 *    maps a route name to its surface so those call sites do not each hand-write it.
 *
 * A deep link is NOT the surface it lands on: arriving at an episode from a shared URL is
 * `deep_link`, and treating it as `player` would count an external share as in-app discovery.
 */

import type { RouteLocationNormalizedLoaded } from 'vue-router'
import type { Source } from './analytics'

/**
 * Route name → the surface a tap on that screen came from.
 *
 * Only routes that can originate a tracked navigation need an entry; anything absent falls back to
 * `other`, which is honest and visible rather than silently mislabelled as something plausible.
 *
 * Note what is deliberately NOT here: `home`. Home's rails are the whole point of the granular
 * `home_*` values, so a Home tap that fell back to a generic `home` would erase the distinction
 * the spec's first six enum values exist to draw. Home taps pass their rail explicitly.
 */
const SOURCE_BY_ROUTE: Readonly<Record<string, Source>> = {
  player: 'player',
  queue: 'queue',
  library: 'library',
  search: 'search',
  browse: 'browse',
  'browse-shows': 'browse',
  'browse-topics': 'browse',
  'browse-people': 'browse',
  catalog: 'browse',
  topic: 'entity_page',
  person: 'entity_page',
  theme: 'entity_page',
  podcast: 'entity_page',
  storyline: 'storyline_page',
  'offline-downloads': 'library',
}

/** The surface for a route name, or `other` when the route originates nothing tracked. */
export function sourceForRoute(name: unknown): Source {
  return (typeof name === 'string' && SOURCE_BY_ROUTE[name]) || 'other'
}

/**
 * The surface for the route the app is currently on.
 *
 * `deep_link` wins over the surface when the app ARRIVED here from outside — a shared link or a
 * notification. Without that, an episode opened from a friend's link would be attributed to the
 * player, counting an external share as in-app discovery and inflating exactly the metric the beta
 * exists to measure honestly.
 */
export function currentSource(route: RouteLocationNormalizedLoaded): Source {
  if (route.query?.utm_source === 'share' || route.query?.shared !== undefined) return 'deep_link'
  return sourceForRoute(route.name)
}
