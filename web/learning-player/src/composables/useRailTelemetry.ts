/**
 * What a Discover rail reports (ADR-163; Discover arc plan, Step 1.2) — usage analytics in Umami,
 * under the listener's opt-out, and nothing else: no rail data goes to the server.
 *
 * - `shown(items, scope)` once per mount, and only when the rail rendered a card — the denominator.
 * - `tapped(item, index, action)` when a card is opened, its "Play from" or Moments tapped; it marks the
 *   next navigation with the rail's source, so `episode_open` and any `capture_created` in that
 *   visit say which rail sent the listener.
 *
 * Buckets and enums only (the analytics contract): never an episode, a quote or a name.
 */
import type { Reason } from '../services/types'
import { markNextNavigation } from '../services/provenance'
import {
  railSource,
  toCountBucket,
  toRankBucket,
  track,
  type RailId,
  type RailReasonKind,
  type RailAction,
} from '../services/analytics'

export interface RailItem {
  why: Reason | null
}

export function useRailTelemetry(rail: RailId) {
  let reported = false

  function shown(items: RailItem[], scope: 'mine' | 'everyone'): void {
    if (reported || items.length === 0) return
    reported = true
    track('discover_rail_shown', {
      rail,
      items: toCountBucket(items.length),
      outside: toCountBucket(items.filter((i) => i.why?.outside_habits).length),
      scope,
    })
  }

  function tapped(item: RailItem, index: number, action: RailAction): void {
    track('discover_rail_tap', {
      rail,
      reason_kind: (item.why?.kind ?? 'none') as RailReasonKind,
      action,
      rank: toRankBucket(index + 1),
      outside_habits: Boolean(item.why?.outside_habits),
    })
    markNextNavigation(railSource(rail))
  }

  return { shown, tapped }
}
