/**
 * Discover rail measurement (ADR-163, Step 1.2): what a rail reports to Umami, and that a tap marks
 * where the listener came from.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest'

const tracked: { name: string; props: unknown }[] = []
vi.mock('../services/analytics', async (orig) => ({
  ...(await orig<typeof import('../services/analytics')>()),
  track: (name: string, props: unknown) => tracked.push({ name, props }),
}))

import type { RouteLocationNormalizedLoaded } from 'vue-router'
import { noteNavigation, sourceOfCurrentNavigation } from '../services/provenance'
import type { Reason } from '../services/types'
import { useRailTelemetry } from './useRailTelemetry'

const why = (over: Partial<Reason> = {}): Reason => ({
  kind: 'rising_topic',
  quote: null,
  subject: null,
  capture: null,
  events: null,
  outside_habits: false,
  ...over,
})
const route = (name: string, path = `/${name}`) =>
  ({ name, path, query: {}, matched: [{}] }) as unknown as RouteLocationNormalizedLoaded

beforeEach(() => {
  tracked.length = 0
})

describe('useRailTelemetry', () => {
  it('reports a rail once per mount, in buckets, and not when it rendered nothing', () => {
    const t = useRailTelemetry('quotes')
    t.shown([], 'mine')
    expect(tracked).toEqual([])
    const items = [{ why: why() }, { why: why({ outside_habits: true }) }, { why: null }]
    t.shown(items, 'mine')
    t.shown(items, 'mine')
    expect(tracked).toEqual([
      { name: 'discover_rail_shown', props: { rail: 'quotes', items: '2-5', outside: '1', scope: 'mine' } },
    ])
  })

  it('a tap reports the reason kind, action and rank bucket — never the episode', () => {
    const t = useRailTelemetry('captured')
    t.tapped({ why: why({ kind: 'captured', outside_habits: true }) }, 3, 'play_from')
    t.tapped({ why: null }, 0, 'open')
    expect(tracked.map((e) => e.props)).toEqual([
      { rail: 'captured', reason_kind: 'captured', action: 'play_from', rank: '4-10', outside_habits: true },
      { rail: 'captured', reason_kind: 'none', action: 'open', rank: '1', outside_habits: false },
    ])
  })

  it('a tap marks the next navigation with the rail, and in-player query changes keep it', () => {
    useRailTelemetry('quotes').tapped({ why: why() }, 0, 'open')
    noteNavigation(route('browse')) // Discover → player: the rail, not `browse`
    expect(sourceOfCurrentNavigation()).toBe('discover_quotes')
    noteNavigation(route('player', '/episode/ep-1'), true) // the player rewrites ?t=: same screen
    expect(sourceOfCurrentNavigation()).toBe('discover_quotes')
    noteNavigation(route('player', '/episode/ep-1')) // a real navigation away: the route again
    expect(sourceOfCurrentNavigation()).toBe('player')
  })
})
