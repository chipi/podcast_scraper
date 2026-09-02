import { describe, expect, it } from 'vitest'
import { routeForDeepLink } from './deepLinks'

/**
 * An inbound URL is attacker-controllable — anything on the device can open one — so these are as
 * much a security boundary as a parser. The rule under test: a closed allow-list of targets,
 * a validated id shape, and a route built from named constants rather than from the URL's path.
 */
describe('routeForDeepLink', () => {
  it('routes the custom scheme to the player', () => {
    expect(routeForDeepLink('closelistening://episode/p06-7217050bc6')).toEqual({
      name: 'player',
      params: { slug: 'p06-7217050bc6' },
    })
  })

  it('accepts the plural host too — links get written by hand', () => {
    expect(routeForDeepLink('closelistening://episodes/p06-7217050bc6')?.name).toBe('player')
  })

  it('routes shows, topics and people', () => {
    expect(routeForDeepLink('closelistening://podcast/p06')).toEqual({
      name: 'podcast',
      params: { feedId: 'p06' },
    })
    expect(routeForDeepLink('closelistening://show/p06')?.name).toBe('podcast')
    expect(routeForDeepLink('closelistening://topic/indexing')?.name).toBe('topic')
    expect(routeForDeepLink('closelistening://person/daniel-cho')?.name).toBe('person')
  })

  it('reads a plain web URL the same way — someone copied the address bar', () => {
    expect(routeForDeepLink('https://closelistening.app/episode/p06-7217050bc6')).toEqual({
      name: 'player',
      params: { slug: 'p06-7217050bc6' },
    })
  })

  it('decodes a percent-escaped id', () => {
    expect(routeForDeepLink('closelistening://episode/show%2Dep01')?.params.slug).toBe('show-ep01')
  })

  it('refuses a target it does not serve', () => {
    expect(routeForDeepLink('closelistening://settings/admin')).toBeNull()
    expect(routeForDeepLink('closelistening://episode')).toBeNull()
  })

  it('NEVER treats the auth callback as a navigation', () => {
    // It carries a token in the fragment and belongs to the auth listener. Routing on it would
    // put a credential through the router, and could consume the callback before auth sees it.
    expect(routeForDeepLink('closelistening://auth#token=abc.def')).toBeNull()
    expect(routeForDeepLink('https://closelistening.app/auth/callback#token=abc')).toBeNull()
  })

  it('rejects an id that is not the shape of anything we mint', () => {
    for (const bad of [
      'closelistening://episode/with space',
      'closelistening://episode/' + 'x'.repeat(200),
      'closelistening://episode/-leading-dash',
    ]) {
      expect(routeForDeepLink(bad), bad).toBeNull()
    }
  })

  it('neutralises traversal rather than routing on it', () => {
    // `URL` normalises `..` away before we see it, so what arrives is an ordinary id — and even
    // then it can only ever become a slug PARAM on the player route, never a path. It resolves to
    // no episode and the view says so. Asserted explicitly because "it happens to be safe" is
    // worth pinning: a future parser that skips `new URL` would reintroduce the risk silently.
    expect(routeForDeepLink('closelistening://episode/../../admin')).toEqual({
      name: 'player',
      params: { slug: 'admin' },
    })
  })

  it('survives junk rather than throwing at the listener', () => {
    expect(routeForDeepLink('not a url')).toBeNull()
    expect(routeForDeepLink('')).toBeNull()
    expect(routeForDeepLink('closelistening://episode/%E0%A4%A')).toBeNull()
  })
})
