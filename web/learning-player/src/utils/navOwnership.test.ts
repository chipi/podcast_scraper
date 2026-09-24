import { describe, expect, it } from 'vitest'
import { OWNED_ROUTES, ownsRoute } from './navOwnership'

/**
 * One information architecture, two navs.
 *
 * The phone bar and the desktop masthead are different components rendering the same IA. They used
 * to disagree: the bar computed ownership from its own map, the masthead fell back to `RouterLink`'s
 * exact-active. On `/search` that lit Discovery on a phone and Search on a desktop — the same URL
 * answering "where am I" two ways depending on window width.
 */
describe('nav route ownership (shared by both navs)', () => {
  it('Discovery owns the search results page (operator 2026-09-20)', () => {
    expect(ownsRoute('browse', 'search')).toBe(true)
    // …and nothing else claims it, or two nav items would light at once.
    for (const owner of Object.keys(OWNED_ROUTES).filter((o) => o !== 'browse')) {
      expect(ownsRoute(owner, 'search'), `${owner} must not claim /search`).toBe(false)
    }
  })

  it('Discovery gathers the corpus indexes and show pages (#14)', () => {
    for (const r of ['browse', 'catalog', 'podcast', 'browse-shows', 'browse-topics', 'browse-people']) {
      expect(ownsRoute('browse', r), `browse should own ${r}`).toBe(true)
    }
  })

  it('NOBODY owns the player — a wrong "you are here" is worse than none', () => {
    for (const owner of Object.keys(OWNED_ROUTES)) {
      expect(ownsRoute(owner, 'player'), `${owner} must not claim the player`).toBe(false)
    }
  })

  it('every route is claimed at most once, so the two navs cannot contradict each other', () => {
    const seen = new Map<string, string>()
    for (const [owner, routes] of Object.entries(OWNED_ROUTES)) {
      for (const r of routes) {
        expect(seen.has(r), `${r} is claimed by both ${seen.get(r)} and ${owner}`).toBe(false)
        seen.set(r, owner)
      }
    }
  })

  it('an unknown owner or a null route never reads active', () => {
    expect(ownsRoute('nope', 'browse')).toBe(false)
    expect(ownsRoute('browse', null)).toBe(false)
    expect(ownsRoute('browse', undefined)).toBe(false)
  })
})
