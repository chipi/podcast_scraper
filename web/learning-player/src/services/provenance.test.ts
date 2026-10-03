import { describe, expect, it } from 'vitest'
import { currentSource, sourceForRoute } from './provenance'
import { SOURCES } from './analytics'
import type { RouteLocationNormalizedLoaded } from 'vue-router'

/** Minimal route stand-in — only the fields `currentSource` reads. */
function route(name: string, query: Record<string, unknown> = {}) {
  return { name, query } as unknown as RouteLocationNormalizedLoaded
}

describe('sourceForRoute', () => {
  it('maps the surfaces that can originate a tracked navigation', () => {
    expect(sourceForRoute('player')).toBe('player')
    expect(sourceForRoute('queue')).toBe('queue')
    expect(sourceForRoute('library')).toBe('library')
    expect(sourceForRoute('search')).toBe('search')
    expect(sourceForRoute('topic')).toBe('entity_page')
    expect(sourceForRoute('person')).toBe('entity_page')
    expect(sourceForRoute('storyline')).toBe('storyline_page')
  })

  it('groups every browse tab under one source', () => {
    // The tab is carried by `browse_tab_view`; splitting the source per tab would make the enum
    // describe two things at once.
    for (const n of ['browse', 'browse-shows', 'browse-topics', 'browse-people', 'catalog']) {
      expect(sourceForRoute(n)).toBe('browse')
    }
  })

  it('does NOT map home — its rails must identify themselves', () => {
    // A generic `home` would erase the distinction the six `home_*` values exist to draw, which is
    // the whole basis of "which rail actually produces discovery".
    expect(sourceForRoute('home')).toBe('other')
  })

  it('falls back to other for anything unmapped, rather than guessing', () => {
    expect(sourceForRoute('settings')).toBe('other')
    expect(sourceForRoute(undefined)).toBe('other')
    expect(sourceForRoute(42)).toBe('other')
  })

  it('only ever returns values in the registry enum', () => {
    const all = ['player', 'queue', 'library', 'search', 'browse', 'topic', 'person', 'theme',
      'podcast', 'storyline', 'offline-downloads', 'catalog', 'home', 'nonsense']
    for (const n of all) {
      expect(SOURCES as readonly string[]).toContain(sourceForRoute(n))
    }
  })
})

describe('currentSource', () => {
  it('uses the surface for an ordinary in-app navigation', () => {
    expect(currentSource(route('player'))).toBe('player')
  })

  it('reports deep_link when the app arrived from outside', () => {
    // An episode opened from a friend's link is NOT in-app discovery. Attributing it to the player
    // would inflate Discovery share with traffic the app did not generate — the one metric the
    // beta most needs to be honest.
    expect(currentSource(route('player', { utm_source: 'share' }))).toBe('deep_link')
    expect(currentSource(route('player', { shared: '1' }))).toBe('deep_link')
  })

  it('does not treat an ordinary query string as a deep link', () => {
    expect(currentSource(route('search', { q: 'anything', scope: 'corpus' }))).toBe('search')
  })
})
