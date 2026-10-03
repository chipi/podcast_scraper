import { describe, expect, it } from 'vitest'
import {
  scrubEventRequestUrl,
  scrubNavigationBreadcrumb,
  stripQuery,
} from './telemetryScrub'

/**
 * #2264 — the search term must not reach GlitchTip.
 *
 * These assert the BEHAVIOUR, not that a hook is wired. A static source check on `main.ts` can
 * only prove that `beforeBreadcrumb` is mentioned, which says nothing about whether it strips
 * anything — exactly the "diagnostic that looks like it is recording something" failure this repo
 * keeps finding. The wiring is checked separately in `__checks__/mobile-invariants.test.ts`.
 */
describe('stripQuery', () => {
  it('drops a query string', () => {
    expect(stripQuery('/search?q=nuclear%20policy')).toBe('/search')
  })

  it('drops a fragment', () => {
    expect(stripQuery('/episode/p09#t=120')).toBe('/episode/p09')
  })

  it('cuts at the FIRST delimiter, so a term containing ? or # cannot survive', () => {
    // A naive `split('?')[0]` on a term that itself contains '?' would be fine, but a regex
    // without the limit, or a lastIndexOf, would leak part of it. This is the case that catches it.
    expect(stripQuery('/search?q=what?#why')).toBe('/search')
    expect(stripQuery('/search?q=a#b?c')).toBe('/search')
  })

  it('leaves a clean path untouched', () => {
    expect(stripQuery('/library')).toBe('/library')
  })

  it('passes non-strings through rather than throwing', () => {
    // Breadcrumb data is `unknown`. A scrubber that throws would take the error report with it.
    expect(stripQuery(undefined)).toBeUndefined()
    expect(stripQuery(null)).toBeNull()
    expect(stripQuery(42)).toBe(42)
  })
})

describe('scrubNavigationBreadcrumb', () => {
  it('strips the term from both ends of a navigation breadcrumb', () => {
    // This is the real shape: the SDK sets from/to to `parseUrl(...).relative`, which is
    // path + query + fragment.
    const crumb = scrubNavigationBreadcrumb({
      category: 'navigation',
      data: { from: '/home', to: '/search?q=interest%20rates&scope=corpus' },
    })
    expect(crumb.data?.to).toBe('/search')
    expect(crumb.data?.from).toBe('/home')
    expect(JSON.stringify(crumb)).not.toContain('interest')
  })

  it('strips a term that is in the FROM side too', () => {
    // Navigating away from a search page puts the term in `from`, which is just as much a leak.
    const crumb = scrubNavigationBreadcrumb({
      category: 'navigation',
      data: { from: '/search?q=secret', to: '/episode/p09' },
    })
    expect(JSON.stringify(crumb)).not.toContain('secret')
  })

  it('leaves other categories alone', () => {
    // Targeted fix: rewriting unrelated breadcrumbs would cost debuggability for no privacy gain.
    const crumb = scrubNavigationBreadcrumb({
      category: 'xhr',
      data: { url: '/api/app/search?q=kept' },
    })
    expect(crumb.data?.url).toBe('/api/app/search?q=kept')
  })

  it('survives a navigation breadcrumb with no data', () => {
    expect(() => scrubNavigationBreadcrumb({ category: 'navigation' })).not.toThrow()
  })
})

describe('scrubEventRequestUrl', () => {
  it('strips the query from request.url when present', () => {
    const event = scrubEventRequestUrl({
      request: { url: 'https://app.example.com/search?q=private' },
    })
    expect(event.request?.url).toBe('https://app.example.com/search')
  })

  it('does nothing when there is no request or url', () => {
    expect(() => scrubEventRequestUrl({})).not.toThrow()
    expect(() => scrubEventRequestUrl({ request: {} })).not.toThrow()
  })
})
