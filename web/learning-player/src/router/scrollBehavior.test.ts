/**
 * Back returns to where the reader was (operator 2026-10-04): opening a person from a section
 * halfway down a page, then Back, must land on that section — it used to land at the top, because
 * `scrollBehavior` ignored the position the browser hands it.
 */
import { afterEach, describe, expect, it, vi } from 'vitest'

vi.mock('../services/deviceStore', () => ({
  getDeviceJson: vi.fn(async () => null),
  setDeviceJson: vi.fn(async () => {}),
  removeDeviceKey: vi.fn(async () => {}),
}))
vi.mock('../services/native', () => ({ isNative: vi.fn(() => false) }))

import type { RouteLocationNormalized } from 'vue-router'
import { router } from './index'

const scrollBehavior = router.options.scrollBehavior!
const route = (over: Partial<RouteLocationNormalized> = {}) =>
  ({ path: '/episode/a', query: {}, hash: '', ...over }) as RouteLocationNormalized

afterEach(() => vi.restoreAllMocks())

describe('router scrollBehavior', () => {
  it('BACK returns the saved position, not the top', async () => {
    vi.spyOn(document.documentElement, 'scrollHeight', 'get').mockReturnValue(5000)
    const saved = { left: 0, top: 1840 }
    expect(await scrollBehavior(route(), route(), saved)).toEqual(saved)
  })

  it('waits for a re-mounted page to grow before handing the position over', async () => {
    // The detail page is a loading line when Back lands; returning 1840 then is clamped to ~0.
    let height = 600
    vi.spyOn(document.documentElement, 'scrollHeight', 'get').mockImplementation(() => height)
    setTimeout(() => (height = 5000), 40)
    const started = Date.now()
    await scrollBehavior(route(), route(), { left: 0, top: 1840 })
    expect(Date.now() - started).toBeGreaterThanOrEqual(35)
  })

  it('a forward navigation still starts at the top', async () => {
    expect(await scrollBehavior(route(), route(), null)).toEqual({ top: 0 })
  })

  it('an anchor link still lands on its section', async () => {
    document.body.innerHTML = '<section id="trends"></section>'
    expect(await scrollBehavior(route({ hash: '#trends' }), route(), null)).toEqual({
      el: '#trends',
      behavior: 'smooth',
      top: 8,
    })
  })

  it('waits for an anchor that renders after the page fetch — a note lands on #notes', async () => {
    document.body.innerHTML = ''
    setTimeout(() => (document.body.innerHTML = '<section id="notes"></section>'), 40)
    const started = Date.now()
    const position = await scrollBehavior(route({ path: '/topic/t', hash: '#notes' }), route(), null)
    // Handing the router '#notes' before it exists makes it scroll nowhere — the reader stays on top.
    expect(Date.now() - started, 'resolved before the notes section rendered').toBeGreaterThanOrEqual(35)
    // Instant (a number), not a smooth scroll toward where the section was at this moment.
    expect(position).toEqual({ top: expect.any(Number) })
  })

  it('opening a SHEET over the page leaves the page where it is', async () => {
    // topic → theme → person stacked as sheets: each adds only its own query key.
    const page = route({ path: '/episode/a', query: { card: 'topic:ai' } })
    const over = route({ path: '/episode/a', query: { card: 'topic:ai', theme: 'tc:x' } })
    expect(await scrollBehavior(over, page, null)).toBe(false)
  })

  it('a query change that is not a sheet still starts at the top', async () => {
    const browse = route({ path: '/browse', query: {} })
    expect(await scrollBehavior(route({ path: '/browse', query: { tab: 'shows' } }), browse, null)).toEqual({
      top: 0,
    })
  })
})
