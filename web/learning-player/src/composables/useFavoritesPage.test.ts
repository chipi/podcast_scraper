import { flushPromises } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ref } from 'vue'
import * as api from '../services/api'
import type { FavoritesPageQuery } from '../services/api'
import type { EpisodeSummary } from '../services/types'
import { useFavoritesStore } from '../stores/favorites'
import { useFavoritesPage } from './useFavoritesPage'

const ep = (slug: string) => ({ slug, title: slug }) as EpisodeSummary

/** A server with `n` saved episodes that pages the way the API does. */
function serverWith(n: number) {
  const all = Array.from({ length: n }, (_, i) => ep(`e${i}`))
  return vi.spyOn(api, 'getFavoritesPage').mockImplementation(async (q: FavoritesPageQuery) => ({
    episodes: all.slice(q.offset ?? 0, (q.offset ?? 0) + q.limit),
    entities: [],
    total: all.length,
    counts: { episode: all.length },
  }))
}

function filters() {
  return { search: ref(''), color: ref<string | null>(null), sort: ref('recent') }
}

beforeEach(() => setActivePinia(createPinia()))
afterEach(() => {
  vi.restoreAllMocks()
  vi.useRealTimers()
})

describe('useFavoritesPage', () => {
  it('asks the server for the first five, then the NEXT five on Show more', async () => {
    const get = serverWith(12)
    const page = useFavoritesPage('episode', filters())
    await page.reload()
    expect(page.episodes.value.map((e) => e.slug)).toEqual(['e0', 'e1', 'e2', 'e3', 'e4'])
    expect(page.total.value).toBe(12)
    expect(page.remaining.value).toBe(7)
    await page.showMore()
    expect(get).toHaveBeenLastCalledWith(expect.objectContaining({ offset: 5, limit: 5 }))
    expect(page.shown.value).toBe(10)
    page.toggle() // 2 left: more
    await flushPromises()
    expect(page.shown.value).toBe(12)
    page.toggle() // none left: back to five, no request
    expect(page.shown.value).toBe(5)
  })

  it('a filter change starts again at five, with the filter sent to the server', async () => {
    const get = serverWith(12)
    const f = filters()
    const page = useFavoritesPage('episode', f)
    await page.reload()
    await page.showMore()
    f.color.value = 'red'
    await flushPromises()
    expect(get).toHaveBeenLastCalledWith(
      expect.objectContaining({ color: 'red', offset: 0, limit: 5 }),
    )
    expect(page.shown.value).toBe(5)
  })

  it('debounces the search before asking', async () => {
    vi.useFakeTimers()
    const get = serverWith(3)
    const f = filters()
    useFavoritesPage('episode', f)
    f.search.value = 's'
    f.search.value = 'sl'
    f.search.value = 'sle'
    expect(get).not.toHaveBeenCalled()
    await vi.advanceTimersByTimeAsync(300)
    expect(get).toHaveBeenCalledTimes(1)
    expect(get).toHaveBeenLastCalledWith(expect.objectContaining({ q: 'sle' }))
  })

  it('a change to the favourites reloads the rows already shown, without collapsing them', async () => {
    const get = serverWith(12)
    const page = useFavoritesPage('episode', filters())
    await page.reload()
    await page.showMore()
    useFavoritesStore()._set([]) // a heart or a colour, confirmed by the server
    await flushPromises()
    expect(get).toHaveBeenLastCalledWith(expect.objectContaining({ offset: 0, limit: 10 }))
    expect(page.shown.value).toBe(10)
  })

  it('an answer that arrives after a newer request is dropped', async () => {
    let release!: () => void
    const slow = new Promise<void>((r) => (release = r))
    vi.spyOn(api, 'getFavoritesPage')
      .mockImplementationOnce(async () => {
        await slow
        return { episodes: [ep('old')], entities: [], total: 1 }
      })
      .mockImplementationOnce(async () => ({ episodes: [ep('new')], entities: [], total: 1 }))
    const page = useFavoritesPage('episode', filters())
    const first = page.reload()
    await page.reload()
    release()
    await first
    expect(page.episodes.value.map((e) => e.slug)).toEqual(['new'])
  })
})
