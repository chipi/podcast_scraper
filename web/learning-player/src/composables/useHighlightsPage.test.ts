import { flushPromises } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ref } from 'vue'
import * as api from '../services/api'
import { pageHighlightsLocally } from '../test/localPagers'
import type { HighlightsPageQuery } from '../services/api'
import type { Highlight } from '../services/types'
import { useCaptureStore } from '../stores/capture'
import { useHighlightsPage } from './useHighlightsPage'
import { useNotesPage } from './useNotesPage'

const hl = (id: string, slug: string, created: number) =>
  ({ id, episode_slug: slug, kind: 'moment', segment_ids: [], created_at: created }) as unknown as Highlight

/** 8 episodes, ep-0 newest; ep-0 holds 7 highlights, the rest one each. */
function server() {
  const all: Highlight[] = []
  for (let e = 0; e < 8; e++) {
    const n = e === 0 ? 7 : 1
    for (let i = 0; i < n; i++) all.push(hl(`h${e}-${i}`, `ep-${e}`, 1000 - e * 10 - i))
  }
  return vi
    .spyOn(api, 'getHighlightsPage')
    .mockImplementation(async (q: HighlightsPageQuery) => pageHighlightsLocally(all, [], q))
}

const filters = () => ({
  search: ref(''),
  color: ref<string | null>(null),
  sort: ref('recent'),
  mutedOnly: ref(false),
})

beforeEach(() => setActivePinia(createPinia()))
afterEach(() => vi.restoreAllMocks())

describe('useHighlightsPage', () => {
  it('five episodes, then the next five; five highlights in an episode, then the rest', async () => {
    const get = server()
    const page = useHighlightsPage(filters())
    await page.reload()
    expect(page.groups.value.map((g) => g.slug)).toEqual(['ep-0', 'ep-1', 'ep-2', 'ep-3', 'ep-4'])
    expect(page.episodeTotal.value).toBe(8)
    expect(page.groups.value[0]!.highlights).toHaveLength(5)
    expect(page.groups.value[0]!.total).toBe(7)

    page.toggleGroups()
    await flushPromises()
    expect(get).toHaveBeenLastCalledWith(expect.objectContaining({ offset: 5, limit: 5 }))
    expect(page.groups.value).toHaveLength(8)

    await page.toggleIn('ep-0')
    expect(get).toHaveBeenLastCalledWith(expect.objectContaining({ episode: 'ep-0', perEpisode: 10 }))
    expect(page.groups.value[0]!.highlights).toHaveLength(7)
  })

  it('renders THROUGH the store: an unsave disappears at once', async () => {
    server()
    const page = useHighlightsPage(filters())
    await page.reload()
    const capture = useCaptureStore()
    capture.highlights = capture.highlights.filter((h) => h.id !== 'h1-0')
    expect(page.groups.value.map((g) => g.slug)).not.toContain('ep-1')
  })

  it('a filter change starts again at five episodes, with the filter sent', async () => {
    const get = server()
    const f = filters()
    const page = useHighlightsPage(f)
    await page.reload()
    f.mutedOnly.value = true
    await flushPromises()
    expect(get).toHaveBeenLastCalledWith(expect.objectContaining({ muted: true, offset: 0, limit: 5 }))
  })
})

describe('a run of taps on "Show more" (2026-10-08)', () => {
  it('notes: taps while the next page is loading are ignored, so its answer lands', async () => {
    let release!: () => void
    const gate = new Promise<void>((r) => (release = r))
    const get = vi.spyOn(api, 'getNotesPage').mockImplementation(async (q) => {
      if ((q.offset ?? 0) > 0) await gate
      const all = Array.from({ length: 7 }, (_, i) => ({ id: `n${i}` }) as never)
      return { items: all.slice(q.offset ?? 0, (q.offset ?? 0) + q.limit), total: 7, counts: {}, highlights: [] }
    })
    const page = useNotesPage(ref(''), ref<string[]>([]))
    await page.reload()
    page.toggle()
    page.toggle()
    page.toggle()
    release()
    await flushPromises()
    expect(get.mock.calls.filter(([q]) => (q.offset ?? 0) > 0)).toHaveLength(1)
    expect(page.items.value).toHaveLength(7)
  })
})

describe('a background revalidation does not swallow a "Show more" tap (2026-10-08)', () => {
  it('notes: a tap during a reload still loads the next page', async () => {
    let releaseReload!: () => void
    const reloadGate = new Promise<void>((r) => (releaseReload = r))
    let calls = 0
    vi.spyOn(api, 'getNotesPage').mockImplementation(async (q) => {
      calls++
      if (calls === 2) await reloadGate // the second call is the background reload
      const all = Array.from({ length: 7 }, (_, i) => ({ id: `n${i}` }) as never)
      return { items: all.slice(q.offset ?? 0, (q.offset ?? 0) + q.limit), total: 7, counts: {}, highlights: [] }
    })
    const page = useNotesPage(ref(''), ref<string[]>([]))
    await page.reload()
    const reloading = page.reload() // a revalidation, still in flight
    page.toggle() // the user taps "Show more" meanwhile
    await flushPromises()
    expect(page.items.value).toHaveLength(7)
    releaseReload()
    await reloading
  })
})

describe('useNotesPage', () => {
  it('counts the chips over ALL notes even while searching, with one extra row-less request', async () => {
    const get = vi.spyOn(api, 'getNotesPage').mockImplementation(async (q) => ({
      items: [],
      total: 0,
      counts: (q.q ? { topic: 1 } : { topic: 3, episode: 2 }) as Record<string, number>,
      highlights: [],
    }))
    const search = ref('risk')
    const page = useNotesPage(search, ref<string[]>([]), { debounceMs: 0 })
    await page.reload()
    expect(get).toHaveBeenCalledWith(expect.objectContaining({ limit: 1 }))
    expect(page.counts.value).toEqual({ topic: 3, episode: 2 })
  })

  it('asks for nothing while disabled (Search before it has run)', async () => {
    const get = vi.spyOn(api, 'getNotesPage')
    const page = useNotesPage(ref(''), ref<string[]>([]), { enabled: () => false })
    await page.reload()
    expect(get).not.toHaveBeenCalled()
    expect(page.items.value).toEqual([])
  })
})
