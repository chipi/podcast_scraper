import { flushPromises } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ref } from 'vue'
import * as api from '../services/api'
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
    .mockImplementation(async (q: HighlightsPageQuery) => api.pageHighlightsLocally(all, [], q))
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

describe('useNotesPage', () => {
  it('counts the chips over ALL notes even while searching, with one extra row-less request', async () => {
    const get = vi.spyOn(api, 'getNotesPage').mockImplementation(async (q) => ({
      items: [],
      total: 0,
      counts: q.q ? { topic: 1 } : { topic: 3, episode: 2 },
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
