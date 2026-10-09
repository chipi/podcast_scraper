// @vitest-environment happy-dom
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import LibraryView from './LibraryView.vue'
import {
  fetchCorpusEpisodes,
  fetchCorpusFeeds,
  type CorpusEpisodeListItem,
} from '../../api/corpusLibraryApi'
import { corpusGraphBaselineLoaderKey } from '../../corpusGraphBaseline'
import { useActiveSearchContextStore } from '../../stores/activeSearchContext'
import { useDashboardNavStore } from '../../stores/dashboardNav'
import { useShellStore } from '../../stores/shell'
import { useSubjectStore } from '../../stores/subject'

/**
 * What the operator does in Library, against realistic rows (2026-10-09).
 *
 * The mount test that pins the corpus-revision reload brought this view into the coverage
 * denominator with 11 of its 73 functions exercised. These tests drive the rest the way an operator
 * does — rows, keyboard, the per-row G / S / show actions, filters, paging, the search-context
 * ordering and the Dashboard handoff — rather than calling internals.
 */
vi.mock('../../api/corpusLibraryApi', async (orig) => ({
  ...(await orig<typeof import('../../api/corpusLibraryApi')>()),
  fetchCorpusFeeds: vi.fn(),
  fetchCorpusEpisodes: vi.fn(),
}))
// The shell store probes /api/health when the path changes; a probe that never answers leaves the
// health this test sets alone.
vi.mock('../../api/httpClient', async (orig) => ({
  ...(await orig<typeof import('../../api/httpClient')>()),
  fetchWithTimeout: () => new Promise(() => {}),
}))

function ep(n: number, over: Partial<CorpusEpisodeListItem> = {}): CorpusEpisodeListItem {
  return {
    metadata_relative_path: `feeds/f${n % 2}/ep${n}.metadata.json`,
    feed_id: `f${n % 2}`,
    episode_id: `ep-${n}`,
    episode_title: `Episode ${n}`,
    publish_date: '2026-01-0' + ((n % 9) + 1),
    duration_seconds: 600 + n,
    episode_number: n,
    summary_preview: `Summary of episode ${n}`,
    has_gi: true,
    gi_relative_path: `feeds/f${n % 2}/ep${n}.gi.json`,
    has_kg: true,
    kg_relative_path: `feeds/f${n % 2}/ep${n}.kg.json`,
    ...over,
  }
}

const FEEDS = [
  { feed_id: 'f0', display_title: 'Show Zero', episode_count: 2 },
  { feed_id: 'f1', display_title: 'Show One', episode_count: 1 },
]

function serve(items: CorpusEpisodeListItem[], opts: { next?: string | null; total?: number } = {}) {
  vi.mocked(fetchCorpusFeeds).mockResolvedValue({ path: '/corpus', feeds: FEEDS })
  vi.mocked(fetchCorpusEpisodes).mockResolvedValue({
    path: '/corpus',
    feed_id: null,
    items,
    next_cursor: opts.next ?? null,
    total: opts.total ?? items.length,
  } as never)
}

const baseline = vi.fn().mockResolvedValue(undefined)

async function mountLibrary() {
  const shell = useShellStore()
  shell.corpusPath = '/corpus'
  shell.healthStatus = 'ok'
  shell.corpusLibraryApiAvailable = true
  const w = mount(LibraryView, {
    global: { provide: { [corpusGraphBaselineLoaderKey as symbol]: baseline } },
    attachTo: document.body,
  })
  await flushPromises()
  return w
}

const rows = (w: ReturnType<typeof mount>) => w.findAll('[data-library-episode-row]')
const lastEpisodesCall = () => vi.mocked(fetchCorpusEpisodes).mock.calls.at(-1)!

beforeEach(() => setActivePinia(createPinia()))
afterEach(() => {
  vi.clearAllMocks()
  vi.useRealTimers()
  document.body.innerHTML = ''
})

describe('LibraryView — the episode list', () => {
  it('lists the episodes with their show names from the catalogue, and an "X of Y" count', async () => {
    serve([ep(1), ep(2)], { next: 'c2', total: 5 })
    const w = await mountLibrary()
    expect(rows(w)).toHaveLength(2)
    expect(rows(w)[0].attributes('aria-label')).toBe('Episode 1, Show One')
    expect(w.text()).toContain('(2 of 5)')
    expect(w.text()).toContain('Summary of episode 1')
    expect(w.find('[role="region"][aria-label^="Episodes"]').attributes('aria-label')).toContain('more available')
  })

  it('selects the first row on load, and a click or Enter selects another', async () => {
    serve([ep(1), ep(2), ep(3)])
    const w = await mountLibrary()
    const subject = useSubjectStore()
    expect(subject.episodeMetadataPath).toBe(ep(1).metadata_relative_path)
    await rows(w)[1].trigger('click')
    expect(subject.episodeMetadataPath).toBe(ep(2).metadata_relative_path)
    await rows(w)[2].trigger('keydown', { key: 'Enter' })
    expect(subject.episodeMetadataPath).toBe(ep(3).metadata_relative_path)
  })

  it('arrow keys move the selection between rows', async () => {
    serve([ep(1), ep(2), ep(3)])
    const w = await mountLibrary()
    const subject = useSubjectStore()
    await rows(w)[0].trigger('keydown', { key: 'ArrowDown' })
    expect(subject.episodeMetadataPath).toBe(ep(2).metadata_relative_path)
    await rows(w)[1].trigger('keydown', { key: 'End' })
    expect(subject.episodeMetadataPath).toBe(ep(3).metadata_relative_path)
  })

  it('Load more asks for the next page and appends it', async () => {
    serve([ep(1), ep(2)], { next: 'cursor-2', total: 3 })
    const w = await mountLibrary()
    vi.mocked(fetchCorpusEpisodes).mockResolvedValueOnce({
      path: '/corpus',
      feed_id: null,
      items: [ep(3)],
      next_cursor: null,
      total: 3,
    } as never)
    await w.findAll('button').find((b) => b.text() === 'Load more')!.trigger('click')
    await flushPromises()
    expect(lastEpisodesCall()[1]).toMatchObject({ cursor: 'cursor-2' })
    expect(rows(w)).toHaveLength(3)
    expect(w.text()).toContain('(3)')
  })

  it('says so when the episodes cannot be read', async () => {
    vi.mocked(fetchCorpusFeeds).mockResolvedValue({ path: '/corpus', feeds: FEEDS })
    vi.mocked(fetchCorpusEpisodes).mockRejectedValue(new Error('corpus unreadable'))
    const w = await mountLibrary()
    expect(w.text()).toContain('corpus unreadable')
    expect(rows(w)).toHaveLength(0)
  })

  it('an empty corpus says no episodes match', async () => {
    serve([])
    const w = await mountLibrary()
    expect(w.text()).toContain('No episodes match.')
    expect(w.find('[role="region"][aria-label^="Episodes"]').attributes('aria-label')).toBe('Episodes, no matches')
  })
})

describe('LibraryView — per-row actions', () => {
  it('the show name scopes the list to that show', async () => {
    serve([ep(1), ep(2)])
    const w = await mountLibrary()
    await w.findAll('[data-testid="library-row-scope-show"]')[0].trigger('click')
    await flushPromises()
    expect(lastEpisodesCall()[1]).toMatchObject({ feedId: 'f1' })
  })

  it('G focuses the episode and switches to the Graph tab', async () => {
    serve([ep(1), ep(2)])
    const w = await mountLibrary()
    await w.findAll('[data-testid="library-row-open-graph"]')[1].trigger('click')
    await flushPromises()
    expect(useSubjectStore().episodeMetadataPath).toBe(ep(2).metadata_relative_path)
    expect(w.emitted('switch-main-tab')?.at(-1)).toEqual(['graph'])
    expect(baseline).toHaveBeenCalled()
  })

  it('S prefills search with the episode title', async () => {
    serve([ep(1)])
    const w = await mountLibrary()
    await w.find('[data-testid="library-row-open-search"]').trigger('click')
    expect(w.emitted('focus-search')?.at(-1)).toEqual([{ feed: '', query: 'Episode 1' }])
  })
})

describe('LibraryView — filters', () => {
  it('Enter in the title or summary filter reloads with that filter', async () => {
    serve([ep(1)])
    const w = await mountLibrary()
    await w.find('[data-testid="library-filter-title"]').setValue('risk')
    await w.find('[data-testid="library-filter-title"]').trigger('keydown', { key: 'Enter' })
    await flushPromises()
    expect(lastEpisodesCall()[1]).toMatchObject({ q: 'risk' })
    await w.find('[data-testid="library-filter-summary"]').setValue('systems')
    await w.find('[data-testid="library-filter-summary"]').trigger('keydown', { key: 'Enter' })
    await flushPromises()
    expect(lastEpisodesCall()[1]).toMatchObject({ q: 'risk', topicQ: 'systems' })
  })

  it('typing in a filter reloads once, after the debounce', async () => {
    serve([ep(1)])
    const w = await mountLibrary()
    vi.useFakeTimers()
    const before = vi.mocked(fetchCorpusEpisodes).mock.calls.length
    await w.find('[data-testid="library-filter-title"]').setValue('r')
    await w.find('[data-testid="library-filter-title"]').setValue('ri')
    await w.find('[data-testid="library-filter-title"]').setValue('ris')
    expect(vi.mocked(fetchCorpusEpisodes).mock.calls.length).toBe(before)
    await vi.advanceTimersByTimeAsync(500)
    expect(vi.mocked(fetchCorpusEpisodes).mock.calls.length).toBe(before + 1)
    expect(lastEpisodesCall()[1]).toMatchObject({ q: 'ris' })
  })

  it('Reset clears every filter and reloads the whole list', async () => {
    serve([ep(1), ep(2)])
    const w = await mountLibrary()
    await w.find('[data-testid="library-filter-title"]').setValue('risk')
    await w.findAll('[data-testid="library-row-scope-show"]')[0].trigger('click')
    await flushPromises()
    w.findComponent({ name: 'LibraryFilterBar' }).vm.$emit('reset')
    await flushPromises()
    const opts = lastEpisodesCall()[1]!
    expect(opts.q).toBeUndefined()
    expect(opts.feedId).toBeUndefined()
    expect((w.find('[data-testid="library-filter-title"]').element as HTMLInputElement).value).toBe('')
  })

  it('the filter bar sets the show, the date and the cluster-only switch', async () => {
    serve([ep(1)])
    const w = await mountLibrary()
    const bar = w.findComponent({ name: 'LibraryFilterBar' })
    bar.vm.$emit('update:feedFilterId', 'f0')
    await flushPromises()
    expect(lastEpisodesCall()[1]).toMatchObject({ feedId: 'f0' })
    bar.vm.$emit('update:topicClusterOnly', true)
    await flushPromises()
    expect(lastEpisodesCall()[1]).toMatchObject({ topicClusterOnly: true })
    vi.useFakeTimers()
    bar.vm.$emit('update:sinceYmd', '2026-01-01')
    await vi.advanceTimersByTimeAsync(500)
    expect(lastEpisodesCall()[1]).toMatchObject({ since: '2026-01-01' })
  })
})

describe('LibraryView — search context and handoff', () => {
  it('an active search floats matching episodes to the top with a "why" line', async () => {
    serve([ep(1), ep(2), ep(3)])
    useActiveSearchContextStore().setContext('risk', [
      { doc_id: 'd3', score: 0.9, text: 'risk lives in the couplings', metadata: { episode_id: 'ep-3', doc_type: 'insight' } },
    ] as never)
    const w = await mountLibrary()
    expect(rows(w)[0].attributes('aria-label')).toContain('Episode 3')
    expect(w.find('[data-testid="library-row-why"]').text()).toContain('risk lives in the couplings')
  })

  it('a Dashboard handoff applies its show, dates and missing-GI filter', async () => {
    serve([ep(1)])
    useDashboardNavStore().setHandoff({ kind: 'library', feedId: 'f1', since: '2026-01-01', until: '2026-02-01', missingGiOnly: true })
    await mountLibrary()
    expect(lastEpisodesCall()[1]).toMatchObject({ feedId: 'f1', since: '2026-01-01', until: '2026-02-01', hasGi: false })
  })
})
