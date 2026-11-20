// @vitest-environment happy-dom
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import LibraryNewsList from './LibraryNewsList.vue'
import { fetchNewsItems } from '../../api/newsApi'
import type { NewsItem } from '../../api/searchApi'
import { useShellStore } from '../../stores/shell'

vi.mock('../../api/newsApi', () => ({ fetchNewsItems: vi.fn() }))

const DE = { id: 'org:daily-example', name: 'Daily Example', domain: 'dailyexample.test' }
const OW = { id: 'org:other-wire', name: 'Other Wire', domain: 'otherwire.test' }
const LB = { id: 'org:late-bulletin', name: 'Late Bulletin', domain: 'latebulletin.test' }

function news(n: number, publisher = DE): NewsItem {
  return {
    content_type: 'news_article',
    item_id: `item:news_article:${n}`,
    title: `Story ${n}`,
    publisher,
    published_at: '2026-11-17T07:30:00Z',
    byline: [],
    serveable: 'metadata',
  }
}

type Opts = { publisherId?: string; limit?: number; offset?: number }

/** A fake items route over `corpus`, newest first, paged and filtered as the server does. */
function serve(corpus: NewsItem[]): void {
  vi.mocked(fetchNewsItems).mockImplementation(async (_root: string, opts: Opts = {}) => {
    const rows = opts.publisherId ? corpus.filter((i) => i.publisher?.id === opts.publisherId) : corpus
    const offset = opts.offset ?? 0
    return { path: '/c', items: rows.slice(offset, offset + (opts.limit ?? 50)), total: rows.length }
  })
}

/** The list's own requests (page size 50), not the publisher sweep. */
function pageCalls(): Opts[] {
  return vi
    .mocked(fetchNewsItems)
    .mock.calls.map(([, opts]) => opts ?? {})
    .filter((o) => o.limit === 50)
}

describe('LibraryNewsList (UXS-019 Library News mode)', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    useShellStore().corpusPath = '/c'
    vi.mocked(fetchNewsItems).mockReset()
  })

  it('lists items, offers their publishers, and heads a publisher page', async () => {
    serve([news(1), news(2, OW)])
    const w = mount(LibraryNewsList)
    await flushPromises()
    expect(w.findAll('[data-testid="search-result-news"]')).toHaveLength(2)
    const options = w.findAll('[data-testid="library-news-publisher"] option').map((o) => o.text())
    expect(options).toEqual(['All publishers', 'Daily Example', 'Other Wire'])
    await w.get('[data-testid="library-news-publisher"]').setValue('org:other-wire')
    await flushPromises()
    expect(pageCalls().at(-1)).toEqual({ publisherId: 'org:other-wire', limit: 50, offset: 0 })
    expect(w.get('[data-testid="library-news-publisher-head"]').text()).toContain('otherwire.test')
    expect(w.findAll('[data-testid="search-result-news"]')).toHaveLength(1)
  })

  it('offers a publisher whose items are past the first page', async () => {
    // Gap 24 (2026-11-20): the select listed the publishers of the loaded pages only, 7 of 17.
    const corpus = [...Array.from({ length: 60 }, (_, i) => news(i)), news(60, LB)]
    serve(corpus)
    const w = mount(LibraryNewsList)
    await flushPromises()
    expect(w.findAll('[data-testid="search-result-news"]')).toHaveLength(50)
    const options = w.findAll('[data-testid="library-news-publisher"] option').map((o) => o.text())
    expect(options).toEqual(['All publishers', 'Daily Example', 'Late Bulletin'])
  })

  it('sweeps a list longer than one sweep page', async () => {
    const corpus = [...Array.from({ length: 500 }, (_, i) => news(i)), news(500, LB)]
    serve(corpus)
    const w = mount(LibraryNewsList)
    await flushPromises()
    const sweeps = vi
      .mocked(fetchNewsItems)
      .mock.calls.map(([, opts]) => opts ?? {})
      .filter((o) => o.limit === 500)
    expect(sweeps).toEqual([
      { limit: 500, offset: 0 },
      { limit: 500, offset: 500 },
    ])
    expect(w.findAll('[data-testid="library-news-publisher"] option').map((o) => o.text())).toContain(
      'Late Bulletin',
    )
  })

  it('loads the next page on request', async () => {
    serve([...Array.from({ length: 50 }, (_, i) => news(i)), news(50)])
    const w = mount(LibraryNewsList)
    await flushPromises()
    await w.get('[data-testid="library-news-more"]').trigger('click')
    await flushPromises()
    expect(pageCalls().at(-1)).toEqual({ publisherId: undefined, limit: 50, offset: 50 })
    expect(w.findAll('[data-testid="search-result-news"]')).toHaveLength(51)
    expect(w.find('[data-testid="library-news-more"]').exists()).toBe(false)
  })

  it('opens on the publisher another surface set in the shell store', async () => {
    useShellStore().libraryNewsPublisherId = 'org:other-wire'
    serve([news(1), news(2, OW)])
    mount(LibraryNewsList)
    await flushPromises()
    expect(pageCalls()[0]).toEqual({ publisherId: 'org:other-wire', limit: 50, offset: 0 })
  })

  it('keeps the publishers the pages gave when the sweep fails', async () => {
    vi.mocked(fetchNewsItems).mockImplementation(async (_root: string, opts: Opts = {}) => {
      if (opts.limit === 500) throw new Error('sweep down')
      return { path: '/c', items: [news(1, OW)], total: 1 }
    })
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const w = mount(LibraryNewsList)
    await flushPromises()
    expect(w.findAll('[data-testid="library-news-publisher"] option').map((o) => o.text())).toEqual([
      'All publishers',
      'Other Wire',
    ])
    expect(w.text()).not.toContain('sweep down')
    expect(w.findAll('[data-testid="search-result-news"]')).toHaveLength(1)
    expect(warn).toHaveBeenCalled()
    warn.mockRestore()
  })
})
