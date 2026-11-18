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

describe('LibraryNewsList (UXS-019 Library News mode)', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    useShellStore().corpusPath = '/c'
    vi.mocked(fetchNewsItems).mockReset()
  })

  it('lists items, offers their publishers, and heads a publisher page', async () => {
    vi.mocked(fetchNewsItems)
      .mockResolvedValueOnce({ path: '/c', items: [news(1), news(2, OW)], total: 2 })
      .mockResolvedValueOnce({ path: '/c', items: [news(2, OW)], total: 1 })
    const w = mount(LibraryNewsList)
    await flushPromises()
    expect(w.findAll('[data-testid="search-result-news"]')).toHaveLength(2)
    const options = w.findAll('[data-testid="library-news-publisher"] option').map((o) => o.text())
    expect(options).toEqual(['All publishers', 'Daily Example', 'Other Wire'])
    await w.get('[data-testid="library-news-publisher"]').setValue('org:other-wire')
    await flushPromises()
    expect(fetchNewsItems).toHaveBeenLastCalledWith('/c', {
      publisherId: 'org:other-wire',
      limit: 50,
      offset: 0,
    })
    expect(w.get('[data-testid="library-news-publisher-head"]').text()).toContain('otherwire.test')
    expect(w.findAll('[data-testid="search-result-news"]')).toHaveLength(1)
  })

  it('loads the next page on request', async () => {
    vi.mocked(fetchNewsItems)
      .mockResolvedValueOnce({ path: '/c', items: [news(1)], total: 2 })
      .mockResolvedValueOnce({ path: '/c', items: [news(2)], total: 2 })
    const w = mount(LibraryNewsList)
    await flushPromises()
    await w.get('[data-testid="library-news-more"]').trigger('click')
    await flushPromises()
    expect(fetchNewsItems).toHaveBeenLastCalledWith('/c', { publisherId: undefined, limit: 50, offset: 1 })
    expect(w.findAll('[data-testid="search-result-news"]')).toHaveLength(2)
    expect(w.find('[data-testid="library-news-more"]').exists()).toBe(false)
  })
})
