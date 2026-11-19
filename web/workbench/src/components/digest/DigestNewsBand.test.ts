// @vitest-environment happy-dom
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import DigestNewsBand from './DigestNewsBand.vue'
import { fetchNewsItems } from '../../api/newsApi'
import type { NewsItem } from '../../api/searchApi'
import { useShellStore } from '../../stores/shell'
import { useSubjectStore } from '../../stores/subject'

vi.mock('../../api/newsApi', () => ({ fetchNewsItems: vi.fn() }))

function news(n: number): NewsItem {
  return {
    content_type: 'news_article',
    item_id: `item:news_article:${n}`,
    title: `Story ${n}`,
    publisher: { id: 'org:daily-example', name: 'Daily Example', domain: 'dailyexample.test' },
    published_at: '2026-11-17T07:30:00Z',
    byline: [],
    serveable: 'summary',
  }
}

describe('DigestNewsBand (UXS-019)', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    const shell = useShellStore()
    shell.corpusPath = '/c'
    shell.healthStatus = 'ok'
    vi.mocked(fetchNewsItems).mockReset()
  })

  it('asks nothing without news', async () => {
    mount(DigestNewsBand, { props: { sinceYmd: '' } })
    await flushPromises()
    expect(fetchNewsItems).not.toHaveBeenCalled()
  })

  it('lists the window, opens an article, and sends See all to the Library in News mode', async () => {
    const shell = useShellStore()
    shell.contentTypes = ['podcast_episode', 'news_article']
    vi.mocked(fetchNewsItems).mockResolvedValue({ path: '/c', items: [news(1), news(2)], total: 12 })
    const w = mount(DigestNewsBand, { props: { sinceYmd: '2026-11-10' } })
    await flushPromises()
    expect(fetchNewsItems).toHaveBeenCalledWith('/c', { since: '2026-11-10', limit: 10 })
    expect(w.findAll('[data-testid="digest-news-row"]')).toHaveLength(2)
    expect(w.get('[data-testid="digest-news-band"]').text()).toContain('12')
    await w.get('[data-testid="digest-news-row"] button').trigger('click')
    expect(useSubjectStore().articleItemId).toBe('item:news_article:1')
    await w.get('[data-testid="digest-news-see-all"]').trigger('click')
    expect(shell.libraryMode).toBe('news')
    expect(w.emitted('switch-main-tab')).toEqual([['library']])
  })

  it('See all opens News mode on every publisher, not the last one opened', async () => {
    // Review 2026-11-19: See all left libraryNewsPublisherId as it was, so after opening a
    // publisher's page from an article, See all opened that publisher's page again.
    const shell = useShellStore()
    shell.contentTypes = ['podcast_episode', 'news_article']
    shell.libraryNewsPublisherId = 'org:daily-example'
    vi.mocked(fetchNewsItems).mockResolvedValue({ path: '/c', items: [news(1)], total: 1 })
    const w = mount(DigestNewsBand, { props: { sinceYmd: '' } })
    await flushPromises()
    await w.get('[data-testid="digest-news-see-all"]').trigger('click')
    expect(shell.libraryMode).toBe('news')
    expect(shell.libraryNewsPublisherId).toBe('')
  })

  it('renders nothing when the window holds no news', async () => {
    useShellStore().contentTypes = ['podcast_episode', 'news_article']
    vi.mocked(fetchNewsItems).mockResolvedValue({ path: '/c', items: [], total: 0 })
    const w = mount(DigestNewsBand, { props: { sinceYmd: '' } })
    await flushPromises()
    expect(w.find('[data-testid="digest-news-band"]').exists()).toBe(false)
  })
})
