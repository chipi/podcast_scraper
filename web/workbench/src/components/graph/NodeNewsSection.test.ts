// @vitest-environment happy-dom
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import NodeNewsSection from './NodeNewsSection.vue'
import { fetchNodeEpisodes } from '../../api/corpusLibraryApi'
import type { NewsItem } from '../../api/searchApi'
import { useShellStore } from '../../stores/shell'
import { useSubjectStore } from '../../stores/subject'

vi.mock('../../api/corpusLibraryApi', () => ({ fetchNodeEpisodes: vi.fn() }))

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

describe('NodeNewsSection (UXS-019 "In the news")', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    const shell = useShellStore()
    shell.corpusPath = '/c'
    shell.healthStatus = 'ok'
    vi.mocked(fetchNodeEpisodes).mockReset()
  })

  it('asks nothing when the corpus holds no news', async () => {
    mount(NodeNewsSection, { props: { nodeId: 'k:topic:risk' } })
    await flushPromises()
    expect(fetchNodeEpisodes).not.toHaveBeenCalled()
  })

  it('lists five, shows all on request, and opens the article card', async () => {
    useShellStore().contentTypes = ['podcast_episode', 'news_article']
    vi.mocked(fetchNodeEpisodes).mockResolvedValue({
      path: '/c',
      node_id: 'topic:risk',
      episodes: [],
      news_items: [1, 2, 3, 4, 5, 6, 7].map(news),
      truncated: false,
      total_matched: null,
    })
    const w = mount(NodeNewsSection, { props: { nodeId: 'k:topic:risk' } })
    await flushPromises()
    expect(fetchNodeEpisodes).toHaveBeenCalledWith('/c', 'k:topic:risk', 1, ['news_article'])
    expect(w.findAll('[data-testid="node-news-item"]')).toHaveLength(5)
    expect(w.get('[data-testid="node-news-item"]').text()).toContain('Daily Example')
    await w.get('[data-testid="node-news-show-all"]').trigger('click')
    expect(w.findAll('[data-testid="node-news-item"]')).toHaveLength(7)
    await w.get('[data-testid="node-news-item"] button').trigger('click')
    const subject = useSubjectStore()
    expect(subject.kind).toBe('article')
    expect(subject.articleItemId).toBe('item:news_article:1')
  })

  it('renders nothing when no news item names the node', async () => {
    useShellStore().contentTypes = ['podcast_episode', 'news_article']
    vi.mocked(fetchNodeEpisodes).mockResolvedValue({
      path: '/c',
      node_id: 'topic:risk',
      episodes: [],
      news_items: [],
      truncated: false,
      total_matched: null,
    })
    const w = mount(NodeNewsSection, { props: { nodeId: 'k:topic:risk' } })
    await flushPromises()
    expect(w.find('[data-testid="node-news-section"]').exists()).toBe(false)
  })
})
