// @vitest-environment happy-dom
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import ArticleRailPanel from './ArticleRailPanel.vue'
import { fetchNewsItem, type NewsItemDetail } from '../../api/newsApi'
import { useShellStore } from '../../stores/shell'
import { useSubjectStore } from '../../stores/subject'

vi.mock('../../api/newsApi', () => ({ fetchNewsItem: vi.fn() }))

function detail(over: Partial<NewsItemDetail['item']> = {}): NewsItemDetail {
  return {
    path: '/c',
    item: {
      content_type: 'news_article',
      item_id: 'item:news_article:abc',
      title: 'Risk desks learn to sit still',
      url: 'https://www.dailyexample.test/risk',
      publisher: { id: 'org:daily-example', name: 'Daily Example', domain: 'dailyexample.test' },
      published_at: '2026-11-17T07:30:00Z',
      language: 'en',
      byline: ['Ana Example', 'Cy Example'],
      serveable: 'summary',
      summary: 'Desks say the hard part is behavioural.',
      ...over,
    },
    kg_relative_path: 'sources/x/risk.kg.json',
    kg_nodes: [
      { id: 'person:ana-example', type: 'Person', label: 'Ana Example', role: 'author' },
      { id: 'topic:risk-management', type: 'Topic', label: 'Risk Management' },
      { id: 'person:skanda-amarnath', type: 'Person', label: 'Skanda Amarnath' },
    ],
  }
}

describe('ArticleRailPanel (UXS-019)', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    useShellStore().corpusPath = '/c'
    vi.mocked(fetchNewsItem).mockReset()
  })

  it('shows the card and links a joined author to the person rail', async () => {
    vi.mocked(fetchNewsItem).mockResolvedValue(detail())
    const subject = useSubjectStore()
    subject.focusArticle('item:news_article:abc', { uiTitle: 'Risk desks' })
    const w = mount(ArticleRailPanel)
    await flushPromises()
    expect(fetchNewsItem).toHaveBeenCalledWith('/c', 'item:news_article:abc')
    expect(w.get('[data-testid="article-title"]').text()).toBe('Risk desks learn to sit still')
    expect(w.get('[data-testid="article-publisher"]').text()).toContain('dailyexample.test')
    expect(w.get('[data-testid="article-summary"]').text()).toContain('behavioural')
    const chips = w.findAll('[data-testid="article-kg-chip"]').map((c) => c.text())
    expect(chips).toEqual(['Risk Management', 'Skanda Amarnath'])
    const byline = w.get('[data-testid="article-byline"]')
    expect(byline.findAll('button').map((b) => b.text())).toEqual(['Ana Example'])
    await byline.get('button').trigger('click')
    // People open in the unified node view (kind graph-node).
    expect(subject.kind).toBe('graph-node')
    expect(subject.graphNodeCyId).toBe('person:ana-example')
  })

  it('shows title and link only for a metadata-only item', async () => {
    vi.mocked(fetchNewsItem).mockResolvedValue(detail({ serveable: 'metadata', summary: null }))
    useSubjectStore().focusArticle('item:news_article:abc')
    const w = mount(ArticleRailPanel)
    await flushPromises()
    expect(w.find('[data-testid="article-summary"]').exists()).toBe(false)
    expect(w.find('[data-testid="article-metadata-only"]').exists()).toBe(true)
    expect(w.find('[data-testid="article-open"]').exists()).toBe(true)
  })
})
