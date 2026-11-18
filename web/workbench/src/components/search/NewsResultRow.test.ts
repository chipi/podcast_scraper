// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import type { NewsItem } from '../../api/searchApi'
import NewsResultRow from './NewsResultRow.vue'

function item(over: Partial<NewsItem> = {}): NewsItem {
  return {
    content_type: 'news_article',
    item_id: 'item:news_article:abc',
    title: 'Risk desks learn to sit still',
    url: 'https://www.dailyexample.test/risk',
    publisher: { id: 'org:daily-example', name: 'Daily Example', domain: 'dailyexample.test' },
    published_at: '2026-11-17T07:30:00Z',
    language: 'en',
    byline: ['Ana Example'],
    editorial_type: 'analysis',
    paywalled: false,
    text_status: 'full',
    serveable: 'summary',
    summary: 'Desks say the hard part is behavioural.',
    ...over,
  }
}

describe('NewsResultRow (UXS-019)', () => {
  it('shows the publisher, byline, title, summary and a link out', () => {
    const w = mount(NewsResultRow, { props: { item: item() } })
    expect(w.get('[data-testid="search-result-news-meta"]').text()).toContain('Daily Example')
    expect(w.get('[data-testid="search-result-news-meta"]').text()).toContain('By Ana Example')
    expect(w.get('[data-testid="search-result-news-title"]').text()).toBe('Risk desks learn to sit still')
    expect(w.get('[data-testid="search-result-news-summary"]').text()).toContain('behavioural')
    expect(w.get('[data-testid="search-result-news-open"]').attributes('href')).toBe(
      'https://www.dailyexample.test/risk',
    )
    expect(w.get('[data-testid="search-result-news-editorial"]').text()).toBe('Analysis')
  })

  it('shows no summary when its rights are metadata only', () => {
    const w = mount(NewsResultRow, { props: { item: item({ serveable: 'metadata' }) } })
    expect(w.find('[data-testid="search-result-news-summary"]').exists()).toBe(false)
  })

  it('marks a teaser and a video summary as from the feed, and reporting gets no badge', () => {
    for (const status of ['teaser', 'video']) {
      const w = mount(NewsResultRow, {
        props: { item: item({ text_status: status, editorial_type: 'reporting' }) },
      })
      expect(w.text()).toContain('Summary from the feed')
      expect(w.find('[data-testid="search-result-news-editorial"]').exists()).toBe(false)
    }
  })

  it('leaves the byline out when there is none and notes a paywall', () => {
    const w = mount(NewsResultRow, { props: { item: item({ byline: [], paywalled: true }) } })
    expect(w.get('[data-testid="search-result-news-meta"]').text()).not.toContain('By')
    expect(w.text()).toContain('Paywalled')
  })

  it('opens the article from the title and the rail button', async () => {
    const w = mount(NewsResultRow, { props: { item: item() } })
    await w.get('[data-testid="search-result-news-title"]').trigger('click')
    await w.get('[data-testid="search-result-news-rail"]').trigger('click')
    expect(w.emitted('open')).toHaveLength(2)
  })
})
