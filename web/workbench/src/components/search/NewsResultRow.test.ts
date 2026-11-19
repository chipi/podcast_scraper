// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import type { NewsItem } from '../../api/searchApi'
import NewsResultRow from './NewsResultRow.vue'
import { useShellStore } from '../../stores/shell'

beforeEach(() => {
  setActivePinia(createPinia())
})

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

  it('names the site when there is no publisher, never a source id', () => {
    const w = mount(NewsResultRow, {
      props: {
        item: item({ publisher: null, site: 'nicolasbouliane.com', source_id: 'source:a8c3140f4b866cb9bc76' }),
      },
    })
    const meta = w.get('[data-testid="search-result-news-meta"]').text()
    expect(meta).toContain('nicolasbouliane.com')
    expect(meta).not.toContain('source:')
  })

  it('shows no summary when its rights are metadata only', () => {
    const w = mount(NewsResultRow, { props: { item: item({ serveable: 'metadata' }) } })
    expect(w.find('[data-testid="search-result-news-summary"]').exists()).toBe(false)
  })

  it('a shown teaser (classed a news summary) has no summary and is not "No summary yet"', () => {
    // The writer summarises full text only; a teaser shows title, metadata and link (2026-11-19).
    const w = mount(NewsResultRow, {
      props: { item: item({ text_status: 'teaser', summary: null, editorial_type: 'reporting' }) },
    })
    expect(w.find('[data-testid="search-result-news-summary"]').exists()).toBe(false)
    expect(w.find('[data-testid="search-result-news-no-summary"]').exists()).toBe(false)
    expect(w.find('[data-testid="search-result-news-editorial"]').exists()).toBe(false)
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

describe('NewsResultRow, the spec points the review found missing (2026-11-19)', () => {
  it('names where its link goes', () => {
    setActivePinia(createPinia())
    const w = mount(NewsResultRow, { props: { item: item() } })
    expect(w.get('[data-testid="search-result-news-open"]').attributes('aria-label')).toBe(
      'Open article on dailyexample.test, opens in a new tab',
    )
    const viaSite = mount(NewsResultRow, { props: { item: item({ publisher: null, site: 'blog.test' }) } })
    expect(viaSite.get('[data-testid="search-result-news-open"]').attributes('aria-label')).toBe(
      'Open article on blog.test, opens in a new tab',
    )
  })

  it('says why there is no summary: metadata only, or not written yet', () => {
    setActivePinia(createPinia())
    const meta = mount(NewsResultRow, { props: { item: item({ serveable: 'metadata', summary: null }) } })
    expect(meta.find('[data-testid="search-result-news-metadata-only"]').exists()).toBe(true)
    const pending = mount(NewsResultRow, { props: { item: item({ summary: null }) } })
    expect(pending.get('[data-testid="search-result-news-no-summary"]').text()).toBe('No summary yet')
    const shown = mount(NewsResultRow, { props: { item: item() } })
    expect(shown.find('[data-testid="search-result-news-no-summary"]').exists()).toBe(false)
  })

  it("opens the publisher's page from its name, and the title is described by the metadata", async () => {
    setActivePinia(createPinia())
    const shell = useShellStore()
    const w = mount(NewsResultRow, { props: { item: item() } })
    await w.get('[data-testid="search-result-news-publisher"]').trigger('click')
    expect(shell.libraryMode).toBe('news')
    expect(shell.libraryNewsPublisherId).toBe('org:daily-example')
    const metaId = w.get('[data-testid="search-result-news-meta"]').attributes('id')
    expect(w.get('[data-testid="search-result-news-title"]').attributes('aria-describedby')).toBe(metaId)
  })
})
