// @vitest-environment happy-dom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'

import ShowsView from './ShowsView.vue'
import { useShellStore } from '../../stores/shell'

/**
 * UXS-015 / RFC-104 — mount tests for the shows-first grid. Verifies the grid
 * renders a card per feed (cover + count), sorts by episode count, emits the
 * selected feed, and shows the empty state when the corpus has no shows.
 */

function res(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function stubFeeds(feeds: unknown[]): void {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url.includes('/api/corpus/feeds')) return res({ path: '/corpus', feeds })
      return res({}, 404)
    }),
  )
}

function withCorpus(): void {
  useShellStore().corpusPath = '/corpus'
}

beforeEach(() => {
  setActivePinia(createPinia())
})

afterEach(() => {
  vi.unstubAllGlobals()
})

const FEEDS = [
  { feed_id: 'small', display_title: 'Small Show', episode_count: 2, description: 'A show.' },
  { feed_id: 'big', display_title: 'Big Show', episode_count: 40, image_url: 'http://x/art.jpg' },
]

describe('ShowsView — mount + behaviour', () => {
  it('renders one card per feed, sorted by episode count desc', async () => {
    stubFeeds(FEEDS)
    withCorpus()
    const w = mount(ShowsView)
    await flushPromises()

    expect(w.find('[data-testid="shows-card-big"]').exists()).toBe(true)
    expect(w.find('[data-testid="shows-card-small"]').exists()).toBe(true)
    const cards = w.findAll('[data-shows-card]')
    expect(cards).toHaveLength(2)
    // Big (40) sorts before Small (2).
    expect(cards[0].attributes('data-testid')).toBe('shows-card-big')
    expect(cards[0].text()).toContain('40 episodes')
    expect(cards[1].text()).toContain('2 episodes')
  })

  it('emits the selected feed on card click', async () => {
    stubFeeds(FEEDS)
    withCorpus()
    const w = mount(ShowsView)
    await flushPromises()

    await w.find('[data-testid="shows-card-big"]').trigger('click')
    const evs = w.emitted('select')
    expect(evs).toBeTruthy()
    expect((evs![0][0] as { feed_id: string }).feed_id).toBe('big')
  })

  it('shows the empty state when the corpus has no shows', async () => {
    stubFeeds([])
    withCorpus()
    const w = mount(ShowsView)
    await flushPromises()

    expect(w.find('[data-testid="shows-grid-empty"]').exists()).toBe(true)
    expect(w.findAll('[data-shows-card]')).toHaveLength(0)
  })

  it('shows an error state when the feeds request fails', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => res({ detail: 'boom' }, 500)),
    )
    withCorpus()
    const w = mount(ShowsView)
    await flushPromises()

    expect(w.find('[data-testid="shows-grid-error"]').exists()).toBe(true)
    expect(w.findAll('[data-shows-card]')).toHaveLength(0)
  })

  it('falls back to feed_id when a show has no display title', async () => {
    stubFeeds([{ feed_id: 'raw-id', display_title: null, episode_count: 1 }])
    withCorpus()
    const w = mount(ShowsView)
    await flushPromises()

    expect(w.find('[data-testid="shows-card-raw-id"]').text()).toContain('raw-id')
    expect(w.find('[data-testid="shows-card-raw-id"]').text()).toContain('1 episode')
  })
})

describe('ShowsView — language badge (V2-C.1)', () => {
  it('labels each show with its language when the corpus holds more than one', async () => {
    stubFeeds([
      { feed_id: 'en1', display_title: 'English Show', episode_count: 3, language: 'en-US' },
      { feed_id: 'es1', display_title: 'Spanish Show', episode_count: 2, language: 'es' },
      { feed_id: 'nolang', display_title: 'Unknown Show', episode_count: 1, language: null },
    ])
    withCorpus()
    const w = mount(ShowsView)
    await flushPromises()

    const badgeOf = (id: string) =>
      w.find(`[data-testid="shows-card-${id}"] [data-testid="language-badge"]`)
    expect(badgeOf('en1').attributes('data-lang')).toBe('en')
    expect(badgeOf('es1').attributes('data-lang')).toBe('es')
    expect(badgeOf('es1').attributes('aria-label')).toBe('Language: Spanish')
    // Unknown is omitted, not guessed.
    expect(badgeOf('nolang').exists()).toBe(false)
  })

  it('shows no badge at all in a single-language corpus (FR7.5)', async () => {
    stubFeeds([
      { feed_id: 'a', display_title: 'A', episode_count: 3, language: 'en' },
      { feed_id: 'b', display_title: 'B', episode_count: 2, language: 'en-GB' },
    ])
    withCorpus()
    const w = mount(ShowsView)
    await flushPromises()
    expect(w.findAll('[data-testid="language-badge"]')).toHaveLength(0)
  })
})
