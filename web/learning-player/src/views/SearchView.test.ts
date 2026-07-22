import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import { useAuthStore } from '../stores/auth'
import SearchView from './SearchView.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })

function makeRouter() {
  return createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: '/search', name: 'search', component: SearchView },
      { path: '/episode/:slug', name: 'player', component: { template: '<div/>' } },
    ],
  })
}

// Default: no entity match — search tests assert passage behaviour without an entity card.
beforeEach(() => {
  vi.spyOn(api, 'resolveEntity').mockResolvedValue({ query: '', entity: null })
})
afterEach(() => vi.restoreAllMocks())

async function mountAt(q: string) {
  const router = makeRouter()
  router.push({ name: 'search', query: { q } })
  await router.isReady()
  const w = mount(SearchView, {
    global: { plugins: [i18n, router, createPinia()], stubs: { teleport: true } },
  })
  await flushPromises()
  return { w, router }
}

describe('SearchView', () => {
  it('renders grounded passages with source + jump, and jumps with ?t=', async () => {
    vi.spyOn(api, 'searchCorpus').mockResolvedValue({
      query: 'memory',
      error: null,
      results: [
        {
          doc_id: 'd1', score: 0.9, text: 'A grounded passage about memory.', source_tier: 'segment',
          metadata: { episode_slug: 'show-x', episode_title: 'Ep X', podcast_title: 'Show' },
          lifted: { quote: { timestamp_start_ms: 20000 } },
        },
      ],
    })
    const { w, router } = await mountAt('memory')
    expect(w.text()).toContain('A grounded passage about memory.')
    expect(w.text()).toContain('Ep X')
    const push = vi.spyOn(router, 'push')
    await w.findAll('button').find((b) => b.text().includes('0:20'))!.trigger('click')
    expect(push).toHaveBeenCalledWith({ name: 'player', params: { slug: 'show-x' }, query: { t: '20' } })
  })

  it('shows the no-index message on error', async () => {
    vi.spyOn(api, 'searchCorpus').mockResolvedValue({ query: 'x', error: 'no_index', results: [] })
    const { w } = await mountAt('x')
    expect(w.text()).toContain('Search needs the library index.')
  })

  it('shows no-results when empty without error', async () => {
    vi.spyOn(api, 'searchCorpus').mockResolvedValue({ query: 'x', error: null, results: [] })
    const { w } = await mountAt('x')
    expect(w.text()).toContain('No grounded passages found.')
  })

  it('surfaces an entity card above passages and opens the full card on tap (3.4)', async () => {
    vi.spyOn(api, 'searchCorpus').mockResolvedValue({ query: 'jane', error: null, results: [] })
    vi.spyOn(api, 'resolveEntity').mockResolvedValue({
      query: 'jane',
      entity: { id: 'person:jane-doe', kind: 'person', label: 'Jane Doe' },
    })
    const getPerson = vi.spyOn(api, 'getPersonCard').mockResolvedValue({
      id: 'person:jane-doe',
      label: 'Jane Doe',
      episode_count: 0,
      episodes: [],
      related_people: [],
      related_topics: [],
    })
    const { w } = await mountAt('jane')
    // Entity hit card shows (Person kicker + name); the no-results line is suppressed.
    expect(w.text()).toContain('Person')
    expect(w.text()).toContain('Jane Doe')
    expect(w.text()).not.toContain('No grounded passages found.')
    // Tapping it opens the full EntityCard overlay.
    await w.findAll('button').find((b) => b.text().includes('View'))!.trigger('click')
    await flushPromises()
    expect(getPerson).toHaveBeenCalledWith('person:jane-doe', undefined)
    expect(w.find('[role="dialog"]').exists()).toBe(true)
  })

  it('hides the Recall scope toggle when signed out', async () => {
    vi.spyOn(api, 'searchCorpus').mockResolvedValue({ query: 'x', error: null, results: [] })
    const { w } = await mountAt('x')
    expect(w.find('[role="tablist"]').exists()).toBe(false)
  })

  it('signed-in: My corpus scope searches scope=mine and shows a recall-specific empty message', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    useAuthStore().user = { user_id: 'u1', email: 'a@b.c', name: 'A' }
    const search = vi.spyOn(api, 'searchCorpus').mockResolvedValue({
      query: 'sleep',
      error: null,
      results: [],
    })
    const router = makeRouter()
    router.push({ name: 'search', query: { q: 'sleep' } })
    await router.isReady()
    const w = mount(SearchView, {
      global: { plugins: [i18n, router, pinia], stubs: { teleport: true } },
    })
    await flushPromises()
    // toggle is visible; default scope=all sent no 'mine'
    expect(w.find('[role="tablist"]').exists()).toBe(true)
    // 4th positional arg is enrich_results=true (#1261-2): the listener always asks the
    // server to decorate hits with related_topics so the "Also about:" chip row can render.
    expect(search).toHaveBeenLastCalledWith('sleep', 12, 'all', true)
    // switch to My corpus → searches scope=mine + recall-empty copy
    await w.findAll('[role="tab"]').find((b) => b.text() === 'My corpus')!.trigger('click')
    await flushPromises()
    expect(search).toHaveBeenLastCalledWith('sleep', 12, 'mine', true)
    expect(w.text()).toContain('Nothing in your corpus on this yet')
  })

  // #1261-2: enriched related-topic chips above episode groups
  it('renders "Also about:" chips from server-decorated hits and opens the topic card on tap', async () => {
    vi.spyOn(api, 'searchCorpus').mockResolvedValue({
      query: 'ai',
      error: null,
      results: [
        {
          doc_id: 'd1',
          score: 0.9,
          text: 'A grounded passage about AI.',
          source_tier: 'insight',
          metadata: {
            episode_slug: 'show-x',
            episode_title: 'Ep X',
            podcast_title: 'Show',
            query_enrichments: {
              related_topics: [
                { topic_id: 'topic:ml', topic_label: 'Machine Learning', similarity: 0.91 },
                { topic_id: 'topic:safety', topic_label: 'AI Safety', similarity: 0.83 },
              ],
            },
          },
        },
      ],
    })
    const getTopic = vi.spyOn(api, 'getTopicCard').mockResolvedValue({
      id: 'topic:ml',
      label: 'Machine Learning',
      cluster_id: null,
      cluster_label: null,
      cluster_size: 0,
      sibling_topics: [],
      episode_count: 0,
      episodes: [],
      related_people: [],
    })
    const { w } = await mountAt('ai')
    const chipRow = w.get('[data-testid="related-topic-chips"]')
    expect(chipRow.text()).toContain('Machine Learning')
    expect(chipRow.text()).toContain('AI Safety')
    // Score-desc: ML (0.91) sorts ahead of Safety (0.83).
    const chipButtons = chipRow.findAll('button')
    expect(chipButtons[0].text()).toBe('Machine Learning')
    await chipButtons[0].trigger('click')
    await flushPromises()
    expect(getTopic).toHaveBeenCalledWith('topic:ml', undefined)
    expect(w.find('[role="dialog"]').exists()).toBe(true)
  })

  it('hides the chip row entirely when no hit carries related_topics decoration', async () => {
    vi.spyOn(api, 'searchCorpus').mockResolvedValue({
      query: 'x',
      error: null,
      results: [
        {
          doc_id: 'd1',
          score: 0.9,
          text: 't',
          source_tier: 'insight',
          metadata: { episode_slug: 'show-x', episode_title: 'Ep X', podcast_title: 'Show' },
        },
      ],
    })
    const { w } = await mountAt('x')
    expect(w.find('[data-testid="related-topic-chips"]').exists()).toBe(false)
  })
})
