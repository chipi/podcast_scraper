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

  // #1261-3: multiple foldable hits on one episode collapse into one summary row
  it('folds N transcript hits per episode into a single "Transcript · N matches" row that expands on tap', async () => {
    vi.spyOn(api, 'searchCorpus').mockResolvedValue({
      query: 'ai',
      error: null,
      results: [
        {
          doc_id: 't1',
          score: 0.9,
          text: 'First matching chunk.',
          source_tier: 'segment',
          metadata: {
            doc_type: 'transcript',
            episode_slug: 'show-x',
            episode_title: 'Ep X',
            podcast_title: 'Show',
          },
        },
        {
          doc_id: 't2',
          score: 0.7,
          text: 'Second matching chunk.',
          source_tier: 'segment',
          metadata: {
            doc_type: 'transcript',
            episode_slug: 'show-x',
            episode_title: 'Ep X',
            podcast_title: 'Show',
          },
        },
        {
          doc_id: 't3',
          score: 0.6,
          text: 'Third matching chunk.',
          source_tier: 'segment',
          metadata: {
            doc_type: 'transcript',
            episode_slug: 'show-x',
            episode_title: 'Ep X',
            podcast_title: 'Show',
          },
        },
      ],
    })
    const { w } = await mountAt('ai')
    const clusterRows = w.findAll('[data-testid="folded-cluster-row"]')
    expect(clusterRows).toHaveLength(1)
    // Collapsed state — only the summary row is present, no excerpts yet.
    expect(w.text()).toContain('3 matches')
    expect(w.text()).not.toContain('First matching chunk.')
    // Expand.
    await clusterRows[0].trigger('click')
    expect(w.text()).toContain('First matching chunk.')
    expect(w.text()).toContain('Second matching chunk.')
    expect(w.text()).toContain('Third matching chunk.')
    // Collapse again.
    await clusterRows[0].trigger('click')
    expect(w.text()).not.toContain('First matching chunk.')
  })

  // #1261-5: matched-field kicker on the episode-group header
  it('renders "Matched: Title · Summary ×2 · Transcript" chips on the episode header', async () => {
    vi.spyOn(api, 'searchCorpus').mockResolvedValue({
      query: 'x',
      error: null,
      results: [
        {
          doc_id: 't1',
          score: 0.9,
          text: 'Title match.',
          source_tier: 'segment',
          metadata: {
            doc_type: 'episode_title',
            episode_slug: 'show-x',
            episode_title: 'Ep X',
            podcast_title: 'Show',
          },
        },
        {
          doc_id: 's1',
          score: 0.8,
          text: 'Summary 1.',
          source_tier: 'segment',
          metadata: {
            doc_type: 'summary_short',
            episode_slug: 'show-x',
            episode_title: 'Ep X',
            podcast_title: 'Show',
          },
        },
        {
          doc_id: 's2',
          score: 0.7,
          text: 'Summary 2.',
          source_tier: 'segment',
          metadata: {
            doc_type: 'summary_short',
            episode_slug: 'show-x',
            episode_title: 'Ep X',
            podcast_title: 'Show',
          },
        },
        {
          doc_id: 'tr1',
          score: 0.6,
          text: 'Transcript match.',
          source_tier: 'segment',
          metadata: {
            doc_type: 'transcript',
            episode_slug: 'show-x',
            episode_title: 'Ep X',
            podcast_title: 'Show',
          },
        },
      ],
    })
    const { w } = await mountAt('x')
    const chips = w.get('[data-testid="matched-fields"]')
    expect(chips.text()).toContain('Matched:')
    expect(chips.text()).toContain('Title')
    expect(chips.text()).toContain('Summary ×2')
    expect(chips.text()).toContain('Transcript')
  })

  it('hides the matched-fields kicker when no hit resolves to an episode-level field', async () => {
    vi.spyOn(api, 'searchCorpus').mockResolvedValue({
      query: 'x',
      error: null,
      results: [
        {
          doc_id: 'kg1',
          score: 0.9,
          text: 't',
          source_tier: 'kg',
          metadata: {
            doc_type: 'kg_topic',
            episode_slug: 'show-x',
            episode_title: 'Ep X',
            podcast_title: 'Show',
          },
        },
      ],
    })
    const { w } = await mountAt('x')
    expect(w.find('[data-testid="matched-fields"]').exists()).toBe(false)
  })

  // #1261-7: year-header grouping — mobile-friendly reshape of the timeline chart
  it('shows year section headers when results span multiple publish years', async () => {
    vi.spyOn(api, 'searchCorpus').mockResolvedValue({
      query: 'ai',
      error: null,
      results: [
        {
          doc_id: 'a',
          score: 0.9,
          text: 'a',
          source_tier: 'insight',
          metadata: {
            doc_type: 'insight',
            episode_slug: 'ep-2024',
            episode_title: 'From 2024',
            publish_date: '2024-04-01',
          },
        },
        {
          doc_id: 'b',
          score: 0.8,
          text: 'b',
          source_tier: 'insight',
          metadata: {
            doc_type: 'insight',
            episode_slug: 'ep-2023',
            episode_title: 'From 2023',
            publish_date: '2023-11-01',
          },
        },
      ],
    })
    const { w } = await mountAt('ai')
    const headers = w.findAll('[data-testid="year-header"]')
    expect(headers).toHaveLength(2)
    expect(headers[0].text()).toContain('2024')
    expect(headers[1].text()).toContain('2023')
  })

  it('hides year headers when results all fall in a single year', async () => {
    vi.spyOn(api, 'searchCorpus').mockResolvedValue({
      query: 'ai',
      error: null,
      results: [
        {
          doc_id: 'a',
          score: 0.9,
          text: 'a',
          source_tier: 'insight',
          metadata: {
            doc_type: 'insight',
            episode_slug: 'ep-2024a',
            episode_title: 'Ep A',
            publish_date: '2024-04-01',
          },
        },
        {
          doc_id: 'b',
          score: 0.8,
          text: 'b',
          source_tier: 'insight',
          metadata: {
            doc_type: 'insight',
            episode_slug: 'ep-2024b',
            episode_title: 'Ep B',
            publish_date: '2024-11-01',
          },
        },
      ],
    })
    const { w } = await mountAt('ai')
    expect(w.findAll('[data-testid="year-header"]')).toHaveLength(0)
  })

  it('keeps insight and kg_topic hits out of the fold — they render as standalone rows', async () => {
    vi.spyOn(api, 'searchCorpus').mockResolvedValue({
      query: 'ai',
      error: null,
      results: [
        {
          doc_id: 'i1',
          score: 0.9,
          text: 'A grounded insight.',
          source_tier: 'insight',
          metadata: {
            doc_type: 'insight',
            episode_slug: 'show-x',
            episode_title: 'Ep X',
            podcast_title: 'Show',
          },
        },
        {
          doc_id: 't1',
          score: 0.7,
          text: 'A transcript chunk.',
          source_tier: 'segment',
          metadata: {
            doc_type: 'transcript',
            episode_slug: 'show-x',
            episode_title: 'Ep X',
            podcast_title: 'Show',
          },
        },
      ],
    })
    const { w } = await mountAt('ai')
    // The insight text shows up without needing to expand anything.
    expect(w.text()).toContain('A grounded insight.')
    // The single-transcript-hit cluster is present but collapsed by default.
    expect(w.findAll('[data-testid="folded-cluster-row"]')).toHaveLength(1)
    expect(w.text()).not.toContain('A transcript chunk.')
  })
})
