import { flushPromises, mount } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
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

afterEach(() => vi.restoreAllMocks())

async function mountAt(q: string) {
  const router = makeRouter()
  router.push({ name: 'search', query: { q } })
  await router.isReady()
  const w = mount(SearchView, { global: { plugins: [i18n, router] } })
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
})
