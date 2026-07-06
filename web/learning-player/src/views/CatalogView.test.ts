import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import type { EpisodeSummary } from '../services/types'
import CatalogView from './CatalogView.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/', name: 'catalog', component: CatalogView },
    { path: '/podcast/:feedId', name: 'podcast', component: { template: '<div/>' } },
    { path: '/episode/:slug', name: 'player', component: { template: '<div/>' } },
  ],
})

function ep(slug: string, title: string): EpisodeSummary {
  return {
    slug, title, feed_id: 'f', podcast_title: 'Show', publish_date: '2024-01-01',
    duration_seconds: 1800, episode_image_url: null, feed_image_url: null, artwork_url: null,
    status: 'ready', summary_preview: 'recap', topics: [], has_transcript: true,
    has_summary: true, has_gi: false, has_kg: false, has_bridge: false,
  }
}

beforeEach(() => setActivePinia(createPinia()))
afterEach(() => vi.restoreAllMocks())

function mountView() {
  return mount(CatalogView, { global: { plugins: [i18n, router] } })
}

describe('CatalogView', () => {
  it('renders episode cards from the API', async () => {
    vi.spyOn(api, 'listEpisodes').mockResolvedValue({
      items: [ep('a-1', 'First'), ep('a-2', 'Second')],
      page: 1, page_size: 20, total: 2, has_more: false,
    })
    const w = mountView()
    await flushPromises()
    expect(w.text()).toContain('First')
    expect(w.text()).toContain('Second')
    expect(w.text()).not.toContain('Load more')
  })

  it('shows Load more and appends the next page', async () => {
    const spy = vi.spyOn(api, 'listEpisodes')
    spy.mockResolvedValueOnce({ items: [ep('a-1', 'First')], page: 1, page_size: 20, total: 2, has_more: true })
    spy.mockResolvedValueOnce({ items: [ep('a-2', 'Second')], page: 2, page_size: 20, total: 2, has_more: false })
    const w = mountView()
    await flushPromises()
    expect(w.text()).toContain('Load more')
    await w.findAll('button').find((b) => b.text() === 'Load more')!.trigger('click')
    await flushPromises()
    expect(w.text()).toContain('Second')
    expect(spy).toHaveBeenCalledTimes(2)
  })

  it('shows the empty state when there are no episodes', async () => {
    vi.spyOn(api, 'listEpisodes').mockResolvedValue({ items: [], page: 1, page_size: 20, total: 0, has_more: false })
    const w = mountView()
    await flushPromises()
    expect(w.text()).toContain('No episodes yet.')
  })

  it('shows an error message when the API fails', async () => {
    vi.spyOn(api, 'listEpisodes').mockRejectedValue(new Error('boom'))
    const w = mountView()
    await flushPromises()
    expect(w.text()).toContain('Couldn’t load episodes.')
  })
})
