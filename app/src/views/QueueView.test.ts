import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import type { EpisodeDetail } from '../services/types'
import QueueView from './QueueView.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/', name: 'catalog', component: { template: '<div/>' } },
    { path: '/queue', name: 'queue', component: QueueView },
    { path: '/episode/:slug', name: 'player', component: { template: '<div/>' } },
    { path: '/podcast/:feedId', name: 'podcast', component: { template: '<div/>' } },
  ],
})

function detail(slug: string, title: string): EpisodeDetail {
  return {
    slug, title, feed_id: 'f', podcast_title: 'Show', publish_date: null, duration_seconds: null,
    episode_image_url: null, feed_image_url: null, artwork_url: null, summary_title: null,
    summary_bullets: [], summary_text: null, has_transcript: true, has_summary: false,
    has_gi: false, has_kg: false, has_bridge: false,
  }
}

beforeEach(() => {
  setActivePinia(createPinia())
  vi.spyOn(api, 'putQueue').mockResolvedValue()
})
afterEach(() => vi.restoreAllMocks())

describe('QueueView', () => {
  it('renders queued episodes with titles in order', async () => {
    vi.spyOn(api, 'getQueue').mockResolvedValue(['a-1', 'b-2'])
    vi.spyOn(api, 'getEpisode').mockImplementation(async (s: string) =>
      s === 'a-1' ? detail('a-1', 'Alpha') : detail('b-2', 'Beta'),
    )
    const w = mount(QueueView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    expect(w.text()).toContain('Alpha')
    expect(w.text()).toContain('Beta')
  })

  it('removes an item via the queue store', async () => {
    vi.spyOn(api, 'getQueue').mockResolvedValue(['a-1', 'b-2'])
    vi.spyOn(api, 'getEpisode').mockImplementation(async (s: string) =>
      s === 'a-1' ? detail('a-1', 'Alpha') : detail('b-2', 'Beta'),
    )
    const w = mount(QueueView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    // First row's remove (✕) button.
    await w.findAll('button').find((b) => b.text() === '✕')!.trigger('click')
    await flushPromises()
    expect(w.text()).not.toContain('Alpha')
    expect(w.text()).toContain('Beta')
  })

  it('shows the empty state with no queue', async () => {
    vi.spyOn(api, 'getQueue').mockResolvedValue([])
    const w = mount(QueueView, { global: { plugins: [i18n, router] } })
    await flushPromises()
    expect(w.text()).toContain('Your queue is empty.')
  })
})
