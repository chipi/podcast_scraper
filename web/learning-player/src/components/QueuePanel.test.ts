import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import type { EpisodeDetail, PlaybackPosition } from '../services/types'
import QueuePanel from './QueuePanel.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const stub = { template: '<div/>' }
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/', name: 'home', component: stub },
    { path: '/episode/:slug', name: 'player', component: stub },
    { path: '/podcast/:feedId', name: 'podcast', component: stub },
  ],
})

function detail(slug: string, title: string): EpisodeDetail {
  return {
    slug,
    title,
    feed_id: 'f',
    podcast_title: 'Show',
    publish_date: '2024-01-01',
    duration_seconds: 1800,
    episode_image_url: null,
    feed_image_url: null,
    artwork_url: null,
    summary_text: null,
    summary_bullets: [],
    has_transcript: true,
    has_summary: false,
    has_gi: false,
    has_kg: false,
    has_bridge: false,
  } as unknown as EpisodeDetail
}

async function mountIt() {
  setActivePinia(createPinia())
  await router.push('/')
  await router.isReady()
  const w = mount(QueuePanel, {
    global: { plugins: [i18n, router], stubs: { teleport: true } },
  })
  await flushPromises()
  return w
}

afterEach(() => vi.restoreAllMocks())

describe('QueuePanel (#1838)', () => {
  it('shows Up next (the queue) and Recently played (history)', async () => {
    vi.spyOn(api, 'getQueue').mockResolvedValue(['q-1'])
    vi.spyOn(api, 'getPlaybackList').mockResolvedValue([
      { slug: 'r-1', position_seconds: 30, finished: false } as PlaybackPosition,
    ])
    vi.spyOn(api, 'getEpisode').mockImplementation(async (slug: string) =>
      slug === 'q-1' ? detail('q-1', 'Queued Ep') : detail('r-1', 'Recent Ep'),
    )
    const w = await mountIt()
    expect(w.find('[data-testid="queue-panel"]').exists()).toBe(true)
    expect(w.text()).toContain('Up next')
    expect(w.text()).toContain('Recently played')
    expect(w.text()).toContain('Queued Ep') // from the queue
    expect(w.get('[data-testid="queue-panel-recent"]').text()).toContain('Recent Ep')
  })

  it('emits close on the close button', async () => {
    vi.spyOn(api, 'getQueue').mockResolvedValue([])
    vi.spyOn(api, 'getPlaybackList').mockResolvedValue([])
    const w = await mountIt()
    await w.get('[data-testid="queue-panel-close"]').trigger('click')
    expect(w.emitted('close')).toBeTruthy()
  })
})
