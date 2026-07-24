import { flushPromises, mount, RouterLinkStub } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import type { Podcast, TrendingEntity } from '../services/types'
import TrendingShowsRail from './TrendingShowsRail.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })

const rows: TrendingEntity[] = [
  { entity_id: 'f0', kind: 'show', label: 'Latent Space', velocity: 1.7, volume: 9, heating_up: true, total: 30, series: [1, 2, 4, 8] },
  { entity_id: 'f1', kind: 'show', label: 'The Daily', velocity: 0.5, volume: 8, heating_up: false, total: 20, series: [8, 5, 3, 2] },
]
const podcasts: Podcast[] = [
  { feed_id: 'f0', title: 'Latent Space', artwork_url: 'https://img/f0.jpg', image_url: null, description: null, episode_count: 30 },
  { feed_id: 'f1', title: 'The Daily', artwork_url: null, image_url: 'https://img/f1.jpg', description: null, episode_count: 20 },
]

const mountIt = (items = rows) => {
  vi.spyOn(api, 'getTrending').mockResolvedValue(items)
  return mount(TrendingShowsRail, {
    props: { title: 'Trending shows', podcasts },
    global: { plugins: [i18n], stubs: { RouterLink: RouterLinkStub } },
  })
}

afterEach(() => vi.restoreAllMocks())

describe('TrendingShowsRail', () => {
  it('renders a cover-art card per trending show, linking to the show page', async () => {
    const w = mountIt()
    await flushPromises()
    const cards = w.findAll('[data-testid="trending-show-card"]')
    expect(cards).toHaveLength(2)
    // entity_id == feed_id → links to the podcast route.
    expect(cards[0].getComponent(RouterLinkStub).props('to')).toEqual({
      name: 'podcast',
      params: { feedId: 'f0' },
    })
    expect(cards[0].text()).toContain('Latent Space')
    expect(cards[0].text()).toContain('1.7×')
  })

  it('joins artwork from the podcasts list by feed_id (artwork_url then image_url)', async () => {
    const w = mountIt()
    await flushPromises()
    const imgs = w.findAll('[data-testid="trending-show-card"] img')
    expect(imgs[0].attributes('src')).toBe('https://img/f0.jpg') // artwork_url wins
    expect(imgs[1].attributes('src')).toBe('https://img/f1.jpg') // falls back to image_url
  })

  it('renders each show its weekly-cadence sparkline (svg)', async () => {
    const w = mountIt()
    await flushPromises()
    expect(w.findAll('[data-testid="trending-show-card"] svg').length).toBe(2)
  })

  it('hides entirely when nothing is trending', async () => {
    const w = mountIt([])
    await flushPromises()
    expect(w.find('[data-testid="trending-shows-rail"]').exists()).toBe(false)
  })
})
