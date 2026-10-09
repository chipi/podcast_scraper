import { flushPromises, mount, RouterLinkStub } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { createMemoryHistory, createRouter } from 'vue-router'
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

// Pinia + a router: ShowTile carries Follow and the heart, which read the library and favourites
// stores and route signed-out taps to sign-in.
const routes = [
  { path: '/', name: 'home', component: { template: '<div/>' } },
  { path: '/login', name: 'login', component: { template: '<div/>' } },
  { path: '/podcast/:feedId', name: 'podcast', component: { template: '<div/>' } },
]

const mountIt = (items = rows, scope: 'corpus' | 'mine' = 'corpus') => {
  vi.spyOn(api, 'getTrending').mockResolvedValue(items)
  setActivePinia(createPinia())
  return mount(TrendingShowsRail, {
    props: { title: 'Trending shows', podcasts, scope },
    global: {
      plugins: [i18n, createPinia(), createRouter({ history: createMemoryHistory(), routes })],
      stubs: { RouterLink: RouterLinkStub },
    },
  })
}

afterEach(() => vi.restoreAllMocks())

describe('TrendingShowsRail', () => {
  it('renders the standard ShowTile per trending show, linking to the show page', async () => {
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
  })

  it('is the standard rail: CardRail, one slot width, three reserved title lines', async () => {
    // Every rail looks the same on every page (operator 2026-10-05). This rail had a second shape —
    // full-width cover bands with a sparkline — that no other rail shared.
    const w = mountIt()
    await flushPromises()
    expect(w.find('ul.lp-rail').exists(), 'not in CardRail').toBe(true)
    const slots = w.findAll('ul.lp-rail > li')
    expect(slots).toHaveLength(2)
    for (const li of slots) expect(li.classes()).toContain('lp-rail-item')
    expect(w.findAll('[data-testid="trending-show-card"] .lp-tile-title')).toHaveLength(2)
    expect(w.find('svg path').exists(), 'the sparkline band is back').toBe(false)
  })

  it('carries Follow and save on every tile, outside the link', async () => {
    const w = mountIt()
    await flushPromises()
    const card = w.findAll('[data-testid="trending-show-card"]')[0]
    expect(card.find('[data-testid="follow-show"]').exists(), 'the tile lost Follow').toBe(true)
    expect(card.find('[data-testid="favorite-button"]').exists(), 'the tile lost the heart').toBe(true)
    const link = card.getComponent(RouterLinkStub)
    expect(link.find('[data-testid="follow-show"]').exists(), 'Follow is nested in the <a>').toBe(false)
    expect(link.find('[data-testid="favorite-button"]').exists(), 'the heart is nested in the <a>').toBe(
      false,
    )
  })

  it('joins artwork from the podcasts list by feed_id (artwork_url then image_url)', async () => {
    const w = mountIt()
    await flushPromises()
    const imgs = w.findAll('[data-testid="trending-show-card"] img')
    expect(imgs[0].attributes('src')).toBe('https://img/f0.jpg') // artwork_url wins
    expect(imgs[1].attributes('src')).toBe('https://img/f1.jpg') // falls back to image_url
  })

  it('a show missing from the catalogue still renders from its trending label', async () => {
    const w = mountIt([{ ...rows[0], entity_id: 'gone', label: 'Left The Corpus' }])
    await flushPromises()
    expect(w.get('[data-testid="trending-show-card"]').text()).toContain('Left The Corpus')
  })

  it('hides entirely when nothing is trending', async () => {
    const w = mountIt([])
    await flushPromises()
    expect(w.find('[data-testid="trending-shows-rail"]').exists()).toBe(false)
  })

  it('hides entirely when nothing is trending — under Everyone', async () => {
    const w = mountIt([], 'corpus')
    await flushPromises()
    expect(w.find('[data-testid="trending-shows-mine-empty"]').exists()).toBe(false)
  })

  // One Mine ⇄ Everyone switch for Discover (operator 2026-10-09).
  it('asks for the scope it is given', async () => {
    mountIt(rows, 'mine')
    await flushPromises()
    expect(api.getTrending).toHaveBeenCalledWith('show', 'mine', 12)
  })

  it('under Mine with nothing to show, SAYS so and offers everyone\'s — never vanishes', async () => {
    const w = mountIt([], 'mine')
    await flushPromises()
    expect(w.find('[data-testid="trending-shows-rail"]').exists()).toBe(true)
    expect(w.get('[data-testid="trending-shows-mine-empty"]').text()).toContain('Your shows appear here')
    await w.get('[data-testid="trending-shows-show-everyone"]').trigger('click')
    expect(w.emitted('show-everyone')).toHaveLength(1)
  })

  it('re-asks when the switch flips', async () => {
    const w = mountIt(rows, 'mine')
    await flushPromises()
    await w.setProps({ scope: 'corpus' })
    await flushPromises()
    expect(api.getTrending).toHaveBeenLastCalledWith('show', 'corpus', 12)
  })
})
