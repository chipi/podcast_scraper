/**
 * Explore's catalog taste (operator 2026-10-11): one server page of the newest episodes or shows,
 * and "See all" to the full list with its filters — never the whole catalogue on Explore.
 */
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import ExploreCatalogTaste from './ExploreCatalogTaste.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/', component: { template: '<div/>' } },
    { path: '/catalog', name: 'catalog', component: { template: '<div/>' } },
    { path: '/shows', name: 'shows', component: { template: '<div/>' } },
    { path: '/podcast/:feedId', name: 'podcast', component: { template: '<div/>' } },
    { path: '/episode/:slug', name: 'player', component: { template: '<div/>' } },
  ],
})

function mountTaste(kind: 'episodes' | 'shows') {
  const pinia = createPinia()
  setActivePinia(pinia)
  return mount(ExploreCatalogTaste, {
    props: { kind },
    global: { plugins: [i18n, router, pinia], stubs: { EpisodeCard: { props: ['episode'], template: '<article data-testid="taste-episode">{{ episode.slug }}</article>' }, ShowTile: { props: ['show'], template: '<div data-testid="taste-show">{{ show.feed_id }}</div>' } } },
  })
}

afterEach(() => vi.restoreAllMocks())

describe('ExploreCatalogTaste', () => {
  it('asks the server for four episodes and links to the full list', async () => {
    const list = vi.spyOn(api, 'listEpisodes').mockResolvedValue({
      items: ['a', 'b', 'c', 'd'].map((slug) => ({ slug }) as never),
      page: 1,
      page_size: 4,
      total: 55,
      has_more: true,
    })
    const w = mountTaste('episodes')
    await flushPromises()
    expect(list).toHaveBeenCalledWith({ page: 1, pageSize: 4 })
    expect(w.findAll('[data-testid="taste-episode"]')).toHaveLength(4)
    const all = w.get('[data-testid="explore-see-all-episodes"]')
    expect(all.text()).toBe('See all 55 episodes ›')
    expect(all.attributes('href')).toBe('/catalog')
  })

  it('asks the server for six shows and links to the full show list', async () => {
    const page = vi.spyOn(api, 'getPodcastsPage').mockResolvedValue({
      items: ['s1', 's2'].map((feed_id) => ({ feed_id }) as never),
      total: 12,
    } as never)
    const w = mountTaste('shows')
    await flushPromises()
    expect(page).toHaveBeenCalledWith({ limit: 6, sort: 'newest', compact: true })
    expect(w.findAll('[data-testid="taste-show"]')).toHaveLength(2)
    const all = w.get('[data-testid="explore-see-all-shows"]')
    expect(all.text()).toBe('See all 12 shows ›')
    expect(all.attributes('href')).toBe('/shows')
  })
})
