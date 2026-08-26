import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import type { Podcast } from '../services/types'
import ShowBrowseView from './ShowBrowseView.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const stub = { template: '<div/>' }

function makeRouter() {
  return createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: '/', name: 'home', component: stub },
      { path: '/browse/shows', name: 'browse-shows', component: ShowBrowseView },
      { path: '/podcast/:feedId', name: 'podcast', component: stub, props: true },
    ],
  })
}

function show(feed_id: string, title: string): Podcast {
  return { feed_id, title, artwork_url: null, image_url: null, description: null, episode_count: 3 }
}

async function mountView(props: Record<string, unknown> = {}) {
  setActivePinia(createPinia())
  const router = makeRouter()
  await router.push({ name: 'browse-shows' })
  await router.isReady()
  const w = mount(ShowBrowseView, { props, global: { plugins: [i18n, router, createPinia()] } })
  await flushPromises()
  return w
}

afterEach(() => vi.restoreAllMocks())

describe('ShowBrowseView', () => {
  it('lists all shows alphabetically, each linking to its podcast page', async () => {
    vi.spyOn(api, 'getPodcasts').mockResolvedValue([show('f-z', 'Zebra Cast'), show('f-a', 'Acme Show')])
    const w = await mountView()
    expect(w.find('[data-testid="show-browse-grid"]').exists()).toBe(true)
    const links = w.findAll('a[href^="/podcast/"]')
    expect(links.length).toBe(2)
    // Alphabetical: Acme before Zebra.
    expect(links[0].attributes('href')).toBe('/podcast/f-a')
  })

  it('filters by name and sorts by episode count', async () => {
    vi.spyOn(api, 'getPodcasts').mockResolvedValue([
      { feed_id: 'f-a', title: 'Acme Show', artwork_url: null, image_url: null, description: null, episode_count: 2 },
      { feed_id: 'f-z', title: 'Zebra Cast', artwork_url: null, image_url: null, description: null, episode_count: 40 },
    ])
    const w = await mountView()
    // Sort by most episodes → Zebra (40) leads Acme (2).
    await w.get('[data-testid="show-browse-sort"]').setValue('episodes')
    expect(w.findAll('a[href^="/podcast/"]')[0].attributes('href')).toBe('/podcast/f-z')
    // Filter narrows to matches only.
    await w.get('[data-testid="show-browse-search"]').setValue('acme')
    const links = w.findAll('a[href^="/podcast/"]')
    expect(links.length).toBe(1)
    expect(links[0].attributes('href')).toBe('/podcast/f-a')
  })

  it('hides heading + back-to-Home when embedded', async () => {
    vi.spyOn(api, 'getPodcasts').mockResolvedValue([])
    const w = await mountView({ embedded: true })
    expect(w.find('[data-testid="browse-back-home"]').exists()).toBe(false)
    expect(w.find('h1').exists()).toBe(false)
  })
})
