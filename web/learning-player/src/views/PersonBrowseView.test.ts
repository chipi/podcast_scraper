import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import PersonBrowseView from './PersonBrowseView.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })

function makeRouter() {
  return createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: '/', name: 'home', component: { template: '<div/>' } },
      { path: '/browse/people', name: 'browse-people', component: PersonBrowseView },
      { path: '/person/:id', name: 'person', component: { template: '<div/>' }, props: true },
    ],
  })
}

async function mountView() {
  setActivePinia(createPinia())
  const router = makeRouter()
  await router.push({ name: 'browse-people' })
  await router.isReady()
  const w = mount(PersonBrowseView, {
    global: { plugins: [i18n, router, createPinia()] },
  })
  await flushPromises()
  return { w, router }
}

afterEach(() => vi.restoreAllMocks())

describe('PersonBrowseView (#1261-6)', () => {
  it('lists trending people with links to /person/:id', async () => {
    vi.spyOn(api, 'getTrending').mockResolvedValue([
      {
        entity_id: 'person:jane-doe',
        kind: 'person',
        label: 'Jane Doe',
        velocity: 0.6,
        volume: 10,
        heating_up: true,
        total: 20,
        series: [],
      },
    ])
    const { w } = await mountView()
    expect(w.find('[data-testid="person-browse-view"]').exists()).toBe(true)
    expect(w.text()).toContain('Jane Doe')
    const link = w.findAll('a[href^="/person/"]')[0]
    expect(link.attributes('href')).toBe('/person/person:jane-doe')
  })

  it('shows the empty message when the endpoint returned nothing', async () => {
    vi.spyOn(api, 'getTrending').mockResolvedValue([])
    const { w } = await mountView()
    expect(w.text()).toContain('Nothing to browse yet')
  })

  it('silent degrade when getTrending rejects — no error surfaced to the listener', async () => {
    vi.spyOn(api, 'getTrending').mockRejectedValue(new Error('offline'))
    const { w } = await mountView()
    expect(w.text()).toContain('Nothing to browse yet')
    expect(w.text()).not.toContain('offline')
  })
})
