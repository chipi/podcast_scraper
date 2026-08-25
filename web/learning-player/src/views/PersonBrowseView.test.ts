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
  it('renders trending people as sparkline rows that open the person page (#12)', async () => {
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
        role: 'host',
      },
    ])
    const { w, router } = await mountView()
    expect(w.find('[data-testid="person-browse-view"]').exists()).toBe(true)
    expect(w.text()).toContain('Jane Doe')
    // Role tag says WHY they trend (host/guest/mentioned).
    expect(w.get('[data-testid="trend-spark-role"]').text().toLowerCase()).toBe('host')
    // #12 — same sparkline treatment as trending topics / Home, not a flat chip grid.
    expect(w.find('[data-testid="trend-sparks"]').exists()).toBe(true)
    const row = w.find('[data-testid="trend-spark-row"]')
    expect(row.exists()).toBe(true)
    const push = vi.spyOn(router, 'push')
    await row.trigger('click')
    expect(push).toHaveBeenCalledWith({ name: 'person', params: { id: 'person:jane-doe' } })
  })

  it('offers a back-to-Home button when standalone (#13)', async () => {
    vi.spyOn(api, 'getTrending').mockResolvedValue([])
    const { w } = await mountView()
    const back = w.find('[data-testid="browse-back-home"]')
    expect(back.exists()).toBe(true)
    expect(back.attributes('href')).toBe('/')
  })

  it('hides the heading + back-to-Home when embedded in the Browse hub', async () => {
    vi.spyOn(api, 'getTrending').mockResolvedValue([])
    setActivePinia(createPinia())
    const router = makeRouter()
    await router.push({ name: 'browse-people' })
    await router.isReady()
    const w = mount(PersonBrowseView, {
      props: { embedded: true },
      global: { plugins: [i18n, router, createPinia()] },
    })
    await flushPromises()
    expect(w.find('[data-testid="browse-back-home"]').exists()).toBe(false)
    expect(w.find('h1').exists()).toBe(false)
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
