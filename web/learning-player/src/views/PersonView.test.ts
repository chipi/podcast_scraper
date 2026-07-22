import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import * as api from '../services/api'
import en from '../i18n/locales/en.json'
import PersonView from './PersonView.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })

function makeRouter() {
  return createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: '/', name: 'home', component: { template: '<div/>' } },
      { path: '/person/:id', name: 'person', component: PersonView, props: true },
    ],
  })
}

beforeEach(() => {
  vi.spyOn(api, 'getPersonCard').mockResolvedValue({
    id: 'person:jane-doe',
    label: 'Jane Doe',
    episode_count: 5,
    episodes: [],
    related_people: [],
    related_topics: [],
  })
})
afterEach(() => vi.restoreAllMocks())

describe('PersonView (#1261-6)', () => {
  it('fetches the person card via the route param and renders the person label', async () => {
    setActivePinia(createPinia())
    const router = makeRouter()
    await router.push({ name: 'person', params: { id: 'person:jane-doe' } })
    await router.isReady()
    const w = mount(PersonView, {
      props: { id: 'person:jane-doe' },
      global: { plugins: [i18n, router, createPinia()], stubs: { teleport: true } },
    })
    await flushPromises()
    expect(api.getPersonCard).toHaveBeenCalledWith('person:jane-doe', undefined)
    expect(w.find('[data-testid="person-view"]').exists()).toBe(true)
    expect(w.text()).toContain('Jane Doe')
  })
})
