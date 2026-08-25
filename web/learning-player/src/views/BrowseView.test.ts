import { flushPromises, mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import en from '../i18n/locales/en.json'
import BrowseView from './BrowseView.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const stub = { template: '<div/>' }

function makeRouter() {
  return createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: '/browse', name: 'browse', component: BrowseView },
      { path: '/catalog', name: 'catalog', component: stub },
      { path: '/browse/topics', name: 'browse-topics', component: stub },
      { path: '/browse/people', name: 'browse-people', component: stub },
    ],
  })
}

async function mountView() {
  const router = makeRouter()
  await router.push({ name: 'browse' })
  await router.isReady()
  const w = mount(BrowseView, { global: { plugins: [i18n, router] } })
  await flushPromises()
  return w
}

describe('BrowseView hub (#14)', () => {
  it('links to the three corpus indexes', async () => {
    const w = await mountView()
    expect(w.find('[data-testid="browse-view"]').exists()).toBe(true)
    expect(w.get('[data-testid="browse-hub-episodes"]').attributes('href')).toBe('/catalog')
    expect(w.get('[data-testid="browse-hub-topics"]').attributes('href')).toBe('/browse/topics')
    expect(w.get('[data-testid="browse-hub-people"]').attributes('href')).toBe('/browse/people')
  })
})
