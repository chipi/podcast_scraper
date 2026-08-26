import { flushPromises, mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import en from '../i18n/locales/en.json'
import BrowseView from './BrowseView.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
// Stub the embedded index views — their own coverage lives in their specs; here we test tab logic.
const stubs = {
  CatalogView: { template: '<div data-testid="stub-episodes" />' },
  ShowBrowseView: { template: '<div data-testid="stub-shows" />' },
  TopicBrowseView: { template: '<div data-testid="stub-topics" />' },
  PersonBrowseView: { template: '<div data-testid="stub-people" />' },
}

function makeRouter(query: Record<string, string> = {}) {
  const router = createRouter({
    history: createMemoryHistory(),
    routes: [{ path: '/browse', name: 'browse', component: BrowseView }],
  })
  void router.push({ name: 'browse', query })
  return router
}

async function mountView(query: Record<string, string> = {}) {
  const router = makeRouter(query)
  await router.isReady()
  const w = mount(BrowseView, { global: { plugins: [i18n, router], stubs } })
  await flushPromises()
  return w
}

describe('BrowseView tabs (#14 revised)', () => {
  it('is a tabbed page with Episodes active by default', async () => {
    const w = await mountView()
    expect(w.find('[data-testid="browse-view"]').exists()).toBe(true)
    for (const key of ['episodes', 'shows', 'topics', 'people']) {
      expect(w.find(`[data-testid="browse-tab-${key}"]`).exists()).toBe(true)
    }
    expect(w.get('[data-testid="browse-tab-episodes"]').attributes('aria-selected')).toBe('true')
    expect(w.get('[data-testid="browse-tab-topics"]').attributes('aria-selected')).toBe('false')
  })

  it('embeds the index views (no navigation) and passes embedded', async () => {
    const w = await mountView()
    // All three panels are mounted (v-show), rendering the embedded index views inline.
    expect(w.find('[data-testid="stub-episodes"]').exists()).toBe(true)
    expect(w.find('[data-testid="stub-shows"]').exists()).toBe(true)
    expect(w.find('[data-testid="stub-topics"]').exists()).toBe(true)
    expect(w.find('[data-testid="stub-people"]').exists()).toBe(true)
  })

  it('switching the tab updates aria-selected', async () => {
    const w = await mountView()
    await w.get('[data-testid="browse-tab-people"]').trigger('click')
    expect(w.get('[data-testid="browse-tab-people"]').attributes('aria-selected')).toBe('true')
    expect(w.get('[data-testid="browse-tab-episodes"]').attributes('aria-selected')).toBe('false')
  })

  it('honours ?tab= for a deep link', async () => {
    const w = await mountView({ tab: 'topics' })
    expect(w.get('[data-testid="browse-tab-topics"]').attributes('aria-selected')).toBe('true')
  })

  it('re-syncs the active tab when ?tab= changes without a remount (kept-alive)', async () => {
    // The view is kept-alive, so setup runs once; a later in-app nav to a new ?tab= must still switch
    // tabs (Home's "Browse people" chip after the hub was opened on Topics).
    const router = makeRouter({ tab: 'topics' })
    await router.isReady()
    const w = mount(BrowseView, { global: { plugins: [i18n, router], stubs } })
    await flushPromises()
    expect(w.get('[data-testid="browse-tab-topics"]').attributes('aria-selected')).toBe('true')

    await router.push({ name: 'browse', query: { tab: 'people' } })
    await flushPromises()
    expect(w.get('[data-testid="browse-tab-people"]').attributes('aria-selected')).toBe('true')
    expect(w.get('[data-testid="browse-tab-topics"]').attributes('aria-selected')).toBe('false')
  })
})
