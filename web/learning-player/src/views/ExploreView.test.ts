import { flushPromises, mount } from '@vue/test-utils'
import { createPinia } from 'pinia'
import { describe, expect, it } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import en from '../i18n/locales/en.json'
import ExploreView from './ExploreView.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
// Explore (ADR-163 §1) = Trends on top + the catalog of Episodes · Shows. Stub the embedded index
// views + Trends (they fetch); here we test the page's composition and the catalog's tab logic.
const stubs = {
  CatalogView: { template: '<div data-testid="stub-episodes" />' },
  ShowBrowseView: { template: '<div data-testid="stub-shows" />' },
  TrendsSection: { template: '<div data-testid="stub-trends" />' },
}

function makeRouter(query: Record<string, string> = {}) {
  const router = createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: '/explore', name: 'explore', component: ExploreView },
      { path: '/search', name: 'search', component: { template: '<div/>' } },
    ],
  })
  void router.push({ name: 'explore', query })
  return router
}

async function mountView(query: Record<string, string> = {}) {
  const router = makeRouter(query)
  await router.isReady()
  const w = mount(ExploreView, { global: { plugins: [i18n, router, createPinia()], stubs } })
  await flushPromises()
  return w
}

describe('ExploreView', () => {
  it('shows Trends, then the catalog of Episodes · Shows (Episodes default)', async () => {
    const w = await mountView()
    expect(w.find('[data-testid="explore-view"]').exists()).toBe(true)
    expect(w.find('[data-testid="stub-trends"]').exists()).toBe(true)
    // The band is only the content containers now — entities live in the dashboard above it.
    for (const key of ['episodes', 'shows']) {
      expect(w.find(`[data-testid="browse-tab-${key}"]`).exists()).toBe(true)
    }
    expect(w.find('[data-testid="browse-tab-topics"]').exists()).toBe(false)
    expect(w.find('[data-testid="browse-tab-people"]').exists()).toBe(false)
    expect(w.get('[data-testid="browse-tab-episodes"]').attributes('aria-selected')).toBe('true')
    expect(w.get('[data-testid="browse-tab-shows"]').attributes('aria-selected')).toBe('false')
  })

  it('embeds the content index views (no navigation); a tab mounts when first opened, then stays', async () => {
    // A tab never opened builds nothing: the hidden Shows grid used to fetch and decode artwork
    // (a 3000px one on prod) while the listener was on Episodes (2026-10-08).
    const w = await mountView()
    expect(w.find('[data-testid="stub-episodes"]').exists()).toBe(true)
    expect(w.find('[data-testid="stub-shows"]').exists()).toBe(false)
    await w.get('[data-testid="browse-tab-shows"]').trigger('click')
    await flushPromises()
    expect(w.find('[data-testid="stub-shows"]').exists()).toBe(true)
    await w.get('[data-testid="browse-tab-episodes"]').trigger('click')
    await flushPromises()
    // Kept after leaving it, so switching back neither refetches nor loses the scroll position.
    expect(w.find('[data-testid="stub-shows"]').exists()).toBe(true)
  })

  it('switching the tab updates aria-selected', async () => {
    const w = await mountView()
    await w.get('[data-testid="browse-tab-shows"]').trigger('click')
    expect(w.get('[data-testid="browse-tab-shows"]').attributes('aria-selected')).toBe('true')
    expect(w.get('[data-testid="browse-tab-episodes"]').attributes('aria-selected')).toBe('false')
  })

  it('honours ?tab= for a deep link', async () => {
    const w = await mountView({ tab: 'shows' })
    expect(w.get('[data-testid="browse-tab-shows"]').attributes('aria-selected')).toBe('true')
  })

  it('re-syncs the active tab when ?tab= changes without a remount (kept-alive)', async () => {
    const router = makeRouter({ tab: 'shows' })
    await router.isReady()
    const w = mount(ExploreView, { global: { plugins: [i18n, router, createPinia()], stubs } })
    await flushPromises()
    expect(w.get('[data-testid="browse-tab-shows"]').attributes('aria-selected')).toBe('true')

    await router.push({ name: 'explore', query: { tab: 'episodes' } })
    await flushPromises()
    expect(w.get('[data-testid="browse-tab-episodes"]').attributes('aria-selected')).toBe('true')
    expect(w.get('[data-testid="browse-tab-shows"]').attributes('aria-selected')).toBe('false')
  })

  it("trending shows' See all arrives on Shows, sorted by trending, as a list", async () => {
    const w = await mountView({ tab: 'shows', sort: 'trending', view: 'list' })
    expect(w.get('[data-testid="browse-tab-shows"]').attributes('aria-selected')).toBe('true')
  })
})
