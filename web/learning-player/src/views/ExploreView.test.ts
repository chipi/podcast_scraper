import { flushPromises, mount } from '@vue/test-utils'
import { createPinia } from 'pinia'
import { describe, expect, it } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import en from '../i18n/locales/en.json'
import ExploreView from './ExploreView.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
// Explore (ADR-163 §1) = Trends on top + Browse, Episodes · Shows: the full lists, as the last release
// had them. Stub the lists + Trends (they fetch); here we test the page's composition and tab logic.
const stubs = {
  CatalogView: { props: { embedded: Boolean }, template: '<div data-testid="stub-episodes" :data-embedded="embedded" />' },
  ShowBrowseView: { props: { embedded: Boolean }, template: '<div data-testid="stub-shows" :data-embedded="embedded" />' },
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

  it('embeds the full list per tab; a tab mounts when first opened, then stays', async () => {
    // A tab never opened builds nothing: the hidden Shows grid used to fetch and decode artwork
    // (a 3000px one on prod) while the listener was on Episodes (2026-10-08).
    const w = await mountView()
    expect(w.get('[data-testid="stub-episodes"]').attributes('data-embedded')).toBe('true')
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



  it('leads with search, then Trends — search moved here from Discover (operator 2026-10-11)', async () => {
    const w = await mountView()
    const html = w.html()
    const at = (id: string) => html.indexOf(`data-testid="${id}"`)
    expect(at('browse-search-section')).toBeGreaterThan(-1)
    expect(at('browse-search-section')).toBeLessThan(at('stub-trends'))
  })

  it('its search box submits to the results page, trimmed; a blank submit goes nowhere', async () => {
    const router = makeRouter()
    await router.isReady()
    const w = mount(ExploreView, { global: { plugins: [i18n, router, createPinia()], stubs } })
    await flushPromises()
    await w.get('[data-testid="browse-search-input"]').setValue('   ')
    await w.get('[data-testid="browse-search-section"] form').trigger('submit')
    await flushPromises()
    expect(router.currentRoute.value.name).toBe('explore')
    await w.get('[data-testid="browse-search-input"]').setValue('  reward hacking  ')
    await w.get('[data-testid="browse-search-section"] form').trigger('submit')
    await flushPromises()
    expect(router.currentRoute.value.name).toBe('search')
    expect(router.currentRoute.value.query.q).toBe('reward hacking')
  })
})
