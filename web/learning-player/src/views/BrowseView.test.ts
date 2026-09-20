import { flushPromises, mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import en from '../i18n/locales/en.json'
import BrowseView from './BrowseView.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
// Discover = the entity dashboard on top + a content band of Episodes · Shows. Stub the embedded
// index views + the dashboard (it fetches trending); here we test the band's tab logic.
const stubs = {
  CatalogView: { template: '<div data-testid="stub-episodes" />' },
  ShowBrowseView: { template: '<div data-testid="stub-shows" />' },
  DiscoveryExplorer: { template: '<div data-testid="stub-explorer" />' },
  TrendingShowsRail: { template: '<div data-testid="stub-trending-shows" />' },
}

function makeRouter(query: Record<string, string> = {}) {
  const router = createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: '/browse', name: 'browse', component: BrowseView },
      { path: '/search', name: 'search', component: { template: '<div/>' } },
    ],
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

describe('BrowseView (Discover)', () => {
  it('shows the discovery dashboard, then a content band of Episodes · Shows (Episodes default)', async () => {
    const w = await mountView()
    expect(w.find('[data-testid="browse-view"]').exists()).toBe(true)
    expect(w.find('[data-testid="stub-explorer"]').exists()).toBe(true)
    // The band is only the content containers now — entities live in the dashboard above it.
    for (const key of ['episodes', 'shows']) {
      expect(w.find(`[data-testid="browse-tab-${key}"]`).exists()).toBe(true)
    }
    expect(w.find('[data-testid="browse-tab-topics"]').exists()).toBe(false)
    expect(w.find('[data-testid="browse-tab-people"]').exists()).toBe(false)
    expect(w.get('[data-testid="browse-tab-episodes"]').attributes('aria-selected')).toBe('true')
    expect(w.get('[data-testid="browse-tab-shows"]').attributes('aria-selected')).toBe('false')
  })

  it('embeds the content index views (no navigation) and passes embedded', async () => {
    const w = await mountView()
    expect(w.find('[data-testid="stub-episodes"]').exists()).toBe(true)
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
    const w = mount(BrowseView, { global: { plugins: [i18n, router], stubs } })
    await flushPromises()
    expect(w.get('[data-testid="browse-tab-shows"]').attributes('aria-selected')).toBe('true')

    await router.push({ name: 'browse', query: { tab: 'episodes' } })
    await flushPromises()
    expect(w.get('[data-testid="browse-tab-episodes"]').attributes('aria-selected')).toBe('true')
    expect(w.get('[data-testid="browse-tab-shows"]').attributes('aria-selected')).toBe('false')
  })

  /**
   * Search folded into Discovery (operator 2026-09-20).
   *
   * It sits between the trending-shows rail and the trends dashboard — the point where the page
   * stops saying "here is what is popular" and starts saying "go find something".
   */
  it('carries a search box that submits to the results page', async () => {
    const w = await mountView()
    const box = w.get('[data-testid="browse-search-section"]')
    expect(box.exists()).toBe(true)

    await w.get('[data-testid="browse-search-input"]').setValue('  reward hacking  ')
    await w.get('[data-testid="browse-search-section"] form').trigger('submit')
    await flushPromises()

    const r = w.vm.$router.currentRoute.value
    expect(r.name).toBe('search')
    // Trimmed: a leading space must not become part of the query.
    expect(r.query.q).toBe('reward hacking')
  })

  it('ignores a blank submit rather than opening an empty results page', async () => {
    const w = await mountView()
    await w.get('[data-testid="browse-search-input"]').setValue('   ')
    await w.get('[data-testid="browse-search-section"] form').trigger('submit')
    await flushPromises()
    expect(w.vm.$router.currentRoute.value.name).toBe('browse')
  })
})
