import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import en from '../i18n/locales/en.json'
import BrowseView from './BrowseView.vue'
import { useAuthStore } from '../stores/auth'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
// Discover = what to hear next (ADR-163): quote cards, trending shows, search. Stub the rails (they
// fetch); here we test the page's composition and its ways out.
const stubs = {
  RecommendationRail: {
    props: ['rail', 'scope', 'fallback'],
    template: '<div :data-testid="`stub-rail-${rail}`" :data-fallback="fallback ? \'yes\' : \'no\'">{{ scope }}</div>',
  },
  TrendingShowsRail: {
    props: ['scope'],
    emits: ['see-all'],
    template: '<div data-testid="stub-trending-shows"><button data-testid="stub-see-all" @click="$emit(\'see-all\')">all</button>{{ scope }}</div>',
  },
}

function makeRouter() {
  const router = createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: '/browse', name: 'browse', component: BrowseView },
      { path: '/explore', name: 'explore', component: { template: '<div/>' } },
      { path: '/search', name: 'search', component: { template: '<div/>' } },
    ],
  })
  void router.push({ name: 'browse' })
  return router
}

async function mountView(pinia = createPinia()) {
  const router = makeRouter()
  await router.isReady()
  const w = mount(BrowseView, { global: { plugins: [i18n, router, pinia], stubs } })
  await flushPromises()
  return w
}

describe('BrowseView (Discover): what to hear next', () => {
  it('leads with the quote cards, then trending shows, then search — no Trends, no catalog', async () => {
    const w = await mountView()
    const html = w.html()
    const at = (id: string) => html.indexOf(`data-testid="${id}"`)
    expect(at('stub-rail-quotes')).toBeGreaterThan(-1)
    expect(at('stub-rail-quotes')).toBeLessThan(at('stub-trending-shows'))
    expect(at('stub-trending-shows')).toBeLessThan(at('browse-search-section'))
    expect(w.find('[data-testid="browse-tab-episodes"]').exists()).toBe(false)
    expect(w.findComponent({ name: 'TrendsSection' }).exists()).toBe(false)
    expect(w.findComponent({ name: 'CatalogView' }).exists()).toBe(false)
  })

  it('Explore is one tap away in the header', async () => {
    const w = await mountView()
    expect(w.get('[data-testid="discover-explore-link"]').attributes('href')).toBe('/explore')
  })

  it("trending shows' See all opens Explore's Shows, sorted by trending, as a list", async () => {
    const w = await mountView()
    await w.get('[data-testid="stub-see-all"]').trigger('click')
    await flushPromises()
    const r = w.vm.$router.currentRoute.value
    expect(r.name).toBe('explore')
    expect(r.query).toEqual({ tab: 'shows', sort: 'trending', view: 'list' })
    expect(r.hash).toBe('#catalog')
  })

  it('carries a search box that submits to the results page', async () => {
    const w = await mountView()
    await w.get('[data-testid="browse-search-input"]').setValue('  reward hacking  ')
    await w.get('[data-testid="browse-search-section"] form').trigger('submit')
    await flushPromises()
    const r = w.vm.$router.currentRoute.value
    expect(r.name).toBe('search')
    expect(r.query.q).toBe('reward hacking') // trimmed
  })

  it('ignores a blank submit rather than opening an empty results page', async () => {
    const w = await mountView()
    await w.get('[data-testid="browse-search-input"]').setValue('   ')
    await w.get('[data-testid="browse-search-section"] form').trigger('submit')
    await flushPromises()
    expect(w.vm.$router.currentRoute.value.name).toBe('browse')
  })

  // One Mine ⇄ Everyone switch for the whole page (operator 2026-10-09): both rails follow it.
  it('a signed-in Discover carries ONE scope switch, and both rails follow it', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('{}', { status: 200 })))
    const pinia = createPinia()
    setActivePinia(pinia)
    useAuthStore().user = { user_id: 'u1', email: 'a@b.c', name: 'A' }
    const w = await mountView(pinia)
    const toggle = w.get('[data-testid="discover-scope"]')
    expect(toggle.attributes('aria-pressed')).toBe('true')
    expect(w.get('[data-testid="stub-trending-shows"]').text()).toContain('mine')
    expect(w.get('[data-testid="stub-rail-quotes"]').text()).toBe('mine')
    // Cold start (ADR-163 §6): only Mine's quote cards fall back to Everyone's, labelled.
    expect(w.get('[data-testid="stub-rail-quotes"]').attributes('data-fallback')).toBe('yes')
    // More from what you captured: signed in, under Mine only.
    expect(w.find('[data-testid="stub-rail-captured"]').exists()).toBe(true)
    await toggle.trigger('click')
    await flushPromises()
    expect(w.get('[data-testid="stub-trending-shows"]').text()).toContain('corpus')
    expect(w.get('[data-testid="stub-rail-quotes"]').text()).toBe('everyone')
    expect(w.get('[data-testid="stub-rail-quotes"]').attributes('data-fallback')).toBe('no')
    expect(w.find('[data-testid="stub-rail-captured"]').exists()).toBe(false)
    expect(w.findAll('[data-testid="discover-scope"]')).toHaveLength(1)
    vi.unstubAllGlobals()
  })
})
