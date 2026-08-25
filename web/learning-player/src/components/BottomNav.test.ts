import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import en from '../i18n/locales/en.json'
import { useAuthStore } from '../stores/auth'
import BottomNav from './BottomNav.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const stub = { template: '<div/>' }
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/', name: 'home', component: stub },
    { path: '/browse', name: 'browse', component: stub },
    { path: '/browse/topics', name: 'browse-topics', component: stub },
    { path: '/browse/people', name: 'browse-people', component: stub },
    { path: '/search', name: 'search', component: stub },
    { path: '/library', name: 'library', component: stub },
    { path: '/profile', name: 'profile', component: stub },
    { path: '/login', name: 'login', component: stub },
    { path: '/episode/:slug', name: 'player', component: stub },
    { path: '/catalog', name: 'catalog', component: stub },
    { path: '/podcast/:feedId', name: 'podcast', component: stub },
  ],
})

beforeEach(() => setActivePinia(createPinia()))

async function mountNav(opts: { signedIn?: boolean; at?: string } = {}) {
  if (opts.signedIn) useAuthStore().user = { user_id: 'u1', email: 'a@b.c', name: 'A' }
  await router.push(opts.at ?? '/')
  await router.isReady()
  return mount(BottomNav, { global: { plugins: [router, i18n] } })
}

describe('BottomNav (#1594)', () => {
  it('offers five destinations including Browse (#14)', async () => {
    // Browse got its own tab: it is the destination that unifies the catalogue, topic and people
    // indexes, and it must be reachable from anywhere on mobile (incl. Search — #6).
    const w = await mountNav()
    expect(w.findAll('[data-testid^="bottom-nav-"]')).toHaveLength(5)
    for (const name of ['home', 'browse', 'search', 'library', 'profile']) {
      expect(w.find(`[data-testid="bottom-nav-${name}"]`).exists()).toBe(true)
    }
    expect(w.get('[data-testid="bottom-nav-browse"]').attributes('href')).toBe('/browse')
  })

  it('marks the current tab with aria-current', async () => {
    const w = await mountNav({ at: '/search' })
    expect(w.get('[data-testid="bottom-nav-search"]').attributes('aria-current')).toBe('page')
    expect(w.get('[data-testid="bottom-nav-home"]').attributes('aria-current')).toBeUndefined()
  })

  it('shows auth-gated tabs signed-out and routes them to sign-in (#1590)', async () => {
    // Hiding Library and Profile would hide the capabilities from exactly the visitors deciding
    // whether to sign up — the same reasoning as the gated controls.
    const w = await mountNav({ at: '/search' })
    const href = w.get('[data-testid="bottom-nav-library"]').attributes('href')
    expect(href).toContain('/login')
    expect(href).toContain('redirect')
  })

  it('links straight through once signed in', async () => {
    const w = await mountNav({ signedIn: true })
    expect(w.get('[data-testid="bottom-nav-library"]').attributes('href')).toBe('/library')
  })

  it('keeps a gated tab visually active on its own route', async () => {
    // The target resolves to /login when signed out, but the tab still OWNS /library — highlighting
    // by the resolved target would leave no tab active.
    const w = await mountNav({ at: '/library' })
    expect(w.get('[data-testid="bottom-nav-library"]').attributes('aria-current')).toBe('page')
  })

  it('is mobile-only and clears the home indicator', async () => {
    // Desktop keeps the header nav, where a top reach costs nothing.
    const nav = (await mountNav()).get('nav')
    expect(nav.classes()).toContain('sm:hidden')
    expect(nav.get('ul').classes().join(' ')).toContain('safe-area-inset-bottom')
  })

  it('every tab meets the 44px touch target', async () => {
    // Apple HIG 44pt / Android 48dp. min-h-[3rem] = 48px.
    const w = await mountNav()
    for (const link of w.findAll('[data-testid^="bottom-nav-"]')) {
      expect(link.classes().join(' ')).toContain('min-h-[3rem]')
    }
  })

  // --- wayfinding on nested routes (S9) ---
  //
  // Exact-name matching left the bar blank on player, podcast and catalog — the routes users spend
  // most of their time on, so it went dark exactly when orientation matters most.

  it('lights up Browse on the catalogue, a show page, and the topic/people indexes (#14)', async () => {
    for (const path of ['/catalog', '/podcast/p03', '/browse/topics', '/browse/people']) {
      const w = await mountNav({ at: path })
      expect(
        w.get('[data-testid="bottom-nav-browse"]').attributes('aria-current'),
        `${path} should light up Browse`,
      ).toBe('page')
      expect(w.get('[data-testid="bottom-nav-search"]').attributes('aria-current')).toBeUndefined()
    }
  })

  it('lights up NOTHING on the player — no tab may claim a path the user might not have taken', async () => {
    const w = await mountNav({ at: '/episode/ep-1' })
    for (const name of ['home', 'browse', 'search', 'library', 'profile']) {
      expect(
        w.get(`[data-testid="bottom-nav-${name}"]`).attributes('aria-current'),
        `${name} must not claim the player route`,
      ).toBeUndefined()
    }
  })
})
