/**
 * The entrances to an episode's Moments reel (operator 2026-10-10): "Play moments" first in the
 * shared ⋯ menu, the "▶ Moments" text action on browsing rows, and the Brief's button.
 */
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import en from '../i18n/locales/en.json'
import EpisodeActions from './EpisodeActions.vue'
import MomentsLink from './MomentsLink.vue'
import { setForcedOffline } from '../composables/useOnline'
import { useDownloadsStore } from '../stores/downloads'
import { afterEach } from 'vitest'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/', name: 'home', component: { template: '<div/>' } },
    { path: '/episode/:slug', name: 'player', component: { template: '<div/>' } },
  ],
})

beforeEach(async () => {
  setActivePinia(createPinia())
  await router.push('/')
})

describe('"Play moments" in the ⋯ menu', () => {
  it('leads the menu when the episode has moments, and opens the Moments view', async () => {
    const w = mount(EpisodeActions, {
      props: { slug: 'ep-1', moments: true },
      global: { plugins: [i18n, router], stubs: { teleport: true } },
      attachTo: document.body,
    })
    await w.get('[data-testid="overflow-trigger"]').trigger('click')
    const items = w.findAll('[role="menuitem"]')
    expect(items[0].attributes('data-testid')).toBe('episode-play-moments')
    expect(items[0].text()).toContain('Play moments')
    await items[0].trigger('click')
    await flushPromises()
    expect(router.currentRoute.value.name).toBe('player')
    expect(router.currentRoute.value.query.moments).toBe('1')
    w.unmount()
  })

  it('is absent when the episode has none', async () => {
    const w = mount(EpisodeActions, {
      props: { slug: 'ep-1' },
      global: { plugins: [i18n, router], stubs: { teleport: true } },
      attachTo: document.body,
    })
    await w.get('[data-testid="overflow-trigger"]').trigger('click')
    expect(w.find('[data-testid="episode-play-moments"]').exists()).toBe(false)
    w.unmount()
  })
})

describe('"▶ Moments" on a row', () => {
  it('is small text pointing at the Moments view, with a full accessible name', () => {
    const w = mount(MomentsLink, { props: { slug: 'ep-1' }, global: { plugins: [i18n, router] } })
    const a = w.get('[data-testid="moments-link"]')
    expect(a.attributes('href')).toBe('/episode/ep-1?moments=1')
    expect(a.attributes('aria-label')).toBe('Play moments')
    expect(a.text()).toBe('▶ Moments')
    // No count, no box (operator 2026-10-10): text only.
    expect(a.classes()).not.toContain('border')
  })
})


describe('"▶ Moments" offline (2026-10-10)', () => {
  afterEach(() => setForcedOffline(false))

  it('greys out, inert and saying why, for an episode that is not downloaded', () => {
    setForcedOffline(true)
    const w = mount(MomentsLink, { props: { slug: 'ep-1' }, global: { plugins: [i18n, router] } })
    expect(w.find('[data-testid="moments-link"]').exists()).toBe(false)
    const off = w.get('[data-testid="moments-link-offline"]')
    expect(off.attributes('aria-disabled')).toBe('true')
    expect(off.attributes('aria-label')).toContain('need a connection')
  })

  it('still plays a downloaded episode\'s reel', () => {
    setForcedOffline(true)
    useDownloadsStore().entries['ep-1'] = { slug: 'ep-1', state: 'downloaded', updatedAt: 1 } as never
    const w = mount(MomentsLink, { props: { slug: 'ep-1' }, global: { plugins: [i18n, router] } })
    expect(w.get('[data-testid="moments-link"]').attributes('href')).toBe('/episode/ep-1?moments=1')
  })
})
