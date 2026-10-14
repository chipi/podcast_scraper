/**
 * The entrances to an episode's Moments reel (operator 2026-10-10): "Play moments" first in the
 * shared ⋯ menu, and the Brief's button (the "▶ Moments" text on rows is gone, 2026-10-14).
 */
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import en from '../i18n/locales/en.json'
import EpisodeActions from './EpisodeActions.vue'
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

describe('"Play moments" offline (2026-10-10)', () => {
  // The ⋯ menu is the one entrance on a list row now (operator 2026-10-14: "▶ Moments" competed
  // with the show's name). Offline it is greyed and says why, unless the episode is downloaded.
  afterEach(() => setForcedOffline(false))
  async function menuItem() {
    const w = mount(EpisodeActions, {
      props: { slug: 'ep-1', moments: true },
      global: { plugins: [i18n, router], stubs: { teleport: true } },
      attachTo: document.body,
    })
    await w.get('[data-testid="overflow-trigger"]').trigger('click')
    return { w, item: w.get('[data-testid="episode-play-moments"]') }
  }

  it('greys out, inert and saying why, for an episode that is not downloaded', async () => {
    setForcedOffline(true)
    const { w, item } = await menuItem()
    expect(item.attributes('disabled')).toBeDefined()
    expect(item.text()).toContain('need a connection')
    w.unmount()
  })

  it('still plays a downloaded episode\'s reel', async () => {
    setForcedOffline(true)
    useDownloadsStore().entries['ep-1'] = { slug: 'ep-1', state: 'downloaded', updatedAt: 1 } as never
    const { w, item } = await menuItem()
    expect(item.attributes('disabled')).toBeUndefined()
    w.unmount()
  })
})
