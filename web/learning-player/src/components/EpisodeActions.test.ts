// @vitest-environment happy-dom
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import en from '../i18n/locales/en.json'
import { useAuthStore } from '../stores/auth'

// Native, so the Download control renders (it is native-only).
vi.mock('../services/native', () => ({ isNative: () => true, openShareSheet: vi.fn() }))
vi.mock('../services/downloadScheduler', () => ({ markForOffline: vi.fn(), getNetworkPolicy: vi.fn(async () => 'wifi-only') }))
vi.mock('../services/downloads', () => ({ deleteEpisode: vi.fn() }))

const EpisodeActions = (await import('./EpisodeActions.vue')).default
const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({ history: createMemoryHistory(), routes: [{ path: '/', component: { template: '<div/>' } }] })

function mountActions(episodeTitle?: string) {
  return mount(EpisodeActions, {
    props: { slug: 'p09-a4bbb5dde3', episodeTitle },
    global: { plugins: [i18n, router], stubs: { teleport: true } },
  })
}

/**
 * Each card's controls name their episode (2026-10-12). On a list every card had the same heart,
 * queue, ⋯ and download, so a screen reader heard "Download for offline, button" five times with
 * nothing to say which episode, and device tests could not tell them apart either.
 */
describe('EpisodeActions: the controls name their episode', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    useAuthStore().user = { user_id: 'u_1', email: 'd@l', name: 'Dev' }
  })

  it('with the episode title, every control says which episode it acts on', () => {
    const w = mountActions('Risk Is a Systems Property')
    const names = w.findAll('button').map((b) => b.attributes('aria-label') ?? '')
    for (const action of [en.fav.add, en.queue.add, en.common.moreActions, en.downloads.download]) {
      const name = names.find((n) => n.startsWith(action))
      expect(name, `no "${action}" control`).toBeTruthy()
      expect(name).toBe(`${action}: Risk Is a Systems Property`)
    }
    // ...and the title is NOT in the page's text: hidden copies of it made every "find the
    // episode by its title" land on a control's label instead of the episode.
    expect(w.text()).not.toContain('Risk Is a Systems Property')
  })

  it('without one, the names stay plain (the player page is about one episode already)', () => {
    const names = mountActions().findAll('button').map((b) => b.attributes('aria-label') ?? '')
    expect(names).toContain('Add to queue')
    expect(names).toContain('More actions')
    expect(names.some((n) => n.includes(':'))).toBe(false)
  })
})
