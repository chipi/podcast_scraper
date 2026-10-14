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

function mountActions(episodeTitle?: string, showDownload = false) {
  return mount(EpisodeActions, {
    props: { slug: 'p09-a4bbb5dde3', episodeTitle, showDownload },
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
    // Download is inline only where it is promoted (Up next); elsewhere it is in the ⋯ (operator
    // 2026-10-12), so it is checked on a promoted row.
    const w = mountActions('Risk Is a Systems Property', true)
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

  it('a row keeps download in the ⋯ unless it is promoted (operator 2026-10-12)', () => {
    const names = mountActions('Risk Is a Systems Property').findAll('button').map((b) => b.attributes('aria-label') ?? '')
    expect(names.some((n) => n.startsWith(en.downloads.download))).toBe(false)
    expect(names.filter((n) => n.startsWith(en.common.moreActions))).toHaveLength(1)
  })

  it('without one, the names stay plain (the player page is about one episode already)', () => {
    const names = mountActions().findAll('button').map((b) => b.attributes('aria-label') ?? '')
    expect(names).toContain('Add to queue')
    expect(names).toContain('More actions')
    expect(names.some((n) => n.includes(':'))).toBe(false)
  })
})

/** "Mark as played" in the ⋯ of Jump back in and Up next (operator 2026-10-14). */
describe('EpisodeActions: Mark as played', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    useAuthStore().user = { user_id: 'u_1', email: 'd@l', name: 'Dev' }
  })

  function mountWith(markPlayed: boolean) {
    return mount(EpisodeActions, {
      props: { slug: 'p09-a4bbb5dde3', episodeTitle: 'Risk', markPlayed },
      global: { plugins: [i18n, router], stubs: { teleport: true } },
    })
  }

  it('is offered only where a surface asks for it', async () => {
    for (const [asked, present] of [
      [true, true],
      [false, false],
    ] as const) {
      const w = mountWith(asked)
      await w.get('[aria-haspopup="menu"]').trigger('click')
      expect(w.find('[data-testid="episode-mark-played"]').exists()).toBe(present)
      w.unmount()
    }
  })

  it('marks the episode played, then offers to undo it', async () => {
    const api = await import('../services/api')
    const mark = vi.spyOn(api, 'markCompleted').mockResolvedValue(['p09-a4bbb5dde3'] as never)
    const w = mountWith(true)
    await w.get('[aria-haspopup="menu"]').trigger('click')
    const item = w.get('[data-testid="episode-mark-played"]')
    expect(item.text()).toBe(en.player.markPlayed)
    await item.trigger('click')
    await new Promise((r) => setTimeout(r, 0))
    expect(mark).toHaveBeenCalledWith('p09-a4bbb5dde3')
    await w.get('[aria-haspopup="menu"]').trigger('click')
    expect(w.get('[data-testid="episode-mark-played"]').text()).toBe(en.player.markUnplayed)
  })
})
