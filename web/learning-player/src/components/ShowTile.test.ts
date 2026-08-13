import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'
import { createMemoryHistory, createRouter } from 'vue-router'
import en from '../i18n/locales/en.json'
import * as api from '../services/api'
import type { Podcast } from '../services/types'
import { useAuthStore } from '../stores/auth'
import { useLibraryStore } from '../stores/library'
import ShowTile from './ShowTile.vue'

const i18n = createI18n({ legacy: false, locale: 'en', messages: { en } })
const router = createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/', name: 'home', component: { template: '<div/>' } },
    { path: '/podcast/:feedId', name: 'podcast', component: { template: '<div/>' } },
    { path: '/login', name: 'login', component: { template: '<div/>' } },
  ],
})

beforeEach(() => setActivePinia(createPinia()))
afterEach(() => vi.restoreAllMocks())

function show(title: string | null, feedId = 'f1'): Podcast {
  return {
    feed_id: feedId,
    title,
    artwork_url: null,
    image_url: null,
    description: null,
    episode_count: 3,
  }
}

function signIn(): void {
  useAuthStore().user = { user_id: 'u1', email: 'a@b.c', name: 'A' }
}

function mountTile(p: Podcast, props: { lines?: 1 | 2; followable?: boolean } = {}) {
  return mount(ShowTile, {
    props: { show: p, ...props },
    global: { plugins: [router, i18n] },
  })
}

describe('ShowTile', () => {
  it('reserves the label box height so grid rows cannot go ragged (#1584)', () => {
    // The bug this component exists to prevent: in a CSS grid the row is as tall as its tallest
    // cell, so an unclamped label makes row height a function of title length. Clamping alone is
    // NOT enough — a 1-line title beside a 2-line one still differs by a line — so the label box
    // must also reserve its full height. Assert both halves.
    const short = mountTile(show('Acquired'))
    const long = mountTile(show('How I Built This with Guy Raz and Friends'))

    for (const w of [short, long]) {
      const label = w.get('div.mt-1')
      expect(label.classes()).toContain('line-clamp-2')
      expect(label.classes()).toContain('min-h-[2.25rem]')
    }

    // Same reserved box regardless of title length — the property that keeps rows uniform.
    expect(short.get('div.mt-1').classes().sort()).toEqual(long.get('div.mt-1').classes().sort())
  })

  it('exposes the full title on hover, since the visible label may be clipped', () => {
    const title = 'How I Built This with Guy Raz and Friends'
    expect(mountTile(show(title)).get('div.mt-1').attributes('title')).toBe(title)
  })

  it('falls back to the feed id when a show has no title', () => {
    const w = mountTile(show(null, 'p09'))
    expect(w.text()).toContain('p09')
  })

  it('single-line variant truncates instead of clamping', () => {
    const label = mountTile(show('Conversations with Tyler'), { lines: 1 }).get('div.mt-1')
    expect(label.classes()).toContain('truncate')
    expect(label.classes()).not.toContain('line-clamp-2')
  })

  it('links to the show page', () => {
    expect(mountTile(show('Acquired', 'p03')).get('a').attributes('href')).toBe('/podcast/p03')
  })

  // --- follow control (#1585) ---
  //
  // The empty "Your shows" state renders these so a user can complete the action where it is
  // offered. Following is otherwise reachable only from a show page, so a tile that merely linked
  // there would describe the action instead of providing it.

  it('has no follow control unless asked', () => {
    expect(mountTile(show('Acquired')).find('[aria-pressed]').exists()).toBe(false)
  })

  it('follows without navigating away from the page it is on', async () => {
    signIn()
    const post = vi.spyOn(api, 'followShow').mockResolvedValue([
      { feed_id: 'p03', feed_url: null, title: 'Acquired', added_at: 1 },
    ])
    const w = mountTile(show('Acquired', 'p03'), { followable: true })
    const btn = w.get('[aria-pressed]')
    expect(btn.attributes('aria-pressed')).toBe('false')

    await btn.trigger('click')
    await flushPromises()

    expect(post).toHaveBeenCalledWith('p03', { title: 'Acquired' })
    expect(w.get('[aria-pressed]').attributes('aria-pressed')).toBe('true')
  })

  it('reflects an existing follow', () => {
    signIn()
    const lib = useLibraryStore()
    lib.items = [{ feed_id: 'p03', feed_url: null, title: 'Acquired', added_at: 1 }]
    lib.loaded = true
    const w = mountTile(show('Acquired', 'p03'), { followable: true })
    expect(w.get('[aria-pressed]').attributes('aria-pressed')).toBe('true')
    expect(w.text()).toContain('Following')
  })

  // --- the sign-in gate (#1590) ---
  //
  // Signed out, the store's optimistic toggle would flip the button, take a 401, and flip back —
  // the control appears to work for one frame and then silently undoes itself. Worse than the
  // hidden control #1590 replaced, because it looks like a failure of the user's own action.

  it('offers the follow control to signed-out visitors, as a sign-in teaser', () => {
    const w = mountTile(show('Acquired', 'p03'), { followable: true })
    const btn = w.get('button')
    expect(btn.attributes('aria-label')).toBe('Sign in to follow')
    // No aria-pressed: nothing is toggled, so claiming a pressed state would be a lie to AT.
    expect(btn.attributes('aria-pressed')).toBeUndefined()
  })

  it('routes a signed-out follow to sign-in instead of calling the API', async () => {
    const post = vi.spyOn(api, 'followShow')
    await router.push('/podcast/p03')
    const w = mountTile(show('Acquired', 'p03'), { followable: true })

    await w.get('button').trigger('click')
    await flushPromises()

    expect(post).not.toHaveBeenCalled()
    expect(router.currentRoute.value.name).toBe('login')
    expect(router.currentRoute.value.query.redirect).toBe('/podcast/p03')
  })
})
