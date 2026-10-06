import { describe, expect, it, vi } from 'vitest'
import { createMemoryHistory, createRouter } from 'vue-router'

/**
 * The restore must make the device's DOWNLOADS visible before it navigates or loads anything.
 *
 * Repro (`make test-ios` phase 2, 2026-10-04): an offline cold launch restored the last place — an
 * episode page — before the shell had pointed the downloads registry at the account. The episode
 * page and the player both looked for a local copy, found none, fell back to the origin URL and
 * showed "Couldn't load the audio from the source" over an episode sitting on the device. With the
 * restore disabled the same journey passed, which is what isolated it.
 */
const store = new Map<string, unknown>()
vi.mock('./native', () => ({ isNative: () => true }))
vi.mock('./deviceStore', () => ({
  getDeviceJson: vi.fn(async (k: string) => store.get(k) ?? null),
  setDeviceJson: vi.fn(async (k: string, v: unknown) => void store.set(k, v)),
}))

const { installLastPlace, LAST_PLACE_KEY } = await import('./lastPlace')

function routerWithPages() {
  const C = { template: '<div/>' }
  return createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: '/', name: 'home', component: C },
      { path: '/episode/:slug', name: 'episode', component: C },
      { path: '/person/:id', name: 'person', component: C },
    ],
  })
}

describe('installLastPlace prepares local sources before restoring', () => {
  it('awaits prepareLocalSources for the account BEFORE navigating to the restored episode page', async () => {
    store.clear()
    store.set(LAST_PLACE_KEY, {
      path: '/episode/p06-5416bc0968',
      userId: 'u1',
      at: Date.now() - 60_000,
      episode: { slug: 'p06-5416bc0968', url: 'http://x/a.mp3', title: 'T', position: 0 },
    })
    const order: string[] = []
    const router = routerWithPages()
    installLastPlace(router, {
      userId: () => 'u1',
      nowPlaying: () => null,
      loadAt: () => order.push('loadAt'),
      ensureAuthLoaded: async () => {},
      prepareLocalSources: async (userId: string) => {
        await Promise.resolve()
        order.push(`prepare:${userId}`)
      },
    })
    router.afterEach((to) => void order.push(`nav:${to.fullPath}`))
    await router.push('/')
    expect(router.currentRoute.value.fullPath).toBe('/episode/p06-5416bc0968')
    expect(order[0]).toBe('prepare:u1')
    expect(order.indexOf('prepare:u1')).toBeLessThan(order.indexOf('nav:/episode/p06-5416bc0968'))
  })

  it('awaits it before loading the episode into the player on another page', async () => {
    store.clear()
    store.set(LAST_PLACE_KEY, {
      path: '/person/person:nora',
      userId: 'u1',
      at: Date.now() - 60_000,
      episode: { slug: 'p06-5416bc0968', url: 'http://x/a.mp3', title: 'T', position: 12 },
    })
    const order: string[] = []
    const router = routerWithPages()
    installLastPlace(router, {
      userId: () => 'u1',
      nowPlaying: () => null,
      loadAt: () => order.push('loadAt'),
      ensureAuthLoaded: async () => {},
      prepareLocalSources: async () => {
        await Promise.resolve()
        order.push('prepare')
      },
    })
    await router.push('/')
    expect(order).toEqual(['prepare', 'loadAt'])
  })

  it('prepares nothing when there is nothing to restore', async () => {
    store.clear()
    const prepare = vi.fn(async () => {})
    const router = routerWithPages()
    installLastPlace(router, {
      userId: () => 'u1',
      nowPlaying: () => null,
      loadAt: () => {},
      ensureAuthLoaded: async () => {},
      prepareLocalSources: prepare,
    })
    await router.push('/')
    expect(prepare).not.toHaveBeenCalled()
  })
})
