import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import { defineComponent, h, nextTick, ref } from 'vue'
import { mount } from '@vue/test-utils'

// Mock the virtual:pwa-register/vue module. vite-plugin-pwa exposes it
// at build time; under vitest the module doesn't exist so we stub it.
// The stub keeps the SAME shape as the real thing so any drift in the
// composable's API against the plugin surfaces here first.
const updateServiceWorker = vi.fn().mockResolvedValue(undefined)
const needRefresh = ref(false)
const offlineReady = ref(false)
let capturedHandlers: {
  onRegisteredSW?: (url: string, reg?: ServiceWorkerRegistration) => void
  onRegisterError?: (err: unknown) => void
} = {}

let registerCalls = 0
vi.mock('virtual:pwa-register/vue', () => ({
  useRegisterSW: (opts: typeof capturedHandlers = {}) => {
    registerCalls += 1
    capturedHandlers = opts
    return {
      needRefresh,
      offlineReady,
      updateServiceWorker,
    }
  },
}))

const native = vi.hoisted(() => ({ value: false }))
vi.mock('../services/native', async (orig) => ({
  ...(await orig<typeof import('../services/native')>()),
  isNative: () => native.value,
}))

// Import AFTER the mock is registered.
import { usePwaUpdate } from './usePwaUpdate'

/** Mount a throwaway host component that calls the composable so its
 *  onMounted / onUnmounted hooks run and the Vue lifecycle proxies work.
 *  All wrappers are auto-tracked and unmounted in afterEach so
 *  visibilitychange listeners don't leak across tests. */
const mountedWrappers: Array<{ unmount: () => void }> = []

function mountHost() {
  const api = { value: null as ReturnType<typeof usePwaUpdate> | null }
  const Host = defineComponent({
    setup() {
      api.value = usePwaUpdate()
      return () => h('div')
    },
  })
  const wrapper = mount(Host)
  mountedWrappers.push(wrapper)
  return { wrapper, api }
}

describe('usePwaUpdate', () => {
  beforeEach(() => {
    native.value = false
    registerCalls = 0
    updateServiceWorker.mockClear()
    needRefresh.value = false
    offlineReady.value = false
    capturedHandlers = {}
  })

  afterEach(() => {
    // Drain all mounted wrappers so their onUnmounted hooks fire and remove
    // the visibilitychange listeners before the next test dispatches events.
    while (mountedWrappers.length) mountedWrappers.pop()?.unmount()
    vi.useRealTimers()
  })

  it('exposes needRefresh + offlineReady from the plugin', () => {
    const { api } = mountHost()
    expect(api.value?.needRefresh).toBe(needRefresh)
    expect(api.value?.offlineReady).toBe(offlineReady)
  })

  it('applyUpdate() calls updateServiceWorker(true) to trigger reload', async () => {
    const { api } = mountHost()
    await api.value?.applyUpdate()
    expect(updateServiceWorker).toHaveBeenCalledTimes(1)
    expect(updateServiceWorker).toHaveBeenCalledWith(true)
  })

  it('dismissUpdate() clears needRefresh without reloading', () => {
    needRefresh.value = true
    const { api } = mountHost()
    api.value?.dismissUpdate()
    expect(needRefresh.value).toBe(false)
    expect(updateServiceWorker).not.toHaveBeenCalled()
  })

  it('onRegisteredSW schedules a periodic registration.update()', async () => {
    vi.useFakeTimers()
    mountHost()

    const reg = { update: vi.fn().mockResolvedValue(undefined) } as unknown as ServiceWorkerRegistration
    capturedHandlers.onRegisteredSW?.('/sw.js', reg)
    // The interval is 15 minutes — fast-forward once and confirm .update() fires.
    vi.advanceTimersByTime(15 * 60 * 1000)
    // Give the microtask queue a chance to drain the .catch handler.
    await Promise.resolve()
    expect(reg.update).toHaveBeenCalledTimes(1)

    // A second tick should fire again — proves it's a repeating interval, not a one-shot.
    vi.advanceTimersByTime(15 * 60 * 1000)
    await Promise.resolve()
    expect(reg.update).toHaveBeenCalledTimes(2)
  })

  it('does not crash when onRegisteredSW receives no registration', () => {
    mountHost()
    // Should be a no-op — no throw, no scheduler.
    expect(() => capturedHandlers.onRegisteredSW?.('/sw.js', undefined)).not.toThrow()
  })

  it('calls registration.update() on visibilitychange when tab becomes visible', async () => {
    const { wrapper } = mountHost()
    const reg = { update: vi.fn().mockResolvedValue(undefined) } as unknown as ServiceWorkerRegistration
    // Stub navigator.serviceWorker.getRegistration
    const originalSW = (globalThis.navigator as unknown as { serviceWorker: unknown }).serviceWorker
    ;(globalThis.navigator as unknown as { serviceWorker: unknown }).serviceWorker = {
      getRegistration: vi.fn().mockResolvedValue(reg),
    }
    try {
      Object.defineProperty(document, 'hidden', { value: false, configurable: true })
      document.dispatchEvent(new Event('visibilitychange'))
      // getRegistration is async; give it a tick.
      await nextTick()
      await Promise.resolve()
      expect(reg.update).toHaveBeenCalled()
    } finally {
      wrapper.unmount()
      ;(globalThis.navigator as unknown as { serviceWorker: unknown }).serviceWorker = originalSW
    }
  })

  it('does NOT call update() on visibilitychange when tab is hidden', async () => {
    const { wrapper } = mountHost()
    const reg = { update: vi.fn() } as unknown as ServiceWorkerRegistration
    const originalSW = (globalThis.navigator as unknown as { serviceWorker: unknown }).serviceWorker
    ;(globalThis.navigator as unknown as { serviceWorker: unknown }).serviceWorker = {
      getRegistration: vi.fn().mockResolvedValue(reg),
    }
    try {
      Object.defineProperty(document, 'hidden', { value: true, configurable: true })
      document.dispatchEvent(new Event('visibilitychange'))
      await nextTick()
      await Promise.resolve()
      expect(reg.update).not.toHaveBeenCalled()
    } finally {
      wrapper.unmount()
      ;(globalThis.navigator as unknown as { serviceWorker: unknown }).serviceWorker = originalSW
    }
  })

  it('cleans up the visibilitychange listener on unmount', async () => {
    const { wrapper } = mountHost()
    const reg = { update: vi.fn() } as unknown as ServiceWorkerRegistration
    const originalSW = (globalThis.navigator as unknown as { serviceWorker: unknown }).serviceWorker
    ;(globalThis.navigator as unknown as { serviceWorker: unknown }).serviceWorker = {
      getRegistration: vi.fn().mockResolvedValue(reg),
    }
    wrapper.unmount()
    Object.defineProperty(document, 'hidden', { value: false, configurable: true })
    document.dispatchEvent(new Event('visibilitychange'))
    await nextTick()
    expect(reg.update).not.toHaveBeenCalled()
    ;(globalThis.navigator as unknown as { serviceWorker: unknown }).serviceWorker = originalSW
  })

  // The native app runs the code it was installed with — no service worker there (operator
  // 2026-10-06). The website's worker, registered inside the app, cached the PREVIOUS install's
  // code and served it on the first launch after every update (measured: installed 21:53:52Z, the
  // first launch ran 21:25:15Z). It never brought anything from the web deploy: its scope is the
  // app's own origin.
  describe('in the native app', () => {
    const unregister = vi.fn().mockResolvedValue(true)
    const deleteCache = vi.fn().mockResolvedValue(true)
    let originalSW: unknown
    let originalCaches: unknown

    beforeEach(() => {
      native.value = true
      unregister.mockClear()
      deleteCache.mockClear()
      originalSW = (globalThis.navigator as unknown as { serviceWorker: unknown }).serviceWorker
      originalCaches = (globalThis as unknown as { caches: unknown }).caches
      ;(globalThis.navigator as unknown as { serviceWorker: unknown }).serviceWorker = {
        getRegistrations: vi.fn().mockResolvedValue([{ unregister }, { unregister }]),
        getRegistration: vi.fn().mockResolvedValue(undefined),
      }
      ;(globalThis as unknown as { caches: unknown }).caches = {
        keys: vi.fn().mockResolvedValue(['workbox-precache-v2', 'app-artwork', 'api-app']),
        delete: deleteCache,
      }
    })
    afterEach(() => {
      ;(globalThis.navigator as unknown as { serviceWorker: unknown }).serviceWorker = originalSW
      ;(globalThis as unknown as { caches: unknown }).caches = originalCaches
    })

    it('never registers a service worker', () => {
      mountHost()
      expect(registerCalls).toBe(0)
    })

    it('removes the workers and caches an earlier version left behind', async () => {
      mountHost()
      await vi.waitFor(() => expect(deleteCache).toHaveBeenCalledTimes(3))
      expect(unregister).toHaveBeenCalledTimes(2)
      expect(deleteCache.mock.calls.map((c) => c[0])).toEqual(['workbox-precache-v2', 'app-artwork', 'api-app'])
    })

    it('never asks to reload, and applying an update is a no-op', async () => {
      const { api } = mountHost()
      expect(api.value?.needRefresh.value).toBe(false)
      await api.value?.applyUpdate()
      expect(updateServiceWorker).not.toHaveBeenCalled()
    })
  })

  it('on the web, registers the worker and leaves a waiting one to the reader (the toast asks)', async () => {
    mountHost()
    expect(registerCalls).toBe(1)
    needRefresh.value = true
    await nextTick()
    expect(updateServiceWorker).not.toHaveBeenCalled()
  })
})
