/**
 * Native (Capacitor / WKWebView) connectivity regression tests.
 *
 * The bug (2026-09-15): `useOnline` trusted `navigator.onLine`, which false-negatives in WKWebView
 * — it reports offline on a live network, especially at cold start. Because the whole data layer
 * fails fast on `isOffline()`, EVERY read errored with "needs a connection" (empty profile / empty
 * collections / "couldn't reach the server") until a reload happened to re-read it as true.
 *
 * These tests pin the fix: on native we ignore `navigator.onLine` entirely and use the OS-level
 * `@capacitor/network` plugin, seeded OPTIMISTICALLY online — only a POSITIVE "disconnected" report
 * flips us offline. They mock the platform as native + control the plugin, and use `resetModules`
 * so the module singleton re-evaluates against the mocks (it reads the platform at import time).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

let netStatus: { connected: boolean } = { connected: true }
let statusChangeCb: ((s: { connected: boolean }) => void) | undefined

vi.mock('@capacitor/core', () => ({ Capacitor: { isNativePlatform: () => true } }))
vi.mock('@capacitor/network', () => ({
  Network: {
    getStatus: () => Promise.resolve(netStatus),
    addListener: (_event: string, cb: (s: { connected: boolean }) => void) => {
      statusChangeCb = cb
      return Promise.resolve({ remove: () => {} })
    },
  },
}))

function setNavigatorOnline(value: boolean): void {
  Object.defineProperty(navigator, 'onLine', { configurable: true, value })
}

describe('useOnline on native (WKWebView)', () => {
  beforeEach(() => {
    vi.resetModules()
    statusChangeCb = undefined
    netStatus = { connected: true }
  })
  afterEach(() => setNavigatorOnline(true))

  it('does NOT go offline just because navigator.onLine is false at boot (the WKWebView false-negative)', async () => {
    setNavigatorOnline(false)
    const { isOffline, useOnline } = await import('./useOnline')
    useOnline() // registers the native listeners; seeds from Network.getStatus()
    // The whole bug in one assertion: a false navigator.onLine must not make native reads fail fast.
    expect(isOffline()).toBe(false)
    await Promise.resolve() // let the seeding getStatus() (connected: true) resolve
    expect(isOffline()).toBe(false)
  })

  it('flips offline ONLY on a positive Network "disconnected" report, and recovers', async () => {
    setNavigatorOnline(false)
    const { isOffline, useOnline } = await import('./useOnline')
    useOnline()
    await Promise.resolve()
    expect(isOffline()).toBe(false)
    statusChangeCb?.({ connected: false }) // OS says genuinely offline
    expect(isOffline()).toBe(true)
    statusChangeCb?.({ connected: true }) // back online
    expect(isOffline()).toBe(false)
  })
})
