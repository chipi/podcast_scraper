/**
 * The offline REASON model (2026-09-16).
 *
 * Why this exists: the app used to carry a boolean, derived only from `@capacitor/network`. That
 * signal reports the DEVICE's radio, so a server that was reachable-but-broken — a reboot had lost
 * its signing secret, the process stayed up, every authed route failed — left the app in its
 * "online" state, rendering a hybrid of cached content and error cards while insisting nothing was
 * wrong. Reproduced on the simulator: with the server stopped and the network up, no offline banner
 * appeared at all.
 *
 * The module holds process-lifetime singleton state, so every test re-imports it fresh.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@capacitor/core', () => ({ Capacitor: { isNativePlatform: () => false } }))
vi.mock('@capacitor/network', () => ({
  Network: { getStatus: vi.fn(async () => ({ connected: true })), addListener: vi.fn() },
}))

beforeEach(() => {
  vi.resetModules()
  try {
    localStorage.removeItem('lp.forceOffline')
  } catch {
    /* storage unavailable in this environment — the module defaults to not-forced anyway */
  }
})

describe('offlineReason', () => {
  it('is null when online and the server is answering', async () => {
    const { offlineReason } = await import('./useOnline')
    expect(offlineReason()).toBeNull()
  })

  it('reports "server" once failures cross the threshold, and NOT before', async () => {
    const { offlineReason, reportServerReachable, isOffline } = await import('./useOnline')
    // One failure is a blip — a single slow or aborted request must not flip the whole UI.
    reportServerReachable(false)
    expect(offlineReason()).toBeNull()
    reportServerReachable(false)
    expect(offlineReason()).toBe('server')
    expect(isOffline()).toBe(true)
  })

  it('a single success clears the degraded state — this is the recovery path', async () => {
    const { offlineReason, reportServerReachable } = await import('./useOnline')
    reportServerReachable(false)
    reportServerReachable(false)
    expect(offlineReason()).toBe('server')
    reportServerReachable(true)
    expect(offlineReason()).toBeNull()
  })

  it('READS are never gated on a degraded server, or it could never recover', async () => {
    const { reportServerReachable, isForcedOffline, isOffline } = await import('./useOnline')
    reportServerReachable(false)
    reportServerReachable(false)
    // `isOffline` is true (banner + writes route to the outbox) but the READ gate is unchanged:
    // a gated read could never succeed, so the app could never discover the server came back.
    expect(isOffline()).toBe(true)
    expect(isForcedOffline()).toBe(false)
  })

  it('the explicit Config switch outranks an inference about the server', async () => {
    const { offlineReason, reportServerReachable, setForcedOffline } = await import('./useOnline')
    reportServerReachable(false)
    reportServerReachable(false)
    expect(offlineReason()).toBe('server')
    setForcedOffline(true)
    expect(offlineReason()).toBe('forced')
    setForcedOffline(false)
    expect(offlineReason()).toBe('server')
  })
})
