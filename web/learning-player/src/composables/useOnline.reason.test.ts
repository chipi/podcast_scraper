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

  it('a route that needs no session answering 200 does not clear a server that refuses the rest', async () => {
    // 2026-10-12, Android degraded-server drill: with the signing secret gone, every read answered
    // 503 except `/corpus/trending-topics` behind Home's search box, and its 200 landed last.
    const { offlineReason, reportServerReachable } = await import('./useOnline')
    reportServerReachable(false, '/api/app/me', 'refused')
    reportServerReachable(true, '/api/app/corpus/trending-topics', 'refused') // a blip before: resets
    reportServerReachable(false, '/api/app/your-week', 'refused')
    reportServerReachable(false, '/api/app/queue', 'refused')
    expect(offlineReason()).toBe('server')
    reportServerReachable(true, '/api/app/corpus/trending-topics', 'refused')
    expect(offlineReason()).toBe('server')
    // Recovery is the failed route itself answering — "Try again", or the next read of it.
    reportServerReachable(true, '/api/app/queue', 'refused')
    expect(offlineReason()).toBeNull()
  })

  it('after a 503 that cleared, the next degradation needs two failures again', async () => {
    const { offlineReason, reportServerReachable } = await import('./useOnline')
    reportServerReachable(false, '/api/app/me', 'refused')
    reportServerReachable(false, '/api/app/queue', 'refused')
    reportServerReachable(true, '/api/app/me', 'refused')
    expect(offlineReason()).toBeNull()
    reportServerReachable(false, '/api/app/push/vapid-key', 'refused') // a 503 by configuration
    expect(offlineReason()).toBeNull()
  })

  it('when the server could not be reached at all, ANY answer proves it is back', async () => {
    const { offlineReason, reportServerReachable } = await import('./useOnline')
    reportServerReachable(false, '/api/app/me', 'unreachable')
    reportServerReachable(false, '/api/app/queue', 'unreachable')
    expect(offlineReason()).toBe('server')
    reportServerReachable(true, '/api/app/corpus/trending-topics', 'refused')
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
