import { beforeEach, describe, expect, it, vi } from 'vitest'

/** #2277 — how the app came back. */
const { stateListeners, device, uptime } = vi.hoisted(() => ({
  stateListeners: [] as Array<(s: { isActive: boolean }) => void>,
  device: new Map<string, unknown>(),
  uptime: vi.fn(async () => ({ ms: 600_000 })),
}))

vi.mock('@capacitor/app', () => ({
  App: {
    addListener: vi.fn(async (_e: string, fn: (s: { isActive: boolean }) => void) => {
      stateListeners.push(fn)
      return { remove: async () => {} }
    }),
  },
}))
vi.mock('@capacitor/core', async (orig) => ({
  ...(await orig<typeof import('@capacitor/core')>()),
  registerPlugin: () => ({ uptime }),
}))
vi.mock('./native', () => ({ isNative: () => true }))
vi.mock('./deviceStore', () => ({
  getDeviceJson: vi.fn(async (k: string) => device.get(k) ?? null),
  setDeviceJson: vi.fn(async (k: string, v: unknown) => void device.set(k, v)),
}))
vi.mock('./analytics', () => ({ track: vi.fn() }))
vi.mock('./lastPlace', () => ({ restoreOutcome: () => 'both' }))

import { track } from './analytics'
import { classifyLaunch, initLifecycle, LIFECYCLE_KEY, toAwayBucket } from './lifecycle'

const tracked = () => (track as unknown as ReturnType<typeof vi.fn>).mock.calls

describe('classifyLaunch', () => {
  it('a page booted with its process is a cold launch', () => {
    expect(classifyLaunch(3_000, 1_200)).toBe('cold')
  })
  it('a page booted inside an older process is a WebView reload', () => {
    expect(classifyLaunch(600_000, 1_200)).toBe('webview_reload')
  })
  it('no uptime is unknown, never a guess', () => {
    expect(classifyLaunch(null, 1_200)).toBe('unknown')
    expect(classifyLaunch(Number.NaN, 1_200)).toBe('unknown')
  })
})

describe('toAwayBucket', () => {
  it('buckets at the documented boundaries', () => {
    expect(toAwayBucket(59_999)).toBe('<1m')
    expect(toAwayBucket(60_000)).toBe('1-5m')
    expect(toAwayBucket(5 * 60_000)).toBe('5-15m')
    expect(toAwayBucket(15 * 60_000)).toBe('15-60m')
    expect(toAwayBucket(60 * 60_000)).toBe('1h+')
  })
  it('none for no background, or nonsense', () => {
    expect(toAwayBucket(null)).toBe('none')
    expect(toAwayBucket(-5)).toBe('none')
  })
})

describe('initLifecycle', () => {
  beforeEach(() => {
    stateListeners.length = 0
    device.clear()
    ;(track as unknown as ReturnType<typeof vi.fn>).mockClear()
  })

  it('reports a launch after a background exit, with the time away and what was restored', async () => {
    device.set(LIFECYCLE_KEY, { state: 'background', at: Date.now() - 10 * 60_000 })
    await initLifecycle(() => {})
    expect(tracked()).toEqual([
      [
        'app_launch',
        { kind: 'webview_reload', previous_exit: 'background', away: '5-15m', restored: 'both' },
      ],
    ])
    expect((device.get(LIFECYCLE_KEY) as { state: string }).state).toBe('foreground')
  })

  it('a previous run that died in the foreground reports foreground with no time away', async () => {
    device.set(LIFECYCLE_KEY, { state: 'foreground', at: Date.now() - 60_000 })
    await initLifecycle(() => {})
    expect(tracked()[0][1]).toMatchObject({ previous_exit: 'foreground', away: 'none' })
  })

  it('first launch says first', async () => {
    await initLifecycle(() => {})
    expect(tracked()[0][1]).toMatchObject({ previous_exit: 'first', away: 'none' })
  })

  it('records background, saves the place, and reports a warm resume', async () => {
    const onBackground = vi.fn()
    await initLifecycle(onBackground)
    stateListeners[0]!({ isActive: false })
    expect(onBackground).toHaveBeenCalledOnce()
    expect((device.get(LIFECYCLE_KEY) as { state: string }).state).toBe('background')
    stateListeners[0]!({ isActive: true })
    expect(tracked()[1]).toEqual(['app_resume', { away: '<1m' }])
  })

  it('does not report a resume without a preceding background', async () => {
    await initLifecycle(() => {})
    stateListeners[0]!({ isActive: true })
    expect(tracked()).toHaveLength(1)
  })
})
