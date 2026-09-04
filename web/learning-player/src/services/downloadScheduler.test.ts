import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useDownloadsStore } from '../stores/downloads'
import * as deviceStore from './deviceStore'

const getStatus = vi.fn()
const addNetListener = vi.fn()
const addAppListener = vi.fn()
const downloadEpisode = vi.fn()
const isNative = vi.fn(() => true)

vi.mock('@capacitor/network', () => ({
  Network: {
    getStatus: (...a: unknown[]) => getStatus(...a),
    addListener: (...a: unknown[]) => addNetListener(...a),
  },
}))
vi.mock('@capacitor/app', () => ({
  App: { addListener: (...a: unknown[]) => addAppListener(...a) },
}))
vi.mock('./downloads', () => ({ downloadEpisode: (s: string) => downloadEpisode(s) }))
vi.mock('./native', () => ({ isNative: () => isNative() }))

const {
  DEFAULT_POLICY,
  POLICY_KEY,
  allows,
  drainQueue,
  getNetworkPolicy,
  markForOffline,
  setNetworkPolicy,
  startDownloadScheduler,
} = await import('./downloadScheduler')

let disk: Record<string, unknown> = {}
const wifi = { connected: true, connectionType: 'wifi' }
const cellular = { connected: true, connectionType: 'cellular' }

beforeEach(() => {
  setActivePinia(createPinia())
  disk = {}
  isNative.mockReturnValue(true)
  vi.spyOn(deviceStore, 'setDeviceJson').mockImplementation(async (k, v) => {
    disk[k] = v
  })
  vi.spyOn(deviceStore, 'getDeviceJson').mockImplementation(async (k) => (disk[k] ?? null) as never)
  getStatus.mockResolvedValue(wifi)
  downloadEpisode.mockResolvedValue(true)
  addNetListener.mockResolvedValue({ remove: vi.fn() })
  addAppListener.mockResolvedValue({ remove: vi.fn() })
})
afterEach(() => {
  vi.restoreAllMocks()
  vi.clearAllMocks()
})

describe('allows — the policy table', () => {
  it('wifi always starts', () => {
    expect(allows('wifi-only', wifi)).toBe(true)
    expect(allows('any', wifi)).toBe(true)
  })
  it('cellular starts only when the user opted in', () => {
    expect(allows('wifi-only', cellular)).toBe(false)
    expect(allows('any', cellular)).toBe(true)
  })
  it('never starts with no connection', () => {
    expect(allows('any', { connected: false, connectionType: 'none' })).toBe(false)
  })
  it('never gambles on an unknown link, even when the user allowed cellular', () => {
    // Guessing wrong costs the user real money on a metered connection.
    expect(allows('any', { connected: true, connectionType: 'unknown' })).toBe(false)
  })
})

describe('network policy', () => {
  it('defaults to wifi-only', async () => {
    await expect(getNetworkPolicy()).resolves.toBe(DEFAULT_POLICY)
    expect(DEFAULT_POLICY).toBe('wifi-only')
  })
  it('persists per device', async () => {
    await setNetworkPolicy('any')
    expect(disk[POLICY_KEY]).toBe('any')
    await expect(getNetworkPolicy()).resolves.toBe('any')
  })
})

describe('drainQueue', () => {
  it('starts queued downloads oldest-first on wifi', async () => {
    const store = useDownloadsStore()
    await store.ensureLoaded()
    store.entries = {
      b: { slug: 'b', state: 'queued', updatedAt: 200 },
      a: { slug: 'a', state: 'queued', updatedAt: 100 },
    }
    await drainQueue()
    expect(downloadEpisode.mock.calls.map((c) => c[0])).toEqual(['a', 'b'])
  })

  it('leaves everything queued on cellular under the default policy', async () => {
    getStatus.mockResolvedValue(cellular)
    const store = useDownloadsStore()
    await store.ensureLoaded()
    await store.mark('a')
    await drainQueue()
    expect(downloadEpisode).not.toHaveBeenCalled()
    expect(store.stateOf('a')).toBe('queued')
  })

  it('stops mid-drain when the link drops', async () => {
    const store = useDownloadsStore()
    await store.ensureLoaded()
    store.entries = {
      a: { slug: 'a', state: 'queued', updatedAt: 1 },
      b: { slug: 'b', state: 'queued', updatedAt: 2 },
    }
    getStatus.mockResolvedValueOnce(wifi).mockResolvedValue({ connected: false, connectionType: 'none' })
    await drainQueue()
    expect(downloadEpisode.mock.calls.map((c) => c[0])).toEqual(['a'])
  })

  it('retries a transient failure but never a permanent one', async () => {
    const store = useDownloadsStore()
    await store.ensureLoaded()
    await store.setFailed('flaky', 'socket reset', 'retryable')
    await store.setFailed('gone', 'not found', 'permanent')
    await drainQueue()
    // Otherwise a corpus-removed episode is retried on every network change, forever.
    expect(downloadEpisode.mock.calls.map((c) => c[0])).toEqual(['flaky'])
    expect(store.stateOf('gone')).toBe('failed')
  })

  it('does not run two drains at once', async () => {
    const store = useDownloadsStore()
    await store.ensureLoaded()
    store.entries = { a: { slug: 'a', state: 'queued', updatedAt: 1 } }
    // Resume and networkStatusChange routinely arrive together.
    await Promise.all([drainQueue(), drainQueue(), drainQueue()])
    expect(downloadEpisode).toHaveBeenCalledTimes(1)
  })

  it('does nothing on the web', async () => {
    isNative.mockReturnValue(false)
    const store = useDownloadsStore()
    await store.ensureLoaded()
    await store.mark('a')
    await drainQueue()
    expect(downloadEpisode).not.toHaveBeenCalled()
  })
})

describe('markForOffline', () => {
  it('flags the episode and starts it when the connection already allows', async () => {
    const store = useDownloadsStore()
    await expect(markForOffline('a')).resolves.toBe(true)
    expect(store.stateOf('a')).toBe('queued')
    await vi.waitFor(() => expect(downloadEpisode).toHaveBeenCalledWith('a'))
  })

  it('reports no change when the episode is already flagged', async () => {
    await markForOffline('a')
    await expect(markForOffline('a')).resolves.toBe(false)
  })
})

describe('startDownloadScheduler', () => {
  it('wires the L1 triggers: network change and app resume', async () => {
    await startDownloadScheduler()
    expect(addNetListener).toHaveBeenCalledWith('networkStatusChange', expect.any(Function))
    expect(addAppListener).toHaveBeenCalledWith('resume', expect.any(Function))
  })

  it('wires nothing on the web', async () => {
    isNative.mockReturnValue(false)
    await startDownloadScheduler()
    expect(addNetListener).not.toHaveBeenCalled()
  })
})
