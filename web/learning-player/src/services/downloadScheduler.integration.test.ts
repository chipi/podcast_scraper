/**
 * Drain ↔ service integration (#1905).
 *
 * `downloadScheduler.test.ts` mocks `downloadEpisode` entirely, so the seam between the drain and
 * the transfer — mark-recreates-the-entry, the per-item cap check, namespace scoping — was never
 * exercised anywhere. This file mocks only the platform edges (Capacitor, api, native) and lets
 * the scheduler drive the REAL service.
 */
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useDownloadsStore } from '../stores/downloads'
import * as api from './api'
import * as deviceStore from './deviceStore'

const getStatus = vi.fn()
const downloadFile = vi.fn()
const getUri = vi.fn()
const stat = vi.fn()
const deleteFile = vi.fn()
const writeFile = vi.fn()
const isNative = vi.fn(() => true)
const localPosition = vi.fn((_s: string) => null as { finished: boolean } | null)

vi.mock('@capacitor/filesystem', () => ({
  Directory: { LibraryNoCloud: 'LIBRARY_NO_CLOUD' },
  Encoding: { UTF8: 'utf8' },
  Filesystem: {
    addListener: async () => ({ remove: vi.fn() }),
    downloadFile: (...a: unknown[]) => downloadFile(...a),
    getUri: (...a: unknown[]) => getUri(...a),
    stat: (...a: unknown[]) => stat(...a),
    deleteFile: (...a: unknown[]) => deleteFile(...a),
    writeFile: (...a: unknown[]) => writeFile(...a),
  },
}))
vi.mock('@capacitor/core', () => ({
  Capacitor: { convertFileSrc: (u: string) => u, isNativePlatform: () => false, getPlatform: () => 'web' },
}))
vi.mock('@capacitor/network', () => ({
  Network: { getStatus: (...a: unknown[]) => getStatus(...a), addListener: async () => ({ remove: vi.fn() }) },
}))
vi.mock('@capacitor/app', () => ({ App: { addListener: async () => ({ remove: vi.fn() }) } }))
vi.mock('./native', () => ({ isNative: () => isNative() }))
vi.mock('./playbackPositions', () => ({ localPosition: (s: string) => localPosition(s) }))

const { drainQueue } = await import('./downloadScheduler')

let disk: Record<string, unknown> = {}

beforeEach(() => {
  setActivePinia(createPinia())
  disk = {}
  isNative.mockReturnValue(true)
  localPosition.mockReturnValue(null)
  getStatus.mockResolvedValue({ connected: true, connectionType: 'wifi' })
  downloadFile.mockResolvedValue({ path: 'x' })
  getUri.mockResolvedValue({ uri: 'file:///a.mp3' })
  stat.mockResolvedValue({ size: 1234, type: 'file', ctime: 0, mtime: 0, uri: 'file:///a.mp3' })
  deleteFile.mockResolvedValue(undefined)
  writeFile.mockResolvedValue({ uri: 'file:///t.json' })
  vi.spyOn(deviceStore, 'setDeviceJson').mockImplementation(async (k, v) => {
    disk[k] = v
  })
  vi.spyOn(deviceStore, 'getDeviceJson').mockImplementation(async (k) => (disk[k] ?? null) as never)
  vi.spyOn(api, 'getAudioSource').mockResolvedValue({
    episode_slug: 'x',
    url: 'https://cdn.example.com/a.mp3',
  } as unknown as Awaited<ReturnType<typeof api.getAudioSource>>)
  vi.spyOn(api, 'getEpisode').mockResolvedValue({
    slug: 'x',
    title: 'Ep',
    podcast_title: 'Show',
    feed_id: 'p05',
    duration_seconds: 100,
    artwork_url: null,
    episode_image_url: null,
    feed_image_url: null,
  } as unknown as Awaited<ReturnType<typeof api.getEpisode>>)
  vi.spyOn(api, 'getSegments').mockResolvedValue({
    segments: [],
  } as unknown as Awaited<ReturnType<typeof api.getSegments>>)
})
afterEach(() => {
  vi.restoreAllMocks()
  vi.clearAllMocks()
})

describe('drain drives the real transfer', () => {
  // NOTE: these mark() directly rather than via markForOffline(), which fires its own
  // un-awaited drain — the in-flight guard would then swallow the drain under test.
  it('takes a queued episode all the way to downloaded', async () => {
    const store = useDownloadsStore()
    await useDownloadsStore().mark('ep-a')
    await drainQueue()

    expect(store.stateOf('ep-a')).toBe('downloaded')
    expect(store.entry('ep-a')?.title).toBe('Ep')
    expect(store.entry('ep-a')?.path).toBe('offline-audio/anon/ep-a.mp3')
  })

  it('leaves it queued on cellular under the default policy, then completes when wifi returns', async () => {
    getStatus.mockResolvedValue({ connected: true, connectionType: 'cellular' })
    const store = useDownloadsStore()
    await useDownloadsStore().mark('ep-b')
    await drainQueue()
    expect(store.stateOf('ep-b')).toBe('queued')
    expect(downloadFile).not.toHaveBeenCalled()

    getStatus.mockResolvedValue({ connected: true, connectionType: 'wifi' })
    await drainQueue()
    expect(store.stateOf('ep-b')).toBe('downloaded')
  })

  it('classifies a corpus-removed episode as permanent and stops retrying it', async () => {
    vi.spyOn(api, 'getAudioSource').mockRejectedValue(new api.ApiError(404, 'gone'))
    const store = useDownloadsStore()
    await useDownloadsStore().mark('ep-gone')
    await drainQueue()
    expect(store.entry('ep-gone')?.errorKind).toBe('permanent')

    // The drain must not pick it back up on the next network change.
    await drainQueue()
    expect(store.stateOf('ep-gone')).toBe('failed')
  })

  it('re-queues and completes a transient failure on the next drain', async () => {
    downloadFile.mockRejectedValueOnce(new Error('socket reset'))
    const store = useDownloadsStore()
    await useDownloadsStore().mark('ep-flaky')
    await drainQueue()
    expect(store.entry('ep-flaky')?.errorKind).toBe('retryable')

    await drainQueue()
    expect(store.stateOf('ep-flaky')).toBe('downloaded')
  })

  it('writes into the account that owns the drain', async () => {
    const store = useDownloadsStore()
    await store.setNamespace('u_alice')
    await store.mark('ep-ns')
    await drainQueue()
    expect(store.entry('ep-ns')?.path).toBe('offline-audio/u_alice/ep-ns.mp3')

    await store.setNamespace('u_bob')
    expect(store.entry('ep-ns')).toBeNull()
  })
})
