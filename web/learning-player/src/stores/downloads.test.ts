import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import * as deviceStore from '../services/deviceStore'
import { REGISTRY_KEY, useDownloadsStore, type DownloadEntry } from './downloads'

function entry(slug: string, over: Partial<DownloadEntry> = {}): DownloadEntry {
  return { slug, state: 'downloaded', updatedAt: 1, ...over }
}

beforeEach(() => {
  setActivePinia(createPinia())
  vi.spyOn(deviceStore, 'setDeviceJson').mockResolvedValue()
  vi.spyOn(deviceStore, 'getDeviceJson').mockResolvedValue(null)
})
afterEach(() => vi.restoreAllMocks())

describe('downloads store', () => {
  it('load() starts empty when the device has no registry yet', async () => {
    const d = useDownloadsStore()
    await d.load()
    expect(d.entries).toEqual({})
    expect(d.loaded).toBe(true)
  })

  it('load() demotes an interrupted download back to queued (there is no resume)', async () => {
    vi.spyOn(deviceStore, 'getDeviceJson').mockResolvedValue({
      a: entry('a', { state: 'downloading' }),
      b: entry('b', { state: 'downloaded' }),
    })
    const d = useDownloadsStore()
    await d.load()
    // Otherwise the UI shows a spinner for a transfer that died with the process.
    expect(d.stateOf('a')).toBe('queued')
    expect(d.stateOf('b')).toBe('downloaded')
  })

  it('load() coalesces concurrent calls onto one read', async () => {
    const d = useDownloadsStore()
    await Promise.all([d.load(), d.load(), d.load()])
    expect(deviceStore.getDeviceJson).toHaveBeenCalledTimes(1)
  })

  it('mark() queues an episode and persists it', async () => {
    const d = useDownloadsStore()
    expect(await d.mark('a')).toBe(true)
    expect(d.stateOf('a')).toBe('queued')
    expect(deviceStore.setDeviceJson).toHaveBeenCalledWith(REGISTRY_KEY, d.entries)
  })

  it('mark() is a no-op for an episode already queued, downloading, or downloaded', async () => {
    const d = useDownloadsStore()
    await d.mark('a')
    // Re-flagging must never restart a live transfer.
    expect(await d.mark('a')).toBe(false)

    d.setDownloading('b')
    expect(await d.mark('b')).toBe(false)

    await d.setDownloaded('c', 'file:///c.mp3', 10)
    expect(await d.mark('c')).toBe(false)
  })

  it('mark() retries a previously failed episode', async () => {
    const d = useDownloadsStore()
    await d.setFailed('a', 'network down')
    expect(await d.mark('a')).toBe(true)
    expect(d.stateOf('a')).toBe('queued')
    expect(d.entry('a')?.error).toBeUndefined()
  })

  it('runs the queued -> downloading -> downloaded lifecycle', async () => {
    const d = useDownloadsStore()
    await d.mark('a')
    d.setDownloading('a')
    expect(d.stateOf('a')).toBe('downloading')

    d.setProgress('a', 0.5)
    expect(d.progressOf('a')).toBe(0.5)

    await d.setDownloaded('a', 'file:///a.mp3', 4242)
    expect(d.stateOf('a')).toBe('downloaded')
    expect(d.entry('a')?.uri).toBe('file:///a.mp3')
    expect(d.entry('a')?.bytes).toBe(4242)
    expect(d.isDownloaded('a')).toBe(true)
    // Progress is transient — a completed download has none to report.
    expect(d.progressOf('a')).toBe(0)
  })

  it('setProgress() clamps to 0..1', () => {
    const d = useDownloadsStore()
    d.setProgress('a', 1.8)
    expect(d.progressOf('a')).toBe(1)
    d.setProgress('a', -3)
    expect(d.progressOf('a')).toBe(0)
  })

  it('setFailed() records the error and drops progress', async () => {
    const d = useDownloadsStore()
    d.setProgress('a', 0.3)
    await d.setFailed('a', 'origin 404')
    expect(d.stateOf('a')).toBe('failed')
    expect(d.entry('a')?.error).toBe('origin 404')
    expect(d.progressOf('a')).toBe(0)
  })

  it('unmark() forgets the entry, and reports whether it had one', async () => {
    const d = useDownloadsStore()
    await d.setDownloaded('a', 'file:///a.mp3', 1)
    expect(await d.unmark('a')).toBe(true)
    expect(d.entry('a')).toBeNull()
    expect(await d.unmark('a')).toBe(false)
  })

  it('a mutation made before the first load survives that load resolving', async () => {
    // Regression: setDownloaded() ran while `loaded` was still false, then the first
    // ensureLoaded() assigned the (empty) stored registry over it and the entry vanished.
    vi.spyOn(deviceStore, 'getDeviceJson').mockResolvedValue({
      old: entry('old', { state: 'downloaded', bytes: 7 }),
    })
    const d = useDownloadsStore()
    await d.setDownloaded('fresh', 'file:///fresh.mp3', 1)
    await d.ensureLoaded()
    expect(d.isDownloaded('fresh')).toBe(true)
    expect(d.isDownloaded('old')).toBe(true)
  })

  it('queued getter returns waiting slugs oldest-first', () => {
    const d = useDownloadsStore()
    d.loaded = true
    d.entries = {
      new: entry('new', { state: 'queued', updatedAt: 300 }),
      old: entry('old', { state: 'queued', updatedAt: 100 }),
      mid: entry('mid', { state: 'queued', updatedAt: 200 }),
      done: entry('done', { state: 'downloaded', updatedAt: 50 }),
    }
    expect(d.queued).toEqual(['old', 'mid', 'new'])
  })

  it('bytesOnDisk counts only completed downloads', () => {
    const d = useDownloadsStore()
    d.loaded = true
    d.entries = {
      a: entry('a', { state: 'downloaded', bytes: 100 }),
      // A partial transfer has no settled size to account for.
      b: entry('b', { state: 'downloading', bytes: 999 }),
      c: entry('c', { state: 'queued' }),
      d: entry('d', { state: 'downloaded', bytes: 25 }),
    }
    expect(d.bytesOnDisk).toBe(125)
    expect(d.downloadedCount).toBe(2)
  })

  it('a failing device write never rejects into a template handler', async () => {
    vi.spyOn(deviceStore, 'setDeviceJson').mockRejectedValue(new Error('storage full'))
    const d = useDownloadsStore()
    await expect(d.mark('a')).resolves.toBe(true)
    // The in-memory flag still flipped; it is simply lost on next launch.
    expect(d.stateOf('a')).toBe('queued')
  })
})
