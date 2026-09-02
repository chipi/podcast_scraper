import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import * as deviceStore from '../services/deviceStore'
import { ANON_NAMESPACE, registryKeyFor, useDownloadsStore, type DownloadEntry } from './downloads'

const REGISTRY_KEY = registryKeyFor(ANON_NAMESPACE)

/**
 * A faithful fake of device storage: a read returns what the last write actually stored. The
 * earlier version returned a fixed snapshot regardless of writes, which let a persist-ordering
 * bug pass its own regression test.
 */
let disk: Record<string, string> = {}

function entry(slug: string, over: Partial<DownloadEntry> = {}): DownloadEntry {
  return { slug, state: 'downloaded', updatedAt: 1, ...over }
}

function seedDisk(entries: Record<string, DownloadEntry>): void {
  disk[REGISTRY_KEY] = JSON.stringify(entries)
}

beforeEach(() => {
  setActivePinia(createPinia())
  disk = {}
  vi.spyOn(deviceStore, 'setDeviceJson').mockImplementation(async (key, value) => {
    disk[key] = JSON.stringify(value)
  })
  vi.spyOn(deviceStore, 'getDeviceJson').mockImplementation(
    async (key) => (disk[key] ? JSON.parse(disk[key]) : null) as never,
  )
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
    seedDisk({ a: entry('a', { state: 'downloading' }), b: entry('b', { state: 'downloaded' }) })
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
    expect(JSON.parse(disk[REGISTRY_KEY]).a.state).toBe('queued')
  })

  it('mark() is a no-op for an episode already queued, downloading, or downloaded', async () => {
    const d = useDownloadsStore()
    await d.mark('a')
    // Re-flagging must never restart a live transfer.
    expect(await d.mark('a')).toBe(false)

    await d.ensureLoaded()
    d.setDownloading('b')
    expect(await d.mark('b')).toBe(false)

    await d.setDownloaded('c', 'file:///c.mp3', 10)
    expect(await d.mark('c')).toBe(false)
  })

  it('mark() retries a previously failed episode and clears the failure', async () => {
    const d = useDownloadsStore()
    await d.ensureLoaded()
    await d.setFailed('a', 'network down')
    expect(await d.mark('a')).toBe(true)
    expect(d.stateOf('a')).toBe('queued')
    expect(d.entry('a')?.error).toBeUndefined()
  })

  it('runs the queued -> downloading -> downloaded lifecycle', async () => {
    const d = useDownloadsStore()
    await d.mark('a')
    d.setDownloading('a', 'offline-audio/a.mp3')
    expect(d.stateOf('a')).toBe('downloading')
    expect(d.entry('a')?.path).toBe('offline-audio/a.mp3')

    d.setProgress('a', 0.5)
    expect(d.progressOf('a')).toBe(0.5)

    await d.setDownloaded('a', 'file:///a.mp3', 4242)
    expect(d.stateOf('a')).toBe('downloaded')
    expect(d.entry('a')?.uri).toBe('file:///a.mp3')
    expect(d.entry('a')?.bytes).toBe(4242)
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

  it('records offline display metadata and artwork on an existing entry', async () => {
    const d = useDownloadsStore()
    await d.mark('a')
    d.setMetadata('a', { title: 'Ep One', showTitle: 'The Show', durationSeconds: 416 })
    d.setArtworkPath('a', 'offline-artwork/a.jpg')
    expect(d.entry('a')?.title).toBe('Ep One')
    expect(d.entry('a')?.showTitle).toBe('The Show')
    expect(d.entry('a')?.durationSeconds).toBe(416)
    expect(d.entry('a')?.artworkPath).toBe('offline-artwork/a.jpg')
  })

  it('does not invent an entry when metadata arrives for an unknown episode', () => {
    const d = useDownloadsStore()
    d.setMetadata('ghost', { title: 'nope' })
    d.setArtworkPath('ghost', 'offline-artwork/ghost.jpg')
    expect(d.entry('ghost')).toBeNull()
  })

  it('setFailed() records the error, its kind, and drops progress', async () => {
    const d = useDownloadsStore()
    await d.ensureLoaded()
    d.setProgress('a', 0.3)
    await d.setFailed('a', 'origin 404', 'permanent')
    expect(d.stateOf('a')).toBe('failed')
    expect(d.entry('a')?.error).toBe('origin 404')
    // Without a kind the drain would retry a gone episode on every network change, forever.
    expect(d.entry('a')?.errorKind).toBe('permanent')
    expect(d.progressOf('a')).toBe(0)
  })

  it('setFailed() defaults to retryable', async () => {
    const d = useDownloadsStore()
    await d.ensureLoaded()
    await d.setFailed('a', 'socket reset')
    expect(d.entry('a')?.errorKind).toBe('retryable')
  })

  it('_forget() drops the record, and reports whether it had one', async () => {
    const d = useDownloadsStore()
    await d.setDownloaded('a', 'file:///a.mp3', 1)
    expect(await d._forget('a')).toBe(true)
    expect(d.entry('a')).toBeNull()
    expect(await d._forget('a')).toBe(false)
  })

  it('a write before the first load does not clobber the stored registry', async () => {
    // Regression: _persist ran unconditionally, so a setter firing before load() overwrote the
    // whole on-disk registry with a near-empty map — and the later read returned that clobbered
    // value, so the in-memory-wins merge recovered nothing. Every prior download's record was
    // lost and its file orphaned.
    seedDisk({ old: entry('old', { state: 'downloaded', bytes: 7 }) })
    const d = useDownloadsStore()

    await d.setDownloaded('fresh', 'file:///fresh.mp3', 1)
    // Nothing may have been written yet: entries is not yet the union of what is on disk.
    expect(JSON.parse(disk[REGISTRY_KEY]).old).toBeTruthy()
    expect(JSON.parse(disk[REGISTRY_KEY]).fresh).toBeUndefined()

    await d.ensureLoaded()
    expect(d.isDownloaded('fresh')).toBe(true)
    expect(d.isDownloaded('old')).toBe(true)
    // ...and the merged union is flushed, so both survive the next launch.
    const flushed = JSON.parse(disk[REGISTRY_KEY])
    expect(Object.keys(flushed).sort()).toEqual(['fresh', 'old'])
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
    const d = useDownloadsStore()
    await d.ensureLoaded()
    vi.spyOn(deviceStore, 'setDeviceJson').mockRejectedValue(new Error('storage full'))
    await expect(d.mark('a')).resolves.toBe(true)
    // The in-memory flag still flipped; it is simply lost on next launch.
    expect(d.stateOf('a')).toBe('queued')
  })

  // #1905 — the registry is per ACCOUNT, because the list of downloaded episodes is listening
  // history. Device settings go the other way and are shared; see DeviceSettings.vue.

  it('loads a different account into a different registry', async () => {
    disk[registryKeyFor('u_alice')] = JSON.stringify({
      a: entry('a', { state: 'downloaded', title: "Alice's" }),
    })
    const d = useDownloadsStore()
    await d.setNamespace('u_alice')
    expect(d.isDownloaded('a')).toBe(true)

    await d.setNamespace('u_bob')
    // Bob must not see, or be able to delete, Alice's downloads.
    expect(d.entries).toEqual({})
    expect(d.isDownloaded('a')).toBe(false)
  })

  it('does not write one account\'s entries into another\'s registry', async () => {
    const d = useDownloadsStore()
    await d.setNamespace('u_alice')
    await d.mark('a')
    await d.setNamespace('u_bob')
    await d.mark('b')

    expect(Object.keys(JSON.parse(disk[registryKeyFor('u_alice')]))).toEqual(['a'])
    expect(Object.keys(JSON.parse(disk[registryKeyFor('u_bob')]))).toEqual(['b'])
  })

  it('keeps an account\'s downloads for a later re-login', async () => {
    const d = useDownloadsStore()
    await d.setNamespace('u_alice')
    await d.setDownloaded('a', 'file:///a.mp3', 1)
    await d.setNamespace(ANON_NAMESPACE)
    await d.setNamespace('u_alice')
    expect(d.isDownloaded('a')).toBe(true)
  })

  it('files are stored under a per-account folder', async () => {
    const d = useDownloadsStore()
    await d.setNamespace('u_alice')
    expect(d.folderFor('offline-audio')).toBe('offline-audio/u_alice')
  })
})

it('a successful download clears the previous attempt error KIND, not just the message', async () => {
  // `errorKind` is read by the drain's retry sweep, so a stale one on a downloaded entry is a
  // wrong input to that decision, not just untidy state (seen on the device, #1925 decision 4).
  const store = useDownloadsStore()
  await store.mark('ep-1')
  await store.setFailed('ep-1', 'boom', 'needs-space')
  expect(store.entry('ep-1')?.errorKind).toBe('needs-space')

  await store.setDownloaded('ep-1', 'file:///x/ep-1.mp3', 42)
  expect(store.entry('ep-1')?.error).toBeUndefined()
  expect(store.entry('ep-1')?.errorKind).toBeUndefined()
})
