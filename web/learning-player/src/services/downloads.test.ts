import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useDownloadsStore } from '../stores/downloads'
import * as api from './api'
import * as deviceStore from './deviceStore'

const addListener = vi.fn()
const downloadFile = vi.fn()
const getUri = vi.fn()
const stat = vi.fn()
const deleteFile = vi.fn()

vi.mock('@capacitor/filesystem', () => ({
  Directory: { LibraryNoCloud: 'LIBRARY_NO_CLOUD', Data: 'DATA', Cache: 'CACHE' },
  Filesystem: {
    addListener: (...a: unknown[]) => addListener(...a),
    downloadFile: (...a: unknown[]) => downloadFile(...a),
    getUri: (...a: unknown[]) => getUri(...a),
    stat: (...a: unknown[]) => stat(...a),
    deleteFile: (...a: unknown[]) => deleteFile(...a),
  },
}))

const { absolutize, deleteEpisode, downloadEpisode, pathFor } = await import('./downloads')

const remove = vi.fn()

beforeEach(() => {
  setActivePinia(createPinia())
  vi.spyOn(deviceStore, 'setDeviceJson').mockResolvedValue()
  vi.spyOn(deviceStore, 'getDeviceJson').mockResolvedValue(null)
  vi.spyOn(api, 'getAudioSource').mockResolvedValue({
    episode_slug: 'a',
    url: 'https://cdn.example.com/a.mp3',
  } as unknown as Awaited<ReturnType<typeof api.getAudioSource>>)
  addListener.mockResolvedValue({ remove })
  downloadFile.mockResolvedValue({ path: 'offline-audio/a.mp3' })
  getUri.mockResolvedValue({ uri: 'file:///Library/offline-audio/a.mp3' })
  stat.mockResolvedValue({ size: 4242, type: 'file', ctime: 0, mtime: 0, uri: 'file:///x' })
  deleteFile.mockResolvedValue(undefined)
})
afterEach(() => {
  vi.restoreAllMocks()
  vi.clearAllMocks()
})

describe('pathFor', () => {
  it('derives the extension from the URL and sanitises the slug', () => {
    expect(pathFor('p05-ee8e', 'https://x/y.m4a')).toBe('offline-audio/p05-ee8e.m4a')
    expect(pathFor('a/../b', 'https://x/y.mp3')).toBe('offline-audio/a_.._b.mp3')
  })
  it('ignores query strings and falls back to mp3', () => {
    expect(pathFor('a', 'https://x/y.mp3?token=1')).toBe('offline-audio/a.mp3')
    expect(pathFor('a', 'https://x/stream')).toBe('offline-audio/a.mp3')
  })
})

describe('absolutize', () => {
  it('passes an absolute origin URL through', () => {
    expect(absolutize('https://cdn.example.com/a.mp3')).toBe('https://cdn.example.com/a.mp3')
  })
  it('resolves a relative fixture URL against the document origin', () => {
    expect(absolutize('/audio/a.mp3')).toBe(`${window.location.origin}/audio/a.mp3`)
  })
  it('refuses a non-http(s) source rather than handing it to the plugin', () => {
    expect(() => absolutize('data:audio/mp3;base64,AAAA')).toThrow(/not an http/)
  })
})

describe('downloadEpisode', () => {
  it('downloads, records the file, and reports success', async () => {
    const store = useDownloadsStore()
    await expect(downloadEpisode('a')).resolves.toBe(true)

    expect(downloadFile).toHaveBeenCalledWith(
      expect.objectContaining({
        url: 'https://cdn.example.com/a.mp3',
        path: 'offline-audio/a.mp3',
        directory: 'LIBRARY_NO_CLOUD',
        progress: true,
        recursive: true,
      }),
    )
    expect(store.isDownloaded('a')).toBe(true)
    expect(store.entry('a')?.uri).toBe('file:///Library/offline-audio/a.mp3')
    expect(store.entry('a')?.bytes).toBe(4242)
    // Recorded at start, so deletion never needs the network again.
    expect(store.entry('a')?.path).toBe('offline-audio/a.mp3')
    expect(remove).toHaveBeenCalled()
  })

  it('short-circuits an episode already on disk', async () => {
    const store = useDownloadsStore()
    await store.setDownloaded('a', 'file:///a.mp3', 1)
    await expect(downloadEpisode('a')).resolves.toBe(true)
    expect(downloadFile).not.toHaveBeenCalled()
  })

  it('reports progress for its own transfer and ignores a sibling', async () => {
    const store = useDownloadsStore()
    await downloadEpisode('a')
    const onProgress = addListener.mock.calls[0][1] as (p: unknown) => void

    onProgress({ url: 'https://cdn.example.com/a.mp3', bytes: 50, contentLength: 200 })
    expect(store.progressOf('a')).toBe(0.25)

    onProgress({ url: 'https://cdn.example.com/OTHER.mp3', bytes: 200, contentLength: 200 })
    expect(store.progressOf('a')).toBe(0.25)
  })

  it('records a failure instead of throwing at the call site', async () => {
    downloadFile.mockRejectedValue(new Error('origin 404'))
    const store = useDownloadsStore()
    await expect(downloadEpisode('a')).resolves.toBe(false)
    expect(store.stateOf('a')).toBe('failed')
    expect(store.entry('a')?.error).toBe('origin 404')
    expect(remove).toHaveBeenCalled()
  })

  it('deletes the orphan when the episode was unmarked mid-transfer', async () => {
    const store = useDownloadsStore()
    // There is no abort, so a cancel drops the record while bytes keep arriving.
    downloadFile.mockImplementation(async () => {
      await store.unmark('a')
      return { path: 'offline-audio/a.mp3' }
    })
    await expect(downloadEpisode('a')).resolves.toBe(false)
    expect(store.entry('a')).toBeNull()
    expect(deleteFile).toHaveBeenCalledWith({
      directory: 'LIBRARY_NO_CLOUD',
      path: 'offline-audio/a.mp3',
    })
  })

  it('does not resurrect a cancelled episode as failed', async () => {
    const store = useDownloadsStore()
    downloadFile.mockImplementation(async () => {
      await store.unmark('a')
      throw new Error('connection reset')
    })
    await expect(downloadEpisode('a')).resolves.toBe(false)
    // A "failed" row on a screen the user just cleared would be a bug, not information.
    expect(store.entry('a')).toBeNull()
    expect(deleteFile).toHaveBeenCalled()
  })
})

describe('deleteEpisode', () => {
  it('drops the record and the bytes', async () => {
    const store = useDownloadsStore()
    await downloadEpisode('a')
    await deleteEpisode('a')
    expect(store.entry('a')).toBeNull()
    expect(deleteFile).toHaveBeenCalledWith({
      directory: 'LIBRARY_NO_CLOUD',
      path: 'offline-audio/a.mp3',
    })
  })

  it('is safe for an episode that was never downloaded', async () => {
    await expect(deleteEpisode('nope')).resolves.toBeUndefined()
    expect(deleteFile).not.toHaveBeenCalled()
  })

  it('survives a filesystem that cannot unlink', async () => {
    deleteFile.mockRejectedValue(new Error('EBUSY'))
    const store = useDownloadsStore()
    await downloadEpisode('a')
    await expect(deleteEpisode('a')).resolves.toBeUndefined()
    expect(store.entry('a')).toBeNull()
  })
})
