import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useDownloadsStore } from '../stores/downloads'
import * as api from './api'
import { ApiError } from './api'
import * as deviceStore from './deviceStore'

const addListener = vi.fn()
const downloadFile = vi.fn()
const getUri = vi.fn()
const stat = vi.fn()
const deleteFile = vi.fn()
const writeFile = vi.fn()
const readFile = vi.fn()
const isNative = vi.fn(() => true)
const localPosition = vi.fn((_s: string) => null as { finished: boolean } | null)

vi.mock('@capacitor/filesystem', () => ({
  Directory: { LibraryNoCloud: 'LIBRARY_NO_CLOUD', Data: 'DATA', Cache: 'CACHE' },
  Encoding: { UTF8: 'utf8' },
  Filesystem: {
    addListener: (...a: unknown[]) => addListener(...a),
    downloadFile: (...a: unknown[]) => downloadFile(...a),
    getUri: (...a: unknown[]) => getUri(...a),
    stat: (...a: unknown[]) => stat(...a),
    deleteFile: (...a: unknown[]) => deleteFile(...a),
    writeFile: (...a: unknown[]) => writeFile(...a),
    readFile: (...a: unknown[]) => readFile(...a),
  },
}))
vi.mock('./native', () => ({ isNative: () => isNative() }))
vi.mock('./playbackPositions', () => ({ localPosition: (s: string) => localPosition(s) }))
vi.mock('@capacitor/core', () => ({
  // isNativePlatform/getPlatform are needed because services/tier.ts reads them transitively.
  Capacitor: {
    convertFileSrc: (u: string) => `capacitor-file://${u}`,
    isNativePlatform: () => false,
    getPlatform: () => 'web',
  },
}))

const {
  DOWNLOAD_CAP_BYTES,
  absolutize,
  artworkPathFor,
  deleteEpisode,
  downloadEpisode,
  localSourceFor,
  localTranscriptFor,
  pathFor,
  reclaimFinished,
  refreshLocalUris,
  transcriptPathFor,
} = await import('./downloads')

const remove = vi.fn()
let disk: Record<string, string> = {}

function audioSource(url = 'https://cdn.example.com/a.mp3') {
  return { episode_slug: 'x', url } as unknown as Awaited<ReturnType<typeof api.getAudioSource>>
}
function episodeDetail(over: Record<string, unknown> = {}) {
  return {
    slug: 'x',
    title: 'Ep One',
    podcast_title: 'The Show',
    duration_seconds: 416,
    artwork_url: 'https://cdn.example.com/art.jpg',
    episode_image_url: null,
    feed_image_url: null,
    ...over,
  } as unknown as Awaited<ReturnType<typeof api.getEpisode>>
}

beforeEach(() => {
  setActivePinia(createPinia())
  disk = {}
  isNative.mockReturnValue(true)
  localPosition.mockReturnValue(null)
  vi.spyOn(deviceStore, 'setDeviceJson').mockImplementation(async (key, value) => {
    disk[key] = JSON.stringify(value)
  })
  vi.spyOn(deviceStore, 'getDeviceJson').mockImplementation(
    async (key) => (disk[key] ? JSON.parse(disk[key]) : null) as never,
  )
  vi.spyOn(api, 'getAudioSource').mockResolvedValue(audioSource())
  vi.spyOn(api, 'getEpisode').mockResolvedValue(episodeDetail())
  addListener.mockResolvedValue({ remove })
  downloadFile.mockResolvedValue({ path: 'x' })
  getUri.mockResolvedValue({ uri: 'file:///Library/offline-audio/a.mp3' })
  stat.mockResolvedValue({ size: 4242, type: 'file', ctime: 0, mtime: 0, uri: 'file:///x' })
  deleteFile.mockResolvedValue(undefined)
  writeFile.mockResolvedValue({ uri: 'file:///t.json' })
  readFile.mockResolvedValue({ data: '{"segments":[{"text":"hi"}]}' })
  vi.spyOn(api, 'getSegments').mockResolvedValue({
    segments: [{ text: 'hi' }],
  } as unknown as Awaited<ReturnType<typeof api.getSegments>>)
})
afterEach(() => {
  vi.restoreAllMocks()
  vi.clearAllMocks()
})

describe('path helpers', () => {
  it('derives the extension from the URL and sanitises the slug', () => {
    expect(pathFor('p05-ee8e', 'https://x/y.m4a')).toBe('offline-audio/anon/p05-ee8e.m4a')
    expect(pathFor('a/../b', 'https://x/y.mp3')).toBe('offline-audio/anon/a_.._b.mp3')
  })
  it('ignores query strings and falls back per asset kind', () => {
    expect(pathFor('a', 'https://x/y.mp3?token=1')).toBe('offline-audio/anon/a.mp3')
    expect(pathFor('a', 'https://x/stream')).toBe('offline-audio/anon/a.mp3')
    expect(artworkPathFor('a', 'https://x/art')).toBe('offline-artwork/anon/a.jpg')
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
  it('refuses to run on the web, where Capacitor would write into IndexedDB', async () => {
    isNative.mockReturnValue(false)
    await expect(downloadEpisode('web1')).resolves.toBe(false)
    expect(downloadFile).not.toHaveBeenCalled()
  })

  it('downloads, records the file and the offline metadata', async () => {
    const store = useDownloadsStore()
    await expect(downloadEpisode('ok1')).resolves.toBe(true)

    expect(downloadFile).toHaveBeenCalledWith(
      expect.objectContaining({
        url: 'https://cdn.example.com/a.mp3',
        path: 'offline-audio/anon/ok1.mp3',
        directory: 'LIBRARY_NO_CLOUD',
        progress: true,
        recursive: true,
      }),
    )
    const e = store.entry('ok1')
    expect(e?.state).toBe('downloaded')
    expect(e?.uri).toBe('file:///Library/offline-audio/a.mp3')
    expect(e?.bytes).toBe(4242)
    expect(e?.path).toBe('offline-audio/anon/ok1.mp3')
    // Needed offline: the API is unreachable exactly when these are rendered.
    expect(e?.title).toBe('Ep One')
    expect(e?.showTitle).toBe('The Show')
    expect(e?.durationSeconds).toBe(416)
    expect(remove).toHaveBeenCalled()
  })

  it('caches the artwork alongside the audio', async () => {
    const store = useDownloadsStore()
    await downloadEpisode('art1')
    expect(downloadFile).toHaveBeenCalledWith(
      expect.objectContaining({ path: 'offline-artwork/anon/art1.jpg' }),
    )
    expect(store.entry('art1')?.artworkPath).toBe('offline-artwork/anon/art1.jpg')
  })

  it('still succeeds when the artwork cannot be fetched', async () => {
    downloadFile.mockImplementation(async (o: { path: string }) =>
      o.path.startsWith('offline-artwork') ? Promise.reject(new Error('403')) : { path: o.path },
    )
    const store = useDownloadsStore()
    await expect(downloadEpisode('art2')).resolves.toBe(true)
    expect(store.isDownloaded('art2')).toBe(true)
    expect(store.entry('art2')?.artworkPath).toBeUndefined()
  })

  it('survives an episode whose metadata cannot be fetched', async () => {
    vi.spyOn(api, 'getEpisode').mockRejectedValue(new Error('offline'))
    await expect(downloadEpisode('meta1')).resolves.toBe(true)
  })

  it('coalesces concurrent calls for the same slug onto one transfer', async () => {
    // Two native transfers truncate-writing one path corrupt the file, and the slice-3 drain
    // makes a tap landing beside a drain tick routine.
    const audioCalls = () =>
      downloadFile.mock.calls.filter((c) => (c[0] as { path: string }).path.endsWith('.mp3')).length
    const [a, b, c] = await Promise.all([
      downloadEpisode('dup1'),
      downloadEpisode('dup1'),
      downloadEpisode('dup1'),
    ])
    expect([a, b, c]).toEqual([true, true, true])
    expect(audioCalls()).toBe(1)
  })

  it('short-circuits an episode already on disk', async () => {
    const store = useDownloadsStore()
    await store.ensureLoaded()
    await store.setDownloaded('done1', 'file:///a.mp3', 1)
    await expect(downloadEpisode('done1')).resolves.toBe(true)
    expect(downloadFile).not.toHaveBeenCalled()
  })

  it('reports progress for its own transfer and ignores a sibling', async () => {
    const store = useDownloadsStore()
    await downloadEpisode('prog1')
    const onProgress = addListener.mock.calls[0][1] as (p: unknown) => void

    onProgress({ url: 'https://cdn.example.com/a.mp3', bytes: 50, contentLength: 200 })
    expect(store.progressOf('prog1')).toBe(0.25)

    onProgress({ url: 'https://cdn.example.com/OTHER.mp3', bytes: 200, contentLength: 200 })
    expect(store.progressOf('prog1')).toBe(0.25)
  })

  it('records a retryable failure instead of throwing at the call site', async () => {
    downloadFile.mockRejectedValue(new Error('socket reset'))
    const store = useDownloadsStore()
    await expect(downloadEpisode('fail1')).resolves.toBe(false)
    expect(store.stateOf('fail1')).toBe('failed')
    expect(store.entry('fail1')?.error).toBe('socket reset')
    expect(store.entry('fail1')?.errorKind).toBe('retryable')
  })

  it('marks a removed episode permanently failed so the drain stops retrying it', async () => {
    vi.spyOn(api, 'getAudioSource').mockRejectedValue(new ApiError(404, 'gone'))
    const store = useDownloadsStore()
    await expect(downloadEpisode('gone1')).resolves.toBe(false)
    expect(store.entry('gone1')?.errorKind).toBe('permanent')
  })

  it('deletes the orphan when the episode was deleted mid-transfer', async () => {
    const store = useDownloadsStore()
    // There is no abort, so a cancel drops the record while bytes keep arriving.
    downloadFile.mockImplementation(async () => {
      await deleteEpisode('cancel1')
      return { path: 'x' }
    })
    await expect(downloadEpisode('cancel1')).resolves.toBe(false)
    expect(store.entry('cancel1')).toBeNull()
    expect(deleteFile).toHaveBeenCalledWith({
      directory: 'LIBRARY_NO_CLOUD',
      path: 'offline-audio/anon/cancel1.mp3',
    })
  })

  it('does not stamp a stale error onto an entry the user re-created', async () => {
    const store = useDownloadsStore()
    downloadFile.mockImplementation(async () => {
      // Cancel, then immediately re-flag: the entry exists again, but it is a NEW one.
      await deleteEpisode('epoch1')
      await store.mark('epoch1')
      throw new Error('connection reset')
    })
    await expect(downloadEpisode('epoch1')).resolves.toBe(false)
    // The fresh entry must stay queued, not inherit the dead transfer's failure.
    expect(store.stateOf('epoch1')).toBe('queued')
    expect(store.entry('epoch1')?.error).toBeUndefined()
  })
})

describe('deleteEpisode', () => {
  it('drops the record, the audio, and the artwork', async () => {
    const store = useDownloadsStore()
    await downloadEpisode('del1')
    await deleteEpisode('del1')
    expect(store.entry('del1')).toBeNull()
    const paths = deleteFile.mock.calls.map((c) => (c[0] as { path: string }).path)
    expect(paths).toContain('offline-audio/anon/del1.mp3')
    expect(paths).toContain('offline-artwork/anon/del1.jpg')
  })

  it('is safe for an episode that was never downloaded', async () => {
    await expect(deleteEpisode('nope1')).resolves.toBeUndefined()
    expect(deleteFile).not.toHaveBeenCalled()
  })

  it('survives a filesystem that cannot unlink', async () => {
    const store = useDownloadsStore()
    await downloadEpisode('busy1')
    deleteFile.mockRejectedValue(new Error('EBUSY'))
    await expect(deleteEpisode('busy1')).resolves.toBeUndefined()
    expect(store.entry('busy1')).toBeNull()
  })
})

describe('localSourceFor', () => {
  it('returns a playable src for a downloaded episode', async () => {
    const store = useDownloadsStore()
    await store.ensureLoaded()
    await store.setDownloaded('src1', 'file:///a.mp3', 1)
    expect(localSourceFor('src1')).toBe('capacitor-file://file:///a.mp3')
  })

  it('returns null for anything not on disk, so the player streams', async () => {
    const store = useDownloadsStore()
    await store.ensureLoaded()
    await store.mark('src2')
    expect(localSourceFor('src2')).toBeNull()
    expect(localSourceFor('never-seen')).toBeNull()
  })

  it('returns null on web even if a record somehow exists', async () => {
    const store = useDownloadsStore()
    await store.ensureLoaded()
    await store.setDownloaded('src3', 'file:///a.mp3', 1)
    isNative.mockReturnValue(false)
    expect(localSourceFor('src3')).toBeNull()
  })
})

describe('refreshLocalUris', () => {
  it('re-derives the URI from the path, because iOS invalidates it on app update', async () => {
    const store = useDownloadsStore()
    await store.ensureLoaded()
    await store.setDownloaded('boot1', 'file:///OLD-CONTAINER/a.mp3', 10)
    store.setDownloading('boot1', 'offline-audio/anon/boot1.mp3')
    await store.setDownloaded('boot1', 'file:///OLD-CONTAINER/a.mp3', 10)
    getUri.mockResolvedValue({ uri: 'file:///NEW-CONTAINER/a.mp3' })

    await refreshLocalUris()
    expect(store.entry('boot1')?.uri).toBe('file:///NEW-CONTAINER/a.mp3')
    expect(store.entry('boot1')?.bytes).toBe(10)
  })

  it('drops a record whose file has vanished rather than handing the player a dead src', async () => {
    const store = useDownloadsStore()
    await store.ensureLoaded()
    store.setDownloading('gone2', 'offline-audio/anon/gone2.mp3')
    await store.setDownloaded('gone2', 'file:///a.mp3', 1)
    stat.mockRejectedValue(new Error('ENOENT'))

    await refreshLocalUris()
    expect(store.entry('gone2')).toBeNull()
  })
})

describe('transcripts', () => {
  it('caches the transcript beside the audio', async () => {
    const store = useDownloadsStore()
    await downloadEpisode('tr1')
    await vi.waitFor(() =>
      expect(store.entry('tr1')?.transcriptPath).toBe('offline-transcripts/anon/tr1.json'),
    )
    expect(writeFile).toHaveBeenCalledWith(
      expect.objectContaining({ path: 'offline-transcripts/anon/tr1.json', encoding: 'utf8' }),
    )
  })

  it('still succeeds when the transcript cannot be fetched', async () => {
    vi.spyOn(api, 'getSegments').mockRejectedValue(new Error('offline'))
    const store = useDownloadsStore()
    await expect(downloadEpisode('tr2')).resolves.toBe(true)
    expect(store.isDownloaded('tr2')).toBe(true)
  })

  it('reads the cached transcript back', async () => {
    const store = useDownloadsStore()
    await store.ensureLoaded()
    await store.setDownloaded('tr3', 'file:///a.mp3', 1)
    store.setTranscriptPath('tr3', transcriptPathFor('tr3'))
    await expect(localTranscriptFor('tr3')).resolves.toEqual({ segments: [{ text: 'hi' }] })
  })

  it('returns null when there is no cached transcript, so the API is used', async () => {
    const store = useDownloadsStore()
    await store.ensureLoaded()
    await store.setDownloaded('tr4', 'file:///a.mp3', 1)
    await expect(localTranscriptFor('tr4')).resolves.toBeNull()
  })

  it('deleteEpisode removes the transcript too', async () => {
    await downloadEpisode('tr5')
    await vi.waitFor(() => expect(writeFile).toHaveBeenCalled())
    await deleteEpisode('tr5')
    const paths = deleteFile.mock.calls.map((c) => (c[0] as { path: string }).path)
    expect(paths).toContain('offline-transcripts/anon/tr5.json')
  })
})

describe('storage cap (#1905)', () => {
  const big = (slug: string, finished: boolean) => {
    localPosition.mockImplementation((s: string) => (s === slug && finished ? { finished } : null))
  }

  it('reclaims a FINISHED episode to make room', async () => {
    const store = useDownloadsStore()
    await store.ensureLoaded()
    store.setDownloading('done', 'offline-audio/anon/done.mp3')
    await store.setDownloaded('done', 'file:///done.mp3', DOWNLOAD_CAP_BYTES)
    big('done', true)

    await expect(downloadEpisode('new1')).resolves.toBe(true)
    // The finished one went, and the new download proceeded.
    expect(store.entry('done')).toBeNull()
    expect(store.isDownloaded('new1')).toBe(true)
  })

  it('refuses rather than deleting something unplayed', async () => {
    const store = useDownloadsStore()
    await store.ensureLoaded()
    store.setDownloading('unplayed', 'offline-audio/anon/unplayed.mp3')
    await store.setDownloaded('unplayed', 'file:///u.mp3', DOWNLOAD_CAP_BYTES)
    // Nothing is finished, so there is nothing safe to reclaim.

    await expect(downloadEpisode('new2')).resolves.toBe(false)
    expect(store.isDownloaded('unplayed')).toBe(true)
    expect(store.entry('new2')?.errorKind).toBe('needs-space')
    expect(downloadFile).not.toHaveBeenCalled()
  })

  it('reclaimFinished stops as soon as it is back under the cap', async () => {
    const store = useDownloadsStore()
    await store.ensureLoaded()
    // Oldest first: the big one goes, and that alone is enough.
    store.setDownloading('old-big', 'offline-audio/anon/old-big.mp3')
    await store.setDownloaded('old-big', 'file:///a.mp3', DOWNLOAD_CAP_BYTES)
    store.setDownloading('new-small', 'offline-audio/anon/new-small.mp3')
    await store.setDownloaded('new-small', 'file:///b.mp3', 10)
    localPosition.mockReturnValue({ finished: true })

    await expect(reclaimFinished()).resolves.toBe(1)
    expect(store.entry('old-big')).toBeNull()
    // Finished too, but no longer needed — reclaiming is not a purge.
    expect(store.isDownloaded('new-small')).toBe(true)
  })

  it('leaves everything alone while there is room', async () => {
    const store = useDownloadsStore()
    await store.ensureLoaded()
    await store.setDownloaded('small', 'file:///s.mp3', 10)
    localPosition.mockReturnValue({ finished: true })
    await expect(reclaimFinished()).resolves.toBe(0)
    expect(store.isDownloaded('small')).toBe(true)
  })
})

describe('account switches mid-activity (#1905)', () => {
  it('a transfer started under one account never writes into another', async () => {
    const store = useDownloadsStore()
    await store.setNamespace('u_alice')
    // The switch lands while the bytes are still arriving.
    downloadFile.mockImplementation(async () => {
      await store.setNamespace('u_bob')
      return { path: 'x' }
    })

    await expect(downloadEpisode('shared')).resolves.toBe(false)
    // Bob must not end up with a record pointing into Alice's folder.
    expect(store.entry('shared')).toBeNull()
    await store.setNamespace('u_alice')
    expect(store.isDownloaded('shared')).toBe(false)
  })

  it('does not hand one account the other\'s in-flight transfer', async () => {
    const store = useDownloadsStore()
    await store.setNamespace('u_alice')
    // Every call gets its own resolver — releasing only the last would hang the first.
    const releases: Array<() => void> = []
    downloadFile.mockImplementation(
      () => new Promise((res) => releases.push(() => res({ path: 'x' }))),
    )
    const alice = downloadEpisode('same-slug')
    await vi.waitFor(() => expect(releases.length).toBe(1))

    await store.setNamespace('u_bob')
    // Same slug, different account: must be its own transfer, not a join onto Alice's.
    const bob = downloadEpisode('same-slug')
    expect(bob).not.toBe(alice)
    await vi.waitFor(() => expect(releases.length).toBe(2))

    releases.forEach((r) => r())
    await Promise.allSettled([alice, bob])
  })
})
