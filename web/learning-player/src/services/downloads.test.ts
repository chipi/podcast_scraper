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
const readdir = vi.fn()
const mkdir = vi.fn()
const rename = vi.fn()
const writeFile = vi.fn()
const readFile = vi.fn()
const isNative = vi.fn(() => true)
const localPosition = vi.fn((_s: string) => null as { finished: boolean } | null)

vi.mock('@capacitor/filesystem', () => ({
  Directory: {
    LibraryNoCloud: 'LIBRARY_NO_CLOUD',
    Documents: 'DOCUMENTS',
    Data: 'DATA',
    Cache: 'CACHE',
  },
  Encoding: { UTF8: 'utf8' },
  Filesystem: {
    addListener: (...a: unknown[]) => addListener(...a),
    downloadFile: (...a: unknown[]) => downloadFile(...a),
    getUri: (...a: unknown[]) => getUri(...a),
    stat: (...a: unknown[]) => stat(...a),
    deleteFile: (...a: unknown[]) => deleteFile(...a),
    readdir: (...a: unknown[]) => readdir(...a),
    mkdir: (...a: unknown[]) => mkdir(...a),
    rename: (...a: unknown[]) => rename(...a),
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
  DEFAULT_CAP_BYTES,
  absolutize,
  artworkPathFor,
  deleteEpisode,
  downloadEpisode,
  backfillKnowledge,
  localKnowledgeFor,
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
  readdir.mockResolvedValue({ files: [] })
  mkdir.mockResolvedValue(undefined)
  rename.mockResolvedValue(undefined)
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
    // Artwork is deliberately fire-and-forget (a missing cover must not fail a good download), so
    // this waits rather than assuming it has settled by the time downloadEpisode resolves.
    await vi.waitFor(() =>
      expect(store.entry('art1')?.artworkPath).toBe('offline-artwork/anon/art1.jpg'),
    )
  })

  /**
   * `nameFor` takes the extension from the URL and falls back to `jpg`, and the artwork endpoint
   * (`/api/app/artwork?ref=…`) has no extension — so every cover was written `.jpg` whatever the
   * server sent. iOS types a local file by its extension, so an SVG stored as `.jpg` arrives as
   * `image/jpeg`, fails to decode, and a fully downloaded episode shows a broken-image box.
   * Observed on the simulator: every stored cover was 2018 bytes of `<svg xmlns=…`.
   */
  describe('the stored cover is named after its BYTES, not its URL', () => {
    /** 18 bytes -> exactly the 24 base64 chars the sniffer reads. */
    const head = (magic: string) => btoa((magic + '\u0000'.repeat(18)).slice(0, 18))

    it('renames an SVG that arrived as .jpg', async () => {
      readFile.mockResolvedValue({ data: head('<svg xmlns="http://') })
      const store = useDownloadsStore()
      await downloadEpisode('svg1')
      await vi.waitFor(() =>
        expect(store.entry('svg1')?.artworkPath).toBe('offline-artwork/anon/svg1.svg'),
      )
      expect(rename).toHaveBeenCalledWith(
        expect.objectContaining({
          from: 'offline-artwork/anon/svg1.jpg',
          to: 'offline-artwork/anon/svg1.svg',
        }),
      )
    })

    it('renames a PNG that arrived as .jpg', async () => {
      readFile.mockResolvedValue({ data: head('\x89PNG\r\n\x1a\n') })
      const store = useDownloadsStore()
      await downloadEpisode('png1')
      await vi.waitFor(() =>
        expect(store.entry('png1')?.artworkPath).toBe('offline-artwork/anon/png1.png'),
      )
    })

    it('leaves a real JPEG alone — no pointless rename', async () => {
      readFile.mockResolvedValue({ data: head('\xff\xd8\xff\xe0JFIF') })
      const store = useDownloadsStore()
      await downloadEpisode('jpg1')
      await vi.waitFor(() =>
        expect(store.entry('jpg1')?.artworkPath).toBe('offline-artwork/anon/jpg1.jpg'),
      )
      expect(
        rename.mock.calls.some((c) => String((c[0] as { from?: string }).from).includes('jpg1')),
        'renamed a file that was already correct',
      ).toBe(false)
    })

    it('keeps the cover when the bytes cannot be read — wrong art beats no art', async () => {
      readFile.mockRejectedValue(new Error('unreadable'))
      const store = useDownloadsStore()
      await downloadEpisode('bad1')
      await vi.waitFor(() =>
        expect(store.entry('bad1')?.artworkPath).toBe('offline-artwork/anon/bad1.jpg'),
      )
    })
  })

  /**
   * A downloaded episode used to carry audio, a transcript and three display fields. Everything the
   * page is actually FOR — summary, insights, topics, people — came from the API, so on a plane it
   * was a player and a wall of transcript.
   */
  describe('the knowledge sidecar (#1905 follow-up)', () => {
    it('stores summary, insights, topics and people beside the audio', async () => {
      vi.spyOn(api, 'getInsights').mockResolvedValue({
        insights: [{ id: 'i1', text: 'An insight' }],
      } as never)
      vi.spyOn(api, 'getEntities').mockResolvedValue({
        topics: [{ id: 't1', label: 'AI' }],
        persons: [{ id: 'p1', label: 'Jane' }],
      } as never)
      const store = useDownloadsStore()
      await downloadEpisode('kn1')
      await vi.waitFor(() =>
        expect(store.entry('kn1')?.knowledgePath).toBe('offline-knowledge/anon/kn1.json'),
      )
      const write = writeFile.mock.calls.find((c) =>
        String((c[0] as { path: string }).path).startsWith('offline-knowledge'),
      )
      expect(write, 'nothing was written to the knowledge folder').toBeTruthy()
      const body = JSON.parse((write![0] as { data: string }).data)
      expect(body.insights).toHaveLength(1)
      expect(body.topics).toHaveLength(1)
      expect(body.persons).toHaveLength(1)
      expect(body.detail, 'the server detail — the summary lives here').toBeTruthy()
    })

    it('reads it back for the player', async () => {
      const store = useDownloadsStore()
      store.entries['kn2'] = {
        slug: 'kn2',
        state: 'downloaded',
        updatedAt: 1,
        knowledgePath: 'offline-knowledge/anon/kn2.json',
      } as never
      readFile.mockResolvedValue({
        data: JSON.stringify({ detail: { slug: 'kn2' }, insights: [1], topics: [], persons: [] }),
      })
      const k = await localKnowledgeFor('kn2')
      expect(k?.insights).toHaveLength(1)
    })

    it('an episode with no sidecar asks the API instead of pretending', async () => {
      const store = useDownloadsStore()
      store.entries['kn3'] = { slug: 'kn3', state: 'downloaded', updatedAt: 1 } as never
      expect(await localKnowledgeFor('kn3')).toBeNull()
    })

    it('keeps the entities when only the insights call fails', async () => {
      // Caught individually, not as one Promise.all: an episode with no insights yet is normal and
      // must not cost the topics and people as well. The outer try/catch already keeps the DOWNLOAD
      // alive, so that is not what the per-call catch is for.
      vi.spyOn(api, 'getInsights').mockRejectedValue(new Error('500'))
      vi.spyOn(api, 'getEntities').mockResolvedValue({
        topics: [{ id: 't1', label: 'AI' }],
        persons: [],
      } as never)
      const store = useDownloadsStore()
      await downloadEpisode('kn6')
      await vi.waitFor(() => expect(store.entry('kn6')?.knowledgePath).toBeTruthy())
      const write = writeFile.mock.calls.find((c) =>
        String((c[0] as { path: string }).path).startsWith('offline-knowledge'),
      )
      const body = JSON.parse((write![0] as { data: string }).data)
      expect(body.topics, 'one failed call took the others down with it').toHaveLength(1)
      expect(body.insights).toEqual([])
    })

    it('backfills episodes downloaded before the sidecar existed', async () => {
      // Otherwise the summary and insights stay missing until the user deletes and re-downloads,
      // which nobody will do and nobody should have to.
      const store = useDownloadsStore()
      store.entries['old1'] = { slug: 'old1', state: 'downloaded', updatedAt: 1 } as never
      store.entries['new1'] = {
        slug: 'new1',
        state: 'downloaded',
        updatedAt: 1,
        knowledgePath: 'offline-knowledge/anon/new1.json',
      } as never
      const getEp = vi.spyOn(api, 'getEpisode')
      await backfillKnowledge()
      expect(store.entry('old1')?.knowledgePath).toBe('offline-knowledge/anon/old1.json')
      const slugs = getEp.mock.calls.map((c) => c[0])
      expect(slugs, 'refetched an episode that already had its knowledge').not.toContain('new1')
    })

    it('skips episodes that are not downloaded', async () => {
      const store = useDownloadsStore()
      store.entries['q1'] = { slug: 'q1', state: 'queued', updatedAt: 1 } as never
      await backfillKnowledge()
      expect(store.entry('q1')?.knowledgePath).toBeUndefined()
    })

    it('writes nothing when the network is down, and leaves it for next launch', async () => {
      const store = useDownloadsStore()
      store.entries['old2'] = { slug: 'old2', state: 'downloaded', updatedAt: 1 } as never
      vi.spyOn(api, 'getEpisode').mockRejectedValue(new Error('offline'))
      vi.spyOn(api, 'getInsights').mockRejectedValue(new Error('offline'))
      vi.spyOn(api, 'getEntities').mockRejectedValue(new Error('offline'))
      await expect(backfillKnowledge()).resolves.toBeUndefined()
      expect(store.entry('old2')?.knowledgePath).toBeUndefined()
    })

    it('a missing sidecar never fails the audio download', async () => {
      vi.spyOn(api, 'getInsights').mockRejectedValue(new Error('500'))
      vi.spyOn(api, 'getEntities').mockRejectedValue(new Error('500'))
      await expect(downloadEpisode('kn4')).resolves.toBe(true)
      expect(useDownloadsStore().isDownloaded('kn4')).toBe(true)
    })

    it('is deleted with the episode, not left as an orphan', async () => {
      const store = useDownloadsStore()
      store.entries['kn5'] = {
        slug: 'kn5',
        state: 'downloaded',
        updatedAt: 1,
        path: 'offline-audio/anon/kn5.mp3',
        knowledgePath: 'offline-knowledge/anon/kn5.json',
      } as never
      await deleteEpisode('kn5')
      const paths = deleteFile.mock.calls.map((c) => (c[0] as { path: string }).path)
      expect(paths).toContain('offline-knowledge/anon/kn5.json')
    })
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
    await store.setDownloaded('done', 'file:///done.mp3', DEFAULT_CAP_BYTES)
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
    await store.setDownloaded('unplayed', 'file:///u.mp3', DEFAULT_CAP_BYTES)
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
    await store.setDownloaded('old-big', 'file:///a.mp3', DEFAULT_CAP_BYTES)
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

describe('orphan reconciliation (#1911)', () => {
  it('deletes files no record references, and keeps the ones that are referenced', async () => {
    const { reconcileDownloadFolders } = await import('./downloads')
    const store = useDownloadsStore()
    await store.ensureLoaded()
    store.setDownloading('keeper', 'offline-audio/anon/keeper.mp3')
    await store.setDownloaded('keeper', 'file:///k.mp3', 10)

    readdir.mockImplementation(async (o: { path: string }) =>
      o.path === 'offline-audio/anon'
        ? { files: [{ name: 'keeper.mp3' }, { name: 'orphan-from-a-failed-try.mp3' }] }
        : { files: [] },
    )

    // A failed transfer leaves bytes bytesOnDisk cannot see, so the cap silently under-counts.
    await expect(reconcileDownloadFolders()).resolves.toBe(1)
    const deleted = deleteFile.mock.calls.map((c) => (c[0] as { path: string }).path)
    expect(deleted).toContain('offline-audio/anon/orphan-from-a-failed-try.mp3')
    expect(deleted).not.toContain('offline-audio/anon/keeper.mp3')
  })

  it('is a no-op when nothing has ever been downloaded', async () => {
    const { reconcileDownloadFolders } = await import('./downloads')
    readdir.mockRejectedValue(new Error('folder does not exist'))
    await expect(reconcileDownloadFolders()).resolves.toBe(0)
  })
})

describe('size preflight (#1911)', () => {
  it('refuses an episode that cannot fit BEFORE starting the transfer', async () => {
    const { downloadEpisode: dl, DEFAULT_CAP_BYTES: cap } = await import('./downloads')
    vi.spyOn(api, 'getAudioSource').mockResolvedValue({
      episode_slug: 'big',
      url: 'https://cdn.example.com/big.mp3',
      content_length: cap + 1,
    } as unknown as Awaited<ReturnType<typeof api.getAudioSource>>)
    const store = useDownloadsStore()

    await expect(dl('big')).resolves.toBe(false)
    expect(store.entry('big')?.errorKind).toBe('needs-space')
    // The point of a preflight: nothing was transferred at all.
    expect(downloadFile).not.toHaveBeenCalled()
  })

  it('proceeds when the episode fits', async () => {
    const { downloadEpisode: dl } = await import('./downloads')
    vi.spyOn(api, 'getAudioSource').mockResolvedValue({
      episode_slug: 'small',
      url: 'https://cdn.example.com/small.mp3',
      content_length: 1024,
    } as unknown as Awaited<ReturnType<typeof api.getAudioSource>>)
    const store = useDownloadsStore()
    await expect(dl('small')).resolves.toBe(true)
    expect(store.isDownloaded('small')).toBe(true)
  })
})

/**
 * The URIs are repaired for the identity that owns the records, not for whoever happened to be
 * signed in at boot. A fresh install that signs in through the UI — or any account switch — used
 * to leave that account's downloads pointing at the PREVIOUS container UUID: the audio element
 * errored and the player swapped its transport for "audio unavailable", with the file sitting on
 * disk the whole time. The simulator tier caught it; nothing else can, because a stale container
 * UUID only exists on a device.
 */
describe('refreshLocalUris across an account switch', () => {
  it('stops rather than writing one account URIs into another registry', async () => {
    const store = useDownloadsStore()
    await store.setNamespace('u_a')
    await store.mark('ep-1')
    await store.setDownloaded('ep-1', 'file:///old/ep-1.mp3', 10)
    store.entries['ep-1'].path = 'offline-audio/u_a/ep-1.mp3'

    // The switch lands while the loop is between entries.
    getUri.mockImplementation(async () => {
      await store.setNamespace('u_b')
      return { uri: 'file:///new/ep-1.mp3' }
    })
    stat.mockResolvedValue({ size: 10 })
    await refreshLocalUris()

    expect(store.namespace).toBe('u_b')
    expect(store.entries['ep-1']).toBeUndefined()
    await store.setNamespace('u_a')
    expect(store.entries['ep-1']?.uri).toBe('file:///old/ep-1.mp3')
  })
})

/**
 * The first download of an account on a device (#1925, decision 4).
 *
 * `downloadFile` gets `recursive: true`, and with `CapacitorHttp` enabled that is not enough — the
 * HTTP plugin serves the transfer and does not honour it, so the call RESOLVES having written
 * nothing and the following `stat` fails. The user sees "Download failed — tap to retry", and the
 * retry fails identically for ever.
 *
 * Every earlier device test seeded a file into the folder first, so the folder always existed by
 * the time anything was measured. Downloading through the UI is what exposed it.
 */
/**
 * Where the bytes actually land (#1925, decision 4).
 *
 * `@capacitor/filesystem` 8.1.2 ignores `directory` in `downloadFile` on iOS — the file appears
 * under `Documents` at the same relative path, while every other call here uses `LibraryNoCloud`.
 * That made EVERY download fail (the following `stat` finds nothing, and the entry is recorded as
 * retryable, so the retry fails identically for ever) and left the audio somewhere iOS backs up to
 * iCloud, which is the exact thing `LibraryNoCloud` was chosen to prevent.
 */
describe('a downloaded file is settled into LibraryNoCloud', () => {
  it('rescues a file the plugin dropped in Documents', async () => {
    const store = useDownloadsStore()
    await store.setNamespace('u_a')
    await store.mark('ep-1')
    // The observed iOS behaviour: nothing at the asked-for location, on the first look only.
    stat.mockRejectedValueOnce(new Error("'stat' failed because file ... does not exist"))

    await downloadEpisode('ep-1')

    expect(rename).toHaveBeenCalledWith(
      expect.objectContaining({
        directory: 'DOCUMENTS',
        toDirectory: 'LIBRARY_NO_CLOUD',
        from: 'offline-audio/u_a/ep-1.mp3',
        to: 'offline-audio/u_a/ep-1.mp3',
      }),
    )
    expect(store.stateOf('ep-1')).toBe('downloaded')
  })

  it('does nothing when the plugin already honoured the directory', async () => {
    // So a fixed plugin (or a non-iOS platform) turns this into a no-op rather than a breakage.
    const store = useDownloadsStore()
    await store.setNamespace('u_a')
    await store.mark('ep-1')

    await downloadEpisode('ep-1')

    expect(rename).not.toHaveBeenCalled()
    expect(store.stateOf('ep-1')).toBe('downloaded')
  })

  it('reports a failure when the file is in NEITHER place', async () => {
    const store = useDownloadsStore()
    await store.setNamespace('u_a')
    await store.mark('ep-1')
    stat.mockRejectedValue(new Error('does not exist'))
    rename.mockRejectedValue(new Error('no such file'))

    await downloadEpisode('ep-1')
    expect(store.stateOf('ep-1')).toBe('failed')
  })
})
