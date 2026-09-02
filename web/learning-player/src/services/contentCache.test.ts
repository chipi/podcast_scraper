import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const isNative = vi.fn(() => false)
const readFile = vi.fn()
const writeFile = vi.fn()
const deleteFile = vi.fn()

vi.mock('./native', () => ({ isNative: () => isNative() }))
vi.mock('@capacitor/filesystem', () => ({
  Directory: { LibraryNoCloud: 'LIBRARY_NO_CLOUD' },
  Encoding: { UTF8: 'utf8' },
  Filesystem: {
    readFile: (...a: unknown[]) => readFile(...a),
    writeFile: (...a: unknown[]) => writeFile(...a),
    deleteFile: (...a: unknown[]) => deleteFile(...a),
  },
}))

import * as deviceStore from './deviceStore'
const { CACHE_KEYS, clearCached, readCached, setCacheNamespace, writeCached } =
  await import('./contentCache')

let disk: Record<string, unknown> = {}

beforeEach(() => {
  disk = {}
  isNative.mockReturnValue(false)
  setCacheNamespace('anon')
  vi.spyOn(deviceStore, 'setDeviceJson').mockImplementation(async (k, v) => {
    disk[k] = v
  })
  vi.spyOn(deviceStore, 'getDeviceJson').mockImplementation(async (k) => (disk[k] ?? null) as never)
  vi.spyOn(deviceStore, 'removeDeviceKey').mockImplementation(async (k) => {
    delete disk[k]
  })
})
afterEach(() => {
  vi.restoreAllMocks()
  vi.clearAllMocks()
})

describe('contentCache (#1909)', () => {
  it('round-trips a value', async () => {
    await writeCached('library', [{ feed_id: 'p05' }])
    await expect(readCached('library')).resolves.toEqual([{ feed_id: 'p05' }])
  })

  it('returns null for a miss rather than throwing', async () => {
    await expect(readCached('nothing')).resolves.toBeNull()
  })

  it('keeps accounts apart', async () => {
    setCacheNamespace('u_alice')
    await writeCached('library', ['alice'])
    setCacheNamespace('u_bob')
    // Bob must not read Alice's library — the whole reason this is not the service-worker cache.
    await expect(readCached('library')).resolves.toBeNull()
    setCacheNamespace('u_alice')
    await expect(readCached('library')).resolves.toEqual(['alice'])
  })

  it('clearCached drops an account\'s content', async () => {
    setCacheNamespace('u_alice')
    await writeCached('library', ['alice'])
    await writeCached('queue', ['a'])
    await clearCached(CACHE_KEYS)
    await expect(readCached('library')).resolves.toBeNull()
    await expect(readCached('queue')).resolves.toBeNull()
  })

  it('a failing write never rejects into the caller', async () => {
    vi.spyOn(deviceStore, 'setDeviceJson').mockRejectedValue(new Error('disk full'))
    // Failing to cache must not fail the read that produced the data.
    await expect(writeCached('library', [1])).resolves.toBeUndefined()
  })

  it('uses the filesystem on native, not device KV', async () => {
    isNative.mockReturnValue(true)
    writeFile.mockResolvedValue({ uri: 'file:///x' })
    readFile.mockResolvedValue({ data: '["from-disk"]' })
    await writeCached('library', ['x'])
    expect(writeFile).toHaveBeenCalledWith(
      expect.objectContaining({ path: 'content-cache/anon/library.json', recursive: true }),
    )
    await expect(readCached('library')).resolves.toEqual(['from-disk'])
    expect(deviceStore.setDeviceJson).not.toHaveBeenCalled()
  })
})
