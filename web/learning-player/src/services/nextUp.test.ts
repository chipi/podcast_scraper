import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useDownloadsStore } from '../stores/downloads'
import * as api from './api'

const localSourceFor = vi.fn((_s: string) => null as string | null)
const localArtworkFor = vi.fn((_s: string) => null as string | null)
vi.mock('./downloads', () => ({
  localSourceFor: (s: string) => localSourceFor(s),
  localArtworkFor: (s: string) => localArtworkFor(s),
}))

const { resolveNextUpFor } = await import('./nextUp')

const NEXT = 'ep-next'
const nextAfter = () => NEXT

beforeEach(() => {
  setActivePinia(createPinia())
  localSourceFor.mockReturnValue(null)
  localArtworkFor.mockReturnValue(null)
})
afterEach(() => {
  vi.restoreAllMocks()
  vi.clearAllMocks()
})

describe('resolveNextUpFor (#1905/#1906)', () => {
  it('returns nothing when there is no current episode or no next', async () => {
    await expect(resolveNextUpFor(null, nextAfter)).resolves.toBeNull()
    await expect(resolveNextUpFor('ep-1', () => null)).resolves.toBeNull()
  })

  it('uses the origin url and API metadata when online', async () => {
    vi.spyOn(api, 'getAudioSource').mockResolvedValue({
      url: 'https://cdn/next.mp3',
    } as unknown as Awaited<ReturnType<typeof api.getAudioSource>>)
    vi.spyOn(api, 'getEpisode').mockResolvedValue({
      title: 'From the API',
      artwork_url: 'https://cdn/art.jpg',
      episode_image_url: null,
      feed_image_url: null,
    } as unknown as Awaited<ReturnType<typeof api.getEpisode>>)

    await expect(resolveNextUpFor('ep-1', nextAfter)).resolves.toMatchObject({
      slug: NEXT,
      url: 'https://cdn/next.mp3',
      title: 'From the API',
    })
  })

  it('advances OFFLINE to a downloaded episode, using the registry for metadata', async () => {
    // The whole point: auto-advance used to stop at the end of every episode with no network,
    // even with the entire queue on disk.
    vi.spyOn(api, 'getAudioSource').mockRejectedValue(new TypeError('Failed to fetch'))
    vi.spyOn(api, 'getEpisode').mockRejectedValue(new TypeError('Failed to fetch'))
    localSourceFor.mockReturnValue('capacitor-file:///next.mp3')
    localArtworkFor.mockReturnValue('capacitor-file:///next.jpg')

    const store = useDownloadsStore()
    await store.ensureLoaded()
    await store.setDownloaded(NEXT, 'file:///next.mp3', 10)
    store.setMetadata(NEXT, { title: 'From the registry', showTitle: 'Show' })

    const up = await resolveNextUpFor('ep-1', nextAfter)
    expect(up).toMatchObject({
      slug: NEXT,
      url: 'capacitor-file:///next.mp3',
      // Without a title here the lock screen keeps showing the PREVIOUS episode — auto-advance
      // runs with no view mounted, so nothing else supplies it.
      title: 'From the registry',
      artwork: 'capacitor-file:///next.jpg',
    })
  })

  it('stops rather than advancing to something it cannot play', async () => {
    vi.spyOn(api, 'getAudioSource').mockRejectedValue(new TypeError('Failed to fetch'))
    vi.spyOn(api, 'getEpisode').mockRejectedValue(new TypeError('Failed to fetch'))
    // Nothing downloaded, no origin url: advancing would load an empty src and stall silently.
    await expect(resolveNextUpFor('ep-1', nextAfter)).resolves.toBeNull()
  })
})
