import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  ApiError,
  addInterest,
  getEpisode,
  getEpisodeStats,
  getMe,
  getMyStats,
  listEpisodes,
  listPodcastEpisodes,
  logListen,
  removeInterest,
} from './api'

function mockFetch(status: number, body: unknown): void {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => ({
      ok: status >= 200 && status < 300,
      status,
      json: async () => body,
    })),
  )
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('getMe', () => {
  it('returns the user on 200', async () => {
    mockFetch(200, { user_id: 'u_1', email: 'dev@localhost', name: 'Dev' })
    const me = await getMe()
    expect(me?.email).toBe('dev@localhost')
  })

  it('returns null on 401 (signed out)', async () => {
    mockFetch(401, { detail: 'Not authenticated.' })
    expect(await getMe()).toBeNull()
  })
})

describe('listEpisodes', () => {
  it('returns the page and requests /api/app/episodes with params', async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => ({ items: [], page: 2, page_size: 10, total: 0, has_more: false }),
    }))
    vi.stubGlobal('fetch', fetchMock)

    const page = await listEpisodes({ page: 2, pageSize: 10, status: 'ready', feedId: 'showa' })
    expect(page.page).toBe(2)
    const url = String(fetchMock.mock.calls[0][0])
    expect(url).toContain('/api/app/episodes')
    expect(url).toContain('page=2')
    expect(url).toContain('page_size=10')
    expect(url).toContain('status=ready')
    expect(url).toContain('feed_id=showa')
  })
})

describe('listPodcastEpisodes', () => {
  it('targets the feed-scoped path', async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => ({ items: [], page: 1, page_size: 20, total: 0, has_more: false }),
    }))
    vi.stubGlobal('fetch', fetchMock)
    await listPodcastEpisodes('show a/b')
    expect(String(fetchMock.mock.calls[0][0])).toContain('/api/app/podcasts/show%20a%2Fb/episodes')
  })
})

describe('getEpisode', () => {
  it('throws ApiError with status on non-2xx', async () => {
    mockFetch(404, { detail: 'Unknown episode slug.' })
    await expect(getEpisode('nope')).rejects.toMatchObject({ status: 404 })
    await expect(getEpisode('nope')).rejects.toBeInstanceOf(ApiError)
  })
})

describe('logListen', () => {
  it('POSTs to the per-episode listen endpoint (best-effort)', async () => {
    const fetchMock = vi.fn(async () => ({ ok: true, status: 200, json: async () => ({}) }))
    vi.stubGlobal('fetch', fetchMock)
    await logListen('show a/b')
    const [url, init] = fetchMock.mock.calls[0]
    expect(String(url)).toContain('/api/app/listen/show%20a%2Fb')
    expect(init).toMatchObject({ method: 'POST' })
  })

  it('never throws even when the request rejects (analytics is best-effort)', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new Error('network down')
      }),
    )
    await expect(logListen('ep')).resolves.toBeUndefined()
  })
})

describe('getMyStats', () => {
  it('GETs the user-stats endpoint and returns the stats', async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => ({
        episodes: 4, shows: 2, listening_seconds: 7200, active_days: 3, day_streak: 2, daily: [],
      }),
    }))
    vi.stubGlobal('fetch', fetchMock)
    const stats = await getMyStats()
    expect(stats?.episodes).toBe(4)
    expect(String(fetchMock.mock.calls[0][0])).toContain('/api/app/me/stats')
  })

  it('returns null when signed out (401)', async () => {
    mockFetch(401, { detail: 'Not authenticated.' })
    expect(await getMyStats()).toBeNull()
  })
})

describe('getEpisodeStats', () => {
  it('GETs the per-episode stats endpoint', async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => ({ slug: 'ep', listeners: 10, opens: 25, insights: 3, daily: [] }),
    }))
    vi.stubGlobal('fetch', fetchMock)
    const stats = await getEpisodeStats('ep')
    expect(stats.opens).toBe(25)
    expect(String(fetchMock.mock.calls[0][0])).toContain('/api/app/episodes/ep/stats')
  })
})

describe('addInterest / removeInterest', () => {
  it('addInterest POSTs to the token path and returns the items', async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => ({ items: ['topic:ai'] }),
    }))
    vi.stubGlobal('fetch', fetchMock)
    const items = await addInterest('topic:ai')
    expect(items).toEqual(['topic:ai'])
    const [url, init] = fetchMock.mock.calls[0]
    expect(String(url)).toContain('/api/app/interests/topic%3Aai')
    expect(init).toMatchObject({ method: 'POST' })
  })

  it('removeInterest DELETEs the token path and returns the remaining items', async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => ({ items: [] }),
    }))
    vi.stubGlobal('fetch', fetchMock)
    const items = await removeInterest('person:jane')
    expect(items).toEqual([])
    const [url, init] = fetchMock.mock.calls[0]
    expect(String(url)).toContain('/api/app/interests/person%3Ajane')
    expect(init).toMatchObject({ method: 'DELETE' })
  })
})
