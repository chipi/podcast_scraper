import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError, getEpisode, getMe, listEpisodes, listPodcastEpisodes } from './api'

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
