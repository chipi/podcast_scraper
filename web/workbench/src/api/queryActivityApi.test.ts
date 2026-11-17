import { afterEach, describe, expect, it, vi } from 'vitest'
import { fetchQueryActivity } from './queryActivityApi'

function mockFetchJson(ok: boolean, body: unknown, text = '', status?: number): void {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => ({
      ok,
      status: status ?? (ok ? 200 : 500),
      text: async () => text,
      json: async () => body,
    })) as unknown as typeof fetch,
  )
}

describe('queryActivityApi', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('GETs query-activity with trimmed path + clamped days', async () => {
    const payload = { total: 2, buckets: [{ date: '2026-06-05', count: 2 }] }
    mockFetchJson(true, payload)
    await expect(fetchQueryActivity('  /c  ', 30)).resolves.toEqual(payload)
    expect(fetch).toHaveBeenCalledWith(
      '/api/corpus/query-activity?path=%2Fc&days=30',
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    )
  })

  it('clamps days into [1, 365]', async () => {
    mockFetchJson(true, { total: 0, buckets: [] })
    await fetchQueryActivity('/c', 9999)
    expect(fetch).toHaveBeenCalledWith(
      '/api/corpus/query-activity?path=%2Fc&days=365',
      expect.anything(),
    )
  })

  it('throws on non-ok', async () => {
    mockFetchJson(false, {}, 'boom', 500)
    await expect(fetchQueryActivity('/c')).rejects.toThrow('boom')
  })
})
