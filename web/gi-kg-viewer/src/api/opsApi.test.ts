import { afterEach, describe, expect, it, vi } from 'vitest'

import { fetchOpsSummary } from './opsApi'

function mockFetch(ok: boolean, body: unknown, text = ''): void {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => ({
      ok,
      status: ok ? 200 : 500,
      text: async () => text,
      json: async () => body,
    })) as unknown as typeof fetch,
  )
}

describe('opsApi', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('GETs /api/ops/summary and returns the summary', async () => {
    const payload = {
      target: 'default',
      live: ['health'],
      unconfigured: ['cost'],
      failed: [],
      sources: { health: { ok: true, source: 'prod_api.health', data: { status: 'ok' } } },
    }
    mockFetch(true, payload)
    const result = await fetchOpsSummary()
    expect(result.live).toEqual(['health'])
    expect(fetch).toHaveBeenCalledWith(
      '/api/ops/summary',
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    )
  })

  it('throws the response text on a non-2xx', async () => {
    mockFetch(false, {}, 'boom')
    await expect(fetchOpsSummary()).rejects.toThrow('boom')
  })
})
