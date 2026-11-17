import { afterEach, describe, expect, it, vi } from 'vitest'

import { fetchLlmGateway, fetchOpsSummary } from './opsApi'

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

  it('GETs /api/ops/llm-gateway and returns the per-key spend snapshot', async () => {
    const payload = {
      configured: true,
      reachable: true,
      keys: [
        { key_alias: 'proj-podcast-prod', spend_usd: 0.42, max_budget_usd: 25, burn_ratio: 0.0168 },
      ],
    }
    mockFetch(true, payload)
    const result = await fetchLlmGateway()
    expect(result.configured).toBe(true)
    expect(result.reachable).toBe(true)
    expect(result.keys[0].key_alias).toBe('proj-podcast-prod')
    expect(fetch).toHaveBeenCalledWith(
      '/api/ops/llm-gateway',
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    )
  })
})
