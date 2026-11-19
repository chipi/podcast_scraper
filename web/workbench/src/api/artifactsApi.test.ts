import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  ARTIFACT_FETCH_TIMEOUT_MS,
  fetchArtifactJson,
} from './artifactsApi'
import { DEFAULT_VIEWER_FETCH_TIMEOUT_MS } from './httpClient'

function mockFetchJson(
  ok: boolean,
  body: unknown,
  text = '',
  status?: number,
): void {
  const st = status ?? (ok ? 200 : 500)
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => ({
      ok,
      status: st,
      text: async () => text,
      json: async () => body,
    })) as unknown as typeof fetch,
  )
}

function lastUrl(): string {
  const calls = vi.mocked(fetch).mock.calls
  return String(calls[calls.length - 1]?.[0])
}

describe('artifactsApi', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('exposes the artifact timeout aliased to the shared default', () => {
    expect(ARTIFACT_FETCH_TIMEOUT_MS).toBe(DEFAULT_VIEWER_FETCH_TIMEOUT_MS)
  })

  it('GETs /api/artifacts/<path> with the trimmed corpus path query', async () => {
    const payload = { foo: 'bar' }
    mockFetchJson(true, payload)
    await expect(
      fetchArtifactJson('  /corpus  ', 'episodes/ep1.json'),
    ).resolves.toEqual(payload)
    const url = lastUrl()
    expect(url).toContain('/api/artifacts/episodes/ep1.json?')
    expect(url).toContain('path=%2Fcorpus')
    expect(vi.mocked(fetch)).toHaveBeenCalledTimes(1)
  })

  it('encodes each path segment but preserves slash separators', async () => {
    mockFetchJson(true, {})
    await fetchArtifactJson('/c', 'a b/c+d/e?f.json')
    const url = lastUrl()
    const pathPart = url.slice(
      '/api/artifacts/'.length,
      url.indexOf('?'),
    )
    // Spaces, plus and question marks are percent-encoded per segment...
    expect(pathPart).toBe('a%20b/c%2Bd/e%3Ff.json')
    // ...while the segment-joining slashes remain literal.
    expect(pathPart.split('/')).toHaveLength(3)
  })

  it('encodes a single (no-slash) relative path', async () => {
    mockFetchJson(true, {})
    await fetchArtifactJson('/c', 'plain file.json')
    const url = lastUrl()
    expect(url).toContain('/api/artifacts/plain%20file.json?')
  })

  it('passes an AbortSignal and the artifact timeout via fetchWithTimeout', async () => {
    mockFetchJson(true, {})
    await fetchArtifactJson('/c', 'ep.json')
    expect(fetch).toHaveBeenCalledWith(
      expect.stringContaining('/api/artifacts/ep.json'),
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    )
  })

  it('throws the response text on a non-ok status', async () => {
    mockFetchJson(false, {}, 'missing artifact', 404)
    await expect(fetchArtifactJson('/c', 'gone.json')).rejects.toThrow(
      'missing artifact',
    )
  })

  it('throws "HTTP <status>" when the error body is empty', async () => {
    mockFetchJson(false, {}, '', 500)
    await expect(fetchArtifactJson('/c', 'x.json')).rejects.toThrow('HTTP 500')
  })

  it('propagates a network error from fetch', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new Error('network down')
      }) as unknown as typeof fetch,
    )
    await expect(fetchArtifactJson('/c', 'x.json')).rejects.toThrow('network down')
  })

  it('propagates malformed JSON on a 200 response', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: true,
        status: 200,
        text: async () => '',
        json: async () => {
          throw new SyntaxError('Unexpected token')
        },
      })) as unknown as typeof fetch,
    )
    await expect(fetchArtifactJson('/c', 'x.json')).rejects.toThrow(
      /Unexpected token/,
    )
  })
})
