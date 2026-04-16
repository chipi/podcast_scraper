import { afterEach, describe, expect, it, vi } from 'vitest'
import { fetchWithTimeout, isAbortOrTimeout } from './httpClient'

describe('httpClient', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('isAbortOrTimeout recognizes DOMException AbortError', () => {
    expect(isAbortOrTimeout(new DOMException('aborted', 'AbortError'))).toBe(true)
  })

  it('isAbortOrTimeout recognizes Error AbortError', () => {
    const e = new Error('aborted')
    e.name = 'AbortError'
    expect(isAbortOrTimeout(e)).toBe(true)
  })

  it('fetchWithTimeout maps fetch AbortError to timeout message', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.reject(new DOMException('Aborted', 'AbortError'))) as typeof fetch,
    )
    await expect(fetchWithTimeout('/api/x')).rejects.toThrow(/Request timed out after \d+ms/)
    expect(vi.mocked(fetch)).toHaveBeenCalledTimes(1)
  })
})
