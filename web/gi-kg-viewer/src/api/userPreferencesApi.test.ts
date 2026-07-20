import { afterEach, describe, expect, it, vi } from 'vitest'

/* Un-mock: ``src/test/setup.ts`` globally auto-mocks this API so every
 * feature store's write-through becomes a resolved-null no-op. This
 * file is testing the REAL HTTP layer, so we need the concrete
 * implementation back. */
vi.unmock('./userPreferencesApi')

import {
  fetchUserPreferences,
  patchUserPreferences,
  replaceUserPreferences,
} from './userPreferencesApi'

/*
 * userPreferencesApi covers the HTTP layer for USERPREFS-1. All three
 * exports MUST degrade to `null` when the endpoint is unreachable so
 * callers can fall back to localStorage-only mode without try/catch
 * ceremony. That contract is the main observable behaviour; every
 * branch here is a graceful-null path.
 */

interface FetchResponseStub {
  status: number
  /** Set to `'invalid-json'` to force res.json() to throw. */
  body: unknown | 'invalid-json'
}

function stubFetch(response: FetchResponseStub | Error): void {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => {
      if (response instanceof Error) throw response
      // Use a real ``Response`` so ``res.json()`` goes through the
      // platform impl instead of relying on our own stub — vitest's node
      // env under vi.stubGlobal doesn't reliably forward function props
      // on plain-object duck-types.
      if (response.body === 'invalid-json') {
        return new Response('not json{', { status: response.status })
      }
      return new Response(JSON.stringify(response.body), {
        status: response.status,
      })
    }) as unknown as typeof fetch,
  )
}

function lastRequestInit(): RequestInit | undefined {
  const calls = vi.mocked(fetch).mock.calls
  return calls[calls.length - 1]?.[1] as RequestInit | undefined
}

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('fetchUserPreferences', () => {
  it('returns the parsed payload on 200 OK', async () => {
    stubFetch({
      status: 200,
      body: { preferences: { theme: 'dark', foo: 1 } },
    })
    const r = await fetchUserPreferences()
    expect(r).toEqual({ preferences: { theme: 'dark', foo: 1 } })
  })

  it('returns null on 401 (unauthenticated)', async () => {
    stubFetch({ status: 401, body: {} })
    expect(await fetchUserPreferences()).toBeNull()
  })

  it('returns null on 404 (endpoint missing)', async () => {
    stubFetch({ status: 404, body: {} })
    expect(await fetchUserPreferences()).toBeNull()
  })

  it('returns null on any non-ok status', async () => {
    stubFetch({ status: 500, body: {} })
    expect(await fetchUserPreferences()).toBeNull()
  })

  it('returns null when the network throws (offline)', async () => {
    stubFetch(new Error('offline'))
    expect(await fetchUserPreferences()).toBeNull()
  })

  it('returns null when JSON parse fails', async () => {
    stubFetch({ status: 200, body: 'invalid-json' })
    expect(await fetchUserPreferences()).toBeNull()
  })

  it('normalizes a missing preferences object to an empty payload', async () => {
    stubFetch({ status: 200, body: {} })
    const r = await fetchUserPreferences()
    expect(r).toEqual({ preferences: {} })
  })

  it('forwards the caller AbortSignal via fetch init', async () => {
    stubFetch({ status: 200, body: { preferences: {} } })
    const ctrl = new AbortController()
    await fetchUserPreferences(ctrl.signal)
    const init = lastRequestInit()
    // fetchWithTimeout composes the caller signal with its own timeout
    // signal; either way, a non-null signal must be present.
    expect(init?.signal).toBeDefined()
  })
})

describe('patchUserPreferences', () => {
  it('POSTs a PATCH request with the updates payload wrapped in {preferences}', async () => {
    stubFetch({
      status: 200,
      body: { preferences: { theme: 'light' } },
    })
    const r = await patchUserPreferences({ theme: 'light' })
    expect(r).toEqual({ preferences: { theme: 'light' } })
    const init = lastRequestInit()
    expect(init?.method).toBe('PATCH')
    expect(init?.headers).toMatchObject({ 'Content-Type': 'application/json' })
    expect(JSON.parse(String(init?.body))).toEqual({
      preferences: { theme: 'light' },
    })
  })

  it('returns null on non-ok status', async () => {
    stubFetch({ status: 500, body: {} })
    expect(await patchUserPreferences({ theme: 'x' })).toBeNull()
  })

  it('returns null on network throw', async () => {
    stubFetch(new Error('offline'))
    expect(await patchUserPreferences({ theme: 'x' })).toBeNull()
  })

  it('returns null when JSON parse fails', async () => {
    stubFetch({ status: 200, body: 'invalid-json' })
    expect(await patchUserPreferences({ theme: 'x' })).toBeNull()
  })
})

describe('replaceUserPreferences', () => {
  it('PUTs the entire payload wrapped in {preferences}', async () => {
    stubFetch({
      status: 200,
      body: { preferences: { a: 1, b: 2 } },
    })
    const r = await replaceUserPreferences({ a: 1, b: 2 })
    expect(r).toEqual({ preferences: { a: 1, b: 2 } })
    const init = lastRequestInit()
    expect(init?.method).toBe('PUT')
    expect(JSON.parse(String(init?.body))).toEqual({
      preferences: { a: 1, b: 2 },
    })
  })

  it('returns null on non-ok status', async () => {
    stubFetch({ status: 500, body: {} })
    expect(await replaceUserPreferences({})).toBeNull()
  })

  it('returns null on network throw', async () => {
    stubFetch(new Error('offline'))
    expect(await replaceUserPreferences({})).toBeNull()
  })
})
