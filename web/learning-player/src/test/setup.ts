/**
 * Global vitest setup — network isolation.
 *
 * happy-dom resolves a relative `/api/...` fetch against its default origin
 * (`http://localhost:3000`). Any component/store that fetches on mount without
 * the test mocking its api layer therefore opens a REAL socket → the
 * `connect ECONNREFUSED 127.0.0.1:3000` flood (~325/run) seen in `make
 * test-app`, plus stray `AbortError`s when the request is still in flight at
 * worker teardown. Unit tests must never touch the network: short-circuit EVERY
 * request to the happy-dom origin to an inert `503` so no unmocked endpoint —
 * present or future — can reach it. Absolute external URLs (none in unit tests)
 * still pass through. A test that needs a real/asserted response still overrides
 * `fetch` or mocks its api module locally (file-scoped, wins over this default).
 *
 * Mirrors `web/gi-kg-viewer/src/test/setup.ts`. Plain closure over the real
 * fetch — not `vi.fn` — so no spy-registry handle survives worker teardown.
 */
{
  const realFetch = globalThis.fetch
  const isHappyDomOrigin = (raw: string): boolean =>
    raw.startsWith('/') ||
    raw.startsWith('http://localhost:3000') ||
    raw.startsWith('http://127.0.0.1:3000') ||
    raw.startsWith('http://[::1]:3000')
  ;(globalThis as unknown as { fetch: typeof fetch }).fetch = ((
    input: RequestInfo | URL,
    init?: RequestInit,
  ) => {
    const raw =
      typeof input === 'string'
        ? input
        : input instanceof URL
          ? input.href
          : (input as Request).url
    if (isHappyDomOrigin(raw)) {
      return Promise.resolve(
        new Response('{}', {
          status: 503,
          headers: { 'content-type': 'application/json' },
        }),
      )
    }
    return realFetch(input as RequestInfo, init)
  }) as typeof fetch
}
