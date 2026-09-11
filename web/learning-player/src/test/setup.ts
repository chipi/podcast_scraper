/**
 * Global vitest setup — Web Storage normalization + network isolation.
 *
 * ## Web Storage (Node >= 24)
 *
 * Node 24 stabilized the Web Storage API and exposes `localStorage`/`sessionStorage` as built-in
 * globals. Launched without a valid `--localstorage-file`, that built-in is a non-persistent stub
 * that is missing `clear()`/`setItem()` — and because it is defined on `globalThis` it SHADOWS the
 * working Storage happy-dom installs per environment. The result is `localStorage.clear is not a
 * function` in every storage-backed test on a modern Node, while CI (pinned to an older Node) stays
 * green — a silent green-CI/red-local split. Replace both globals with an in-memory Storage so the
 * suite behaves identically across Node versions; the descriptor is configurable, so this sticks.
 *
 * ## Network isolation
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
  class MemoryStorage implements Storage {
    private m = new Map<string, string>()
    get length(): number {
      return this.m.size
    }
    key(i: number): string | null {
      return Array.from(this.m.keys())[i] ?? null
    }
    getItem(k: string): string | null {
      return this.m.has(k) ? (this.m.get(k) as string) : null
    }
    setItem(k: string, v: string): void {
      this.m.set(String(k), String(v))
    }
    removeItem(k: string): void {
      this.m.delete(k)
    }
    clear(): void {
      this.m.clear()
    }
  }
  for (const key of ['localStorage', 'sessionStorage'] as const) {
    const store = new MemoryStorage()
    Object.defineProperty(globalThis, key, { configurable: true, writable: true, value: store })
    if (typeof window !== 'undefined') {
      Object.defineProperty(window, key, { configurable: true, writable: true, value: store })
    }
  }
}

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
