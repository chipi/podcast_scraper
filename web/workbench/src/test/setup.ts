/**
 * USERPREFS-1 — global vitest setup.
 *
 * Mocks the user-preferences HTTP wrappers so every feature store's
 * write-through to `/api/app/preferences` becomes an inert
 * resolved-null. Under happy-dom, an unmocked write hits
 * `http://localhost:3000/api/app/preferences` (happy-dom's default
 * location origin), gets ECONNREFUSED, and the dangling rejected
 * promise keeps the fork's event loop alive → vitest can't finalize
 * → coverage aggregation stalls after all tests have completed.
 *
 * Use plain `async () => null` factories, NOT
 * `vi.fn().mockResolvedValue(null)`. The vi.fn spy objects hold
 * internal call/invocation state that hooks into vitest's spy
 * registry; combined with a live pre-bound `Promise.resolve(null)`,
 * they retain handles across the worker teardown boundary, causing
 * happy-dom's AsyncTaskManager.abortAll to fire against still-pending
 * refs and stalling the worker indefinitely on Mac (git-bisected to
 * commit 50909e00 — the mock-with-vi.fn was introduced there and
 * silently regressed test-ui on macOS; Linux CI happened not to
 * expose it, which is why the pre-existing harden note flagged only
 * ECONNREFUSED symptoms).
 *
 * Individual tests remain free to `vi.mock('../api/userPreferencesApi',
 * ...)` locally with `vi.fn()` when they want to observe call
 * arguments — that's file-scoped and torn down per-file, so no
 * suite-wide leak.
 */
import { afterEach, vi } from 'vitest'

/* NETWORK-ISOLATION — unit tests must never open a real socket.
 *
 * happy-dom resolves a relative ``/api/...`` fetch against its default origin
 * (``http://localhost:3000``). Any component that fetches on mount without the
 * test mocking its api module therefore opens a REAL connection → ECONNREFUSED,
 * and ``httpClient``'s 120 s ``AbortSignal.timeout`` leaves an abort that fires
 * at worker teardown (``AbortError``) — which, under the ``forks`` pool +
 * coverage on macOS, stalls finalize (the "Timeout terminating forks worker"
 * seen in ``make test-ui``). USERPREFS-1 (below) plugged one endpoint
 * (``/api/app/preferences``); this guard generalises it: short-circuit EVERY
 * request to the happy-dom origin to an inert ``503`` so no unmocked endpoint —
 * present or future — can reach the network. Absolute external URLs (none in
 * unit tests) still pass through. A test that needs a real/asserted response
 * still overrides ``fetch`` or ``vi.mock``s its api module locally (file-scoped,
 * torn down per file, so it wins over this default).
 *
 * Plain closure over the real fetch — NOT ``vi.fn`` — so no spy-registry handle
 * survives the worker-teardown boundary (see the userPreferences note below for
 * why vi.fn here would re-introduce the very stall this removes). */
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

vi.mock('../api/userPreferencesApi', () => ({
  fetchUserPreferences: async () => null,
  patchUserPreferences: async () => null,
  replaceUserPreferences: async () => null,
}))

/* Close any live BroadcastChannel opened by ``useUserPreferencesStore``
 * between tests. See ``src/stores/userPreferences.ts`` for the registry
 * — real browsers close channels on tab unload; tests must drain them
 * or happy-dom's AsyncTaskManager teardown stalls the worker. This is
 * surgical (only touches the specific leak) instead of disposing the
 * whole Pinia scope, which would race any fire-and-forget
 * ``import().then(useStore)`` still in flight (see artifacts.ts:237). */
afterEach(() => {
  ;(
    globalThis as unknown as { __closeAllUserPreferencesChannels?: () => void }
  ).__closeAllUserPreferencesChannels?.()
})
