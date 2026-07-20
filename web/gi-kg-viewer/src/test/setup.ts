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
