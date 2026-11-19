// @vitest-environment happy-dom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { localYmdDaysAgo } from '../utils/localCalendarDate'

/**
 * Stack-test (#666 follow-up): the production graph time-lens defaults
 * to "last 7 days" via ``seedFromCorpusLensIfNeeded``. Static-fixture
 * test corpora (publish dates from 2025-09) get filtered out by that
 * default and the auto-load picks 0 artifacts. ``VITE_DEFAULT_GRAPH_LENS_DAYS``
 * is the build-time override so stack-test images can opt into "all
 * time" without changing the production code default.
 *
 * These tests cover all four branches of the env-var resolver:
 *   - unset    → 7-day default (production)
 *   - ``"0"``  → empty string (= all time)
 *   - ``""``   → empty string (= all time)
 *   - ``"30"`` → ``localYmdDaysAgo(30)``
 *   - ``"abc"``→ falls back to 7-day default (defensive)
 */

const ENV_KEY = 'VITE_DEFAULT_GRAPH_LENS_DAYS'

beforeEach(() => {
  setActivePinia(createPinia())
  vi.resetModules()
  vi.unstubAllEnvs()
})

afterEach(() => {
  vi.unstubAllEnvs()
  vi.resetModules()
})

describe('seedFromCorpusLensIfNeeded — VITE_DEFAULT_GRAPH_LENS_DAYS resolution', () => {
  it('defaults to last 7 days when the env var is not set', async () => {
    vi.stubEnv(ENV_KEY, '')
    // ``vi.stubEnv("", "")`` sets it to empty string, so route to the
    // unset branch by stubbing back to undefined via reset (covered
    // separately below) — for "unset" semantics we go through the
    // explicit path here:
    vi.unstubAllEnvs()
    const { useGraphExplorerStore } = await import('./graphExplorer')
    const store = useGraphExplorerStore()
    store.seedFromCorpusLensIfNeeded()
    expect(store.sinceYmd).toBe(localYmdDaysAgo(7))
  })

  it('"0" → empty string (= all time, no lower bound)', async () => {
    vi.stubEnv(ENV_KEY, '0')
    const { useGraphExplorerStore } = await import('./graphExplorer')
    const store = useGraphExplorerStore()
    store.seedFromCorpusLensIfNeeded()
    expect(store.sinceYmd).toBe('')
  })

  it('empty string → empty string (= all time, explicit form)', async () => {
    vi.stubEnv(ENV_KEY, '')
    const { useGraphExplorerStore } = await import('./graphExplorer')
    const store = useGraphExplorerStore()
    store.seedFromCorpusLensIfNeeded()
    expect(store.sinceYmd).toBe('')
  })

  it('positive integer → localYmdDaysAgo(N)', async () => {
    vi.stubEnv(ENV_KEY, '30')
    const { useGraphExplorerStore } = await import('./graphExplorer')
    const store = useGraphExplorerStore()
    store.seedFromCorpusLensIfNeeded()
    expect(store.sinceYmd).toBe(localYmdDaysAgo(30))
  })

  it('non-numeric / unparseable → fall back to 7-day default', async () => {
    vi.stubEnv(ENV_KEY, 'abc')
    const { useGraphExplorerStore } = await import('./graphExplorer')
    const store = useGraphExplorerStore()
    store.seedFromCorpusLensIfNeeded()
    expect(store.sinceYmd).toBe(localYmdDaysAgo(7))
  })

  it('seeded flag prevents reseed once set', async () => {
    vi.stubEnv(ENV_KEY, '0')
    const { useGraphExplorerStore } = await import('./graphExplorer')
    const store = useGraphExplorerStore()
    store.seedFromCorpusLensIfNeeded()
    expect(store.sinceYmd).toBe('')
    // Mutate to a custom value, then re-seed — should be a no-op.
    store.setSinceYmd('2024-01-01')
    expect(store.sinceYmd).toBe('2024-01-01')
    store.seedFromCorpusLensIfNeeded()
    expect(store.sinceYmd).toBe('2024-01-01')
  })
})
