import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { MAX_SAVED_QUERIES, useSavedQueriesStore } from './savedQueries'
import { useUserPreferencesStore } from './userPreferences'
import { useAuthStore } from './auth'

describe('useSavedQueriesStore (#1261-8)', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    // Stub the network — every set() writes locally but the PATCH silently
    // no-ops. Reads use the preferences store's in-memory ref anyway.
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ preferences: {} }), { status: 200 }),
    )
  })

  afterEach(() => vi.restoreAllMocks())

  it('starts empty when no prefs have been hydrated', () => {
    const s = useSavedQueriesStore()
    expect(s.list).toEqual([])
    expect(s.count).toBe(0)
  })

  it('save() pushes to the front and mirrors into userPreferences', async () => {
    const s = useSavedQueriesStore()
    const prefs = useUserPreferencesStore()
    await s.save('AI regulation', 'all', 1_000)
    expect(s.list).toEqual([{ q: 'AI regulation', scope: 'all', saved_at: 1_000 }])
    expect(prefs.get('lp.savedQueries')).toEqual(s.list)
  })

  it('save() dedupes case-insensitively by (query, scope) and lifts the entry to the front', async () => {
    const s = useSavedQueriesStore()
    await s.save('sleep science', 'all', 1)
    await s.save('memory research', 'all', 2)
    await s.save('SLEEP Science', 'all', 3) // dedupe by normalize, replace older
    expect(s.list.map((it) => it.q)).toEqual(['SLEEP Science', 'memory research'])
    expect(s.list[0].saved_at).toBe(3)
  })

  it('same query on different scopes counts as two entries', async () => {
    const s = useSavedQueriesStore()
    await s.save('AI', 'all', 1)
    await s.save('AI', 'mine', 2)
    expect(s.list).toHaveLength(2)
    expect(s.isSaved('AI', 'all')).toBe(true)
    expect(s.isSaved('AI', 'mine')).toBe(true)
  })

  it('caps the list at MAX_SAVED_QUERIES (oldest entries drop off)', async () => {
    const s = useSavedQueriesStore()
    for (let i = 0; i < MAX_SAVED_QUERIES + 5; i++) {
      await s.save(`q${i}`, 'all', i)
    }
    expect(s.list).toHaveLength(MAX_SAVED_QUERIES)
    // Most-recent first.
    expect(s.list[0].q).toBe(`q${MAX_SAVED_QUERIES + 4}`)
  })

  it('remove() drops one entry; missing entries are a no-op', async () => {
    const s = useSavedQueriesStore()
    await s.save('a', 'all', 1)
    await s.save('b', 'all', 2)
    await s.remove('a', 'all')
    expect(s.list.map((it) => it.q)).toEqual(['b'])
    await s.remove('never-saved', 'all') // no throw
    expect(s.list.map((it) => it.q)).toEqual(['b'])
  })

  it('clear() empties the list and writes an empty array to prefs', async () => {
    const s = useSavedQueriesStore()
    const prefs = useUserPreferencesStore()
    await s.save('a', 'all')
    await s.clear()
    expect(s.list).toEqual([])
    expect(prefs.get('lp.savedQueries')).toEqual([])
  })

  it('save() drops blank / whitespace-only queries silently', async () => {
    const s = useSavedQueriesStore()
    await s.save('   ', 'all')
    await s.save('', 'all')
    expect(s.list).toEqual([])
  })

  it('isSaved() is case-insensitive and scope-aware', async () => {
    const s = useSavedQueriesStore()
    await s.save('Sleep Science', 'all')
    expect(s.isSaved('sleep science', 'all')).toBe(true)
    expect(s.isSaved('sleep science', 'mine')).toBe(false)
    expect(s.isSaved('  SLEEP SCIENCE  ', 'all')).toBe(true)
  })

  /**
   * A prefs refresh landing MID-WRITE must not revert the write.
   *
   * `prefs.set` applies locally then PATCHes. A refresh resolving in that window carries a server
   * snapshot that predates our write; applying it dropped the query, `isSaved()` went false, and a
   * second tap re-SAVED instead of removing — the user saw "Saved ✓" refuse to toggle off. It only
   * reproduces when a refresh interleaves two taps, so it surfaced as a flaky e2e (the same spec
   * failed on desktop-chrome, then on mobile-chrome) rather than as a bug report.
   *
   * The stale snapshot is injected while the PATCH is still pending — exactly the window the bug
   * lived in. Without the echo-suppression guard this test fails on the final assertion.
   */
  it('a stale prefs refresh arriving mid-write does not revert the write (toggle-off works)', async () => {
    const s = useSavedQueriesStore()
    const prefs = useUserPreferencesStore()
    // `hydrate()` no-ops for a signed-out visitor, and preferences are per-account anyway.
    useAuthStore().$patch({ user: { user_id: 'u_test', email: 't@e2e.local' } as never })

    let releasePatch: (() => void) | undefined
    vi.spyOn(globalThis, 'fetch').mockImplementation((_url, init) => {
      // The write: hold it open so the refresh below lands INSIDE the write window.
      if ((init as RequestInit | undefined)?.method === 'PATCH') {
        return new Promise((resolve) => {
          releasePatch = () =>
            resolve(new Response(JSON.stringify({ preferences: {} }), { status: 200 }))
        })
      }
      // The refresh: the server has not seen the PATCH yet, so it returns the PRE-save list.
      return Promise.resolve(
        new Response(JSON.stringify({ preferences: { 'lp.savedQueries': [] } }), { status: 200 }),
      )
    })

    const saving = s.save('AI regulation', 'all', 1_000)
    await prefs.hydrate() // stale snapshot arrives mid-write
    releasePatch?.()
    await saving

    expect(s.isSaved('AI regulation', 'all')).toBe(true)

    // The tap that was broken: with the write reverted, this took the save branch and stayed saved.
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ preferences: {} }), { status: 200 }),
    )
    await s.remove('AI regulation', 'all')
    expect(s.isSaved('AI regulation', 'all')).toBe(false)
    expect(s.list).toEqual([])
  })

  /**
   * The mirror must be SYNCHRONOUS, and that is not a style preference.
   *
   * With Vue's default `flush: 'pre'` the watcher callback is queued and can run AFTER the write
   * has settled and `writingLocally` is back to false — a guard that reads as protecting the write
   * window while letting through the exact interleaving it exists for. My first version of this fix
   * shipped that way, and the guard test above passed against it.
   *
   * Asserted as a property rather than a race: outside any write, a prefs mutation must land in
   * `items` with NO await at all. `prefs.set` assigns `preferences.value` synchronously before it
   * PATCHes, so a sync watcher has already run by the next statement; a queued one has not.
   */
  it('mirrors a prefs change synchronously, with no tick in between', async () => {
    const s = useSavedQueriesStore()
    const prefs = useUserPreferencesStore()
    expect(s.list).toEqual([])

    // Deliberately NOT awaited: the assertion is about what is true before the microtask queue runs.
    void prefs.set('lp.savedQueries', [{ q: 'from elsewhere', scope: 'all', saved_at: 5 }])

    expect(
      s.list.map((it) => it.q),
      'the mirror must be flush:"sync" — a queued watcher has not run yet at this point',
    ).toEqual(['from elsewhere'])
  })

  /**
   * OVERLAPPING writes — why the guard counts rather than flags.
   *
   * The Save button has no pending state, so a second tap during the first PATCH is reachable, and
   * rapid Save/un-Save is exactly what the original e2e flake did. With a boolean, the FIRST PATCH
   * to resolve clears it while the second is still in flight, and a refresh landing in that window
   * reverts the write the user just made — the same bug, one interaction deeper.
   */
  it('a refresh during the SECOND of two overlapping writes still cannot revert', async () => {
    const s = useSavedQueriesStore()
    const prefs = useUserPreferencesStore()
    useAuthStore().$patch({ user: { user_id: 'u_test', email: 't@e2e.local' } as never })

    const release: Array<() => void> = []
    vi.spyOn(globalThis, 'fetch').mockImplementation((_url, init) => {
      if ((init as RequestInit | undefined)?.method === 'PATCH') {
        return new Promise((resolve) => {
          release.push(() =>
            resolve(new Response(JSON.stringify({ preferences: {} }), { status: 200 })),
          )
        })
      }
      return Promise.resolve(
        new Response(JSON.stringify({ preferences: { 'lp.savedQueries': [] } }), { status: 200 }),
      )
    })

    const first = s.save('one', 'all', 1)
    const second = s.save('two', 'all', 2)
    expect(release).toHaveLength(2)

    // The FIRST write settles while the second is still open — a boolean guard opens here.
    release[0]!()
    await first

    await prefs.hydrate() // stale snapshot, inside the second write's window

    release[1]!()
    await second

    expect(s.isSaved('one', 'all'), 'the first write must survive').toBe(true)
    expect(s.isSaved('two', 'all'), 'the second write must survive').toBe(true)
  })
})
