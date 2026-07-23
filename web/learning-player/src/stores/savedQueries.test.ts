import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { MAX_SAVED_QUERIES, useSavedQueriesStore } from './savedQueries'
import { useUserPreferencesStore } from './userPreferences'

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
})
