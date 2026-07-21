// @vitest-environment happy-dom
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('posthog-js', () => ({ default: { capture: vi.fn() } }))

import { useSavedQueriesStore } from './savedQueries'
import { useUserPreferencesStore } from './userPreferences'

/**
 * Search v3 §S7 — Saved + Recent writer contract. Every test resets a
 * fresh Pinia + stubs the network layer to keep the store isolated from
 * the real /api/app/preferences endpoint. The wrapper is thin — most of
 * the behaviour under test is dedupe, ring-buffer capping, and shape.
 */
describe('useSavedQueriesStore (Search v3 §S7)', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    // ``userPrefs.set`` fires a network PATCH via patchUserPreferences;
    // stub that off so tests don't need a network. The store's mirror
    // updates synchronously, which is what our reads depend on.
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: true,
        json: async () => ({ preferences: {} }),
      })) as unknown as typeof fetch,
    )
  })

  it('readers return empty arrays when no writes have happened yet', () => {
    const store = useSavedQueriesStore()
    expect(store.listSaved()).toEqual([])
    expect(store.listRecent()).toEqual([])
  })

  it('pushRecent adds a new entry with a timestamp; empty string is a no-op', async () => {
    const store = useSavedQueriesStore()
    await store.pushRecent('')
    await store.pushRecent('   ')
    expect(store.listRecent()).toEqual([])
    await store.pushRecent('llm strategy')
    const recent = store.listRecent()
    expect(recent).toHaveLength(1)
    expect(recent[0].q).toBe('llm strategy')
    expect(typeof recent[0].ts).toBe('number')
  })

  it('pushRecent de-dupes on q — repeat moves the entry to the front', async () => {
    const store = useSavedQueriesStore()
    await store.pushRecent('one')
    await store.pushRecent('two')
    await store.pushRecent('one') // re-push
    const recent = store.listRecent()
    expect(recent.map((e) => e.q)).toEqual(['one', 'two'])
  })

  it('pushRecent trims whitespace before storing (dedupe uses trimmed value)', async () => {
    const store = useSavedQueriesStore()
    await store.pushRecent('  climate  ')
    await store.pushRecent('climate')
    const recent = store.listRecent()
    expect(recent).toHaveLength(1)
    expect(recent[0].q).toBe('climate')
  })

  it('pushRecent caps the ring buffer at RECENT_MAX = 20 (newest first)', async () => {
    const store = useSavedQueriesStore()
    for (let i = 0; i < 25; i++) {
      await store.pushRecent(`q-${i}`)
    }
    const recent = store.listRecent()
    expect(recent).toHaveLength(20)
    // Newest first: 24 → 5 (25 - 20 dropped).
    expect(recent[0].q).toBe('q-24')
    expect(recent[recent.length - 1].q).toBe('q-5')
  })

  it('listRecent with a limit slice returns the first N entries', async () => {
    const store = useSavedQueriesStore()
    for (let i = 0; i < 8; i++) await store.pushRecent(`q-${i}`)
    expect(store.listRecent(3).map((e) => e.q)).toEqual(['q-7', 'q-6', 'q-5'])
    expect(store.listRecent(0)).toEqual([])
  })

  it('saveQuery creates a new entry with id + label + ts', async () => {
    const store = useSavedQueriesStore()
    const entry = await store.saveQuery('cell therapy', 'Cell Therapy Landscape')
    expect(entry).not.toBeNull()
    expect(entry!.q).toBe('cell therapy')
    expect(entry!.label).toBe('Cell Therapy Landscape')
    expect(entry!.id).toMatch(/^sq-/)
    expect(typeof entry!.ts).toBe('number')
    expect(store.listSaved()).toHaveLength(1)
    expect(store.listSaved()[0].id).toBe(entry!.id)
  })

  it('saveQuery is idempotent on q — second save updates label + moves to front', async () => {
    const store = useSavedQueriesStore()
    await store.saveQuery('a')
    await store.saveQuery('b')
    const secondSaveOfA = await store.saveQuery('a', 'A rename')
    expect(store.listSaved()).toHaveLength(2)
    expect(store.listSaved()[0].q).toBe('a')
    expect(store.listSaved()[0].label).toBe('A rename')
    expect(store.listSaved()[0].id).toBe(secondSaveOfA!.id)
    expect(store.listSaved()[1].q).toBe('b')
  })

  it('saveQuery on empty / whitespace-only q is a no-op returning null', async () => {
    const store = useSavedQueriesStore()
    expect(await store.saveQuery('')).toBeNull()
    expect(await store.saveQuery('   ')).toBeNull()
    expect(store.listSaved()).toEqual([])
  })

  it('isSaved reflects the trimmed q lookup', async () => {
    const store = useSavedQueriesStore()
    await store.saveQuery('climate')
    expect(store.isSaved('climate')).toBe(true)
    expect(store.isSaved('  climate  ')).toBe(true) // trims
    expect(store.isSaved('unsaved')).toBe(false)
    expect(store.isSaved('')).toBe(false)
  })

  it('removeSaved drops the entry by id; no-op when id missing', async () => {
    const store = useSavedQueriesStore()
    const a = await store.saveQuery('a')
    await store.saveQuery('b')
    await store.removeSaved(a!.id)
    expect(store.listSaved().map((e) => e.q)).toEqual(['b'])
    await store.removeSaved('nonexistent') // no throw
    expect(store.listSaved()).toHaveLength(1)
  })

  it('clearRecent wipes the ring buffer', async () => {
    const store = useSavedQueriesStore()
    await store.pushRecent('a')
    await store.pushRecent('b')
    await store.clearRecent()
    expect(store.listRecent()).toEqual([])
  })

  it('resetAll wipes both Saved and Recent', async () => {
    const store = useSavedQueriesStore()
    await store.saveQuery('a')
    await store.pushRecent('b')
    await store.resetAll()
    expect(store.listSaved()).toEqual([])
    expect(store.listRecent()).toEqual([])
  })

  it('writes are visible through useUserPreferencesStore.get (shared mirror)', async () => {
    const store = useSavedQueriesStore()
    await store.saveQuery('climate')
    await store.pushRecent('recent-1')
    const prefs = useUserPreferencesStore()
    const saved = prefs.get<{ q: string }[]>('search.savedQueries')
    const recent = prefs.get<{ q: string }[]>('search.recentQueries')
    expect(saved).toBeDefined()
    expect(saved![0].q).toBe('climate')
    expect(recent).toBeDefined()
    expect(recent![0].q).toBe('recent-1')
  })
})
