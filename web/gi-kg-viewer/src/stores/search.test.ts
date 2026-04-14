import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { useSearchStore } from './search'

describe('useSearchStore', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('applyLibrarySearchHandoff normalizes feed and query', () => {
    const s = useSearchStore()
    s.applyLibrarySearchHandoff('  f1  ', 'alpha bravo')
    expect(s.filters.feed).toBe('f1')
    expect(s.query).toBe('alpha bravo')
    expect(s.feedFilterDisplayLabel).toBeNull()
    expect(s.feedFilterHandoffPristine).toBe(false)
  })

  it('applyLibrarySearchHandoff pairs catalog title with feed id for UI', () => {
    const s = useSearchStore()
    s.applyLibrarySearchHandoff('f1', 'q', { feedDisplayTitle: 'Mock Show' })
    expect(s.filters.feed).toBe('f1')
    expect(s.feedFilterDisplayLabel).toBe('Mock Show')
    expect(s.feedFilterHandoffPristine).toBe(true)
  })

  it('commitFeedFilterUiInput clears handoff title pairing', () => {
    const s = useSearchStore()
    s.applyLibrarySearchHandoff('f1', 'q', { feedDisplayTitle: 'Mock Show' })
    s.commitFeedFilterUiInput('x')
    expect(s.feedFilterHandoffPristine).toBe(false)
    expect(s.feedFilterDisplayLabel).toBeNull()
    expect(s.filters.feed).toBe('x')
  })

  it('applyLibrarySearchHandoff sets since from Digest ISO option', () => {
    const s = useSearchStore()
    s.filters.since = '1999-01-01'
    s.applyLibrarySearchHandoff('', 'topic q', { since: '2024-06-01T00:00:00Z' })
    expect(s.filters.since).toBe('2024-06-01')
  })

  it('clearResults clears validation error', async () => {
    const s = useSearchStore()
    await s.runSearch('/mock/corpus')
    expect(s.error).toBe('Enter a search query.')
    s.clearResults()
    expect(s.error).toBeNull()
  })

  it('runSearch stores lift_stats from API', async () => {
    const s = useSearchStore()
    s.query = 'climate'
    globalThis.fetch = vi.fn(async () => {
      return new Response(
        JSON.stringify({
          query: 'climate',
          results: [],
          lift_stats: { transcript_hits_returned: 2, lift_applied: 1 },
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      )
    }) as typeof fetch
    await s.runSearch('/mock/corpus')
    expect(s.liftStats).toEqual({
      transcript_hits_returned: 2,
      lift_applied: 1,
    })
  })
})
