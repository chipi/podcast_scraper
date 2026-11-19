import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { searchCorpus, type SearchHit } from '../api/searchApi'
import { useSearchStore } from './search'

vi.mock('../api/searchApi', () => ({
  searchCorpus: vi.fn(),
}))

describe('useSearchStore runSearch overlapping requests', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.mocked(searchCorpus).mockReset()
  })

  it('last started search wins; stale completion does not clobber results or loading', async () => {
    const store = useSearchStore()
    store.query = 'climate'

    let releaseFirst!: () => void
    const firstGate = new Promise<void>((resolve) => {
      releaseFirst = resolve
    })

    let call = 0
    const hit = (text: string): SearchHit => ({
      doc_id: 'd',
      score: 0.5,
      metadata: {},
      text,
    })
    vi.mocked(searchCorpus).mockImplementation(async () => {
      call += 1
      if (call === 1) {
        await firstGate
      }
      return {
        query: 'climate',
        results: call === 1 ? [hit('old')] : [hit('new')],
        lift_stats: null,
      }
    })

    const p1 = store.runSearch('/corpus')
    await Promise.resolve()
    expect(store.loading).toBe(true)

    const p2 = store.runSearch('/corpus')
    await p2

    expect(store.loading).toBe(false)
    expect(store.results).toHaveLength(1)
    expect(store.results[0]?.text).toBe('new')

    releaseFirst()
    await p1

    expect(store.loading).toBe(false)
    expect(store.results[0]?.text).toBe('new')
  })
})
