import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('posthog-js', () => ({ default: { capture: vi.fn() } }))

const searchCorpusMock = vi.fn()
vi.mock('../api/searchApi', () => ({
  searchCorpus: (...args: unknown[]) => searchCorpusMock(...args),
}))

import { useSearchStore } from './search'

/**
 * Search v3 §S4b — ``search.runOperator`` covers the client half of the
 * server operator flow. The store overwrites the visible hit set with the
 * operator page (so the client can index cluster ``hit_indices`` back into
 * ``results``) and always over-fetches ``top_k × 3``.
 */
describe('useSearchStore.runOperator (Search v3 §S4b)', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    searchCorpusMock.mockReset()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('is a no-op when the query is empty', async () => {
    const s = useSearchStore()
    s.query = '   '
    await s.runOperator('/tmp/corpus', 'cluster')
    expect(searchCorpusMock).not.toHaveBeenCalled()
    expect(s.clusters).toBeNull()
    expect(s.operatorLoading).toBeNull()
  })

  it('is a no-op when the corpus path is empty', async () => {
    const s = useSearchStore()
    s.query = 'llm'
    await s.runOperator('  ', 'cluster')
    expect(searchCorpusMock).not.toHaveBeenCalled()
  })

  it('cluster: sends operator=cluster + top_k×3 and stores clusters back on the store', async () => {
    const clusters = [
      { cluster_id: 'tc:env', cluster_kind: 'topic_cluster', label: 'Env', size: 2, hit_indices: [0, 1] },
    ]
    searchCorpusMock.mockResolvedValueOnce({
      results: [{ doc_id: 'd:1' }, { doc_id: 'd:2' }],
      clusters,
    })
    const s = useSearchStore()
    s.query = 'climate'
    s.filters.topK = 10
    await s.runOperator('/tmp/corpus', 'cluster')
    expect(searchCorpusMock).toHaveBeenCalledTimes(1)
    const [, opts] = searchCorpusMock.mock.calls[0]
    expect(opts.operator).toBe('cluster')
    expect(opts.topK).toBe(30)
    expect(s.clusters).toEqual(clusters)
    expect(s.consensusPairs).toBeNull()
    expect(s.results).toHaveLength(2)
    expect(s.operatorLoading).toBeNull()
    expect(s.operatorError).toBeNull()
  })

  it('consensus: stores consensus_pairs on the store', async () => {
    const pairs = [
      {
        topic_id: 't:x',
        person_a_id: 'p:a',
        person_b_id: 'p:b',
        insight_a_id: 'i:1',
        insight_b_id: 'i:2',
        insight_a_text: '',
        insight_b_text: '',
        contradiction_score: 0.1,
      },
    ]
    searchCorpusMock.mockResolvedValueOnce({
      results: [{ doc_id: 'd:1' }],
      consensus_pairs: pairs,
    })
    const s = useSearchStore()
    s.query = 'climate'
    await s.runOperator('/tmp/corpus', 'consensus')
    expect(s.consensusPairs).toEqual(pairs)
    expect(s.clusters).toBeNull()
  })

  it('top_k×3 is capped at 100 so pathological topK values do not run away', async () => {
    searchCorpusMock.mockResolvedValueOnce({ results: [] })
    const s = useSearchStore()
    s.query = 'climate'
    s.filters.topK = 50
    await s.runOperator('/tmp/corpus', 'cluster')
    const [, opts] = searchCorpusMock.mock.calls[0]
    expect(opts.topK).toBe(100)
  })

  it('mapped server error surfaces via operatorError instead of raising', async () => {
    searchCorpusMock.mockResolvedValueOnce({
      results: [],
      error: 'no_index',
      detail: null,
    })
    const s = useSearchStore()
    s.query = 'x'
    await s.runOperator('/tmp/corpus', 'cluster')
    expect(s.operatorError).toContain('No vector index')
    // Clusters not populated on server error.
    expect(s.clusters).toBeNull()
  })

  it('network failure surfaces via operatorError; loading returns to null', async () => {
    searchCorpusMock.mockRejectedValueOnce(new Error('offline'))
    const s = useSearchStore()
    s.query = 'x'
    await s.runOperator('/tmp/corpus', 'cluster')
    expect(s.operatorError).toBe('offline')
    expect(s.operatorLoading).toBeNull()
  })

  it('clearResults wipes cluster + consensus state alongside results', async () => {
    searchCorpusMock.mockResolvedValueOnce({
      results: [{ doc_id: 'd:1' }],
      clusters: [
        { cluster_id: 'x', cluster_kind: 'topic', label: 'X', size: 1, hit_indices: [0] },
      ],
    })
    const s = useSearchStore()
    s.query = 'x'
    await s.runOperator('/tmp/corpus', 'cluster')
    expect(s.clusters).toHaveLength(1)
    s.clearResults()
    expect(s.clusters).toBeNull()
    expect(s.consensusPairs).toBeNull()
    expect(s.operatorError).toBeNull()
  })

  // #1259-4 followup — activeOperator is promoted to the store so
  // external surfaces (Cmd-K palette) can toggle the operator panel
  // visible without touching the ResultSetOperatorBar chip.
  it('activeOperator is null by default and round-trips arbitrary assignments', () => {
    const s = useSearchStore()
    expect(s.activeOperator).toBeNull()
    s.activeOperator = 'cluster'
    expect(s.activeOperator).toBe('cluster')
    s.activeOperator = 'compare'
    expect(s.activeOperator).toBe('compare')
    s.activeOperator = null
    expect(s.activeOperator).toBeNull()
  })
})
