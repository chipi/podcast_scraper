import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('posthog-js', () => ({ default: { capture: vi.fn() } }))

const compareSubjectsMock = vi.fn()
const searchCorpusMock = vi.fn()
vi.mock('../api/searchApi', () => ({
  compareSubjects: (...args: unknown[]) => compareSubjectsMock(...args),
  searchCorpus: (...args: unknown[]) => searchCorpusMock(...args),
}))

import { useSearchStore } from './search'
import type { CompareSubjectRef } from '../api/searchApi'

/**
 * Search v3 §S8 — ``search.runCompare`` covers the client half of the
 * server compare flow. The store does NOT touch ``results`` /
 * ``clusters`` / ``consensusPairs`` — the compare panel renders from
 * ``compareResult`` only.
 */
describe('useSearchStore.runCompare (Search v3 §S8)', () => {
  const subjectA: CompareSubjectRef = { kind: 'person', id: 'Alice', label: 'Alice' }
  const subjectB: CompareSubjectRef = { kind: 'person', id: 'Bob', label: 'Bob' }

  beforeEach(() => {
    setActivePinia(createPinia())
    compareSubjectsMock.mockReset()
    searchCorpusMock.mockReset()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('is a no-op when the corpus path is empty', async () => {
    const s = useSearchStore()
    await s.runCompare('  ', subjectA, subjectB)
    expect(compareSubjectsMock).not.toHaveBeenCalled()
    expect(s.compareLoading).toBe(false)
  })

  it('is a no-op when either subject id is empty', async () => {
    const s = useSearchStore()
    await s.runCompare('/tmp/corpus', { kind: 'person', id: '  ' }, subjectB)
    expect(compareSubjectsMock).not.toHaveBeenCalled()
    await s.runCompare('/tmp/corpus', subjectA, { kind: 'person', id: '' })
    expect(compareSubjectsMock).not.toHaveBeenCalled()
  })

  it('stores the response payload and turns off loading on success', async () => {
    const response = {
      pack_a: {
        subject: subjectA,
        query: 'x',
        query_type: 'semantic',
        rendered: 'A rendered',
        token_count: 3,
        max_tokens: 2000,
        top_insight_id: 'insight:a1',
        top_insight_text: 'A',
        supporting_segment_ids: [],
        supporting_segment_texts: [],
        coverage_summary: { episode_count: 2 },
        confidence_p50: 0.8,
        result_count: 3,
        grounded: true,
      },
      pack_b: {
        subject: subjectB,
        query: 'x',
        query_type: 'semantic',
        rendered: 'B rendered',
        token_count: 3,
        max_tokens: 2000,
        top_insight_id: 'insight:b1',
        top_insight_text: 'B',
        supporting_segment_ids: [],
        supporting_segment_texts: [],
        coverage_summary: { episode_count: 1 },
        confidence_p50: 0.5,
        result_count: 1,
        grounded: true,
      },
      judge_summary: 'Alice shows higher confidence',
    }
    compareSubjectsMock.mockResolvedValueOnce(response)
    const s = useSearchStore()
    s.query = 'x'
    await s.runCompare('/tmp/corpus', subjectA, subjectB)
    expect(compareSubjectsMock).toHaveBeenCalledTimes(1)
    const [callA, callB, opts] = compareSubjectsMock.mock.calls[0]
    expect(callA).toEqual(subjectA)
    expect(callB).toEqual(subjectB)
    expect(opts.path).toBe('/tmp/corpus')
    expect(opts.q).toBe('x')
    expect(s.compareResult).toEqual(response)
    expect(s.compareLoading).toBe(false)
    expect(s.compareError).toBeNull()
  })

  it('surfaces a server error via compareError and clears compareResult', async () => {
    compareSubjectsMock.mockResolvedValueOnce({
      pack_a: { grounded: false },
      pack_b: { grounded: false },
      judge_summary: null,
      error: 'no_corpus_path',
    })
    const s = useSearchStore()
    s.query = 'x'
    await s.runCompare('/tmp/corpus', subjectA, subjectB)
    expect(s.compareResult).toBeNull()
    // no_corpus_path has no friendly mapping in mapSearchError; surface the raw code.
    expect(s.compareError).toBe('no_corpus_path')
  })

  it('surfaces a network / thrown error via compareError', async () => {
    compareSubjectsMock.mockRejectedValueOnce(new Error('boom'))
    const s = useSearchStore()
    s.query = 'x'
    await s.runCompare('/tmp/corpus', subjectA, subjectB)
    expect(s.compareError).toBe('boom')
    expect(s.compareResult).toBeNull()
    expect(s.compareLoading).toBe(false)
  })

  it('clearCompare wipes result + error and does not touch results/clusters', async () => {
    const s = useSearchStore()
    s.compareResult = { pack_a: {}, pack_b: {}, judge_summary: null } as never
    s.compareError = 'x'
    s.results = [{ doc_id: 'd:1', score: 0.5, metadata: {}, text: 'x' }] as never
    s.clearCompare()
    expect(s.compareResult).toBeNull()
    expect(s.compareError).toBeNull()
    expect(s.results).toHaveLength(1)
  })

  it('drops a stale in-flight response when a fresh runCompare fires', async () => {
    let resolveA: (value: unknown) => void = () => {}
    const inflightA = new Promise((r) => {
      resolveA = r
    })
    compareSubjectsMock.mockImplementationOnce(() => inflightA)
    const fresh = {
      pack_a: { grounded: true, subject: subjectA } as never,
      pack_b: { grounded: true, subject: subjectB } as never,
      judge_summary: 'fresh',
    }
    compareSubjectsMock.mockResolvedValueOnce(fresh)
    const s = useSearchStore()
    s.query = 'x'
    const firstCall = s.runCompare('/tmp/corpus', subjectA, subjectB)
    // Fire a second call before the first resolves — the store must ignore
    // the first response when it eventually arrives.
    await s.runCompare('/tmp/corpus', subjectA, subjectB)
    expect(s.compareResult?.judge_summary).toBe('fresh')
    resolveA({ pack_a: {}, pack_b: {}, judge_summary: 'stale' } as never)
    await firstCall
    // Stale response must not overwrite the fresh one.
    expect(s.compareResult?.judge_summary).toBe('fresh')
  })
})
