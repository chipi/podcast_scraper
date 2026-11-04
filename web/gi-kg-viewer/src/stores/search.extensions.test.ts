import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { searchCorpus } from '../api/searchApi'
import { searchListeners, searchRequestParams } from '../extensions'
import { useSearchStore } from './search'

vi.mock('../api/searchApi', () => ({ searchCorpus: vi.fn() }))
vi.mock('../lib/analytics', () => ({ track: vi.fn() }))

describe('useSearchStore extension points', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.mocked(searchCorpus).mockReset()
  })
  afterEach(() => {
    searchRequestParams.reset()
    searchListeners.reset()
  })

  it('sends registered parameters and tells listeners about the run and the response', async () => {
    const seen: string[] = []
    searchRequestParams.register({ id: 'p', params: () => ({ extra_flag: 'true' }) })
    searchListeners.register({
      id: 'l',
      onClear: () => seen.push('clear'),
      onResponse: (body) => seen.push(`response:${String(body.extra_field)}`),
    })
    vi.mocked(searchCorpus).mockResolvedValue({ query: 'q', results: [], extra_field: 'x' } as never)
    const s = useSearchStore()
    s.query = 'q'
    await s.runSearch('/corpus')
    expect(vi.mocked(searchCorpus).mock.calls[0][1].extraParams).toEqual({ extra_flag: 'true' })
    expect(seen).toEqual(['clear', 'response:x'])
    s.clearResults()
    expect(seen).toEqual(['clear', 'response:x', 'clear'])
  })

  it('requestOptions carries the filters, a topK override and the registered parameters', () => {
    searchRequestParams.register({ id: 'p', params: () => ({ a: '1' }) })
    const s = useSearchStore()
    s.filters.topK = 10
    const o = s.requestOptions('/corpus', 30)
    expect(o).toMatchObject({ path: '/corpus', topK: 30, extraParams: { a: '1' } })
    expect(s.requestOptions('/corpus').topK).toBe(10)
  })
})
