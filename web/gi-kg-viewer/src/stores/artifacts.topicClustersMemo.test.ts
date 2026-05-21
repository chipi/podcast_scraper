import { beforeEach, describe, expect, it, vi } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'

import { useArtifactsStore } from './artifacts'
import { fetchTopicClustersFromApi } from '../api/corpusTopicClustersApi'

vi.mock('../api/corpusTopicClustersApi', () => ({
  fetchTopicClustersFromApi: vi.fn(),
}))

/**
 * #769 — topic-clusters memoization contract.
 *
 * ``syncTopicClustersForCurrentCorpus`` is called at the top of every
 * ``ensureTopicClusterCompoundVisible`` invocation, which the App.vue
 * orchestrator can fire 2-3 times per first-open graph click. The HTTP
 * fetch is identical each time; memoize-per-root.
 *
 * Each test below pins one invalidation site or memoize behavior the
 * prior attempt regressed on.
 */
describe('useArtifactsStore — topic-clusters memoization (#769)', () => {
  const okResult = {
    status: 'ok' as const,
    document: { clusters: [] },
    schemaWarning: null,
  }

  beforeEach(() => {
    setActivePinia(createPinia())
    vi.mocked(fetchTopicClustersFromApi).mockReset()
    vi.mocked(fetchTopicClustersFromApi).mockResolvedValue(okResult)
  })

  it('caches a successful fetch — second call for same root skips HTTP', async () => {
    const store = useArtifactsStore()
    store.setCorpusPath('/c')

    await store.syncTopicClustersForCurrentCorpus()
    await store.syncTopicClustersForCurrentCorpus()
    await store.syncTopicClustersForCurrentCorpus()

    expect(fetchTopicClustersFromApi).toHaveBeenCalledTimes(1)
    expect(store.topicClustersDoc).toEqual({ clusters: [] })
  })

  it('invalidates on corpus path change — new root triggers a fresh HTTP fetch', async () => {
    const store = useArtifactsStore()
    store.setCorpusPath('/a')
    await store.syncTopicClustersForCurrentCorpus()
    expect(fetchTopicClustersFromApi).toHaveBeenCalledTimes(1)

    store.setCorpusPath('/b')
    await store.syncTopicClustersForCurrentCorpus()
    expect(fetchTopicClustersFromApi).toHaveBeenCalledTimes(2)
  })

  it('does NOT re-fetch when setCorpusPath is called with the same root', async () => {
    const store = useArtifactsStore()
    store.setCorpusPath('/c')
    await store.syncTopicClustersForCurrentCorpus()
    expect(fetchTopicClustersFromApi).toHaveBeenCalledTimes(1)

    // Same root again — no invalidation.
    store.setCorpusPath('/c')
    await store.syncTopicClustersForCurrentCorpus()
    expect(fetchTopicClustersFromApi).toHaveBeenCalledTimes(1)
  })

  it('invalidates on clearSelection — next sync fetches again', async () => {
    const store = useArtifactsStore()
    store.setCorpusPath('/c')
    await store.syncTopicClustersForCurrentCorpus()
    expect(fetchTopicClustersFromApi).toHaveBeenCalledTimes(1)

    store.clearSelection()
    expect(store.topicClustersDoc).toBeNull()

    await store.syncTopicClustersForCurrentCorpus()
    expect(fetchTopicClustersFromApi).toHaveBeenCalledTimes(2)
  })

  it('invalidates on missing/error responses — next sync retries', async () => {
    const store = useArtifactsStore()
    store.setCorpusPath('/c')

    // First call: server returns 'missing'.
    vi.mocked(fetchTopicClustersFromApi).mockResolvedValueOnce({
      status: 'missing' as const,
    })
    await store.syncTopicClustersForCurrentCorpus()
    expect(fetchTopicClustersFromApi).toHaveBeenCalledTimes(1)
    expect(store.topicClustersDoc).toBeNull()

    // Second call: server now returns 'ok'. Must re-fetch (not skipped).
    vi.mocked(fetchTopicClustersFromApi).mockResolvedValueOnce(okResult)
    await store.syncTopicClustersForCurrentCorpus()
    expect(fetchTopicClustersFromApi).toHaveBeenCalledTimes(2)
    expect(store.topicClustersDoc).toEqual({ clusters: [] })
  })

  it('invalidates when the doc has been nulled outside the API path', async () => {
    const store = useArtifactsStore()
    store.setCorpusPath('/c')
    await store.syncTopicClustersForCurrentCorpus()
    expect(fetchTopicClustersFromApi).toHaveBeenCalledTimes(1)

    // Simulate any path that nulls the doc without changing the sentinel
    // (defense-in-depth — the cache key requires BOTH root match AND
    // non-null doc to skip).
    store.$patch({ topicClustersDoc: null })

    await store.syncTopicClustersForCurrentCorpus()
    expect(fetchTopicClustersFromApi).toHaveBeenCalledTimes(2)
  })

  it('does not memoize fetch failures — next call retries', async () => {
    const store = useArtifactsStore()
    store.setCorpusPath('/c')

    vi.mocked(fetchTopicClustersFromApi).mockRejectedValueOnce(new Error('network'))
    await store.syncTopicClustersForCurrentCorpus()
    expect(fetchTopicClustersFromApi).toHaveBeenCalledTimes(1)
    expect(store.topicClustersDoc).toBeNull()
    expect(store.topicClustersLoadState).toBe('error')

    vi.mocked(fetchTopicClustersFromApi).mockResolvedValueOnce(okResult)
    await store.syncTopicClustersForCurrentCorpus()
    expect(fetchTopicClustersFromApi).toHaveBeenCalledTimes(2)
    expect(store.topicClustersDoc).toEqual({ clusters: [] })
  })

  it('empty/whitespace corpus path is a no-op (no HTTP, no state change)', async () => {
    const store = useArtifactsStore()
    store.setCorpusPath('   ')

    await store.syncTopicClustersForCurrentCorpus()
    expect(fetchTopicClustersFromApi).not.toHaveBeenCalled()
    expect(store.topicClustersDoc).toBeNull()
  })
})
