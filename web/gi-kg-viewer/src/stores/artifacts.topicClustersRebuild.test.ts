import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'

import { useArtifactsStore } from './artifacts'
import {
  fetchTopicClustersFromApi,
  postTopicClustersRebuild,
} from '../api/corpusTopicClustersApi'

vi.mock('../api/corpusTopicClustersApi', () => ({
  fetchTopicClustersFromApi: vi.fn(),
  postTopicClustersRebuild: vi.fn(),
}))

/**
 * task-#14 — operator-facing topic-clusters rebuild.
 *
 * ``rebuildTopicClusters`` replaces the CLI/SSH-only build with a button: POST the
 * rebuild, then poll the reader (clearing the #769 memo each tick) until the clusters
 * appear so the dashboard card flips to "Loaded" on its own.
 */
describe('useArtifactsStore — rebuildTopicClusters (task-#14)', () => {
  const okResult = { status: 'ok' as const, document: { clusters: [] }, schemaWarning: null }

  beforeEach(() => {
    setActivePinia(createPinia())
    vi.useFakeTimers()
    vi.mocked(fetchTopicClustersFromApi).mockReset()
    vi.mocked(postTopicClustersRebuild).mockReset()
    vi.mocked(postTopicClustersRebuild).mockResolvedValue({
      accepted: true,
      corpus_path: '/c',
    })
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('POSTs the rebuild with the trimmed root, then polls until Loaded', async () => {
    // First poll still missing, second poll ok — proves it keeps polling, not one-shot.
    vi.mocked(fetchTopicClustersFromApi)
      .mockResolvedValueOnce({ status: 'missing' as const })
      .mockResolvedValue(okResult)

    const store = useArtifactsStore()
    store.setCorpusPath('  /c  ')

    const p = store.rebuildTopicClusters()
    expect(store.topicClustersRebuilding).toBe(true)
    await vi.advanceTimersByTimeAsync(3200) // two 1500ms poll ticks
    await p

    expect(postTopicClustersRebuild).toHaveBeenCalledExactlyOnceWith('/c')
    expect(store.topicClustersLoadState).toBe('ok')
    expect(store.topicClustersRebuilding).toBe(false)
  })

  it('is a no-op when the corpus path is empty', async () => {
    const store = useArtifactsStore()
    store.setCorpusPath('   ')

    await store.rebuildTopicClusters()

    expect(postTopicClustersRebuild).not.toHaveBeenCalled()
    expect(store.topicClustersRebuilding).toBe(false)
  })

  it('surfaces an error and clears the rebuilding flag when the POST fails', async () => {
    vi.mocked(postTopicClustersRebuild).mockRejectedValueOnce(new Error('HTTP 409'))
    const store = useArtifactsStore()
    store.setCorpusPath('/c')

    const p = store.rebuildTopicClusters()
    await vi.advanceTimersByTimeAsync(0)
    await p

    expect(store.topicClustersLoadState).toBe('error')
    expect(store.topicClustersErrorDetail).toContain('409')
    expect(store.topicClustersRebuilding).toBe(false)
  })

  it('ignores a concurrent call while a rebuild is already in flight', async () => {
    vi.mocked(fetchTopicClustersFromApi).mockResolvedValue(okResult)
    const store = useArtifactsStore()
    store.setCorpusPath('/c')

    const p1 = store.rebuildTopicClusters()
    await store.rebuildTopicClusters() // returns immediately — guard hit
    await vi.advanceTimersByTimeAsync(1600)
    await p1

    expect(postTopicClustersRebuild).toHaveBeenCalledTimes(1)
  })
})
