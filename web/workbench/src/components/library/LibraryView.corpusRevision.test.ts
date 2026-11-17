// @vitest-environment happy-dom
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import LibraryView from './LibraryView.vue'
import { fetchCorpusEpisodes, fetchCorpusFeeds } from '../../api/corpusLibraryApi'
import { useShellStore } from '../../stores/shell'

vi.mock('../../api/corpusLibraryApi', async (orig) => ({
  ...(await orig<typeof import('../../api/corpusLibraryApi')>()),
  fetchCorpusFeeds: vi.fn().mockResolvedValue({ path: '/corpus', feeds: [] }),
  fetchCorpusEpisodes: vi.fn().mockResolvedValue({ path: '/corpus', items: [], next_cursor: null }),
}))
// The shell store probes /api/health when the path changes; a probe that never answers leaves the
// health this test sets alone.
vi.mock('../../api/httpClient', async (orig) => ({
  ...(await orig<typeof import('../../api/httpClient')>()),
  fetchWithTimeout: () => new Promise(() => {}),
}))

/**
 * Library is kept alive and reloads only on a path or health change; a pipeline job finishing
 * bumps `shell.corpusRevision`, and the episodes must re-read on it (2026-10-09).
 */
describe('viewer Library — a finished job reloads the corpus lists', () => {
  beforeEach(() => setActivePinia(createPinia()))
  afterEach(() => vi.clearAllMocks())

  it('re-reads feeds and episodes when the corpus revision moves', async () => {
    const shell = useShellStore()
    shell.corpusPath = '/corpus'
    shell.healthStatus = 'ok'
    mount(LibraryView, { global: { stubs: { teleport: true } } })
    await flushPromises()
    const feeds = vi.mocked(fetchCorpusFeeds).mock.calls.length
    const eps = vi.mocked(fetchCorpusEpisodes).mock.calls.length
    expect(feeds, 'the first load did not run').toBeGreaterThan(0)
    shell.noteCorpusChanged()
    await flushPromises()
    expect(vi.mocked(fetchCorpusFeeds).mock.calls.length).toBe(feeds + 1)
    expect(vi.mocked(fetchCorpusEpisodes).mock.calls.length).toBeGreaterThan(eps)
  })
})
