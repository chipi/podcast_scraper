// @vitest-environment happy-dom
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import DigestView from './DigestView.vue'
import { fetchCorpusDigest } from '../../api/digestApi'
import { useShellStore } from '../../stores/shell'

vi.mock('../../api/digestApi', async (orig) => ({
  ...(await orig<typeof import('../../api/digestApi')>()),
  fetchCorpusDigest: vi.fn().mockResolvedValue({ path: '/corpus', rows: [], topics: [] }),
}))
vi.mock('../../api/corpusLibraryApi', async (orig) => ({
  ...(await orig<typeof import('../../api/corpusLibraryApi')>()),
  fetchCorpusFeeds: vi.fn().mockResolvedValue({ path: '/corpus', feeds: [] }),
}))
// The shell store probes /api/health when the path changes; a probe that never answers leaves the
// health this test sets alone.
vi.mock('../../api/httpClient', async (orig) => ({
  ...(await orig<typeof import('../../api/httpClient')>()),
  fetchWithTimeout: () => new Promise(() => {}),
}))

/**
 * Digest is kept alive and reloads only on a path / health / API-availability change; a pipeline
 * job finishing bumps `shell.corpusRevision`, and the digest must re-read on it (2026-10-09).
 */
describe('viewer Digest — a finished job reloads the digest', () => {
  beforeEach(() => setActivePinia(createPinia()))
  afterEach(() => vi.clearAllMocks())

  it('re-reads the digest when the corpus revision moves', async () => {
    const shell = useShellStore()
    shell.corpusPath = '/corpus'
    shell.healthStatus = 'ok'
    shell.corpusDigestApiAvailable = true
    shell.corpusLibraryApiAvailable = true
    mount(DigestView, { global: { stubs: { teleport: true } } })
    await flushPromises()
    const before = vi.mocked(fetchCorpusDigest).mock.calls.length
    expect(before, 'the first load did not run').toBeGreaterThan(0)
    shell.noteCorpusChanged()
    await flushPromises()
    expect(vi.mocked(fetchCorpusDigest).mock.calls.length).toBe(before + 1)
  })
})
