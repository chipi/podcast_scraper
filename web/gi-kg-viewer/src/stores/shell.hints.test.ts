import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useShellStore } from './shell'

describe('useShellStore /api/artifacts hints', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('populates corpusHints when response includes hints', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: true,
        json: async () => ({
          path: '/mock/corpus/feeds/rss_x/metadata',
          artifacts: [],
          hints: [
            'Unified semantic index is under /mock/corpus. Set corpus root to that directory.',
          ],
        }),
      })) as unknown as typeof fetch,
    )

    const shell = useShellStore()
    shell.corpusPath = '/mock/corpus/feeds/rss_x/metadata'
    await shell.fetchArtifactList()

    expect(shell.corpusHints.length).toBe(1)
    expect(shell.corpusHints[0]).toContain('Unified semantic index')
    expect(vi.mocked(fetch)).toHaveBeenCalled()
  })

  it('defaults corpusHints to empty when hints omitted', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: true,
        json: async () => ({
          path: '/x',
          artifacts: [{ name: 'a.gi.json', relative_path: 'a.gi.json', kind: 'gi', size_bytes: 1 }],
        }),
      })) as unknown as typeof fetch,
    )

    const shell = useShellStore()
    shell.corpusPath = '/x'
    await shell.fetchArtifactList()

    expect(shell.corpusHints).toEqual([])
  })
})
