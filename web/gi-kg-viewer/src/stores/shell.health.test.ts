import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useShellStore } from './shell'

describe('useShellStore /api/health discovery flags', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('normalizes status ok/OK/Ok to ok so watchers do not thrash on casing', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: true,
        json: async () => ({
          status: 'OK',
          corpus_library_api: true,
        }),
      })) as unknown as typeof fetch,
    )
    const shell = useShellStore()
    await shell.fetchHealth()
    expect(shell.healthStatus).toBe('ok')
  })

  it('sets corpusLibraryApiAvailable and corpusDigestApiAvailable from health JSON', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: true,
        json: async () => ({
          status: 'ok',
          corpus_library_api: true,
          corpus_digest_api: true,
        }),
      })) as unknown as typeof fetch,
    )
    const shell = useShellStore()
    await shell.fetchHealth()
    expect(shell.healthStatus).toBe('ok')
    expect(shell.healthStatusDisplay).toBe('OK')
    expect(shell.corpusLibraryApiAvailable).toBe(true)
    expect(shell.corpusDigestApiAvailable).toBe(true)
    expect(shell.corpusBinaryApiAvailable).toBe(true)
    expect(shell.artifactsApiAvailable).toBe(true)
    expect(shell.searchApiAvailable).toBe(true)
    expect(shell.exploreApiAvailable).toBe(true)
    expect(shell.indexRoutesApiAvailable).toBe(true)
    expect(shell.corpusMetricsApiAvailable).toBe(true)
    expect(shell.cilQueriesApiAvailable).toBe(true)
  })

  it('infers digest when corpus_digest_api is omitted but catalog is available', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: true,
        json: async () => ({
          status: 'ok',
          corpus_library_api: true,
        }),
      })) as unknown as typeof fetch,
    )
    const shell = useShellStore()
    await shell.fetchHealth()
    expect(shell.corpusLibraryApiAvailable).toBe(true)
    expect(shell.corpusDigestApiAvailable).toBe(true)
  })

  it('disables digest when corpus_digest_api is explicitly false', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: true,
        json: async () => ({
          status: 'ok',
          corpus_library_api: true,
          corpus_digest_api: false,
        }),
      })) as unknown as typeof fetch,
    )
    const shell = useShellStore()
    await shell.fetchHealth()
    expect(shell.corpusLibraryApiAvailable).toBe(true)
    expect(shell.corpusDigestApiAvailable).toBe(false)
  })

  it('clears digest flag on fetch failure', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new Error('network')
      }) as unknown as typeof fetch,
    )
    const shell = useShellStore()
    await shell.fetchHealth()
    expect(shell.corpusLibraryApiAvailable).toBe(false)
    expect(shell.corpusDigestApiAvailable).toBe(false)
    expect(shell.corpusBinaryApiAvailable).toBe(false)
    expect(shell.artifactsApiAvailable).toBe(false)
    expect(shell.searchApiAvailable).toBe(false)
    expect(shell.exploreApiAvailable).toBe(false)
    expect(shell.indexRoutesApiAvailable).toBe(false)
    expect(shell.corpusMetricsApiAvailable).toBe(false)
    expect(shell.cilQueriesApiAvailable).toBe(false)
  })

  it('disables CIL queries when cil_queries_api is explicitly false', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: true,
        json: async () => ({
          status: 'ok',
          corpus_library_api: true,
          corpus_digest_api: true,
          cil_queries_api: false,
        }),
      })) as unknown as typeof fetch,
    )
    const shell = useShellStore()
    await shell.fetchHealth()
    expect(shell.cilQueriesApiAvailable).toBe(false)
    expect(shell.searchApiAvailable).toBe(true)
  })

  it('disables artifacts when artifacts_api is explicitly false', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: true,
        json: async () => ({
          status: 'ok',
          corpus_library_api: true,
          corpus_digest_api: true,
          artifacts_api: false,
        }),
      })) as unknown as typeof fetch,
    )
    const shell = useShellStore()
    await shell.fetchHealth()
    expect(shell.artifactsApiAvailable).toBe(false)
    expect(shell.searchApiAvailable).toBe(true)
  })
})
