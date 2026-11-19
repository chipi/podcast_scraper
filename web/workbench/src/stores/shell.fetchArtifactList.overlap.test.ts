import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useShellStore } from './shell'

describe('useShellStore fetchArtifactList overlapping requests', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('last started fetch wins and stale completion does not clobber list or loading', async () => {
    let releaseFirst!: () => void
    const firstGate = new Promise<void>((resolve) => {
      releaseFirst = resolve
    })

    let call = 0
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        call += 1
        if (call === 1) {
          await firstGate
        }
        const name = call === 1 ? 'first.gi.json' : 'second.gi.json'
        return {
          ok: true,
          json: async () => ({
            path: '/corpus',
            artifacts: [
              {
                name,
                relative_path: name,
                kind: 'gi',
                size_bytes: 1,
                mtime_utc: '2020-01-01T00:00:00Z',
              },
            ],
          }),
        }
      }) as unknown as typeof fetch,
    )

    const shell = useShellStore()
    shell.corpusPath = '/corpus'

    const p1 = shell.fetchArtifactList()
    await Promise.resolve()
    expect(shell.artifactsLoading).toBe(true)

    const p2 = shell.fetchArtifactList()
    await p2

    expect(shell.artifactsLoading).toBe(false)
    expect(shell.artifactList).toHaveLength(1)
    expect(shell.artifactList[0]?.relative_path).toBe('second.gi.json')

    releaseFirst()
    await p1

    expect(shell.artifactsLoading).toBe(false)
    expect(shell.artifactList[0]?.relative_path).toBe('second.gi.json')
  })
})
