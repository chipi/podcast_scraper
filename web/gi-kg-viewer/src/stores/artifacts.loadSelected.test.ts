import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { fetchArtifactJson } from '../api/artifactsApi'
import type { ArtifactData } from '../types/artifact'
import { useArtifactsStore } from './artifacts'
import { useGraphExpansionStore } from './graphExpansion'

vi.mock('../api/artifactsApi', () => ({
  fetchArtifactJson: vi.fn(),
}))

const emptyArtifact: ArtifactData = { nodes: [], edges: [] }

describe('useArtifactsStore loadSelected single-flight', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.mocked(fetchArtifactJson).mockReset()
  })

  it('overlapping loadSelected leaves last run authoritative and clears loading', async () => {
    const store = useArtifactsStore()
    store.setCorpusPath('/c')
    store.selectAllListed(['first.gi.json', 'second.gi.json'])

    let releaseFirst!: () => void
    const firstGate = new Promise<void>((resolve) => {
      releaseFirst = resolve
    })
    let firstGiCalls = 0
    vi.mocked(fetchArtifactJson).mockImplementation(async (_c, rel) => {
      if (rel === 'first.gi.json') {
        firstGiCalls += 1
        if (firstGiCalls === 1) await firstGate
      }
      return emptyArtifact
    })

    const p1 = store.loadSelected()
    await Promise.resolve()
    expect(store.loading).toBe(true)

    const p2 = store.loadSelected()
    await p2

    expect(store.loading).toBe(false)
    expect(store.parsedList).toHaveLength(2)

    releaseFirst()
    await p1

    expect(store.loading).toBe(false)
    expect(store.parsedList).toHaveLength(2)
    expect(firstGiCalls).toBe(2)
  })

  it('loadSelected resets graph expansion by default', async () => {
    const store = useArtifactsStore()
    const graphExpansion = useGraphExpansionStore()
    graphExpansion.recordExpand('n:episode:1', ['extra.gi.json'])
    store.setCorpusPath('/c')
    store.selectAllListed(['only.gi.json'])
    vi.mocked(fetchArtifactJson).mockResolvedValue(emptyArtifact)

    await store.loadSelected()

    expect(graphExpansion.isExpanded('n:episode:1')).toBe(false)
  })

  it('loadSelected with preserveExpansion keeps expansion state', async () => {
    const store = useArtifactsStore()
    const graphExpansion = useGraphExpansionStore()
    graphExpansion.recordExpand('n:episode:1', ['extra.gi.json'])
    store.setCorpusPath('/c')
    store.selectAllListed(['only.gi.json', 'extra.gi.json'])
    vi.mocked(fetchArtifactJson).mockResolvedValue(emptyArtifact)

    await store.loadSelected({ preserveExpansion: true })

    expect(graphExpansion.isExpanded('n:episode:1')).toBe(true)
  })

  it('#586: does not clear parsedList mid-load (prior list visible until atomic swap)', async () => {
    // Regression guard: the root cause of "renderer is null" during
    // RFC-076 expand was an intermediate ``parsedList = []`` at the
    // start of loadSelected. That empty state triggered GraphCanvas'
    // filteredArtifact watcher → redraw() → destroyCy() → while a
    // pending COSE-layout rAF was still holding a reference to the
    // now-destroyed renderer. Keep a prior list visible across the
    // async load window so no empty-state redraw races the teardown.
    const store = useArtifactsStore()
    store.setCorpusPath('/c')
    store.selectAllListed(['prior.gi.json'])
    vi.mocked(fetchArtifactJson).mockResolvedValue(emptyArtifact)

    // Prime the store with a prior load.
    await store.loadSelected()
    expect(store.parsedList).toHaveLength(1)

    // Now start a second load but capture parsedList **while** it is
    // in flight. The prior list must still be present.
    let releaseSecond!: () => void
    const secondGate = new Promise<void>((resolve) => {
      releaseSecond = resolve
    })
    vi.mocked(fetchArtifactJson).mockImplementation(async () => {
      await secondGate
      return emptyArtifact
    })
    store.selectAllListed(['next.gi.json'])
    const p = store.loadSelected()
    await Promise.resolve()

    // Key assertion: the list is **not** empty mid-flight.
    expect(store.parsedList.length).toBeGreaterThan(0)

    releaseSecond()
    await p
    expect(store.parsedList).toHaveLength(1)
  })

  it('#586: clears parsedList only when the load resolves to nothing', async () => {
    const store = useArtifactsStore()
    store.setCorpusPath('/c')
    store.selectAllListed(['only.gi.json'])
    vi.mocked(fetchArtifactJson).mockResolvedValue(emptyArtifact)
    await store.loadSelected()
    expect(store.parsedList).toHaveLength(1)

    // A subsequent empty selection clears the list (explicit empty).
    store.selectAllListed([])
    await store.loadSelected()
    expect(store.parsedList).toHaveLength(0)
  })

  /**
   * #768 — parallelization contract.
   *
   * The fix dispatches all per-file ``fetchArtifactJson`` calls
   * concurrently via ``Promise.all`` (the prior implementation awaited
   * each one serially, which dominated wall-clock for cross-surface
   * "Open in graph" handoffs on cold corpora).
   *
   * Each of the six tests below pins a specific invariant the first
   * attempt at parallelization broke (see issue #768 hypotheses).
   * Together they replace ad-hoc "manual real-corpus" validation with
   * structural guards.
   */

  it('#768: fetches main set concurrently — N inflight before any resolves', async () => {
    const store = useArtifactsStore()
    store.setCorpusPath('/c')
    // .kg.json-only selection (no .gi.json present) so the sibling-bridge
    // path stays dormant — keeps the concurrency assertion tight on the
    // main set without coupling to the parallel-sibling-fetch optimisation.
    store.selectAllListed([
      'a.kg.json',
      'b.kg.json',
      'c.kg.json',
      'd.kg.json',
    ])

    let inflight = 0
    let maxInflight = 0
    let release!: () => void
    const gate = new Promise<void>((resolve) => {
      release = resolve
    })
    vi.mocked(fetchArtifactJson).mockImplementation(async () => {
      inflight += 1
      maxInflight = Math.max(maxInflight, inflight)
      await gate
      inflight -= 1
      return emptyArtifact
    })

    const p = store.loadSelected()
    // Let every fetch's first ``await`` register.
    await Promise.resolve()
    await Promise.resolve()
    // Sequential would have ``maxInflight === 1``; parallel must see all 4.
    expect(maxInflight).toBe(4)

    release()
    await p
    expect(store.parsedList).toHaveLength(4)
  })

  it('#768: sibling-bridge fetch runs concurrently with main set (5th inflight)', async () => {
    const store = useArtifactsStore()
    store.setCorpusPath('/c')
    // 4 main files + a .gi.json → sibling-bridge candidate also dispatches.
    store.selectAllListed([
      'ep.gi.json',
      'a.kg.json',
      'b.kg.json',
      'c.kg.json',
    ])

    let inflight = 0
    let maxInflight = 0
    let release!: () => void
    const gate = new Promise<void>((resolve) => {
      release = resolve
    })
    vi.mocked(fetchArtifactJson).mockImplementation(async () => {
      inflight += 1
      maxInflight = Math.max(maxInflight, inflight)
      await gate
      inflight -= 1
      return emptyArtifact
    })

    const p = store.loadSelected()
    await Promise.resolve()
    await Promise.resolve()
    // 4 main + 1 sibling = 5 concurrent. Sequential would be 1 (sibling
    // wouldn't even fire until the main loop finished).
    expect(maxInflight).toBe(5)

    release()
    await p
  })

  it('#768: stale-check between dispatch and join — superseded run does not write parsedList', async () => {
    const store = useArtifactsStore()
    store.setCorpusPath('/c')
    store.selectAllListed(['first.gi.json'])

    let firstRelease!: () => void
    const firstGate = new Promise<void>((resolve) => {
      firstRelease = resolve
    })
    let firstCalls = 0
    vi.mocked(fetchArtifactJson).mockImplementation(async (_c, rel) => {
      if (rel === 'first.gi.json') {
        firstCalls += 1
        if (firstCalls === 1) {
          // Block the first run's fetch so we can supersede it.
          await firstGate
        }
      }
      return { nodes: [{ id: rel, type: 'X', properties: {} }], edges: [] }
    })

    const p1 = store.loadSelected()
    await Promise.resolve()
    expect(store.loading).toBe(true)

    // Supersede with second selection mid-flight.
    store.selectAllListed(['second.gi.json'])
    const p2 = store.loadSelected()
    await p2

    expect(store.parsedList).toHaveLength(1)
    expect(store.parsedList[0]?.data.nodes?.[0]?.id).toBe('second.gi.json')

    // Unblock the stale run. Its post-join writes MUST be suppressed.
    firstRelease()
    await p1

    // ``parsedList`` still reflects the second (authoritative) run.
    expect(store.parsedList).toHaveLength(1)
    expect(store.parsedList[0]?.data.nodes?.[0]?.id).toBe('second.gi.json')
    expect(store.loading).toBe(false)
  })

  it('#768: sibling-bridge fallback fires when no .bridge.json in selection', async () => {
    const store = useArtifactsStore()
    store.setCorpusPath('/c')
    store.selectAllListed(['ep.gi.json'])

    const calls: string[] = []
    vi.mocked(fetchArtifactJson).mockImplementation(async (_c, rel) => {
      calls.push(rel)
      if (rel === 'ep.bridge.json') {
        return {
          schema_version: '1.0',
          episode_id: 'ep-1',
          identities: [],
        } as unknown as ArtifactData
      }
      return emptyArtifact
    })

    await store.loadSelected()

    // Sibling-bridge fetch fired in parallel with the main set.
    expect(calls).toContain('ep.gi.json')
    expect(calls).toContain('ep.bridge.json')
    expect(store.bridgeDocument?.episode_id).toBe('ep-1')
  })

  it('#768: in-selection bridge wins over sibling-bridge fallback', async () => {
    const store = useArtifactsStore()
    store.setCorpusPath('/c')
    // Both .gi.json AND .bridge.json explicitly selected. Sibling-bridge
    // path must NOT overwrite the in-selection bridge document.
    store.selectAllListed(['ep.gi.json', 'ep.bridge.json'])

    const calls: string[] = []
    vi.mocked(fetchArtifactJson).mockImplementation(async (_c, rel) => {
      calls.push(rel)
      if (rel === 'ep.bridge.json') {
        return {
          schema_version: '1.0',
          episode_id: 'in-selection',
          identities: [],
        } as unknown as ArtifactData
      }
      return emptyArtifact
    })

    await store.loadSelected()

    // The bridge path appears ONCE (no duplicate sibling fetch).
    const bridgeCount = calls.filter((r) => r === 'ep.bridge.json').length
    expect(bridgeCount).toBe(1)
    expect(store.bridgeDocument?.episode_id).toBe('in-selection')
  })

  it('#768: main-fetch failure fails fast — loadError set, parsedList preserves prior list', async () => {
    const store = useArtifactsStore()
    store.setCorpusPath('/c')

    // Prime a known-good prior list.
    store.selectAllListed(['ok.gi.json'])
    vi.mocked(fetchArtifactJson).mockResolvedValue(emptyArtifact)
    await store.loadSelected()
    expect(store.parsedList).toHaveLength(1)

    // Now switch to a selection where one main fetch will throw.
    store.selectAllListed(['ok.gi.json', 'broken.gi.json'])
    vi.mocked(fetchArtifactJson).mockImplementation(async (_c, rel) => {
      if (rel === 'broken.gi.json') {
        throw new Error('HTTP 500')
      }
      return emptyArtifact
    })

    await store.loadSelected()

    expect(store.loadError).toContain('HTTP 500')
    // Prior list preserved (no mid-load empty wipe — #586 contract).
    expect(store.parsedList).toHaveLength(1)
    expect(store.loading).toBe(false)
  })

  it('#768: sibling-bridge failure is silent — main set still applied', async () => {
    const store = useArtifactsStore()
    store.setCorpusPath('/c')
    store.selectAllListed(['ep.gi.json'])

    vi.mocked(fetchArtifactJson).mockImplementation(async (_c, rel) => {
      if (rel === 'ep.bridge.json') {
        throw new Error('HTTP 404')
      }
      return emptyArtifact
    })

    await store.loadSelected()

    expect(store.loadError).toBeNull()
    expect(store.parsedList).toHaveLength(1)
    expect(store.bridgeDocument).toBeNull()
  })
})
