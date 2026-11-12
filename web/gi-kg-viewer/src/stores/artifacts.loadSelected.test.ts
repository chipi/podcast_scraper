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
    store.selectAllListed(['first.kg.json', 'second.kg.json'])

    let releaseFirst!: () => void
    const firstGate = new Promise<void>((resolve) => {
      releaseFirst = resolve
    })
    let firstGiCalls = 0
    vi.mocked(fetchArtifactJson).mockImplementation(async (_c, rel) => {
      if (rel === 'first.kg.json') {
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
    graphExpansion.recordExpand('n:episode:1', ['extra.kg.json'])
    store.setCorpusPath('/c')
    store.selectAllListed(['only.kg.json'])
    vi.mocked(fetchArtifactJson).mockResolvedValue(emptyArtifact)

    await store.loadSelected()

    expect(graphExpansion.isExpanded('n:episode:1')).toBe(false)
  })

  it('loadSelected with preserveExpansion keeps expansion state', async () => {
    const store = useArtifactsStore()
    const graphExpansion = useGraphExpansionStore()
    graphExpansion.recordExpand('n:episode:1', ['extra.kg.json'])
    store.setCorpusPath('/c')
    store.selectAllListed(['only.kg.json', 'extra.kg.json'])
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
    store.selectAllListed(['prior.kg.json'])
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
    store.selectAllListed(['next.kg.json'])
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
    store.selectAllListed(['only.kg.json'])
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
    // A .kg.json-only selection, as every viewer load is: the assertion is on the main set alone.
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

  it('#768: stale-check between dispatch and join — superseded run does not write parsedList', async () => {
    const store = useArtifactsStore()
    store.setCorpusPath('/c')
    store.selectAllListed(['first.kg.json'])

    let firstRelease!: () => void
    const firstGate = new Promise<void>((resolve) => {
      firstRelease = resolve
    })
    let firstCalls = 0
    vi.mocked(fetchArtifactJson).mockImplementation(async (_c, rel) => {
      if (rel === 'first.kg.json') {
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
    store.selectAllListed(['second.kg.json'])
    const p2 = store.loadSelected()
    await p2

    expect(store.parsedList).toHaveLength(1)
    expect(store.parsedList[0]?.data.nodes?.[0]?.id).toBe('second.kg.json')

    // Unblock the stale run. Its post-join writes MUST be suppressed.
    firstRelease()
    await p1

    // ``parsedList`` still reflects the second (authoritative) run.
    expect(store.parsedList).toHaveLength(1)
    expect(store.parsedList[0]?.data.nodes?.[0]?.id).toBe('second.kg.json')
    expect(store.loading).toBe(false)
  })

  it('#768: main-fetch failure fails fast — loadError set, parsedList preserves prior list', async () => {
    const store = useArtifactsStore()
    store.setCorpusPath('/c')

    // Prime a known-good prior list.
    store.selectAllListed(['ok.kg.json'])
    vi.mocked(fetchArtifactJson).mockResolvedValue(emptyArtifact)
    await store.loadSelected()
    expect(store.parsedList).toHaveLength(1)

    // Now switch to a selection where one main fetch will throw.
    store.selectAllListed(['ok.kg.json', 'broken.kg.json'])
    vi.mocked(fetchArtifactJson).mockImplementation(async (_c, rel) => {
      if (rel === 'broken.kg.json') {
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

})
