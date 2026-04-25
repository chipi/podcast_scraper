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
})
