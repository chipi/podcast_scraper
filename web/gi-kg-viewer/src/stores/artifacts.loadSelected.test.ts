import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { fetchArtifactJson } from '../api/artifactsApi'
import { useArtifactsStore } from './artifacts'

vi.mock('../api/artifactsApi', () => ({
  fetchArtifactJson: vi.fn(),
}))

const emptyArtifact = { nodes: [] as unknown[], edges: [] as unknown[] }

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
})
