import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { fetchArtifactJson } from '../api/artifactsApi'
import { artifactLoadHooks, artifactTransforms, displayOverrides, graphEgoViews } from '../extensions'
import type { ArtifactData, ParsedArtifact } from '../types/artifact'
import { useArtifactsStore } from './artifacts'
import { useGraphFilterStore } from './graphFilters'

vi.mock('../api/artifactsApi', () => ({ fetchArtifactJson: vi.fn() }))

const gi = (episodeId: string): ArtifactData =>
  ({
    episode_id: episodeId,
    model_version: 'm1',
    prompt_version: 'p1',
    nodes: [
      { id: `episode:${episodeId}`, type: 'Episode', properties: { publish_date: '2026-01-01' } },
      { id: `g:insight:${episodeId}:1`, type: 'Insight', properties: { episode_id: episodeId, text: 'x' } },
    ],
    edges: [],
  }) as unknown as ArtifactData

const art = (name: string, types: string[]): ParsedArtifact => ({
  name,
  kind: 'both',
  episodeId: null,
  nodes: types.length,
  edges: 0,
  nodeTypes: {},
  data: { nodes: types.map((t, i) => ({ id: `${name}:${i}`, type: t })), edges: [] },
})

beforeEach(() => {
  setActivePinia(createPinia())
  vi.mocked(fetchArtifactJson).mockReset()
})
afterEach(() => {
  artifactTransforms.reset()
  artifactLoadHooks.reset()
  displayOverrides.reset()
  graphEgoViews.reset()
})

describe('artifacts store extension points', () => {
  it('runs transforms in order over the merged graph', async () => {
    artifactTransforms.register({ id: 'b', order: 2, transform: (a) => (a ? { ...a, name: `${a.name}+b` } : a) })
    artifactTransforms.register({ id: 'a', order: 1, transform: (a) => (a ? { ...a, name: `${a.name}+a` } : a) })
    const store = useArtifactsStore()
    store.setCorpusPath('/c')
    store.selectAllListed(['ep1.gi.json'])
    vi.mocked(fetchArtifactJson).mockResolvedValue(gi('ep1'))
    await store.loadSelected()
    expect(store.displayArtifact?.name.endsWith('+a+b')).toBe(true)
  })

  it('tells load hooks about the load, waits for beforeAssign, and reports resets and corpus changes', async () => {
    const seen: string[] = []
    let release!: () => void
    artifactLoadHooks.register({
      id: 'h',
      onLoadStart: (o) => seen.push(`start:${o.preserveExpansion}`),
      beforeAssign: () =>
        new Promise<void>((r) => {
          seen.push('beforeAssign')
          release = r
        }),
      onReset: (why) => seen.push(`reset:${why}`),
      onCorpusPathChange: (next, prev) => seen.push(`path:${prev}->${next}`),
    })
    const store = useArtifactsStore()
    store.setCorpusPath('/c')
    store.setCorpusPath('/c ')
    store.selectAllListed(['ep1.gi.json'])
    vi.mocked(fetchArtifactJson).mockResolvedValue(gi('ep1'))
    const loading = store.loadSelected()
    await vi.waitFor(() => expect(seen).toContain('beforeAssign'))
    expect(store.parsedList).toHaveLength(0)
    release()
    await loading
    expect(store.parsedList.length).toBeGreaterThan(0)
    store.clearSelection()
    expect(seen).toEqual(['path:->/c', 'start:false', 'beforeAssign', 'reset:clear'])
  })
})

describe('graph filters extension points', () => {
  it('builds on the display graph while no override is active', () => {
    vi.spyOn(useArtifactsStore(), 'displayArtifact', 'get').mockReturnValue(art('base', ['Insight']))
    displayOverrides.register({ id: 'o', active: () => false, artifact: () => art('alt', ['Bubble']) })
    expect(useGraphFilterStore().fullArtifact?.name).toBe('base')
  })

  it('uses an active override with its forced types, and a registered ego view', () => {
    vi.spyOn(useArtifactsStore(), 'displayArtifact', 'get').mockReturnValue(art('base', ['Insight']))
    displayOverrides.register({
      id: 'o',
      active: () => true,
      artifact: () => art('alt', ['Bubble', 'Topic']),
      forceVisibleTypes: ['Bubble'],
    })
    const gf = useGraphFilterStore()
    expect(gf.fullArtifact?.name).toBe('alt')
    gf.state!.allowedTypes = { ...gf.state!.allowedTypes, Bubble: false }
    expect(gf.filteredArtifact?.data?.nodes?.some((n) => n.type === 'Bubble')).toBe(true)
    graphEgoViews.register({ id: 'ego', view: (base, focus) => (focus === 'x' ? { ...base, name: 'ego-x' } : null) })
    expect(gf.viewWithEgo('x')?.name).toBe('ego-x')
    expect(gf.viewWithEgo('y')?.name).not.toBe('ego-x')
  })
})
