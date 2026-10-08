// @vitest-environment happy-dom
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { fetchArtifactJson } from '../api/artifactsApi'
import type { ArtifactData } from '../types/artifact'
import { useArtifactsStore } from './artifacts'
import { useGraphExpansionStore } from './graphExpansion'
import { useGraphExplorerStore } from './graphExplorer'

vi.mock('../api/artifactsApi', () => ({
  fetchArtifactJson: vi.fn(),
}))

const emptyArtifact: ArtifactData = { nodes: [], edges: [] }

/** Build a File whose .text() yields the given JSON, with an optional mtime. */
function jsonFile(
  name: string,
  data: unknown,
  lastModified = Date.UTC(2026, 0, 1),
): File {
  const f = new File([JSON.stringify(data)], name, {
    type: 'application/json',
    lastModified,
  })
  return f
}

/** A minimal FileList-like wrapper around an array of File objects. */
function fileList(files: File[]): FileList {
  const list = {
    length: files.length,
    item: (i: number) => files[i] ?? null,
  } as unknown as FileList
  files.forEach((f, i) => {
    ;(list as unknown as Record<number, File>)[i] = f
  })
  return list
}

function giData(episodeId: string): ArtifactData {
  return {
    episode_id: episodeId,
    model_version: 'm1',
    prompt_version: 'p1',
    nodes: [
      {
        id: `episode:${episodeId}`,
        type: 'Episode',
        properties: { publish_date: '2026-01-01' },
      },
      {
        id: `g:insight:${episodeId}:1`,
        type: 'Insight',
        properties: { episode_id: episodeId, text: 'insight one' },
      },
    ],
    edges: [],
  } as unknown as ArtifactData
}

function kgData(episodeId: string): ArtifactData {
  return {
    episode_id: episodeId,
    extraction: { extracted_at: '2026-01-01' },
    nodes: [
      { id: `episode:${episodeId}`, type: 'Episode', properties: {} },
      { id: `entity:person:ada`, type: 'Entity', properties: { name: 'Ada' } },
    ],
    edges: [],
  } as unknown as ArtifactData
}

beforeEach(() => {
  setActivePinia(createPinia())
  vi.mocked(fetchArtifactJson).mockReset()
})

describe('useArtifactsStore — loadFromLocalFiles', () => {
  it('no-ops on null / empty selection', async () => {
    const store = useArtifactsStore()
    await store.loadFromLocalFiles(null)
    expect(store.parsedList).toHaveLength(0)
    expect(store.loading).toBe(false)
    await store.loadFromLocalFiles(fileList([]))
    expect(store.parsedList).toHaveLength(0)
  })

  it('parses .gi.json and .kg.json files and marks manual selection', async () => {
    const store = useArtifactsStore()
    const files = fileList([
      jsonFile('ep1.gi.json', giData('ep1')),
      jsonFile('ep1.kg.json', kgData('ep1')),
    ])
    await store.loadFromLocalFiles(files)
    expect(store.parsedList).toHaveLength(2)
    expect(store.manualGraphSelection).toBe(true)
    expect(store.selectedRelPaths).toEqual(['ep1.gi.json', 'ep1.kg.json'])
    expect(store.loading).toBe(false)
    expect(store.loadError).toBeNull()
  })

  it('builds a merged GI+KG display artifact from local files', async () => {
    const store = useArtifactsStore()
    await store.loadFromLocalFiles(
      fileList([
        jsonFile('ep1.gi.json', giData('ep1')),
        jsonFile('ep1.kg.json', kgData('ep1')),
      ]),
    )
    const display = store.displayArtifact
    expect(display).not.toBeNull()
    expect(display!.kind).toBe('both')
    expect(store.giArts).toHaveLength(1)
    expect(store.kgArts).toHaveLength(1)
  })

  it('ignores non-artifact files; errors when nothing usable remains', async () => {
    const store = useArtifactsStore()
    await store.loadFromLocalFiles(
      fileList([jsonFile('notes.txt.json', { a: 1 })]),
    )
    expect(store.parsedList).toHaveLength(0)
    expect(store.loadError).toBe('No .gi.json or .kg.json files in selection.')
  })

  it('captures a matching sibling .bridge.json by episode stem', async () => {
    const store = useArtifactsStore()
    await store.loadFromLocalFiles(
      fileList([
        jsonFile('ep1.gi.json', giData('ep1')),
        jsonFile('ep1.bridge.json', {
          schema_version: '1.0',
          episode_id: 'ep1',
          identities: [],
        }),
      ]),
    )
    expect(store.bridgeDocument).not.toBeNull()
    expect(store.bridgeDocument?.episode_id).toBe('ep1')
  })

  it('does not attach a bridge whose stem matches no kept artifact', async () => {
    const store = useArtifactsStore()
    await store.loadFromLocalFiles(
      fileList([
        jsonFile('ep1.gi.json', giData('ep1')),
        jsonFile('other.bridge.json', {
          schema_version: '1.0',
          episode_id: 'other',
          identities: [],
        }),
      ]),
    )
    expect(store.bridgeDocument).toBeNull()
  })

  it('sets loadError when a file contains invalid JSON', async () => {
    const store = useArtifactsStore()
    const bad = new File(['{ not json'], 'ep1.gi.json', {
      type: 'application/json',
      lastModified: Date.UTC(2026, 0, 1),
    })
    await store.loadFromLocalFiles(fileList([bad]))
    expect(store.loadError).toBeTruthy()
    expect(store.loading).toBe(false)
  })

  it('resets graph expansion state on local-file load', async () => {
    const store = useArtifactsStore()
    const expansion = useGraphExpansionStore()
    expansion.recordExpand('n:episode:1', ['extra.gi.json'])
    await store.loadFromLocalFiles(fileList([jsonFile('ep1.gi.json', giData('ep1'))]))
    // resetExpansionState is dispatched via a dynamic import; flush its
    // promise chain on a macrotask before asserting.
    await new Promise((r) => setTimeout(r, 0))
    expect(expansion.isExpanded('n:episode:1')).toBe(false)
  })
})

describe('useArtifactsStore — selection actions', () => {
  it('toggleSelection adds then removes a path', () => {
    const store = useArtifactsStore()
    store.toggleSelection('a.gi.json')
    expect(store.selectedRelPaths).toEqual(['a.gi.json'])
    store.toggleSelection('b.gi.json')
    expect(store.selectedRelPaths).toEqual(['a.gi.json', 'b.gi.json'])
    store.toggleSelection('a.gi.json')
    expect(store.selectedRelPaths).toEqual(['b.gi.json'])
  })

  it('selectAllListed replaces the selection with a copy', () => {
    const store = useArtifactsStore()
    const src = ['a.gi.json', 'b.gi.json']
    store.selectAllListed(src)
    expect(store.selectedRelPaths).toEqual(src)
    expect(store.selectedRelPaths).not.toBe(src)
  })

  it('deselectAllListed empties the selection', () => {
    const store = useArtifactsStore()
    store.selectAllListed(['a.gi.json'])
    store.deselectAllListed()
    expect(store.selectedRelPaths).toEqual([])
  })

  it('clearManualGraphSelection flips the manual flag off', () => {
    const store = useArtifactsStore()
    store.selectAllListed(['a.gi.json'])
    void store.loadRelativeArtifacts(['a.gi.json'])
    expect(store.manualGraphSelection).toBe(true)
    store.clearManualGraphSelection()
    expect(store.manualGraphSelection).toBe(false)
  })

  it('clearSelection wipes selection and list', async () => {
    const store = useArtifactsStore()
    store.setCorpusPath('/c')
    store.selectAllListed(['ep1.gi.json'])
    vi.mocked(fetchArtifactJson).mockResolvedValue(giData('ep1'))
    await store.loadSelected()
    expect(store.parsedList.length).toBeGreaterThan(0)

    store.clearSelection()
    expect(store.selectedRelPaths).toEqual([])
    expect(store.parsedList).toHaveLength(0)
    expect(store.bridgeDocument).toBeNull()
    expect(store.loadError).toBeNull()
    expect(store.manualGraphSelection).toBe(false)
  })

  it('setCorpusPath stores the raw string', () => {
    const store = useArtifactsStore()
    store.setCorpusPath('/some/corpus')
    expect(store.corpusPath).toBe('/some/corpus')
  })
})

describe('useArtifactsStore — displayArtifact / getters', () => {
  it('is null with no artifacts and single GI returns that artifact', async () => {
    const store = useArtifactsStore()
    expect(store.displayArtifact).toBeNull()

    store.setCorpusPath('/c')
    store.selectAllListed(['ep1.gi.json'])
    vi.mocked(fetchArtifactJson).mockResolvedValue(giData('ep1'))
    await store.loadSelected()
    expect(store.displayArtifact?.kind).toBe('gi')
    expect(store.giArts).toHaveLength(1)
    expect(store.kgArts).toHaveLength(0)
  })

})

describe('useArtifactsStore — appendRelativeArtifacts / removeRelativeArtifacts', () => {
  it('appendRelativeArtifacts no-ops without a corpus path', async () => {
    const store = useArtifactsStore()
    await store.appendRelativeArtifacts(['a.gi.json'])
    expect(fetchArtifactJson).not.toHaveBeenCalled()
    expect(store.selectedRelPaths).toEqual([])
  })

  it('appendRelativeArtifacts adds only new normalized paths and reloads', async () => {
    const store = useArtifactsStore()
    store.setCorpusPath('/c')
    store.selectAllListed(['ep1.gi.json'])
    vi.mocked(fetchArtifactJson).mockImplementation(async (_c, rel) =>
      rel === 'ep1.gi.json' ? giData('ep1') : giData('ep2'),
    )
    await store.loadSelected()

    await store.appendRelativeArtifacts(['ep2.gi.json', 'ep1.gi.json'])
    // ep1 already present (dedup); ep2 added.
    expect(store.selectedRelPaths).toEqual(['ep1.gi.json', 'ep2.gi.json'])
    expect(store.parsedList).toHaveLength(2)
  })

  it('appendRelativeArtifacts no-ops when nothing new', async () => {
    const store = useArtifactsStore()
    store.setCorpusPath('/c')
    store.selectAllListed(['ep1.gi.json'])
    vi.mocked(fetchArtifactJson).mockResolvedValue(giData('ep1'))
    await store.loadSelected()
    const calls = vi.mocked(fetchArtifactJson).mock.calls.length

    await store.appendRelativeArtifacts(['ep1.gi.json'])
    // No reload because nothing new was appended.
    expect(vi.mocked(fetchArtifactJson).mock.calls.length).toBe(calls)
  })

  it('removeRelativeArtifacts drops paths and reloads', async () => {
    const store = useArtifactsStore()
    store.setCorpusPath('/c')
    store.selectAllListed(['ep1.gi.json', 'ep2.gi.json'])
    vi.mocked(fetchArtifactJson).mockImplementation(async (_c, rel) =>
      rel === 'ep1.gi.json' ? giData('ep1') : giData('ep2'),
    )
    await store.loadSelected()
    expect(store.parsedList).toHaveLength(2)

    await store.removeRelativeArtifacts(['ep2.gi.json'])
    expect(store.selectedRelPaths).toEqual(['ep1.gi.json'])
    expect(store.parsedList).toHaveLength(1)
  })

  it('removeRelativeArtifacts no-ops when path is absent', async () => {
    const store = useArtifactsStore()
    store.setCorpusPath('/c')
    store.selectAllListed(['ep1.gi.json'])
    vi.mocked(fetchArtifactJson).mockResolvedValue(giData('ep1'))
    await store.loadSelected()
    const calls = vi.mocked(fetchArtifactJson).mock.calls.length

    await store.removeRelativeArtifacts(['nope.gi.json'])
    expect(store.selectedRelPaths).toEqual(['ep1.gi.json'])
    expect(vi.mocked(fetchArtifactJson).mock.calls.length).toBe(calls)
  })

  it('removeRelativeArtifacts no-ops without corpus or empty input', async () => {
    const store = useArtifactsStore()
    store.selectAllListed(['ep1.gi.json'])
    await store.removeRelativeArtifacts(['ep1.gi.json'])
    expect(fetchArtifactJson).not.toHaveBeenCalled()
    store.setCorpusPath('/c')
    await store.removeRelativeArtifacts([])
    expect(fetchArtifactJson).not.toHaveBeenCalled()
  })
})

describe('useArtifactsStore — load source getter', () => {
  it('setLoadSource / clearLoadSource drive currentLoadSource', () => {
    const store = useArtifactsStore()
    expect(store.currentLoadSource).toBeNull()
    store.setLoadSource('digest-external')
    expect(store.currentLoadSource).toBe('digest-external')
    store.clearLoadSource()
    expect(store.currentLoadSource).toBeNull()
  })

})

describe('useArtifactsStore — loadRelativeArtifacts', () => {
  it('trims, filters empties and loads', async () => {
    const store = useArtifactsStore()
    store.setCorpusPath('/c')
    vi.mocked(fetchArtifactJson).mockResolvedValue(giData('ep1'))
    await store.loadRelativeArtifacts(['  ep1.gi.json  ', '', '   '])
    expect(store.selectedRelPaths).toEqual(['ep1.gi.json'])
    expect(store.manualGraphSelection).toBe(true)
    expect(store.parsedList).toHaveLength(1)
  })
})

describe('useArtifactsStore — loadSelected guard branches', () => {
  it('errors when corpus path is empty', async () => {
    const store = useArtifactsStore()
    store.selectAllListed(['ep1.gi.json'])
    await store.loadSelected()
    expect(store.loadError).toBe(
      'Set corpus path and select at least one artifact file.',
    )
    expect(store.parsedList).toHaveLength(0)
  })

  it('errors when selection only carries a bridge (no gi/kg output)', async () => {
    const store = useArtifactsStore()
    store.setCorpusPath('/c')
    store.selectAllListed(['ep1.bridge.json'])
    vi.mocked(fetchArtifactJson).mockResolvedValue({
      schema_version: '1.0',
      episode_id: 'ep1',
      identities: [],
    } as unknown as ArtifactData)
    await store.loadSelected()
    expect(store.loadError).toBe('No .gi.json or .kg.json files in selection.')
    expect(store.parsedList).toHaveLength(0)
  })

  it('does not apply the date lens (local picks keep older fixtures)', async () => {
    const store = useArtifactsStore()
    const explorer = useGraphExplorerStore()
    explorer.setSinceYmd('2099-01-01')
    await store.loadFromLocalFiles(
      fileList([jsonFile('old.gi.json', giData('old'), Date.UTC(2000, 0, 1))]),
    )
    // Local file load ignores the lens — the old episode is kept.
    expect(store.parsedList).toHaveLength(1)
  })
})
