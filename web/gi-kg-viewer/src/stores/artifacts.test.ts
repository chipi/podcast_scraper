// @vitest-environment happy-dom
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { fetchArtifactJson } from '../api/artifactsApi'
import type { ArtifactData } from '../types/artifact'
import { artifactReaders } from '../extensions'
import { artifactFileReadable, readableArtifactSuffixes, useArtifactsStore } from './artifacts'
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

  it('ignores non-artifact files; errors when nothing usable remains', async () => {
    const store = useArtifactsStore()
    await store.loadFromLocalFiles(
      fileList([jsonFile('notes.txt.json', { a: 1 })]),
    )
    expect(store.parsedList).toHaveLength(0)
    expect(store.loadError).toBe('No graph artifact files (.kg.json) in selection.')
  })

  it('sets loadError when a file contains invalid JSON', async () => {
    const store = useArtifactsStore()
    const bad = new File(['{ not json'], 'ep1.kg.json', {
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
    expansion.recordExpand('n:episode:1', ['extra.kg.json'])
    await store.loadFromLocalFiles(fileList([jsonFile('ep1.kg.json', kgData('ep1'))]))
    // resetExpansionState is dispatched via a dynamic import; flush its
    // promise chain on a macrotask before asserting.
    await new Promise((r) => setTimeout(r, 0))
    expect(expansion.isExpanded('n:episode:1')).toBe(false)
  })
})

describe('useArtifactsStore — selection actions', () => {
  it('toggleSelection adds then removes a path', () => {
    const store = useArtifactsStore()
    store.toggleSelection('a.kg.json')
    expect(store.selectedRelPaths).toEqual(['a.kg.json'])
    store.toggleSelection('b.kg.json')
    expect(store.selectedRelPaths).toEqual(['a.kg.json', 'b.kg.json'])
    store.toggleSelection('a.kg.json')
    expect(store.selectedRelPaths).toEqual(['b.kg.json'])
  })

  it('selectAllListed replaces the selection with a copy', () => {
    const store = useArtifactsStore()
    const src = ['a.kg.json', 'b.kg.json']
    store.selectAllListed(src)
    expect(store.selectedRelPaths).toEqual(src)
    expect(store.selectedRelPaths).not.toBe(src)
  })

  it('deselectAllListed empties the selection', () => {
    const store = useArtifactsStore()
    store.selectAllListed(['a.kg.json'])
    store.deselectAllListed()
    expect(store.selectedRelPaths).toEqual([])
  })

  it('clearManualGraphSelection flips the manual flag off', () => {
    const store = useArtifactsStore()
    store.selectAllListed(['a.kg.json'])
    void store.loadRelativeArtifacts(['a.kg.json'])
    expect(store.manualGraphSelection).toBe(true)
    store.clearManualGraphSelection()
    expect(store.manualGraphSelection).toBe(false)
  })

  it('clearSelection wipes selection and list', async () => {
    const store = useArtifactsStore()
    store.setCorpusPath('/c')
    store.selectAllListed(['ep1.kg.json'])
    vi.mocked(fetchArtifactJson).mockResolvedValue(kgData('ep1'))
    await store.loadSelected()
    expect(store.parsedList.length).toBeGreaterThan(0)

    store.clearSelection()
    expect(store.selectedRelPaths).toEqual([])
    expect(store.parsedList).toHaveLength(0)
    expect(store.loadError).toBeNull()
    expect(store.manualGraphSelection).toBe(false)
  })

  it('setCorpusPath stores the raw string', () => {
    const store = useArtifactsStore()
    store.setCorpusPath('/some/corpus')
    expect(store.corpusPath).toBe('/some/corpus')
  })
})

describe('artifactFileReadable', () => {
  afterEach(() => artifactReaders.reset())

  it('takes KG files, and the files a registered reader takes', () => {
    expect(artifactFileReadable('a/ep.kg.json')).toBe(true)
    expect(artifactFileReadable('a/ep.gi.json')).toBe(false)
    expect(artifactFileReadable('a/ep.bridge.json')).toBe(false)
    artifactReaders.register({ id: 'x', suffixes: ['.GI.json'], read: () => null })
    expect(artifactFileReadable('a/ep.gi.json')).toBe(true)
  })

  it('lists KG, then each registered reader\'s suffixes once (the offline picker\'s accept list)', () => {
    expect(readableArtifactSuffixes()).toEqual(['.kg.json'])
    artifactReaders.register({ id: 'a', suffixes: ['.gi.json', '.bridge.json'], read: () => null })
    artifactReaders.register({ id: 'b', suffixes: ['.kg.json'], read: () => null })
    expect(readableArtifactSuffixes()).toEqual(['.kg.json', '.gi.json', '.bridge.json'])
  })
})

describe('useArtifactsStore — displayArtifact', () => {
  it('is null with no artifacts, a single KG as is, and several KG merged', async () => {
    const store = useArtifactsStore()
    expect(store.displayArtifact).toBeNull()
    await store.loadFromLocalFiles(fileList([jsonFile('ep1.kg.json', kgData('ep1'))]))
    expect(store.displayArtifact?.kind).toBe('kg')
    expect(store.displayArtifact?.name).toBe('ep1.kg.json')
    await store.loadFromLocalFiles(
      fileList([jsonFile('ep1.kg.json', kgData('ep1')), jsonFile('ep2.kg.json', kgData('ep2'))]),
    )
    expect(store.displayArtifact?.name).toContain('Merged KG')
  })

  it('draws nothing from a .gi.json the viewer has no reader for', async () => {
    const store = useArtifactsStore()
    await store.loadFromLocalFiles(fileList([jsonFile('ep1.gi.json', { nodes: [], edges: [] })]))
    expect(store.parsedList).toHaveLength(0)
    expect(store.loadError).toBe('No graph artifact files (.kg.json) in selection.')
  })
})

describe('useArtifactsStore — appendRelativeArtifacts / removeRelativeArtifacts', () => {
  it('appendRelativeArtifacts no-ops without a corpus path', async () => {
    const store = useArtifactsStore()
    await store.appendRelativeArtifacts(['a.kg.json'])
    expect(fetchArtifactJson).not.toHaveBeenCalled()
    expect(store.selectedRelPaths).toEqual([])
  })

  it('appendRelativeArtifacts adds only new normalized paths and reloads', async () => {
    const store = useArtifactsStore()
    store.setCorpusPath('/c')
    store.selectAllListed(['ep1.kg.json'])
    vi.mocked(fetchArtifactJson).mockImplementation(async (_c, rel) =>
      rel === 'ep1.kg.json' ? kgData('ep1') : kgData('ep2'),
    )
    await store.loadSelected()

    await store.appendRelativeArtifacts(['ep2.kg.json', 'ep1.kg.json'])
    // ep1 already present (dedup); ep2 added.
    expect(store.selectedRelPaths).toEqual(['ep1.kg.json', 'ep2.kg.json'])
    expect(store.parsedList).toHaveLength(2)
  })

  it('appendRelativeArtifacts no-ops when nothing new', async () => {
    const store = useArtifactsStore()
    store.setCorpusPath('/c')
    store.selectAllListed(['ep1.kg.json'])
    vi.mocked(fetchArtifactJson).mockResolvedValue(kgData('ep1'))
    await store.loadSelected()
    const calls = vi.mocked(fetchArtifactJson).mock.calls.length

    await store.appendRelativeArtifacts(['ep1.kg.json'])
    // No reload because nothing new was appended.
    expect(vi.mocked(fetchArtifactJson).mock.calls.length).toBe(calls)
  })

  it('removeRelativeArtifacts drops paths and reloads', async () => {
    const store = useArtifactsStore()
    store.setCorpusPath('/c')
    store.selectAllListed(['ep1.kg.json', 'ep2.kg.json'])
    vi.mocked(fetchArtifactJson).mockImplementation(async (_c, rel) =>
      rel === 'ep1.kg.json' ? kgData('ep1') : kgData('ep2'),
    )
    await store.loadSelected()
    expect(store.parsedList).toHaveLength(2)

    await store.removeRelativeArtifacts(['ep2.kg.json'])
    expect(store.selectedRelPaths).toEqual(['ep1.kg.json'])
    expect(store.parsedList).toHaveLength(1)
  })

  it('removeRelativeArtifacts no-ops when path is absent', async () => {
    const store = useArtifactsStore()
    store.setCorpusPath('/c')
    store.selectAllListed(['ep1.kg.json'])
    vi.mocked(fetchArtifactJson).mockResolvedValue(kgData('ep1'))
    await store.loadSelected()
    const calls = vi.mocked(fetchArtifactJson).mock.calls.length

    await store.removeRelativeArtifacts(['nope.kg.json'])
    expect(store.selectedRelPaths).toEqual(['ep1.kg.json'])
    expect(vi.mocked(fetchArtifactJson).mock.calls.length).toBe(calls)
  })

  it('removeRelativeArtifacts no-ops without corpus or empty input', async () => {
    const store = useArtifactsStore()
    store.selectAllListed(['ep1.kg.json'])
    await store.removeRelativeArtifacts(['ep1.kg.json'])
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
    vi.mocked(fetchArtifactJson).mockResolvedValue(kgData('ep1'))
    await store.loadRelativeArtifacts(['  ep1.kg.json  ', '', '   '])
    expect(store.selectedRelPaths).toEqual(['ep1.kg.json'])
    expect(store.manualGraphSelection).toBe(true)
    expect(store.parsedList).toHaveLength(1)
  })
})

describe('useArtifactsStore — loadSelected guard branches', () => {
  it('errors when corpus path is empty', async () => {
    const store = useArtifactsStore()
    store.selectAllListed(['ep1.kg.json'])
    await store.loadSelected()
    expect(store.loadError).toBe(
      'Set corpus path and select at least one artifact file.',
    )
    expect(store.parsedList).toHaveLength(0)
  })

  it('errors when the selection carries no graph artifact (a bridge alone)', async () => {
    const store = useArtifactsStore()
    store.setCorpusPath('/c')
    store.selectAllListed(['ep1.bridge.json'])
    vi.mocked(fetchArtifactJson).mockResolvedValue({
      schema_version: '1.0',
      episode_id: 'ep1',
      identities: [],
    } as unknown as ArtifactData)
    await store.loadSelected()
    expect(store.loadError).toBe('No graph artifact files (.kg.json) in selection.')
    expect(store.parsedList).toHaveLength(0)
  })

  it('does not apply the date lens (local picks keep older fixtures)', async () => {
    const store = useArtifactsStore()
    const explorer = useGraphExplorerStore()
    explorer.setSinceYmd('2099-01-01')
    await store.loadFromLocalFiles(
      fileList([jsonFile('old.kg.json', kgData('old'), Date.UTC(2000, 0, 1))]),
    )
    // Local file load ignores the lens — the old episode is kept.
    expect(store.parsedList).toHaveLength(1)
  })
})
