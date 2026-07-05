// @vitest-environment happy-dom
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { fetchArtifactJson } from '../api/artifactsApi'
import { fetchThemeClustersFromApi, fetchTopicClustersFromApi } from '../api/corpusTopicClustersApi'
import { fetchResolveEpisodeArtifacts } from '../api/corpusLibraryApi'
import type { ArtifactData } from '../types/artifact'
import type {
  TopicClustersDocument,
  TopicClustersFetchResult,
} from '../api/corpusTopicClustersApi'
import type { CorpusResolveEpisodesResponse } from '../api/corpusLibraryApi'
import { useArtifactsStore } from './artifacts'
import { useGraphExpansionStore } from './graphExpansion'
import { useGraphExplorerStore } from './graphExplorer'

vi.mock('../api/artifactsApi', () => ({
  fetchArtifactJson: vi.fn(),
}))
vi.mock('../api/corpusTopicClustersApi', () => ({
  fetchTopicClustersFromApi: vi.fn(),
  fetchThemeClustersFromApi: vi.fn(),
}))
vi.mock('../api/corpusLibraryApi', () => ({
  fetchResolveEpisodeArtifacts: vi.fn(),
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

const okClustersResult: TopicClustersFetchResult = {
  status: 'ok',
  document: { clusters: [] },
  schemaWarning: null,
}

beforeEach(() => {
  setActivePinia(createPinia())
  vi.mocked(fetchArtifactJson).mockReset()
  vi.mocked(fetchTopicClustersFromApi).mockReset()
  vi.mocked(fetchTopicClustersFromApi).mockResolvedValue(okClustersResult)
  vi.mocked(fetchThemeClustersFromApi).mockReset()
  vi.mocked(fetchThemeClustersFromApi).mockResolvedValue({ status: 'missing' })
  vi.mocked(fetchResolveEpisodeArtifacts).mockReset()
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
    expect(store.topicClustersLoadState).toBe('local_files')
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

  it('clearSelection wipes selection, list and topic-cluster state', async () => {
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
    expect(store.topicClustersDoc).toBeNull()
    expect(store.topicClustersLoadState).toBe('idle')
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

  it('overlays topic clusters onto the display artifact when doc present', async () => {
    const store = useArtifactsStore()
    store.setCorpusPath('/c')
    store.selectAllListed(['ep1.gi.json'])

    // GI with a Topic node the cluster will adopt as a compound child.
    const giWithTopic = {
      episode_id: 'ep1',
      model_version: 'm1',
      prompt_version: 'p1',
      nodes: [
        { id: 'episode:ep1', type: 'Episode', properties: {} },
        { id: 'topic:ai', type: 'Topic', properties: { label: 'AI' } },
      ],
      edges: [],
    } as unknown as ArtifactData
    vi.mocked(fetchArtifactJson).mockResolvedValue(giWithTopic)

    const clusterDoc: TopicClustersDocument = {
      clusters: [
        {
          graph_compound_parent_id: 'tc:1',
          canonical_label: 'AI cluster',
          members: [{ topic_id: 'topic:ai', label: 'AI' }],
        },
      ],
    }
    vi.mocked(fetchTopicClustersFromApi).mockResolvedValue({
      status: 'ok',
      document: clusterDoc,
      schemaWarning: null,
    })

    await store.loadSelected()
    const display = store.displayArtifact
    const ids = (display?.data.nodes ?? []).map((n) => String(n!.id))
    expect(ids).toContain('tc:1')
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

  it('clearSiblingMergeBanner resets the banner refs', () => {
    const store = useArtifactsStore()
    store.maybeMergeClusterSiblingEpisodes // referenced to keep import warm
    store.clearSiblingMergeBanner()
    expect(store.siblingMergeLine).toBeNull()
    expect(store.siblingMergeError).toBe(false)
  })
})

describe('useArtifactsStore — maybeMergeClusterSiblingEpisodes', () => {
  /**
   * Loads one GI episode (ep1), a topic-clusters doc that lists ep1 and a
   * sibling ep2, then drives the auto-merge. ep2 should be resolved + appended.
   */
  async function primeWithCluster(store: ReturnType<typeof useArtifactsStore>) {
    store.setCorpusPath('/c')
    store.selectAllListed(['ep1.gi.json'])
    vi.mocked(fetchArtifactJson).mockImplementation(async (_c, rel) =>
      rel.includes('ep2') ? giData('ep2') : giData('ep1'),
    )
    const clusterDoc: TopicClustersDocument = {
      clusters: [
        {
          graph_compound_parent_id: 'tc:1',
          members: [{ topic_id: 'topic:x', episode_ids: ['ep1', 'ep2'] }],
        },
      ],
    }
    vi.mocked(fetchTopicClustersFromApi).mockResolvedValue({
      status: 'ok',
      document: clusterDoc,
      schemaWarning: null,
    })
    await store.loadSelected()
  }

  it('skips when last load was external', async () => {
    const store = useArtifactsStore()
    await primeWithCluster(store)
    store.setLoadSource('digest-external')
    await store.maybeMergeClusterSiblingEpisodes(true)
    expect(fetchResolveEpisodeArtifacts).not.toHaveBeenCalled()
  })

  it('skips when the graph tab is not active', async () => {
    const store = useArtifactsStore()
    await primeWithCluster(store)
    await store.maybeMergeClusterSiblingEpisodes(false)
    expect(fetchResolveEpisodeArtifacts).not.toHaveBeenCalled()
  })

  it('skips when no corpus path or no topic-clusters doc', async () => {
    const store = useArtifactsStore()
    // No corpus path, no doc.
    await store.maybeMergeClusterSiblingEpisodes(true)
    expect(fetchResolveEpisodeArtifacts).not.toHaveBeenCalled()
  })

  it('resolves and appends sibling episode artifacts', async () => {
    const store = useArtifactsStore()
    await primeWithCluster(store)

    const resolveResp: CorpusResolveEpisodesResponse = {
      path: '/c',
      resolved: [
        {
          episode_id: 'ep2',
          publish_date: '2026-02-01',
          gi_relative_path: 'ep2.gi.json',
          kg_relative_path: null,
          bridge_relative_path: null,
        },
      ],
      missing_episode_ids: [],
    }
    vi.mocked(fetchResolveEpisodeArtifacts).mockResolvedValue(resolveResp)

    await store.maybeMergeClusterSiblingEpisodes(true)

    expect(fetchResolveEpisodeArtifacts).toHaveBeenCalledWith('/c', ['ep2'])
    expect(store.selectedRelPaths).toContain('ep2.gi.json')
    expect(store.siblingMergeError).toBe(false)
    expect(store.siblingMergeLine).toContain('+1 new')
  })

  it('reports a miss count in the banner line', async () => {
    const store = useArtifactsStore()
    store.setCorpusPath('/c')
    store.selectAllListed(['ep1.gi.json'])
    vi.mocked(fetchArtifactJson).mockResolvedValue(giData('ep1'))
    const clusterDoc: TopicClustersDocument = {
      clusters: [
        {
          graph_compound_parent_id: 'tc:1',
          members: [{ topic_id: 'topic:x', episode_ids: ['ep1', 'ep2', 'ep3'] }],
        },
      ],
    }
    vi.mocked(fetchTopicClustersFromApi).mockResolvedValue({
      status: 'ok',
      document: clusterDoc,
      schemaWarning: null,
    })
    await store.loadSelected()

    vi.mocked(fetchResolveEpisodeArtifacts).mockResolvedValue({
      path: '/c',
      resolved: [],
      missing_episode_ids: ['ep2', 'ep3'],
    })

    await store.maybeMergeClusterSiblingEpisodes(true)
    expect(store.siblingMergeLine).toContain('2 misses')
  })

  it('surfaces a resolve error in the banner', async () => {
    const store = useArtifactsStore()
    await primeWithCluster(store)
    vi.mocked(fetchResolveEpisodeArtifacts).mockRejectedValue(new Error('resolve 500'))

    await store.maybeMergeClusterSiblingEpisodes(true)
    expect(store.siblingMergeError).toBe(true)
    expect(store.siblingMergeLine).toContain('resolve 500')
  })

  it('does not re-run for an unchanged selection snapshot', async () => {
    const store = useArtifactsStore()
    await primeWithCluster(store)
    vi.mocked(fetchResolveEpisodeArtifacts).mockResolvedValue({
      path: '/c',
      resolved: [
        {
          episode_id: 'ep2',
          publish_date: '2026-02-01',
          gi_relative_path: 'ep2.gi.json',
          kg_relative_path: null,
          bridge_relative_path: null,
        },
      ],
      missing_episode_ids: [],
    })
    await store.maybeMergeClusterSiblingEpisodes(true)
    const calls = vi.mocked(fetchResolveEpisodeArtifacts).mock.calls.length

    // Same artifact set → snapshot key matches → no second resolve.
    await store.maybeMergeClusterSiblingEpisodes(true)
    expect(vi.mocked(fetchResolveEpisodeArtifacts).mock.calls.length).toBe(calls)
  })

  it('updates snapshot and skips resolve when no candidates remain', async () => {
    const store = useArtifactsStore()
    store.setCorpusPath('/c')
    store.selectAllListed(['ep1.gi.json'])
    vi.mocked(fetchArtifactJson).mockResolvedValue(giData('ep1'))
    // Cluster references only ep1 (already loaded) → zero candidates.
    vi.mocked(fetchTopicClustersFromApi).mockResolvedValue({
      status: 'ok',
      document: {
        clusters: [
          {
            graph_compound_parent_id: 'tc:1',
            members: [{ topic_id: 'topic:x', episode_ids: ['ep1'] }],
          },
        ],
      },
      schemaWarning: null,
    })
    await store.loadSelected()

    await store.maybeMergeClusterSiblingEpisodes(true)
    expect(fetchResolveEpisodeArtifacts).not.toHaveBeenCalled()
  })
})

describe('useArtifactsStore — ensureTopicClusterCompoundVisible', () => {
  it('returns true immediately for an empty compound id', async () => {
    const store = useArtifactsStore()
    store.setCorpusPath('/c')
    expect(await store.ensureTopicClusterCompoundVisible('  ')).toBe(true)
  })

  it('returns false without a corpus path', async () => {
    const store = useArtifactsStore()
    expect(await store.ensureTopicClusterCompoundVisible('tc:1')).toBe(false)
  })

  it('returns true when the cluster id is not in the doc', async () => {
    const store = useArtifactsStore()
    store.setCorpusPath('/c')
    vi.mocked(fetchTopicClustersFromApi).mockResolvedValue({
      status: 'ok',
      document: { clusters: [{ graph_compound_parent_id: 'tc:other', members: [] }] },
      schemaWarning: null,
    })
    expect(await store.ensureTopicClusterCompoundVisible('tc:missing')).toBe(true)
  })

  it('returns true when the compound is already visible in the merged graph', async () => {
    const store = useArtifactsStore()
    store.setCorpusPath('/c')
    store.selectAllListed(['ep1.gi.json'])
    // GI with a topic node so the overlay can build the tc:1 compound.
    vi.mocked(fetchArtifactJson).mockResolvedValue({
      episode_id: 'ep1',
      model_version: 'm1',
      prompt_version: 'p1',
      nodes: [
        { id: 'episode:ep1', type: 'Episode', properties: {} },
        { id: 'topic:ai', type: 'Topic', properties: { label: 'AI' } },
      ],
      edges: [],
    } as unknown as ArtifactData)
    const doc: TopicClustersDocument = {
      clusters: [
        {
          graph_compound_parent_id: 'tc:1',
          canonical_label: 'AI',
          members: [{ topic_id: 'topic:ai', episode_ids: ['ep1'] }],
        },
      ],
    }
    vi.mocked(fetchTopicClustersFromApi).mockResolvedValue({
      status: 'ok',
      document: doc,
      schemaWarning: null,
    })
    await store.loadSelected()
    // Overlay already produced tc:1 in displayArtifact.
    expect(await store.ensureTopicClusterCompoundVisible('tc:1')).toBe(true)
  })

  it('returns false when the cluster lists no episodes', async () => {
    const store = useArtifactsStore()
    store.setCorpusPath('/c')
    vi.mocked(fetchTopicClustersFromApi).mockResolvedValue({
      status: 'ok',
      document: {
        clusters: [{ graph_compound_parent_id: 'tc:1', members: [{ topic_id: 't' }] }],
      },
      schemaWarning: null,
    })
    expect(await store.ensureTopicClusterCompoundVisible('tc:1')).toBe(false)
  })

  it('resolves + appends episodes until the compound appears', async () => {
    const store = useArtifactsStore()
    store.setCorpusPath('/c')
    // Start empty; cluster lists ep1 whose GI carries the topic node.
    vi.mocked(fetchArtifactJson).mockResolvedValue({
      episode_id: 'ep1',
      model_version: 'm1',
      prompt_version: 'p1',
      nodes: [
        { id: 'episode:ep1', type: 'Episode', properties: {} },
        { id: 'topic:ai', type: 'Topic', properties: { label: 'AI' } },
      ],
      edges: [],
    } as unknown as ArtifactData)
    const doc: TopicClustersDocument = {
      clusters: [
        {
          graph_compound_parent_id: 'tc:1',
          canonical_label: 'AI',
          members: [{ topic_id: 'topic:ai', episode_ids: ['ep1'] }],
        },
      ],
    }
    vi.mocked(fetchTopicClustersFromApi).mockResolvedValue({
      status: 'ok',
      document: doc,
      schemaWarning: null,
    })
    vi.mocked(fetchResolveEpisodeArtifacts).mockResolvedValue({
      path: '/c',
      resolved: [
        {
          episode_id: 'ep1',
          publish_date: '2026-01-01',
          gi_relative_path: 'ep1.gi.json',
          kg_relative_path: null,
          bridge_relative_path: null,
        },
      ],
      missing_episode_ids: [],
    })

    const ok = await store.ensureTopicClusterCompoundVisible('tc:1')
    expect(ok).toBe(true)
    expect(store.parsedList.length).toBeGreaterThan(0)
    expect(store.selectedRelPaths).toContain('ep1.gi.json')
  })
})

describe('useArtifactsStore — maybeBootstrapGraphFromTopicClusterOnly', () => {
  it('returns false for an empty id', async () => {
    const store = useArtifactsStore()
    store.setCorpusPath('/c')
    expect(await store.maybeBootstrapGraphFromTopicClusterOnly('')).toBe(false)
  })

  it('returns false when the cluster has no listed episodes', async () => {
    const store = useArtifactsStore()
    store.setCorpusPath('/c')
    vi.mocked(fetchTopicClustersFromApi).mockResolvedValue({
      status: 'ok',
      document: {
        clusters: [{ graph_compound_parent_id: 'tc:1', members: [{ topic_id: 't' }] }],
      },
      schemaWarning: null,
    })
    expect(await store.maybeBootstrapGraphFromTopicClusterOnly('tc:1')).toBe(false)
  })

  it('loads from the cluster and returns true when artifacts arrive', async () => {
    const store = useArtifactsStore()
    store.setCorpusPath('/c')
    vi.mocked(fetchArtifactJson).mockResolvedValue(giData('ep1'))
    const doc: TopicClustersDocument = {
      clusters: [
        {
          graph_compound_parent_id: 'tc:1',
          members: [{ topic_id: 'topic:ai', episode_ids: ['ep1'] }],
        },
      ],
    }
    vi.mocked(fetchTopicClustersFromApi).mockResolvedValue({
      status: 'ok',
      document: doc,
      schemaWarning: null,
    })
    vi.mocked(fetchResolveEpisodeArtifacts).mockResolvedValue({
      path: '/c',
      resolved: [
        {
          episode_id: 'ep1',
          publish_date: '2026-01-01',
          gi_relative_path: 'ep1.gi.json',
          kg_relative_path: null,
          bridge_relative_path: null,
        },
      ],
      missing_episode_ids: [],
    })

    const ok = await store.maybeBootstrapGraphFromTopicClusterOnly('tc:1')
    expect(ok).toBe(true)
    expect(store.parsedList.length).toBeGreaterThan(0)
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
