// @vitest-environment happy-dom
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ref } from 'vue'

import type { ParsedArtifact } from '../types/artifact'
import { parseArtifact } from '../utils/parsing'

/*
 * graphFilters store — tier 8 top-down routing + SuperTheme force-through.
 * Both branches are safety-critical: getting either wrong strands the user
 * with an empty top-down canvas or with the full graph mounted when the
 * user asked for the slice.
 *
 * We drive the store's collaborators via vi.mock: the artifacts store
 * (fullArtifact source), the load-mode store (top-down flag), and the
 * navigation store (only touched by viewWithEgo, which we don't test
 * here).
 */

const mockDisplayArtifact = ref<ParsedArtifact | null>(null)
const mockTopDownArtifact = ref<ParsedArtifact | null>(null)
const mockIsTopDown = ref(false)

vi.mock('./artifacts', () => ({
  useArtifactsStore: () => ({
    get displayArtifact() {
      return mockDisplayArtifact.value
    },
    get topDownDisplayArtifact() {
      return mockTopDownArtifact.value
    },
    get topicClustersDoc() {
      return null
    },
  }),
}))

vi.mock('./graphLoadMode', () => ({
  useGraphLoadModeStore: () => ({
    get isTopDown() {
      return mockIsTopDown.value
    },
  }),
}))

vi.mock('./graphNavigation', () => ({
  useGraphNavigationStore: () => ({ trailNodeIds: [] }),
}))

import { useGraphFilterStore } from './graphFilters'

function parsedGi(nodes: unknown[]): ParsedArtifact {
  return parseArtifact('ep1.gi.json', {
    episode_id: 'ep1',
    model_version: 'gpt-4o',
    prompt_version: 'v2',
    nodes,
    edges: [],
  })
}

function parsedTopDown(): ParsedArtifact {
  return parseArtifact('topdown.gi.json', {
    episode_id: 'ep1',
    model_version: 'gpt-4o',
    prompt_version: 'v2',
    nodes: [
      { id: 'sth:health', type: 'SuperTheme', properties: { label: 'Health' } },
      { id: 'sth:gear', type: 'SuperTheme', properties: { label: 'Gear' } },
    ],
    edges: [],
  })
}

const displayArt = () =>
  parsedGi([
    { id: 'episode:ep1', type: 'Episode', properties: { title: 'Ep 1' } },
    { id: 'topic:business-markets', type: 'Topic', properties: { label: 'Business' } },
    { id: 'i1', type: 'Insight', properties: { text: 'A', episode_id: 'ep1' } },
  ])

describe('useGraphFilterStore', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    mockDisplayArtifact.value = null
    mockTopDownArtifact.value = null
    mockIsTopDown.value = false
  })

  describe('fullArtifact routing (tier 8-1)', () => {
    it('returns displayArtifact when NOT in top-down mode', () => {
      mockDisplayArtifact.value = displayArt()
      mockTopDownArtifact.value = parsedTopDown()
      mockIsTopDown.value = false
      const s = useGraphFilterStore()
      // displayArtifact has an Episode; topDown does not.
      expect(s.state?.allowedTypes.Episode).toBe(true)
      expect(s.state?.allowedTypes.SuperTheme).toBeUndefined()
    })

    it('returns topDownDisplayArtifact when in top-down mode AND slice is available', () => {
      mockDisplayArtifact.value = displayArt()
      mockTopDownArtifact.value = parsedTopDown()
      mockIsTopDown.value = true
      const s = useGraphFilterStore()
      // top-down slice has SuperTheme; the display Episode should be absent
      expect(s.state?.allowedTypes.SuperTheme).toBe(true)
      expect(s.state?.allowedTypes.Episode).toBeUndefined()
    })

    it('falls back to displayArtifact when top-down mode is on but the slice is null', () => {
      mockDisplayArtifact.value = displayArt()
      mockTopDownArtifact.value = null
      mockIsTopDown.value = true
      const s = useGraphFilterStore()
      expect(s.state?.allowedTypes.Episode).toBe(true)
      expect(s.state?.allowedTypes.SuperTheme).toBeUndefined()
    })
  })

  describe('filteredArtifact SuperTheme force-through (tier 8-4)', () => {
    it('forces allowedTypes.SuperTheme = true in top-down mode even if the user toggled it off', () => {
      mockDisplayArtifact.value = displayArt()
      mockTopDownArtifact.value = parsedTopDown()
      mockIsTopDown.value = true
      const s = useGraphFilterStore()
      // User "disables" SuperTheme — the slice must NOT be emptied.
      if (s.state) s.state.allowedTypes.SuperTheme = false
      const filtered = s.filteredArtifact
      expect(filtered).not.toBeNull()
      const nodeTypes = filtered?.data.nodes.map((n) => n.type) ?? []
      // SuperTheme still present despite the user toggling it off.
      expect(nodeTypes).toContain('SuperTheme')
    })

    it('respects the user toggle for OTHER types (Topic) even in top-down mode', () => {
      const td = parseArtifact('topdown-with-topic.gi.json', {
        episode_id: 'ep1',
        model_version: 'gpt-4o',
        prompt_version: 'v2',
        nodes: [
          { id: 'sth:health', type: 'SuperTheme', properties: { label: 'Health' } },
          { id: 'topic:x', type: 'Topic', properties: { label: 'X' } },
        ],
        edges: [],
      })
      mockDisplayArtifact.value = displayArt()
      mockTopDownArtifact.value = td
      mockIsTopDown.value = true
      const s = useGraphFilterStore()
      // Disable Topic — force-through only pins SuperTheme, so Topic should drop.
      if (s.state) s.state.allowedTypes.Topic = false
      const filtered = s.filteredArtifact
      const nodeTypes = filtered?.data.nodes.map((n) => n.type) ?? []
      expect(nodeTypes).toContain('SuperTheme')
      expect(nodeTypes).not.toContain('Topic')
    })

    it('does NOT force SuperTheme through when NOT in top-down mode (respect user filter)', () => {
      const withSuperTheme = parseArtifact('mixed.gi.json', {
        episode_id: 'ep1',
        model_version: 'gpt-4o',
        prompt_version: 'v2',
        nodes: [
          { id: 'sth:x', type: 'SuperTheme', properties: { label: 'X' } },
          { id: 'topic:x', type: 'Topic', properties: { label: 'X' } },
        ],
        edges: [],
      })
      mockDisplayArtifact.value = withSuperTheme
      mockTopDownArtifact.value = null
      mockIsTopDown.value = false
      const s = useGraphFilterStore()
      if (s.state) s.state.allowedTypes.SuperTheme = false
      const filtered = s.filteredArtifact
      const nodeTypes = filtered?.data.nodes.map((n) => n.type) ?? []
      // SuperTheme respects the user's off toggle when not in top-down mode.
      expect(nodeTypes).not.toContain('SuperTheme')
    })
  })
})
