// @vitest-environment happy-dom
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import { useGraphNavigationStore } from './graphNavigation'

describe('useGraphNavigationStore', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('starts empty', () => {
    const s = useGraphNavigationStore()
    expect(s.pendingFocusNodeId).toBeNull()
    expect(s.pendingFocusFallbackNodeId).toBeNull()
    expect(s.libraryHighlightSourceIds).toEqual([])
    expect(s.pendingFocusCameraIncludeRawIds).toEqual([])
    expect(s.graphEgoFocusCyId).toBeNull()
    expect(s.requestFitAfterLoad).toBe(false)
    expect(s.topicClusterCanvasCollapsedIds).toEqual([])
  })

  describe('requestFocusNode', () => {
    it('sets the trimmed primary id', () => {
      const s = useGraphNavigationStore()
      s.requestFocusNode('  node:a  ')
      expect(s.pendingFocusNodeId).toBe('node:a')
      expect(s.pendingFocusFallbackNodeId).toBeNull()
      expect(s.pendingFocusCameraIncludeRawIds).toEqual([])
    })

    it('maps an empty/whitespace primary id to null', () => {
      const s = useGraphNavigationStore()
      s.requestFocusNode('   ')
      expect(s.pendingFocusNodeId).toBeNull()
    })

    it('records a trimmed fallback when provided', () => {
      const s = useGraphNavigationStore()
      s.requestFocusNode('node:a', '  ep:1 ')
      expect(s.pendingFocusFallbackNodeId).toBe('ep:1')
    })

    it('treats an empty-string fallback as null', () => {
      const s = useGraphNavigationStore()
      s.requestFocusNode('node:a', '   ')
      expect(s.pendingFocusFallbackNodeId).toBeNull()
    })

    it('treats an explicit null fallback as null', () => {
      const s = useGraphNavigationStore()
      s.requestFocusNode('node:a', null)
      expect(s.pendingFocusFallbackNodeId).toBeNull()
    })

    it('trims + filters cameraIncludeRawIds', () => {
      const s = useGraphNavigationStore()
      s.requestFocusNode('node:a', null, ['  tc:1 ', '', '   ', 'tc:2'])
      expect(s.pendingFocusCameraIncludeRawIds).toEqual(['tc:1', 'tc:2'])
    })

    it('treats null cameraIncludeRawIds as empty', () => {
      const s = useGraphNavigationStore()
      s.requestFocusNode('node:a', null, null)
      expect(s.pendingFocusCameraIncludeRawIds).toEqual([])
    })

    it('clears prior focus first so a repeat request still notifies', () => {
      const s = useGraphNavigationStore()
      s.requestFocusNode('node:a', 'ep:1', ['tc:1'])
      // A second call for a different node resets fallback + camera ids.
      s.requestFocusNode('node:b')
      expect(s.pendingFocusNodeId).toBe('node:b')
      expect(s.pendingFocusFallbackNodeId).toBeNull()
      expect(s.pendingFocusCameraIncludeRawIds).toEqual([])
    })
  })

  it('clearPendingFocus resets all three pending fields', () => {
    const s = useGraphNavigationStore()
    s.requestFocusNode('node:a', 'ep:1', ['tc:1'])
    s.clearPendingFocus()
    expect(s.pendingFocusNodeId).toBeNull()
    expect(s.pendingFocusFallbackNodeId).toBeNull()
    expect(s.pendingFocusCameraIncludeRawIds).toEqual([])
  })

  describe('library episode highlights', () => {
    it('sets trimmed, non-empty ids', () => {
      const s = useGraphNavigationStore()
      s.setLibraryEpisodeHighlights([' ep:1 ', '', 'ep:2', '   '])
      expect(s.libraryHighlightSourceIds).toEqual(['ep:1', 'ep:2'])
    })

    it('clears highlights', () => {
      const s = useGraphNavigationStore()
      s.setLibraryEpisodeHighlights(['ep:1'])
      s.clearLibraryEpisodeHighlights()
      expect(s.libraryHighlightSourceIds).toEqual([])
    })
  })

  describe('setGraphEgoFocusCyId', () => {
    it('stores a trimmed id', () => {
      const s = useGraphNavigationStore()
      s.setGraphEgoFocusCyId('  cy-7 ')
      expect(s.graphEgoFocusCyId).toBe('cy-7')
    })

    it('maps empty/whitespace/null to null', () => {
      const s = useGraphNavigationStore()
      s.setGraphEgoFocusCyId('cy-7')
      s.setGraphEgoFocusCyId('   ')
      expect(s.graphEgoFocusCyId).toBeNull()
      s.setGraphEgoFocusCyId(null)
      expect(s.graphEgoFocusCyId).toBeNull()
    })
  })

  it('toggles requestFitAfterLoad on and off', () => {
    const s = useGraphNavigationStore()
    s.setRequestFitAfterLoad()
    expect(s.requestFitAfterLoad).toBe(true)
    s.clearRequestFitAfterLoad()
    expect(s.requestFitAfterLoad).toBe(false)
  })

  describe('topic-cluster canvas collapse', () => {
    it('toggles an id in and out of the collapsed set', () => {
      const s = useGraphNavigationStore()
      s.toggleTopicClusterCanvasCollapsed(' tc:1 ')
      expect(s.topicClusterCanvasCollapsedIds).toEqual(['tc:1'])
      expect(s.isTopicClusterCanvasCollapsed('tc:1')).toBe(true)
      s.toggleTopicClusterCanvasCollapsed('tc:1')
      expect(s.topicClusterCanvasCollapsedIds).toEqual([])
      expect(s.isTopicClusterCanvasCollapsed('tc:1')).toBe(false)
    })

    it('ignores an empty id', () => {
      const s = useGraphNavigationStore()
      s.toggleTopicClusterCanvasCollapsed('   ')
      expect(s.topicClusterCanvasCollapsedIds).toEqual([])
    })

    it('isTopicClusterCanvasCollapsed trims its argument', () => {
      const s = useGraphNavigationStore()
      s.toggleTopicClusterCanvasCollapsed('tc:1')
      expect(s.isTopicClusterCanvasCollapsed('  tc:1 ')).toBe(true)
    })

    it('clears all collapsed ids', () => {
      const s = useGraphNavigationStore()
      s.toggleTopicClusterCanvasCollapsed('tc:1')
      s.toggleTopicClusterCanvasCollapsed('tc:2')
      s.clearTopicClusterCanvasCollapsed()
      expect(s.topicClusterCanvasCollapsedIds).toEqual([])
    })
  })

  describe('#6 breadcrumb trail', () => {
    it('appends navigated nodes in order', () => {
      const s = useGraphNavigationStore()
      s.addToTrail('topic:a')
      s.addToTrail('person:b')
      expect(s.trailNodeIds).toEqual(['topic:a', 'person:b'])
    })

    it('LRU-touches a re-navigated node to the newest position', () => {
      const s = useGraphNavigationStore()
      s.addToTrail('a')
      s.addToTrail('b')
      s.addToTrail('a') // re-visit → moves to end
      expect(s.trailNodeIds).toEqual(['b', 'a'])
    })

    it('never exceeds the budget, pruning oldest first', () => {
      const s = useGraphNavigationStore()
      for (let i = 0; i < 40; i++) s.addToTrail(`n${i}`)
      expect(s.trailNodeIds.length).toBe(28)
      expect(s.trailNodeIds[0]).toBe('n12') // n0..n11 pruned
      expect(s.trailNodeIds[27]).toBe('n39')
    })

    it('ignores blanks and the pinned ego origin', () => {
      const s = useGraphNavigationStore()
      s.setGraphEgoFocusCyId('origin')
      s.addToTrail('  ')
      s.addToTrail('origin') // always in view → not a trail node
      s.addToTrail('topic:a')
      expect(s.trailNodeIds).toEqual(['topic:a'])
    })

    it('resets the trail when the ego origin changes', () => {
      const s = useGraphNavigationStore()
      s.setGraphEgoFocusCyId('origin1')
      s.addToTrail('a')
      s.addToTrail('b')
      s.setGraphEgoFocusCyId('origin2') // new origin → fresh trail
      expect(s.trailNodeIds).toEqual([])
    })

    it('clearTrail empties it', () => {
      const s = useGraphNavigationStore()
      s.addToTrail('a')
      s.clearTrail()
      expect(s.trailNodeIds).toEqual([])
    })
  })
})
