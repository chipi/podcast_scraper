// @vitest-environment happy-dom
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useSubjectStore } from './subject'
import { useGraphNavigationStore } from './graphNavigation'

describe('useSubjectStore', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('focusEpisode sets kind and clears graph fields', () => {
    const s = useSubjectStore()
    s.focusGraphNode('topic:x')
    s.focusEpisode('metadata/a.json')
    expect(s.kind).toBe('episode')
    expect(s.episodeMetadataPath).toBe('metadata/a.json')
    expect(s.graphNodeCyId).toBeNull()
  })

  it('focusGraphNode sets kind and clears episode path', () => {
    const s = useSubjectStore()
    s.focusEpisode('metadata/b.json')
    s.focusGraphNode('insight:y')
    expect(s.kind).toBe('graph-node')
    expect(s.graphNodeCyId).toBe('insight:y')
    expect(s.episodeMetadataPath).toBeNull()
  })

  it('clearSubject resets all', () => {
    const s = useSubjectStore()
    s.focusEpisode('m.json', { graphConnectionsCyId: 'topic:z' })
    s.clearSubject()
    expect(s.kind).toBeNull()
    expect(s.episodeMetadataPath).toBeNull()
    expect(s.graphConnectionsCyId).toBeNull()
  })

  it('rejects blank paths as clear', () => {
    const s = useSubjectStore()
    s.focusEpisode('ok.json')
    s.focusEpisode('  ')
    expect(s.kind).toBeNull()
  })

  it('stores optional episode UI label from focusEpisode opts', () => {
    const s = useSubjectStore()
    s.focusEpisode('metadata/a.json', { uiTitle: 'Hello world' })
    expect(s.episodeUiLabel).toBe('Hello world')
  })

  it('focusEpisode same path updates graph anchor without clearing metadata path', () => {
    const s = useSubjectStore()
    s.focusEpisode('metadata/a.json', { uiTitle: 'T1' })
    expect(s.episodeMetadataPath).toBe('metadata/a.json')
    s.focusEpisode('metadata/a.json', {
      graphConnectionsCyId: 'g:episode:x',
      uiTitle: 'T2',
    })
    expect(s.episodeMetadataPath).toBe('metadata/a.json')
    expect(s.graphConnectionsCyId).toBe('g:episode:x')
    expect(s.episodeUiLabel).toBe('T2')
  })

  // H1.6 follow-up — Episode panel knew the UUID but
  // ``focusEpisode`` was called without ``opts.episodeId``, nulling the
  // field. Lock in that opts.episodeId is preserved when supplied AND that
  // re-focusing the same path with episodeId updates it without clearing.
  it('focusEpisode preserves opts.episodeId on initial focus', () => {
    const s = useSubjectStore()
    s.focusEpisode('metadata/a.json', {
      uiTitle: 'T',
      episodeId: 'ep-uuid-123',
    })
    expect(s.kind).toBe('episode')
    expect(s.episodeMetadataPath).toBe('metadata/a.json')
    expect(s.episodeId).toBe('ep-uuid-123')
  })

  it('focusEpisode same path with episodeId updates without nulling', () => {
    const s = useSubjectStore()
    s.focusEpisode('metadata/a.json', { episodeId: 'ep-uuid-1' })
    s.focusEpisode('metadata/a.json', {
      graphConnectionsCyId: 'g:episode:x',
      episodeId: 'ep-uuid-1',
    })
    expect(s.episodeId).toBe('ep-uuid-1')
    expect(s.graphConnectionsCyId).toBe('g:episode:x')
  })

  // --- focusEpisode: trimming, opts normalization, branch coverage --------

  it('focusEpisode trims the metadata path before storing', () => {
    const s = useSubjectStore()
    s.focusEpisode('  metadata/a.json  ')
    expect(s.kind).toBe('episode')
    expect(s.episodeMetadataPath).toBe('metadata/a.json')
  })

  it('focusEpisode with no opts leaves optional fields null', () => {
    const s = useSubjectStore()
    s.focusEpisode('metadata/a.json')
    expect(s.graphConnectionsCyId).toBeNull()
    expect(s.episodeUiLabel).toBeNull()
    expect(s.episodeId).toBeNull()
  })

  it('focusEpisode treats whitespace-only opts as null', () => {
    const s = useSubjectStore()
    s.focusEpisode('metadata/a.json', {
      graphConnectionsCyId: '   ',
      uiTitle: '   ',
      episodeId: '   ',
    })
    expect(s.graphConnectionsCyId).toBeNull()
    expect(s.episodeUiLabel).toBeNull()
    expect(s.episodeId).toBeNull()
  })

  it('focusEpisode treats explicit null opts as null', () => {
    const s = useSubjectStore()
    s.focusEpisode('metadata/a.json', {
      graphConnectionsCyId: null,
      uiTitle: null,
      episodeId: null,
    })
    expect(s.graphConnectionsCyId).toBeNull()
    expect(s.episodeUiLabel).toBeNull()
    expect(s.episodeId).toBeNull()
  })

  it('focusEpisode trims a long uiTitle to 72 chars with an ellipsis', () => {
    const s = useSubjectStore()
    const long = 'x'.repeat(100)
    s.focusEpisode('metadata/a.json', { uiTitle: long })
    expect(s.episodeUiLabel).toHaveLength(72)
    expect(s.episodeUiLabel?.endsWith('…')).toBe(true)
    expect(s.episodeUiLabel).toBe(`${'x'.repeat(71)}…`)
  })

  it('focusEpisode keeps a uiTitle of exactly 72 chars untruncated', () => {
    const s = useSubjectStore()
    const exact = 'y'.repeat(72)
    s.focusEpisode('metadata/a.json', { uiTitle: exact })
    expect(s.episodeUiLabel).toBe(exact)
  })

  it('focusEpisode trims surrounding whitespace off a uiTitle', () => {
    const s = useSubjectStore()
    s.focusEpisode('metadata/a.json', { uiTitle: '  Trimmed  ' })
    expect(s.episodeUiLabel).toBe('Trimmed')
  })

  it('blank focusEpisode clears all previously set fields', () => {
    const s = useSubjectStore()
    s.focusEpisode('metadata/a.json', {
      graphConnectionsCyId: 'g:x',
      uiTitle: 'T',
      episodeId: 'ep-1',
    })
    s.focusEpisode('')
    expect(s.kind).toBeNull()
    expect(s.episodeMetadataPath).toBeNull()
    expect(s.episodeUiLabel).toBeNull()
    expect(s.episodeId).toBeNull()
    expect(s.graphConnectionsCyId).toBeNull()
  })

  it('focusEpisode to a different path clears prior graph/topic/person', () => {
    const s = useSubjectStore()
    s.focusEpisode('metadata/a.json', { episodeId: 'ep-1' })
    s.focusEpisode('metadata/b.json')
    expect(s.episodeMetadataPath).toBe('metadata/b.json')
    expect(s.episodeId).toBeNull()
  })

  it('focusEpisode same path clears stale graphNode/topic/person, keeps path', () => {
    const s = useSubjectStore()
    s.focusEpisode('metadata/a.json')
    // Simulate stale cross-kind ids on the same-episode re-focus branch.
    s.focusGraphNode('node:1')
    s.focusEpisode('metadata/a.json') // different kind now → not sameEpisode
    s.focusTopic('topic:1')
    s.focusEpisode('metadata/a.json') // again different kind → full clear
    expect(s.episodeMetadataPath).toBe('metadata/a.json')
    expect(s.graphNodeCyId).toBeNull()
    expect(s.topicId).toBeNull()
    expect(s.personId).toBeNull()
  })

  it('focusEpisode same path (sameEpisode branch) nulls stale cross-kind ids but keeps path/episodeId', () => {
    const s = useSubjectStore()
    s.focusEpisode('metadata/a.json', { episodeId: 'ep-1' })
    // Inject stale cross-kind ids directly (returned refs are writable);
    // this reproduces the real-world state where the rail still holds a
    // graph/topic/person id when "Open in graph" re-focuses the same episode.
    s.graphNodeCyId = 'stale-node'
    s.topicId = 'stale-topic'
    s.personId = 'stale-person'
    s.focusEpisode('metadata/a.json', { graphConnectionsCyId: 'g:x' })
    expect(s.episodeMetadataPath).toBe('metadata/a.json')
    // episodeId is re-derived from opts at the end of focusEpisode, so an
    // absent opts.episodeId nulls it even on the same-episode branch.
    expect(s.episodeId).toBeNull()
    expect(s.graphNodeCyId).toBeNull()
    expect(s.topicId).toBeNull()
    expect(s.personId).toBeNull()
    expect(s.graphConnectionsCyId).toBe('g:x')
  })

  it('focusEpisode same path is not "same" when prior kind is not episode', () => {
    const s = useSubjectStore()
    s.focusTopic('topic:keep?')
    // kind is 'topic', path null → sameEpisode false → full clear path taken.
    s.focusEpisode('metadata/a.json')
    expect(s.kind).toBe('episode')
    expect(s.topicId).toBeNull()
  })

  // --- focusGraphNode -----------------------------------------------------

  it('focusGraphNode trims the cy node id', () => {
    const s = useSubjectStore()
    s.focusGraphNode('  node:1  ')
    expect(s.kind).toBe('graph-node')
    expect(s.graphNodeCyId).toBe('node:1')
  })

  it('focusGraphNode with blank id clears and nulls kind', () => {
    const s = useSubjectStore()
    s.focusEpisode('metadata/a.json', { episodeId: 'ep-1' })
    s.focusGraphNode('   ')
    expect(s.kind).toBeNull()
    expect(s.graphNodeCyId).toBeNull()
    expect(s.episodeMetadataPath).toBeNull()
    expect(s.episodeId).toBeNull()
  })

  // --- focusTopic / focusEntity -------------------------------------------

  it('focusTopic focuses the topic as a graph node (unified node view)', () => {
    const s = useSubjectStore()
    s.focusEpisode('metadata/a.json', { episodeId: 'ep-1' })
    s.focusTopic('  topic:42  ')
    expect(s.kind).toBe('graph-node')
    expect(s.graphNodeCyId).toBe('topic:42')
    expect(s.topicId).toBeNull()
    expect(s.episodeMetadataPath).toBeNull()
    expect(s.episodeId).toBeNull()
  })

  it('focusTopic with blank id clears and nulls kind', () => {
    const s = useSubjectStore()
    s.focusTopic('topic:1')
    s.focusTopic('   ')
    expect(s.kind).toBeNull()
    expect(s.topicId).toBeNull()
  })

  it('focusEntity focuses the entity as a graph node (unified node view)', () => {
    const s = useSubjectStore()
    s.focusEntity('  entity:7  ')
    expect(s.kind).toBe('graph-node')
    expect(s.graphNodeCyId).toBe('entity:7')
    expect(s.topicId).toBeNull()
  })

  it('focusEntity with blank id clears', () => {
    const s = useSubjectStore()
    s.focusEntity('entity:1')
    s.focusEntity('')
    expect(s.kind).toBeNull()
    expect(s.topicId).toBeNull()
  })

  // --- focusPerson --------------------------------------------------------

  it('focusPerson sets kind and personId, clears other fields', () => {
    const s = useSubjectStore()
    s.focusTopic('topic:1')
    s.focusPerson('  person:9  ')
    expect(s.kind).toBe('graph-node')
    expect(s.graphNodeCyId).toBe('person:9')
    expect(s.topicId).toBeNull()
  })

  it('focusPerson with blank id clears and nulls kind', () => {
    const s = useSubjectStore()
    s.focusPerson('person:1')
    s.focusPerson('   ')
    expect(s.kind).toBeNull()
    expect(s.personId).toBeNull()
  })

  // --- kind transitions ---------------------------------------------------

  it('cycles through all kinds keeping only the active id set', () => {
    const s = useSubjectStore()
    s.focusEpisode('metadata/a.json', { episodeId: 'ep-1' })
    expect(s.kind).toBe('episode')

    s.focusGraphNode('node:1')
    expect(s.kind).toBe('graph-node')
    expect(s.episodeMetadataPath).toBeNull()
    expect(s.episodeId).toBeNull()

    s.focusTopic('topic:1')
    expect(s.kind).toBe('graph-node')
    expect(s.graphNodeCyId).toBe('topic:1')

    s.focusPerson('person:1')
    expect(s.kind).toBe('graph-node')
    expect(s.graphNodeCyId).toBe('person:1')

    s.clearSubject()
    expect(s.kind).toBeNull()
    expect(s.personId).toBeNull()
  })

  // --- setEpisodeId -------------------------------------------------------

  it('setEpisodeId stores a trimmed value', () => {
    const s = useSubjectStore()
    s.setEpisodeId('  ep-uuid-5  ')
    expect(s.episodeId).toBe('ep-uuid-5')
  })

  it('setEpisodeId with whitespace-only nulls the field', () => {
    const s = useSubjectStore()
    s.setEpisodeId('ep-1')
    s.setEpisodeId('   ')
    expect(s.episodeId).toBeNull()
  })

  it('setEpisodeId with null nulls the field', () => {
    const s = useSubjectStore()
    s.setEpisodeId('ep-1')
    s.setEpisodeId(null)
    expect(s.episodeId).toBeNull()
  })

  // --- setEpisodeUiLabel --------------------------------------------------

  it('setEpisodeUiLabel stores a trimmed label', () => {
    const s = useSubjectStore()
    s.setEpisodeUiLabel('  Label  ')
    expect(s.episodeUiLabel).toBe('Label')
  })

  it('setEpisodeUiLabel truncates a long label to 72 chars', () => {
    const s = useSubjectStore()
    s.setEpisodeUiLabel('z'.repeat(100))
    expect(s.episodeUiLabel).toHaveLength(72)
    expect(s.episodeUiLabel?.endsWith('…')).toBe(true)
  })

  it('setEpisodeUiLabel with whitespace-only nulls the label', () => {
    const s = useSubjectStore()
    s.setEpisodeUiLabel('something')
    s.setEpisodeUiLabel('   ')
    expect(s.episodeUiLabel).toBeNull()
  })

  it('setEpisodeUiLabel with null nulls the label', () => {
    const s = useSubjectStore()
    s.setEpisodeUiLabel('something')
    s.setEpisodeUiLabel(null)
    expect(s.episodeUiLabel).toBeNull()
  })

  // --- clearSubject -------------------------------------------------------

  it('clearSubject from a fresh store is a no-op', () => {
    const s = useSubjectStore()
    s.clearSubject()
    expect(s.kind).toBeNull()
    expect(s.episodeMetadataPath).toBeNull()
    expect(s.topicId).toBeNull()
    expect(s.personId).toBeNull()
    expect(s.graphNodeCyId).toBeNull()
  })
})

// The window.__GIKG_SUBJECT__ dev hook is gated behind import.meta.env.DEV.
// Stub DEV true and (re)import the store fresh so the gated getters/mutators
// register, then exercise them to lock the E2E correctness-assertion contract.
describe('useSubjectStore — DEV window hook (__GIKG_SUBJECT__)', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.stubEnv('DEV', true)
    vi.resetModules()
    delete (window as unknown as { __GIKG_SUBJECT__?: object }).__GIKG_SUBJECT__
  })

  afterEach(() => {
    vi.unstubAllEnvs()
    delete (window as unknown as { __GIKG_SUBJECT__?: object }).__GIKG_SUBJECT__
  })

  type DevHook = {
    kind: unknown
    episodeMetadataPath: unknown
    episodeId: unknown
    graphNodeCyId: unknown
    topicId: unknown
    personId: unknown
    focusTopic: (id: string) => void
    focusEntity: (id: string) => void
    clearSubject: () => void
  }

  const getHook = (): DevHook =>
    (window as unknown as { __GIKG_SUBJECT__: DevHook }).__GIKG_SUBJECT__

  it('registers the hook on window under DEV', async () => {
    const { useSubjectStore: useFresh } = await import('./subject')
    useFresh()
    expect(getHook()).toBeTruthy()
  })

  it('hook getters reflect store state across all kinds', async () => {
    const { useSubjectStore: useFresh } = await import('./subject')
    const s = useFresh()
    const hook = getHook()

    s.focusEpisode('metadata/a.json', { episodeId: 'ep-1' })
    expect(hook.kind).toBe('episode')
    expect(hook.episodeMetadataPath).toBe('metadata/a.json')
    expect(hook.episodeId).toBe('ep-1')

    s.focusGraphNode('node:1')
    expect(hook.graphNodeCyId).toBe('node:1')

    s.focusTopic('topic:1')
    expect(hook.graphNodeCyId).toBe('topic:1')

    s.focusPerson('person:1')
    expect(hook.graphNodeCyId).toBe('person:1')
  })

  it('hook mutators (focusTopic/focusEntity/clearSubject) drive the store', async () => {
    const { useSubjectStore: useFresh } = await import('./subject')
    const s = useFresh()
    const hook = getHook()

    hook.focusTopic('topic:hooked')
    expect(s.kind).toBe('graph-node')
    expect(s.graphNodeCyId).toBe('topic:hooked')

    hook.focusEntity('entity:hooked')
    expect(s.kind).toBe('graph-node')
    expect(s.graphNodeCyId).toBe('entity:hooked')

    hook.clearSubject()
    expect(s.kind).toBeNull()
    expect(s.graphNodeCyId).toBeNull()
  })
})

// #1049 — positionTrackerTopicId state lifecycle.
describe('useSubjectStore — positionTrackerTopicId (#1049)', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('selectTopicForPositionTracker sets the topic when a Person is focused', () => {
    const s = useSubjectStore()
    s.focusPerson('person:alice')
    s.selectTopicForPositionTracker('topic:ai')
    expect(s.positionTrackerTopicId).toBe('topic:ai')
  })

  it('selectTopicForPositionTracker is a no-op when no Person is focused', () => {
    const s = useSubjectStore()
    s.focusEpisode('metadata/a.json')
    s.selectTopicForPositionTracker('topic:ai')
    expect(s.positionTrackerTopicId).toBeNull()
  })

  it('selectTopicForPositionTracker clears on empty / whitespace input', () => {
    const s = useSubjectStore()
    s.focusPerson('person:alice')
    s.selectTopicForPositionTracker('topic:ai')
    s.selectTopicForPositionTracker('   ')
    expect(s.positionTrackerTopicId).toBeNull()
  })

  it('clearPositionTrackerTopic resets the topic without dropping the Person', () => {
    const s = useSubjectStore()
    s.focusPerson('person:alice')
    s.selectTopicForPositionTracker('topic:ai')
    s.clearPositionTrackerTopic()
    expect(s.positionTrackerTopicId).toBeNull()
    expect(s.kind).toBe('graph-node')
    expect(s.graphNodeCyId).toBe('person:alice')
  })

  it('focusPerson(new person) clears any stale Position Tracker topic', () => {
    const s = useSubjectStore()
    s.focusPerson('person:alice')
    s.selectTopicForPositionTracker('topic:ai')
    s.focusPerson('person:bob')
    expect(s.positionTrackerTopicId).toBeNull()
    expect(s.graphNodeCyId).toBe('person:bob')
  })

  it('clearSubject also clears the Position Tracker topic', () => {
    const s = useSubjectStore()
    s.focusPerson('person:alice')
    s.selectTopicForPositionTracker('topic:ai')
    s.clearSubject()
    expect(s.positionTrackerTopicId).toBeNull()
    expect(s.kind).toBeNull()
  })

  it('focusEpisode (re-open same episode) does not leak a prior Position Tracker topic', () => {
    const s = useSubjectStore()
    // Land on an episode first so the "sameEpisode" branch is exercised below.
    s.focusEpisode('metadata/a.json')
    // Now focus a Person, select a Topic for Position Tracker.
    s.focusPerson('person:alice')
    s.selectTopicForPositionTracker('topic:ai')
    // Re-open the same episode. The else-branch in focusEpisode used to skip
    // positionTrackerTopicId — fix #1049-FU-review verifies it clears now.
    s.focusEpisode('metadata/a.json')
    expect(s.positionTrackerTopicId).toBeNull()
    expect(s.kind).toBe('episode')
  })
})

// Back navigation — node→node chains (topic → entity → person → co-speaker)
// push the prior subject so the rail Back affordance can return to it.
describe('useSubjectStore — Back navigation history', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('starts with empty history; the first node focus records nothing', () => {
    const s = useSubjectStore()
    expect(s.canGoBack).toBe(false)
    s.focusTopic('topic:1')
    expect(s.canGoBack).toBe(false)
  })

  it('node→node navigation records history and back() restores the previous node', () => {
    const s = useSubjectStore()
    s.focusTopic('topic:geopolitics')
    s.focusEntity('entity:strait-of-hormuz')
    expect(s.canGoBack).toBe(true)
    expect(s.graphNodeCyId).toBe('entity:strait-of-hormuz')
    s.back()
    expect(s.kind).toBe('graph-node')
    expect(s.graphNodeCyId).toBe('topic:geopolitics')
    expect(s.canGoBack).toBe(false)
  })

  it('back() through a multi-step chain returns nodes in reverse order', () => {
    const s = useSubjectStore()
    s.focusTopic('topic:a')
    s.focusEntity('entity:b')
    s.focusPerson('person:c')
    expect(s.graphNodeCyId).toBe('person:c')
    s.back()
    expect(s.graphNodeCyId).toBe('entity:b')
    s.back()
    expect(s.graphNodeCyId).toBe('topic:a')
    expect(s.canGoBack).toBe(false)
  })

  it('re-focusing the same node does not push history', () => {
    const s = useSubjectStore()
    s.focusTopic('topic:1')
    s.focusGraphNode('topic:1')
    expect(s.canGoBack).toBe(false)
  })

  it('back() with empty history is a no-op', () => {
    const s = useSubjectStore()
    s.focusTopic('topic:1')
    s.back()
    expect(s.kind).toBe('graph-node')
    expect(s.graphNodeCyId).toBe('topic:1')
  })

  it('clearSubject drops the Back history', () => {
    const s = useSubjectStore()
    s.focusTopic('topic:1')
    s.focusPerson('person:2')
    expect(s.canGoBack).toBe(true)
    s.clearSubject()
    expect(s.canGoBack).toBe(false)
  })

  it('back() restores the Position Tracker topic captured in the snapshot', () => {
    const s = useSubjectStore()
    s.focusPerson('person:alice')
    s.selectTopicForPositionTracker('topic:ai')
    s.focusPerson('person:bob')
    s.back()
    expect(s.graphNodeCyId).toBe('person:alice')
    expect(s.positionTrackerTopicId).toBe('topic:ai')
  })

  it('focusTopic/focusPerson/focusEntity request a graph focus — 2-way sync (#6)', () => {
    const s = useSubjectStore()
    const nav = useGraphNavigationStore()
    s.focusTopic('topic:ai')
    expect(nav.pendingFocusNodeId).toBe('topic:ai')
    s.focusPerson('person:alice')
    expect(nav.pendingFocusNodeId).toBe('person:alice')
    s.focusEntity('org:acme')
    expect(nav.pendingFocusNodeId).toBe('org:acme')
  })

  it('a direct focusGraphNode (a graph click) does NOT re-request a graph focus (#6)', () => {
    const s = useSubjectStore()
    const nav = useGraphNavigationStore()
    s.focusGraphNode('topic:ai') // graph-originated → the node is already selected there
    expect(nav.pendingFocusNodeId).toBeNull()
  })
})
