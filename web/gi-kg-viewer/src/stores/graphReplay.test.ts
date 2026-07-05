// @vitest-environment happy-dom
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useGraphReplayStore } from './graphReplay'
import { useGraphNavigationStore } from './graphNavigation'

const SESSION = [
  { action: 'graph_node_tap', id: 'topic:a' },
  { action: 'graph_rail_nav', to_id: 'person:b' },
  { action: 'graph_rail_nav', to_id: 'topic:c' },
  { action: 'graph_recenter', target_id: 'topic:z' }, // resets the trail
  { action: 'graph_rail_nav', to_id: 'person:d' },
]

describe('useGraphReplayStore', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.useFakeTimers()
  })
  afterEach(() => vi.useRealTimers())

  it('load flips inactive → active at step 0', () => {
    const r = useGraphReplayStore()
    expect(r.active).toBe(false)
    r.load('s1', SESSION)
    expect(r.active).toBe(true)
    expect(r.sessionId).toBe('s1')
    expect(r.step).toBe(0)
    expect(r.total).toBe(5)
  })

  it('reconstructs the trail + focus as you step (reuses the D#6 trail)', () => {
    const r = useGraphReplayStore()
    const nav = useGraphNavigationStore()
    r.load('s1', SESSION)
    r.setStep(3) // tap a, nav b, nav c
    expect(nav.trailNodeIds).toEqual(['person:b', 'topic:c'])
    expect(nav.pendingFocusNodeId).toBe('topic:c')
    r.setStep(5) // + re-centre z (resets trail) + nav d
    expect(nav.trailNodeIds).toEqual(['person:d'])
    expect(nav.pendingFocusNodeId).toBe('person:d')
  })

  it('scrubbing back rewinds deterministically', () => {
    const r = useGraphReplayStore()
    const nav = useGraphNavigationStore()
    r.load('s1', SESSION)
    r.setStep(5)
    r.setStep(2) // tap a, nav b
    expect(nav.trailNodeIds).toEqual(['person:b'])
  })

  it('timed play advances then stops at the end', () => {
    const r = useGraphReplayStore()
    r.load('s1', SESSION)
    r.play()
    expect(r.playing).toBe(true)
    vi.advanceTimersByTime(800 * 6)
    expect(r.step).toBe(5)
    expect(r.playing).toBe(false) // auto-stopped at the end
  })

  it('exit clears the session + trail', () => {
    const r = useGraphReplayStore()
    const nav = useGraphNavigationStore()
    r.load('s1', SESSION)
    r.setStep(3)
    r.exit()
    expect(r.active).toBe(false)
    expect(nav.trailNodeIds).toEqual([])
  })
})
