import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('posthog-js', () => ({ default: { capture: vi.fn() } }))

import { useRecentActivityStore } from './recentActivity'

describe('useRecentActivityStore (#1259-3)', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  afterEach(() => vi.restoreAllMocks())

  it('pushSubject dedupes on kind + id and moves the entry to the front', async () => {
    const store = useRecentActivityStore()
    await store.pushSubject({ kind: 'topic', id: 'topic:a', label: 'A' })
    await store.pushSubject({ kind: 'topic', id: 'topic:b', label: 'B' })
    await store.pushSubject({ kind: 'topic', id: 'topic:a', label: 'A (updated)' })
    const list = store.listRecentSubjects()
    expect(list.map((e) => e.id)).toEqual(['topic:a', 'topic:b'])
    expect(list[0].label).toBe('A (updated)')
  })

  it('pushSubject treats "topic:a" and "person:a" as distinct (different kind)', async () => {
    const store = useRecentActivityStore()
    await store.pushSubject({ kind: 'topic', id: 'a' })
    await store.pushSubject({ kind: 'person', id: 'a' })
    const list = store.listRecentSubjects()
    expect(list.length).toBe(2)
    expect(list.map((e) => `${e.kind}:${e.id}`).sort()).toEqual(['person:a', 'topic:a'])
  })

  it('pushSubject caps the ring buffer at 20', async () => {
    const store = useRecentActivityStore()
    for (let i = 0; i < 25; i += 1) {
      await store.pushSubject({ kind: 'topic', id: `topic:t${i}` })
    }
    const list = store.listRecentSubjects()
    expect(list).toHaveLength(20)
    // Newest first — most recently pushed at index 0.
    expect(list[0].id).toBe('topic:t24')
    expect(list[19].id).toBe('topic:t5')
  })

  it('pushSubject silently no-ops on blank id', async () => {
    const store = useRecentActivityStore()
    await store.pushSubject({ kind: 'topic', id: '   ' })
    expect(store.listRecentSubjects()).toEqual([])
  })

  it('pushHandoff dedupes on trail + target and moves the entry to the front', async () => {
    const store = useRecentActivityStore()
    await store.pushHandoff({ trail: 'Library → Graph', target: 'topic:x' })
    await store.pushHandoff({ trail: 'Library → Graph', target: 'topic:y' })
    await store.pushHandoff({ trail: 'Library → Graph', target: 'topic:x' })
    const list = store.listRecentHandoffs()
    expect(list.map((e) => e.target)).toEqual(['topic:x', 'topic:y'])
  })

  it('pushHandoff no-ops when trail or target is blank', async () => {
    const store = useRecentActivityStore()
    await store.pushHandoff({ trail: '', target: 'topic:x' })
    await store.pushHandoff({ trail: 'X', target: '   ' })
    expect(store.listRecentHandoffs()).toEqual([])
  })

  it('listRecentSubjects respects the optional limit', async () => {
    const store = useRecentActivityStore()
    for (let i = 0; i < 5; i += 1) {
      await store.pushSubject({ kind: 'topic', id: `topic:${i}` })
    }
    expect(store.listRecentSubjects(3)).toHaveLength(3)
    expect(store.listRecentSubjects(0)).toEqual([])
    expect(store.listRecentSubjects(999)).toHaveLength(5)
  })

  it('clearRecentSubjects wipes the ring', async () => {
    const store = useRecentActivityStore()
    await store.pushSubject({ kind: 'topic', id: 'topic:a' })
    await store.clearRecentSubjects()
    expect(store.listRecentSubjects()).toEqual([])
  })

  it('clearRecentHandoffs wipes the handoff ring', async () => {
    const store = useRecentActivityStore()
    await store.pushHandoff({ trail: 'X → Y', target: 'topic:a' })
    await store.clearRecentHandoffs()
    expect(store.listRecentHandoffs()).toEqual([])
  })
})
